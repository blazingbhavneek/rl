from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from algo.base import BaseAlgo
from backprop.base import BaseBackprop
from client.base import BaseClient, ClientContext
from tasksets.base import Problem, Score


@dataclass
class RolloutResult:
    problem: Problem
    completion_text: str
    token_ids: List[int]
    prompt_text: str
    prompt_token_ids: List[int]
    score: Score
    reward: float
    passed: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProblemRollouts:
    problem: Problem
    rollouts: List[RolloutResult]
    best_rollout: Optional[RolloutResult]
    any_passed: bool
    peer_solution: Optional[str]


@dataclass
class TeacherHintResult:
    problem: Problem
    failed_rollout: RolloutResult
    teacher_hint: str
    student_retry: RolloutResult
    passed_after_hint: bool
    attempt_number: int


@dataclass
class SFTPair:
    problem_id: str
    messages: List[Dict[str, Any]]
    source: str = "teacher_hint_sft"


@dataclass
class StepResult:
    rl_stats: Dict[str, float]
    sft_stats: Dict[str, float]
    n_problems: int
    n_passed_rl: int
    n_passed_after_hint: int
    step: int
    problem_rollouts: List[ProblemRollouts] = field(default_factory=list)


@dataclass
class RewardConfig:
    reward_compile: float = 1.0
    reward_per_test: float = 1.0
    reward_length_penalty: float = 0.01
    min_completion_tokens: int = 64
    reward_error_engage: float = 0.1


class BasePipeline(ABC):
    def __init__(
        self,
        student_client: BaseClient,
        verifier,
        algo: BaseAlgo,
        backprop: BaseBackprop,
        optimizer: torch.optim.Optimizer,
        scheduler,
        reward_config: RewardConfig,
        sft_output_dir: Optional[str],
        extensions: Optional[List[Any]] = None,
        max_grad_norm: float = 1.0,
    ) -> None:
        self.student_client = student_client
        self.verifier = verifier
        self.algo = algo
        self.backprop = backprop
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.reward_config = reward_config
        self.sft_output_dir = Path(sft_output_dir) if sft_output_dir else None
        self.extensions = list(extensions or [])
        self.max_grad_norm = float(max_grad_norm)

        if self.sft_output_dir is not None:
            self.sft_output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def run_step(self, problems: List[Problem], step: int) -> StepResult:
        ...

    def _rollout(self, problem: Problem, context: ClientContext, n: int) -> List[RolloutResult]:
        client_result = self.student_client.run(problem=problem, context=context, n=n)
        progress_cb = context.extra.get("_on_generation_done") if context.extra else None

        rollouts: List[RolloutResult] = []
        for idx, completion in enumerate(client_result.completions):
            score = self.verifier.verify(problem, completion)
            reward = self._compute_reward(score=score, completion_text=completion, context=context)
            passed = bool(score.total > 0 and score.passed == score.total)

            tok_ids = client_result.token_ids[idx] if idx < len(client_result.token_ids) else []
            rollouts.append(
                RolloutResult(
                    problem=problem,
                    completion_text=completion,
                    token_ids=list(tok_ids),
                    prompt_text=client_result.prompt_text,
                    prompt_token_ids=list(client_result.prompt_token_ids),
                    score=score,
                    reward=float(reward),
                    passed=passed,
                    metadata=dict(client_result.metadata),
                )
            )
            if callable(progress_cb):
                try:
                    progress_cb(idx + 1, len(client_result.completions), len(tok_ids))
                except Exception:
                    pass

        return rollouts

    def _compute_reward(
        self,
        score: Score,
        completion_text: str,
        context: ClientContext,
    ) -> float:
        r = 0.0
        if score.compiled:
            r += self.reward_config.reward_compile

        if score.total > 0:
            # Scale by pass ratio, not raw pass count, to keep reward bounded.
            r += self.reward_config.reward_per_test * (float(score.passed) / float(score.total))

        n_tokens = int(context.extra.get("completion_tokens", 0))
        if n_tokens <= 0:
            try:
                n_tokens = len(self.student_client.tokenizer.encode(completion_text, add_special_tokens=False))
            except Exception:
                n_tokens = len(completion_text.split())
        # Penalize suspiciously short completions (original behavior),
        # not long completions.
        if n_tokens < self.reward_config.min_completion_tokens:
            r -= self.reward_config.reward_length_penalty * (
                self.reward_config.min_completion_tokens - n_tokens
            )

        if context.pass_number >= 2 and score.error:
            r += self.reward_config.reward_error_engage

        return float(r)

    def _run_sft_phase(self, pairs: List[SFTPair]) -> Dict[str, float]:
        if not pairs:
            return {
                "n_pairs_trained": 0.0,
                "sft_loss": 0.0,
                "mean_sft_logp": 0.0,
            }

        device = next(self.backprop.model.parameters()).device
        self.optimizer.zero_grad(set_to_none=True)

        per_pair_stats: List[Dict[str, float]] = []
        total_pairs = len(pairs)
        for pair in pairs:
            if not pair.messages or len(pair.messages) < 2:
                continue
            if pair.messages[-1].get("role") != "assistant":
                continue

            prompt_messages = pair.messages[:-1]
            completion_text = pair.messages[-1].get("content", "")
            prompt_text = self.student_client.apply_chat_template(prompt_messages)

            prompt_token_ids = self.student_client.tokenizer.encode(
                prompt_text,
                add_special_tokens=False,
            )
            completion_token_ids = self.student_client.tokenizer.encode(
                completion_text,
                add_special_tokens=False,
            )

            if not completion_token_ids:
                continue

            prompt_ids = torch.tensor([prompt_token_ids], dtype=torch.long, device=device)
            completion_ids = torch.tensor([completion_token_ids], dtype=torch.long, device=device)
            completion_mask = torch.ones_like(completion_ids, dtype=torch.float32, device=device)

            def sft_loss_fn(log_probs: Tensor, gen_idx: int, hidden_comp: Optional[Tensor] = None) -> Tensor:
                del hidden_comp
                mask = completion_mask[gen_idx].to(log_probs.device, non_blocking=True)
                return -(log_probs * mask).sum() / mask.sum().clamp(min=1.0)

            bp_stats = self.backprop.backward_on_batch(
                model=self.backprop.model,
                prompt_ids=prompt_ids,
                completion_ids=completion_ids,
                completion_mask=completion_mask,
                loss_fn=sft_loss_fn,
                loss_scale=1.0 / max(1, total_pairs),
                lora_path=None,
            )
            per_pair_stats.append(
                {
                    "sft_loss": float(bp_stats.get("loss", 0.0)),
                    "mean_sft_logp": float(bp_stats.get("mean_logp", 0.0)),
                }
            )

        if not per_pair_stats:
            self.optimizer.zero_grad(set_to_none=True)
            return {
                "n_pairs_trained": 0.0,
                "sft_loss": 0.0,
                "mean_sft_logp": 0.0,
            }

        opt_stats = self._optimizer_step()
        sft_stats = self._mean_stats(per_pair_stats)
        sft_stats.update(opt_stats)
        sft_stats["n_pairs_trained"] = float(len(per_pair_stats))
        return sft_stats

    def _optimizer_step(self) -> Dict[str, float]:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.backprop.model.parameters(),
            self.max_grad_norm,
        )
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        return {"grad_norm": float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm)}

    def _save_sft_pair(self, pair: SFTPair) -> None:
        if self.sft_output_dir is None:
            return
        out_path = self.sft_output_dir / "sft_pairs.jsonl"
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(pair), ensure_ascii=False) + "\n")

    def _tensorize_problem_rollouts(
        self,
        rollouts: Sequence[RolloutResult],
        device: torch.device,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if not rollouts:
            raise ValueError("rollouts cannot be empty")

        prompt_ids = torch.tensor([rollouts[0].prompt_token_ids], dtype=torch.long, device=device)

        max_len = max((len(r.token_ids) for r in rollouts), default=1)
        if max_len <= 0:
            max_len = 1

        completion_ids = torch.zeros((len(rollouts), max_len), dtype=torch.long, device=device)
        completion_mask = torch.zeros((len(rollouts), max_len), dtype=torch.float32, device=device)

        for i, r in enumerate(rollouts):
            toks = r.token_ids or []
            if not toks:
                continue
            ln = len(toks)
            completion_ids[i, :ln] = torch.tensor(toks, dtype=torch.long, device=device)
            completion_mask[i, :ln] = 1.0

        return prompt_ids, completion_ids, completion_mask

    @staticmethod
    def _mean_stats(stats_list: Sequence[Dict[str, float]]) -> Dict[str, float]:
        if not stats_list:
            return {}
        keys = sorted({k for d in stats_list for k in d.keys()})
        out: Dict[str, float] = {}
        for k in keys:
            vals = [float(d[k]) for d in stats_list if k in d]
            if vals:
                out[k] = float(sum(vals) / len(vals))
        return out
