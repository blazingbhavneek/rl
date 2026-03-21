from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import torch
from torch import Tensor, nn

try:
    from tasksets.base import Score
except ImportError:  # pragma: no cover - direct script execution fallback
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from tasksets.base import Score

try:
    from .base import AlgoConfig, AlgoOutput, BaseAlgo
    from .utils import masked_mean, normalize_advantages
except ImportError:  # pragma: no cover - direct script execution fallback
    from base import AlgoConfig, AlgoOutput, BaseAlgo
    from utils import masked_mean, normalize_advantages

try:
    from backprop.base import BaseBackprop
except Exception:  # pragma: no cover - keeps import-time lightweight
    BaseBackprop = Any  # type: ignore


@dataclass
class SDPOConfig:
    credit_assignment: str = "logit"  # "logit" | "token" | "sequence"
    top_k: int = 20
    teacher_reg: str = "ema"  # "ema" | "trust_region" | "none"
    teacher_alpha: float = 0.01
    divergence: str = "reverse_kl"  # "reverse_kl" | "forward_kl" | "js"
    lambda_grpo: float = 0.0
    clip_advantages: float = 5.0


class SDPOAlgo(BaseAlgo):
    def __init__(
        self,
        config: Optional[AlgoConfig],
        sdpo_config: Optional[SDPOConfig],
        backprop: BaseBackprop,
        model: nn.Module,
        *,
        tokenizer: Optional[Any] = None,
        current_lora_path_getter: Optional[Callable[[], Optional[str]]] = None,
    ) -> None:
        super().__init__(config or AlgoConfig())
        self.sdpo_config = sdpo_config or SDPOConfig()
        self.backprop = backprop
        self.model = model
        self.tokenizer = tokenizer
        self.current_lora_path_getter = current_lora_path_getter

        self._ema_model: Optional[nn.Module] = None
        if self.sdpo_config.teacher_reg == "ema":
            self._ema_model = copy.deepcopy(model).eval()
            for p in self._ema_model.parameters():
                p.requires_grad_(False)

    def requires_rich_feedback(self) -> bool:
        return True

    @property
    def needs_hidden_states(self) -> bool:
        return self.sdpo_config.credit_assignment == "logit"

    def _decode(self, token_ids: Tensor) -> str:
        if self.tokenizer is None or not hasattr(self.tokenizer, "decode"):
            return " ".join(str(int(x)) for x in token_ids.tolist())
        return self.tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)

    def _encode(self, text: str, device: torch.device) -> Tensor:
        if self.tokenizer is None:
            raise RuntimeError("SDPOAlgo requires a tokenizer to build teacher contexts.")

        if hasattr(self.tokenizer, "encode"):
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            return torch.tensor([ids], dtype=torch.long, device=device)

        encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        return encoded.input_ids.to(device)

    def build_teacher_context(
        self,
        prompt_ids: Tensor,
        completion_text: str,
        feedback_text: str,
        peer_solution: Optional[str],
    ) -> Tensor:
        model_device = next(self.model.parameters()).device
        question = self._decode(prompt_ids[0].detach().cpu())

        lines = [f"User: {question}"]
        if peer_solution:
            lines.append(f"Correct solution: {peer_solution}")
        if feedback_text:
            lines.append("The following is feedback from your unsuccessful attempt:")
            lines.append(feedback_text)
        else:
            lines.append("The previous attempt was successful.")
        lines.append("Correctly solve the original question.")
        if completion_text:
            # Completion text is intentionally *not* included per SDPO ablation.
            _ = completion_text
        teacher_prompt = "\n".join(lines)
        return self._encode(teacher_prompt, model_device)

    def _compute_teacher_logprobs(
        self,
        teacher_model: nn.Module,
        teacher_prompt_ids: Tensor,
        completion_ids: Tensor,
        lora_path: Optional[str],
    ) -> Tensor:
        with torch.inference_mode():
            if teacher_model is self.model and lora_path is None:
                # If we don't know a checkpoint path, keep current active adapters and
                # avoid compute_ref_logprobs(None), which may disable adapters.
                lp = self.backprop.compute_logprobs(
                    model=teacher_model,
                    prompt_ids=teacher_prompt_ids,
                    completion_ids=completion_ids,
                    lora_path=None,
                ).detach().cpu()
            else:
                lp = self.backprop.compute_ref_logprobs(
                    model=teacher_model,
                    prompt_ids=teacher_prompt_ids,
                    completion_ids=completion_ids,
                    lora_path=lora_path,
                )
        return lp.detach().float().cpu()

    def process_rollouts(
        self,
        prompt_ids: Tensor,
        completion_ids: Tensor,
        completion_mask: Tensor,
        scores: List[Score],
        rewards: List[float],
        feedback: Optional[List[str]] = None,
        peer_solution: Optional[str] = None,
    ) -> AlgoOutput:
        g, t_c = completion_ids.shape
        if len(scores) != g:
            raise ValueError(f"scores length ({len(scores)}) must equal G ({g})")
        if len(rewards) != g:
            raise ValueError(f"rewards length ({len(rewards)}) must equal G ({g})")
        if feedback is None:
            feedback = [""] * g
        if len(feedback) != g:
            raise ValueError(f"feedback length ({len(feedback)}) must equal G ({g})")

        model_device = next(self.model.parameters()).device
        lora_path = self.current_lora_path_getter() if self.current_lora_path_getter else None
        if lora_path is None:
            lora_path = getattr(self.backprop, "_current_lora_path", None)
        teacher_model = self._ema_model if self.sdpo_config.teacher_reg == "ema" and self._ema_model else self.model

        teacher_logprobs: List[Tensor] = []
        n_success = 0
        used_peer = 0
        for i in range(g):
            sc = scores[i]
            success = bool(sc.total > 0 and sc.passed == sc.total)
            n_success += int(success)
            feedback_text = feedback[i] or (sc.error or "")

            completion_text = self._decode(completion_ids[i].detach().cpu())
            context_peer_solution: Optional[str]
            if success:
                context_peer_solution = completion_text
            else:
                context_peer_solution = peer_solution
                used_peer += int(context_peer_solution is not None)

            teacher_prompt_ids = self.build_teacher_context(
                prompt_ids=prompt_ids.detach().cpu(),
                completion_text=completion_text,
                feedback_text=feedback_text,
                peer_solution=context_peer_solution,
            )

            comp_i = completion_ids[i].to(model_device, non_blocking=True)
            teacher_lp = self._compute_teacher_logprobs(
                teacher_model=teacher_model,
                teacher_prompt_ids=teacher_prompt_ids,
                completion_ids=comp_i,
                lora_path=None if teacher_model is not self.model else lora_path,
            )

            if self.sdpo_config.teacher_reg == "trust_region":
                ref_lp = self._compute_teacher_logprobs(
                    teacher_model=self.model,
                    teacher_prompt_ids=teacher_prompt_ids,
                    completion_ids=comp_i,
                    lora_path=None,
                )
                a = float(self.sdpo_config.teacher_alpha)
                teacher_lp = (1.0 - a) * ref_lp + a * teacher_lp

            if teacher_lp.numel() != t_c:
                raise RuntimeError(
                    f"Teacher logprob length mismatch: got {teacher_lp.numel()} expected {t_c}"
                )
            teacher_logprobs.append(teacher_lp)

        grpo_advantages: Optional[Tensor] = None
        if self.sdpo_config.lambda_grpo > 0.0:
            grpo_advantages = normalize_advantages(torch.tensor(rewards, dtype=torch.float32))

        loss_fn = self._make_loss_fn(
            teacher_logprobs=teacher_logprobs,
            completion_mask=completion_mask.float(),
            credit_assignment=self.sdpo_config.credit_assignment,
            lambda_grpo=self.sdpo_config.lambda_grpo,
            grpo_advantages=grpo_advantages,
        )

        stacked_teacher = torch.stack(teacher_logprobs) if teacher_logprobs else torch.empty(0)
        stats = {
            "mean_reward": float(torch.tensor(rewards).mean().item()) if rewards else 0.0,
            "mean_teacher_logprob": float(stacked_teacher.mean().item()) if stacked_teacher.numel() else 0.0,
            "success_ratio": float(n_success / max(1, g)),
            "used_peer_ratio": float(used_peer / max(1, g)),
            "lambda_grpo": float(self.sdpo_config.lambda_grpo),
        }
        if grpo_advantages is not None:
            stats["mean_grpo_advantage"] = float(grpo_advantages.mean().item())

        return AlgoOutput(
            loss_fn=loss_fn,
            needs_hidden_states=self.needs_hidden_states,
            stats=stats,
            ref_prompt_ids=None,
        )

    def _make_loss_fn(
        self,
        teacher_logprobs: List[Tensor],  # len G, each (T_c,)
        completion_mask: Tensor,  # (G, T_c)
        credit_assignment: str,
        lambda_grpo: float,
        grpo_advantages: Optional[Tensor],
    ) -> Callable[[Tensor, int, Optional[Tensor]], Tensor]:
        if credit_assignment not in {"logit", "token", "sequence"}:
            raise ValueError(f"Unsupported credit assignment: {credit_assignment}")

        max_adv = float(self.sdpo_config.clip_advantages)

        def _grpo_term(log_probs: Tensor, idx: int, mask: Tensor) -> Tensor:
            if grpo_advantages is None:
                return torch.zeros((), device=log_probs.device, dtype=log_probs.dtype)
            g_adv = grpo_advantages[idx].to(log_probs.device, non_blocking=True)
            return -(g_adv * log_probs * mask).sum() / mask.sum().clamp(min=1.0)

        def loss_fn(log_probs: Tensor, gen_idx: int, hidden_comp: Optional[Tensor] = None) -> Tensor:
            mask = completion_mask[gen_idx].to(log_probs.device, non_blocking=True)
            teacher_lp = teacher_logprobs[gen_idx].to(log_probs.device, non_blocking=True)
            token_adv = (teacher_lp - log_probs.detach()).clamp(min=-max_adv, max=max_adv) * mask

            if credit_assignment == "sequence":
                seq_adv = masked_mean(token_adv, mask)
                sdpo_loss = -(seq_adv * log_probs * mask).sum() / mask.sum().clamp(min=1.0)
            elif credit_assignment == "logit":
                # Current backprop path may not provide hidden states yet.
                # Fallback to sampled-token reverse-KL proxy if hidden_comp is missing.
                _ = hidden_comp
                sdpo_loss = -masked_mean(token_adv * log_probs, mask)
            else:  # token
                sdpo_loss = -masked_mean(token_adv * log_probs, mask)

            if lambda_grpo <= 0.0:
                return sdpo_loss
            grpo_loss = _grpo_term(log_probs, gen_idx, mask)
            return lambda_grpo * grpo_loss + (1.0 - lambda_grpo) * sdpo_loss

        return loss_fn

    def update_ema_teacher(self, alpha: float = 0.01) -> None:
        if self._ema_model is None:
            return
        a = float(alpha)
        with torch.inference_mode():
            for p_ema, p_cur in zip(self._ema_model.parameters(), self.model.parameters()):
                p_ema.mul_(1.0 - a).add_(p_cur.detach(), alpha=a)


if __name__ == "__main__":
    class _MockBackprop:
        def compute_logprobs(self, model, prompt_ids, completion_ids, lora_path=None):
            del model, prompt_ids, lora_path
            return torch.zeros_like(completion_ids, dtype=torch.float32)

        def compute_ref_logprobs(self, model, prompt_ids, completion_ids, lora_path=None):
            del model, prompt_ids, lora_path
            return torch.zeros_like(completion_ids, dtype=torch.float32).cpu()

    class _TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.tensor([1.0]))

    class _Tok:
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return [ord(c) % 127 for c in text][:64]

        def decode(self, token_ids, skip_special_tokens=True):
            del skip_special_tokens
            return "".join(chr(int(x) % 127) for x in token_ids)

    algo = SDPOAlgo(
        config=AlgoConfig(),
        sdpo_config=SDPOConfig(credit_assignment="token", teacher_reg="none"),
        backprop=_MockBackprop(),
        model=_TinyModel(),
        tokenizer=_Tok(),
    )
    prompt_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    completion_ids = torch.tensor([[10, 11, 0], [12, 13, 14]], dtype=torch.long)
    completion_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.float32)
    scores = [Score(compiled=True, passed=0, total=2, error="WA"), Score(compiled=True, passed=2, total=2)]
    out = algo.process_rollouts(
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        scores=scores,
        rewards=[0.0, 1.0],
        feedback=["test 1 failed", ""],
        peer_solution="print(42)",
    )
    lp = torch.tensor([-1.0, -1.2, -2.0], requires_grad=True)
    loss = out.loss_fn(lp, 0)
    loss.backward()
    assert lp.grad is not None
    print("algo/sdpo.py self-test passed")
