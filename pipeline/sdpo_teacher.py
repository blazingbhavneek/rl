from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm.auto import tqdm

from algo.sdpo import SDPOAlgo
from client.base import BaseClient, ClientContext
from tasksets.base import Problem

from .base import (
    BasePipeline,
    ProblemRollouts,
    RewardConfig,
    RolloutResult,
    SFTPair,
    StepResult,
    TeacherHintResult,
)

log = logging.getLogger(__name__)


class SDPOTeacherPipeline(BasePipeline):
    def __init__(
        self,
        student_client: BaseClient,
        teacher_client: BaseClient,
        verifier,
        algo: SDPOAlgo,
        backprop,
        optimizer,
        scheduler,
        reward_config: RewardConfig,
        sft_output_dir: str,
        n_rollouts: int = 8,
        max_hint_attempts: int = 3,
        sft_pass_threshold: float = 1.0,
        extensions: Optional[List[Any]] = None,
        show_tqdm: bool = True,
        max_grad_norm: float = 1.0,
    ) -> None:
        super().__init__(
            student_client=student_client,
            verifier=verifier,
            algo=algo,
            backprop=backprop,
            optimizer=optimizer,
            scheduler=scheduler,
            reward_config=reward_config,
            sft_output_dir=sft_output_dir,
            extensions=extensions,
            max_grad_norm=max_grad_norm,
        )
        self.teacher_client = teacher_client
        self.n_rollouts = int(n_rollouts)
        self.max_hint_attempts = int(max_hint_attempts)
        self.sft_pass_threshold = float(sft_pass_threshold)
        self.show_tqdm = bool(show_tqdm)

    def run_step(self, problems: List[Problem], step: int) -> StepResult:
        self.optimizer.zero_grad(set_to_none=True)

        problem_rollouts, rl_stats = self._run_rl_phase(problems=problems, step=step)

        sft_stats: Dict[str, float] = {}
        sft_pairs: List[SFTPair] = []
        if self.sft_output_dir is not None:
            log.info("teacher phase start: problems=%s", len(problem_rollouts))
            teacher_stats, sft_pairs = self._run_teacher_phase(problem_rollouts)
            sft_phase_stats = self._run_sft_phase(sft_pairs)
            sft_stats = {**teacher_stats, **sft_phase_stats}
            log.info(
                "teacher phase done: hints=%s passed_after_hint=%s pairs_saved=%s pairs_trained=%s",
                int(sft_stats.get("n_hints_given", 0.0)),
                int(sft_stats.get("n_passed_after_hint", 0.0)),
                int(sft_stats.get("n_pairs_saved", 0.0)),
                int(sft_stats.get("n_pairs_trained", 0.0)),
            )

        step_stats = {
            **rl_stats,
            **{f"sft_{k}": v for k, v in sft_stats.items()},
        }
        for ext in self.extensions:
            ext.on_step_end(step, step_stats)

        return StepResult(
            rl_stats=rl_stats,
            sft_stats=sft_stats,
            n_problems=len(problems),
            n_passed_rl=sum(1 for pr in problem_rollouts if pr.any_passed),
            n_passed_after_hint=int(sft_stats.get("n_passed_after_hint", 0.0)),
            step=step,
            problem_rollouts=problem_rollouts,
        )

    def _run_rl_phase(
        self,
        problems: List[Problem],
        step: int,
    ) -> Tuple[List[ProblemRollouts], Dict[str, float]]:
        del step
        problem_rollouts: List[ProblemRollouts] = []

        problem_iter = enumerate(problems)
        if self.show_tqdm:
            problem_iter = enumerate(tqdm(problems, total=len(problems), desc="rollout", leave=False))
        for pidx, problem in problem_iter:
            gen_bar = None
            extra = {}
            if self.show_tqdm:
                gen_start = time.perf_counter()
                gen_tokens = {"n": 0}

                def _on_generation_done(done: int, total: int, tok_count: int) -> None:
                    del total
                    gen_tokens["n"] += int(tok_count)
                    elapsed = max(1e-6, time.perf_counter() - gen_start)
                    toks_per_sec = gen_tokens["n"] / elapsed
                    gen_bar.update(max(0, int(done) - int(gen_bar.n)))
                    gen_bar.set_postfix_str(f"tok/s={toks_per_sec:.1f}")

                gen_bar = tqdm(
                    total=self.n_rollouts,
                    desc=f"gen[{pidx+1}/{len(problems)}]",
                    leave=False,
                )
                extra["_on_generation_done"] = _on_generation_done

            context = ClientContext(pass_number=1, extra=extra)
            rollouts = self._rollout(problem=problem, context=context, n=self.n_rollouts)
            if gen_bar is not None:
                gen_bar.n = self.n_rollouts
                gen_bar.refresh()
                gen_bar.close()

            best_rollout = max(rollouts, key=lambda r: r.reward) if rollouts else None
            passed_rollouts = [r for r in rollouts if r.passed]
            any_passed = bool(passed_rollouts)
            peer_solution = (
                max(passed_rollouts, key=lambda r: r.reward).completion_text if passed_rollouts else None
            )

            problem_rollouts.append(
                ProblemRollouts(
                    problem=problem,
                    rollouts=rollouts,
                    best_rollout=best_rollout,
                    any_passed=any_passed,
                    peer_solution=peer_solution,
                )
            )

        for ext in self.extensions:
            problem_rollouts = ext.on_rollouts_complete(problem_rollouts)

        device = next(self.backprop.model.parameters()).device
        rl_problem_stats: List[Dict[str, float]] = []
        total_bp = int(sum(len(pr.rollouts) for pr in problem_rollouts))
        n_problem_batches = max(1, sum(1 for pr in problem_rollouts if pr.rollouts))
        bp_done = 0

        bp_iter = problem_rollouts
        if self.show_tqdm:
            bp_iter = tqdm(problem_rollouts, total=len(problem_rollouts), desc="rl-backprop", leave=False)
        for pr in bp_iter:
            if not pr.rollouts:
                continue

            prompt_ids, completion_ids, completion_mask = self._tensorize_problem_rollouts(pr.rollouts, device)
            rewards = [r.reward for r in pr.rollouts]
            scores = [r.score for r in pr.rollouts]
            feedback = [r.score.error or "" for r in pr.rollouts]

            algo_out = self.algo.process_rollouts(
                prompt_ids=prompt_ids,
                completion_ids=completion_ids,
                completion_mask=completion_mask,
                scores=scores,
                rewards=rewards,
                feedback=feedback,
                peer_solution=pr.peer_solution,
            )

            if algo_out.ref_prompt_ids is not None and hasattr(self.algo, "bind_ref_logprobs"):
                ref_logprobs = torch.zeros_like(completion_mask)
                for i in range(completion_ids.shape[0]):
                    ln = int(completion_mask[i].sum().item())
                    if ln <= 0:
                        continue
                    lp = self.backprop.compute_ref_logprobs(
                        model=self.backprop.model,
                        prompt_ids=algo_out.ref_prompt_ids.to(device),
                        completion_ids=completion_ids[i, :ln],
                        lora_path=None,
                    )
                    ref_logprobs[i, :ln] = lp.to(device)
                self.algo.bind_ref_logprobs(ref_logprobs)

            prev_cb = getattr(self.backprop, "progress_callback", None)
            setattr(self.backprop, "progress_callback", None)
            bp_stats = self.backprop.backward_on_batch(
                model=self.backprop.model,
                prompt_ids=prompt_ids,
                completion_ids=completion_ids,
                completion_mask=completion_mask,
                loss_fn=algo_out.loss_fn,
                # Normalize gradients by generations and number of problems in this step
                # (equivalent to mean loss over rollout batch).
                loss_scale=1.0 / max(1, int(completion_ids.shape[0]) * n_problem_batches),
                lora_path=None,
            )
            setattr(self.backprop, "progress_callback", prev_cb)
            bp_done += int(completion_ids.shape[0])

            merged = {**algo_out.stats, **{f"bp_{k}": float(v) for k, v in bp_stats.items()}}
            rl_problem_stats.append(merged)
            if self.show_tqdm and hasattr(bp_iter, "set_postfix"):
                running = self._mean_stats(rl_problem_stats)
                live_loss = float(running.get("bp_loss", running.get("loss", 0.0)))
                live_reward = float(running.get("mean_reward", 0.0))
                live_kl = float(running.get("bp_kl_loss", running.get("kl_loss", 0.0)))
                live_tlogp = float(running.get("mean_teacher_logprob", 0.0))
                live_sr = float(running.get("success_ratio", 0.0))
                bp_iter.set_postfix(
                    loss=f"{live_loss:.4f}",
                    rew=f"{live_reward:.3f}",
                    kl=f"{live_kl:.4f}",
                    tlogp=f"{live_tlogp:.3f}",
                    succ=f"{live_sr:.2f}",
                )

        extra_losses = []
        for ext in self.extensions:
            extra = ext.extra_loss(self.backprop.model, problem_rollouts)
            if extra is not None:
                extra_losses.append(extra)

        if extra_losses:
            extra_total = torch.stack([x if x.dim() == 0 else x.mean() for x in extra_losses]).sum()
            extra_total.backward()
            rl_problem_stats.append({"extra_loss": float(extra_total.item())})

        opt_stats = self._optimizer_step()
        if hasattr(self.algo, "update_ema_teacher"):
            try:
                self.algo.update_ema_teacher()
            except Exception:
                pass

        rl_stats = self._mean_stats(rl_problem_stats)
        rl_stats.update(opt_stats)
        rl_stats["n_problem_rollouts"] = float(len(problem_rollouts))
        return problem_rollouts, rl_stats

    def _run_teacher_phase(
        self,
        problem_rollouts: List[ProblemRollouts],
    ) -> Tuple[Dict[str, float], List[SFTPair]]:
        n_hints_given = 0
        n_passed_after_hint = 0
        n_pairs_saved = 0
        pairs: List[SFTPair] = []

        teacher_iter = problem_rollouts
        if self.show_tqdm:
            teacher_iter = tqdm(problem_rollouts, total=len(problem_rollouts), desc="teacher-hints", leave=False)
        for pr in teacher_iter:
            failed = [r for r in pr.rollouts if not r.passed]
            for fr in failed:
                out = self._run_hint_loop(pr.problem, fr, pr.peer_solution)
                n_hints_given += int(out.get("hints", 0))
                n_passed_after_hint += int(out.get("passed", 0))
                n_pairs_saved += int(out.get("pairs", 0))
                pairs.extend(out.get("pair_objects", []))
            if self.show_tqdm and hasattr(teacher_iter, "set_postfix"):
                teacher_iter.set_postfix(
                    hints=n_hints_given,
                    passed=n_passed_after_hint,
                    pairs=n_pairs_saved,
                )

        return (
            {
                "n_hints_given": float(n_hints_given),
                "n_passed_after_hint": float(n_passed_after_hint),
                "n_pairs_saved": float(n_pairs_saved),
            },
            pairs,
        )

    def _run_hint_loop(
        self,
        problem: Problem,
        failed_rollout: RolloutResult,
        peer_solution: Optional[str],
    ) -> Dict[str, int]:
        previous_hints: List[str] = []
        previous_retries: List[RolloutResult] = []

        hints = 0
        pairs = 0
        passed = 0
        pair_objects: List[SFTPair] = []

        current_failed = failed_rollout
        for attempt in range(1, self.max_hint_attempts + 1):
            teacher_messages = self._build_teacher_messages(
                problem=problem,
                failed_rollout=current_failed,
                peer_solution=peer_solution,
                attempt=attempt,
                previous_hints=previous_hints,
                previous_retries=previous_retries,
            )
            hint_text = self._generate_teacher_hint(problem, teacher_messages)
            hints += 1

            student_retry = self._run_student_retry(problem, current_failed, hint_text)
            hint_result = TeacherHintResult(
                problem=problem,
                failed_rollout=current_failed,
                teacher_hint=hint_text,
                student_retry=student_retry,
                passed_after_hint=student_retry.passed,
                attempt_number=attempt,
            )

            if hint_result.passed_after_hint and self._score_ratio(student_retry) >= self.sft_pass_threshold:
                pair = self._build_sft_pair(problem, current_failed, hint_text, student_retry)
                self._save_sft_pair(pair)
                pairs += 1
                passed += 1
                pair_objects.append(pair)
                break

            previous_hints.append(hint_text)
            previous_retries.append(student_retry)
            current_failed = student_retry

        return {"hints": hints, "pairs": pairs, "passed": passed, "pair_objects": pair_objects}

    @staticmethod
    def _score_ratio(rollout: RolloutResult) -> float:
        if rollout.score.total <= 0:
            return 0.0
        return float(rollout.score.passed) / float(rollout.score.total)

    def _generate_teacher_hint(self, problem: Problem, teacher_messages: List[Dict[str, Any]]) -> str:
        if hasattr(self.teacher_client, "run_messages"):
            out = self.teacher_client.run_messages(teacher_messages, n=1)
            if out.completions:
                return out.completions[0]

        # Fallback via generic client API.
        ctx = ClientContext(
            pass_number=2,
            best_code=teacher_messages[-2]["content"] if len(teacher_messages) >= 2 else None,
            error_context=teacher_messages[-1]["content"],
        )
        out = self.teacher_client.run(problem=problem, context=ctx, n=1)
        return out.completions[0] if out.completions else "Provide a concrete debugging hint."

    def _run_student_retry(
        self,
        problem: Problem,
        failed_rollout: RolloutResult,
        hint_text: str,
    ) -> RolloutResult:
        student_context = ClientContext(
            pass_number=2,
            best_code=failed_rollout.completion_text,
            error_context=(
                "Your solution is incorrect. Here is some guidance:\n\n"
                f"{hint_text}\n\n"
                "Please reconsider your approach and try again."
            ),
        )
        return self._rollout(problem=problem, context=student_context, n=1)[0]

    def _build_teacher_messages(
        self,
        problem: Problem,
        failed_rollout: RolloutResult,
        peer_solution: Optional[str],
        attempt: int,
        previous_hints: Optional[List[str]] = None,
        previous_retries: Optional[List[RolloutResult]] = None,
    ) -> List[Dict[str, str]]:
        previous_hints = previous_hints or []
        previous_retries = previous_retries or []

        system = (
            "You are a programming tutor. Identify exactly what went wrong in the student's reasoning "
            "and provide targeted guidance. Do NOT provide the final solution."
        )

        chunks = [
            f"Problem:\n{problem.statement}",
            f"Student's attempt:\n{failed_rollout.completion_text}",
            f"Execution result:\n{failed_rollout.score.error or 'Unknown failure'}",
        ]

        if peer_solution:
            chunks.append(
                "Note: another student solved this correctly. Use it only for your internal diagnosis; "
                "do NOT reveal the answer.\n"
                f"Correct solution (teacher-only reference):\n{peer_solution}"
            )

        if attempt > 1 and previous_hints and previous_retries:
            last_hint = previous_hints[-1]
            last_retry = previous_retries[-1]
            chunks.append(
                "Previous hint you gave:\n"
                f"{last_hint}\n\n"
                "Student retry result:\n"
                f"{last_retry.score.error or 'still incorrect'}\n"
                "The student still failed. Give a different angle."
            )

        chunks.append(
            "Explain the exact reasoning flaw and how to fix it. "
            "Do not provide the full final answer."
        )

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(chunks)},
        ]

    def _build_sft_pair(
        self,
        problem: Problem,
        failed_rollout: RolloutResult,
        hint_text: str,
        student_retry: RolloutResult,
    ) -> SFTPair:
        system_prompt = getattr(self.student_client, "system_prompt", "")
        return SFTPair(
            problem_id=problem.id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": problem.statement},
                {"role": "assistant", "content": failed_rollout.completion_text},
                {
                    "role": "user",
                    "content": (
                        "Your solution is incorrect. Here is some guidance:\n\n"
                        f"{hint_text}\n\n"
                        "Please reconsider your approach and try again."
                    ),
                },
                {"role": "assistant", "content": student_retry.completion_text},
            ],
            source="teacher_hint_sft",
        )
