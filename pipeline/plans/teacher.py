"""
Drop-in replacements for GRPOPipeline._teacher_refine and GRPOPipeline._sft_step,
plus the TrainConfig additions needed.

Changes:
  - Teacher is stateful per candidate (one AgentClient, history builds across attempts)
  - Sequential hint loop: hint → student retry → new error → new hint → ...
  - ref_answer can be str | list[str] | None; samples from list
  - SFT fallback on unsolved candidates using ground truth (sampled) or None (skip)
  - Separate teacher host/model/key config in TrainConfig
"""

from __future__ import annotations

import asyncio
import random
from typing import Optional

import torch
from torch import Tensor
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# TrainConfig additions (add these fields to your existing dataclass)
# ---------------------------------------------------------------------------

# --- teacher (add to TrainConfig) ---
# teacher_base_url: Optional[str] = None   # if None, falls back to engine_base_url
# teacher_api_key: Optional[str] = None    # if None, falls back to engine_api_key
# teacher_model: Optional[str] = None      # if None, falls back to model_path


# ---------------------------------------------------------------------------
# Updated train() teacher construction (replace the block in train())
# ---------------------------------------------------------------------------

def _build_teacher_client(cfg, profile):
    """
    Build the default teacher AgentClient using dedicated teacher config
    fields when provided, falling back to the engine/student config.
    Paste this into GRPOPipeline.train() to replace the existing block.
    """
    from client.agent import AgentClient

    return AgentClient(
        base_url=cfg.teacher_base_url or cfg.engine_base_url,
        api_key=cfg.teacher_api_key or cfg.engine_api_key,
        temperature=cfg.teacher_temperature,
        max_output_tokens=cfg.max_tokens,
        system_prompt=(
            "You are a coding tutor helping a student fix their C code. "
            "Each turn you will see the student's latest attempt and its "
            "compiler/test output. Give one concise targeted hint per turn. "
            "Do NOT reveal the full solution."
        ),
        model=cfg.teacher_model or cfg.model_path,
        tools=[],
        max_turns=cfg.teacher_max_turns,
        extra_body=profile.teacher_extra_body,
    )


# ---------------------------------------------------------------------------
# Helper: resolve ref_answer → single string or None
# ---------------------------------------------------------------------------

def _sample_ref_answer(problem) -> Optional[str]:
    """
    problem.metadata['answer'] may be:
      - a non-empty str   → use as-is
      - a non-empty list  → sample one element
      - empty / missing   → return None
    """
    raw = (problem.metadata or {}).get("answer", None)
    if not raw:
        return None
    if isinstance(raw, list):
        valid = [s for s in raw if s and isinstance(s, str)]
        return random.choice(valid) if valid else None
    if isinstance(raw, str):
        return raw.strip() or None
    return None


# ---------------------------------------------------------------------------
# _teacher_refine  (replace the existing method on GRPOPipeline)
# ---------------------------------------------------------------------------

async def _teacher_refine(
    self,
    candidates: list[dict],
    generate_fn,
    score_fn,
    teacher_client,           # used as a config template; a fresh stateful
                              # AgentClient is created per candidate
) -> tuple[list[dict], int, int]:
    """
    For each failed candidate:
      1. Create one stateful AgentClient (history persists across attempts).
      2. Sequential loop up to max_hint_attempts:
           - Teacher sees problem + latest attempt + latest error log → hint
           - Student retries with hint + error log
           - If passes: mark solved, break
           - Else: update current_text/score and loop (teacher sees new error next turn)
      3. If never solved:
           - sft_target = sampled ref answer  (str | None)
           - None → skip SFT for this candidate (GRPO already captured signal)
    """
    cfg = self.cfg
    sem = asyncio.Semaphore(max(1, cfg.engine_semaphore_limit))
    pbar = tqdm(
        total=len(candidates) * cfg.max_hint_attempts,
        desc="teacher-refine",
        leave=False,
    )

    from client.agent import AgentClient

    async def _refine_one(rec: dict) -> tuple[dict, int, int]:
        problem      = rec["problem"]
        current_text = str(rec["text"])
        current_score = rec["score"]
        best_text, best_score, best_reward = current_text, current_score, float(rec["reward"])
        solved       = False
        local_hints  = 0

        # Resolve reference answer once per candidate (str | None).
        ref_answer = _sample_ref_answer(problem)

        # ------------------------------------------------------------------
        # One stateful teacher per candidate.
        # History accumulates across attempts so the teacher knows what it
        # already suggested and can escalate its guidance.
        # ------------------------------------------------------------------
        system_prompt = (
            "You are a coding tutor helping a student fix their C code. "
            "Each turn you will see the student's latest attempt and its "
            "compiler/test output. Give one concise targeted hint per turn. "
            "Do NOT reveal the full solution."
        )
        if ref_answer:
            system_prompt += (
                "\n\n[Reference solution — for YOUR guidance ONLY, "
                "do NOT reveal to the student]:\n"
                f"{ref_answer}"
            )

        local_teacher = AgentClient(
            base_url=teacher_client.llm.openai_api_base,
            api_key=teacher_client.llm.openai_api_key,
            temperature=teacher_client.llm.temperature,
            max_output_tokens=teacher_client.llm.max_tokens,
            system_prompt=system_prompt,
            model=teacher_client.llm.model_name,
            tools=[],
            max_turns=cfg.teacher_max_turns,
            extra_body=self.profile.teacher_extra_body,
        )

        # ------------------------------------------------------------------
        # Sequential hint loop.
        # Each iteration: teacher sees the *latest* attempt + error,
        # student retries with that hint + the same error.
        # ------------------------------------------------------------------
        for attempt in range(cfg.max_hint_attempts):

            # Build a compact status string from the latest score.
            if not current_score.compiled:
                status = "did not compile"
            else:
                status = (
                    f"compiled, passed {current_score.passed}/"
                    f"{current_score.total} tests"
                )
            error_block = ""
            if current_score.error:
                error_block += f"\nError: {current_score.error}"
            if current_score.details:
                error_block += f"\nTest details:\n{current_score.details}"

            # Teacher prompt — history carries all previous hints automatically.
            teacher_prompt = (
                f"Problem:\n{problem.statement}\n\n"
                f"Student attempt (try {attempt + 1}):\n{current_text}\n\n"
                f"Verifier result: {status}"
                f"{error_block}\n\n"
                "Give one concise hint to fix this."
            )

            async with sem:
                _, hint = await local_teacher.run(prompt=teacher_prompt)
            local_hints += 1

            # Student retry: sees the hint AND the full error log.
            retry_messages = [
                {"role": "system", "content": cfg.system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"{problem.statement}\n\n"
                        f"Your previous attempt:\n{current_text}\n\n"
                        f"Verifier result: {status}"
                        f"{error_block}\n\n"
                        f"Tutor hint:\n{hint}\n\n"
                        "Fix your solution."
                    ),
                },
            ]
            retry_text = await generate_fn(retry_messages)
            pbar.update(1)

            retry_sc, retry_reward, retry_passed = await score_fn(problem, retry_text)
            retry_reward *= cfg.hint_reward_discount

            # Update current attempt for the next teacher turn.
            current_text  = retry_text
            current_score = retry_sc

            if retry_reward > best_reward:
                best_text, best_score, best_reward = retry_text, retry_sc, retry_reward

            if retry_passed:
                solved = True
                # Consume remaining pbar slots for this candidate.
                pbar.update(cfg.max_hint_attempts - attempt - 1)
                break

        # ------------------------------------------------------------------
        # SFT target resolution.
        #
        #   solved            → sft_target = None  (no SFT needed)
        #   not solved + ref  → sft_target = ref_answer (ground truth)
        #   not solved + none → sft_target = None  (skip; GRPO handles it)
        #
        # Intentionally NOT using best_text as a fallback: an unverified
        # failing attempt is noisy SFT signal and can confuse the model.
        # ------------------------------------------------------------------
        sft_target = None
        if not solved and ref_answer:
            sft_target = ref_answer   # already sampled above

        return (
            dict(
                problem=problem,
                messages=[
                    {"role": "system", "content": cfg.system_prompt},
                    {"role": "user",   "content": str(problem.statement)},
                ],
                text=best_text,
                score=best_score,
                reward=best_reward,
                passed=solved,
                sft_target=sft_target,   # str or None
            ),
            local_hints,
            int(solved),
        )

    # Run all candidates concurrently (semaphore limits engine pressure).
    results = await asyncio.gather(*[_refine_one(rec) for rec in candidates])
    pbar.close()

    refined, hints_given, passed_after_hint = [], 0, 0
    for result, h, p in results:
        refined.append(result)
        hints_given += h
        passed_after_hint += p

    return refined, hints_given, passed_after_hint


# ---------------------------------------------------------------------------
# _sft_step  (replace the existing method on GRPOPipeline)
# ---------------------------------------------------------------------------

def _sft_step(self, *, refined: list[dict], train_model) -> float:
    """
    SFT on unsolved candidates that have a ground-truth sft_target.
    Uses sft_target (sampled ref answer) rather than the model's own output,
    so the gradient is anchored to a known-correct completion.
    """

    # Only supervise on unsolved candidates that have a ref answer.
    sft_candidates = [
        r for r in refined
        if not r["passed"] and r.get("sft_target")
    ]
    if not sft_candidates:
        return 0.0

    def _sft_loss_batch(
        batch_log_probs: Tensor,
        batch_mask: Tensor,
        hidden_comp=None,
    ) -> Tensor:
        mask    = batch_mask.to(batch_log_probs.device).float()
        lengths = mask.sum(dim=1).clamp(min=1.0)
        return (-((batch_log_probs * mask).sum(dim=1) / lengths)).mean()

    def _sft_loss(log_probs: Tensor, gen_idx: int, hidden_comp=None) -> Tensor:
        return -log_probs.mean()

    setattr(_sft_loss, "loss_fn_batch", _sft_loss_batch)

    bp = train_model.backward(
        messages=[r["messages"] for r in sft_candidates],
        completion_texts=[r["sft_target"] for r in sft_candidates],
        loss_fn=_sft_loss,
        loss_scale=1.0 / max(1, len(sft_candidates)),
    )

    return float(bp.get("loss", 0.0))
