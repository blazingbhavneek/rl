from __future__ import annotations

from typing import Callable, List, Optional

import torch
from torch import Tensor

try:
    from taskset.base import Score
except ImportError:  # pragma: no cover - direct script execution fallback
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from taskset.base import Score

try:
    from .base import AlgoConfig, AlgoOutput, BaseAlgo
    from .utils import apply_ppo_clip, compute_kl_penalty, normalize_advantages
except ImportError:  # pragma: no cover - direct script execution fallback
    from base import AlgoConfig, AlgoOutput, BaseAlgo
    from utils import apply_ppo_clip, compute_kl_penalty, normalize_advantages


class GRPOAlgo(BaseAlgo):
    def __init__(self, config: Optional[AlgoConfig] = None) -> None:
        super().__init__(config or AlgoConfig())
        self._ref_logprobs: Optional[Tensor] = None
        self._old_logprobs: Optional[Tensor] = None

    def requires_rich_feedback(self) -> bool:
        return False

    @property
    def needs_hidden_states(self) -> bool:
        return False

    def bind_ref_logprobs(self, ref_logprobs: Optional[Tensor]) -> None:
        self._ref_logprobs = ref_logprobs

    def bind_old_logprobs(self, old_logprobs: Optional[Tensor]) -> None:
        self._old_logprobs = old_logprobs

    def compute_advantages(self, rewards: Tensor) -> Tensor:
        if self.config.norm_advantages:
            return normalize_advantages(rewards)
        return rewards.float()

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
        del completion_ids, scores, feedback, peer_solution
        g = completion_mask.shape[0]
        if len(rewards) != g:
            raise ValueError(f"rewards length ({len(rewards)}) must equal G ({g})")

        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        # GRPO uses scalar rollout rewards and converts them to group-relative advantages.
        # This is the key sparse-signal baseline SDPO is compared against.
        advantages = self.compute_advantages(rewards_t)

        self._ref_logprobs = None
        self._old_logprobs = None
        ref_prompt_ids = prompt_ids if self.config.kl_coeff > 0.0 else None

        loss_fn = self._make_loss_fn(
            advantages=advantages,
            completion_mask=completion_mask.float(),
        )

        stats = {
            "mean_reward": float(rewards_t.mean().item()) if rewards_t.numel() else 0.0,
            "std_reward": float(rewards_t.std(unbiased=False).item()) if rewards_t.numel() else 0.0,
            "mean_advantage": float(advantages.mean().item()) if advantages.numel() else 0.0,
            "std_advantage": float(advantages.std(unbiased=False).item()) if advantages.numel() else 0.0,
        }
        return AlgoOutput(
            loss_fn=loss_fn,
            needs_hidden_states=self.needs_hidden_states,
            stats=stats,
            ref_prompt_ids=ref_prompt_ids,
        )

    def _make_loss_fn(
        self,
        advantages: Tensor,  # (G,)
        completion_mask: Tensor,  # (G, T_c)
    ) -> Callable[[Tensor, int, Optional[Tensor]], Tensor]:
        def loss_fn(log_probs: Tensor, gen_idx: int, hidden_comp: Optional[Tensor] = None) -> Tensor:
            del hidden_comp
            g_adv = advantages[gen_idx].to(log_probs.device, non_blocking=True)
            g_mask = completion_mask[gen_idx].to(log_probs.device, non_blocking=True)
            g_len = g_mask.sum().clamp(min=1.0)

            if self._old_logprobs is not None:
                # PPO-style clipped policy-ratio objective over sampled completion tokens.
                old_lp = self._old_logprobs[gen_idx].to(log_probs.device, non_blocking=True)
                ratio = torch.exp(log_probs - old_lp)
                clipped_ratio = apply_ppo_clip(
                    ratio=ratio,
                    advantage=torch.full_like(ratio, g_adv),
                    clip_low=self.config.clip_ratio_low,
                    clip_high=self.config.clip_ratio_high,
                )
                surrogate = torch.minimum(ratio * g_adv, clipped_ratio * g_adv)
                pg_term = -(surrogate * g_mask).sum() / g_len
            else:
                # On-policy fallback: weighted log-prob objective with scalar advantage.
                pg_term = -(g_adv * log_probs * g_mask).sum() / g_len

            kl_term = torch.zeros((), device=log_probs.device, dtype=log_probs.dtype)
            if self._ref_logprobs is not None:
                ref_lp = self._ref_logprobs[gen_idx].to(log_probs.device, non_blocking=True)
                kl_term = compute_kl_penalty(log_probs.detach(), ref_lp, g_mask).to(log_probs.dtype)
            return pg_term + (self.config.kl_coeff * kl_term)

        return loss_fn
