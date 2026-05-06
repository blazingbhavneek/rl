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

        # Stores log probabilities from a reference model 
        # Used to calculate KL divergence penalty—keeps the new model from diverging too much from where it started.
        self._ref_logprobs: Optional[Tensor] = None

        # Stores log probabilities from your current training model BEFORE the gradient update. Used for PPO-style clipping
        # compute the ratio of (new policy / old policy) to see how much the model changed
        self._old_logprobs: Optional[Tensor] = None

    # Signals whether this algorithm needs detailed per-token feedback
    def requires_rich_feedback(self) -> bool:
        return False

    @property
    def needs_hidden_states(self) -> bool:
        return False
    
    # Setter methods. They store references to tensors that will be used in the loss computation.
    def bind_ref_logprobs(self, ref_logprobs: Optional[Tensor]) -> None:
        self._ref_logprobs = ref_logprobs

    def bind_old_logprobs(self, old_logprobs: Optional[Tensor]) -> None:
        self._old_logprobs = old_logprobs

    def compute_advantages(self, rewards: Tensor) -> Tensor:
        return normalize_advantages(rewards.float())

    # main entry point of algorithm
    def process_rollouts(
        self,
        # [G, T_p] prompt token ids replicated per rollout in the group.
        prompt_ids: Tensor,
        # [G, T_c] candidate completion token ids (may be padded to common length).
        completion_ids: Tensor,
        # [G, T_c] 1 for real completion tokens, 0 for padding.
        completion_mask: Tensor,
        # Per-rollout verifier/test outcomes (passed/total/compiled/details).
        scores: List[Score],
        # Scalar reward per rollout candidate.
        rewards: List[float],
        # Optional compiler/runtime feedback (unused in core GRPO loss here).
        feedback: Optional[List[str]] = None,
        # Optional best peer solution reference (unused in core GRPO loss here).
        peer_solution: Optional[str] = None,
    ) -> AlgoOutput:
        # These inputs are part of a shared interface but not used by this implementation path.
        del completion_ids, feedback, peer_solution

        g = completion_mask.shape[0]
        if len(scores) != g:
            raise ValueError(f"scores length ({len(scores)}) must equal G ({g})")
        if len(rewards) != g:
            raise ValueError(f"rewards length ({len(rewards)}) must equal G ({g})")

        # Convert rollout rewards to tensor for group-level advantage computation.
        rewards_t = torch.tensor(rewards, dtype=torch.float32)

        advantages = self.compute_advantages(rewards_t)

        # If KL coefficient > 0, prepare to pass prompt_ids to the loss function (needed to compute KL penalty later)
        ref_prompt_ids = prompt_ids if self.config.kl_coeff > 0.0 else None

        # Make the loss function which will be used for backprop
        loss_fn = self._make_loss_fn(
            advantages=advantages,
            completion_mask=completion_mask.float(),
        )

        # Return stats for logging
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
        advantages: Tensor,      # (G,)
        completion_mask: Tensor,  # (G, T_c)
    ) -> Callable[[Tensor, int, Optional[Tensor]], Tensor]:
        # This function does not compute the loss right away.
        # Instead, it builds and returns small functions that already
        # remember `advantages` and `completion_mask`.

        def loss_fn(
            log_probs: Tensor,
            gen_idx: int,
            hidden_comp: Optional[Tensor] = None,
        ) -> Tensor:
            del hidden_comp

            # Pick the data for one rollout in the group.
            # `g_adv` is one number: how good/bad this rollout was.
            # `g_mask` says which completion tokens are real tokens
            # and which ones are just padding.
            g_adv = advantages[gen_idx].to(log_probs.device, non_blocking=True)
            g_mask = completion_mask[gen_idx].to(log_probs.device, non_blocking=True)

            # Count how many real completion tokens we have.
            # We divide by this later so longer completions do not
            # automatically produce a bigger loss.
            g_len = g_mask.sum().clamp(min=1.0)

            if self._old_logprobs is not None:
                # If we saved the model's old log-probs, we can do PPO.
                # PPO compares the new policy to the old one and prevents
                # the update from changing too much in one step.
                old_lp = self._old_logprobs[gen_idx].to(log_probs.device, non_blocking=True)

                # log_probs - old_lp is log(new / old), so exp(...) gives
                # the usual PPO probability ratio: new_policy / old_policy.
                ratio = torch.exp(log_probs - old_lp)

                # This applies PPO clipping.
                # The clipped version acts like a safety limit on the update.
                clipped_ratio = apply_ppo_clip(
                    ratio=ratio,
                    advantage=torch.full_like(ratio, g_adv),
                    clip_low=self.config.clip_ratio_low,
                    clip_high=self.config.clip_ratio_high,
                )

                # PPO uses the smaller of the unclipped and clipped objective.
                # This is the key trick that makes PPO more stable.
                surrogate = torch.minimum(ratio * g_adv, clipped_ratio * g_adv)
                pg_term = -(surrogate * g_mask).sum() / g_len
            else:
                # If we do not have old log-probs, we fall back to a simpler loss.
                # This is basically: increase log-probs for good rollouts,
                # decrease them for bad rollouts.
                pg_term = -(g_adv * log_probs * g_mask).sum() / g_len

            # Start with no KL penalty.
            kl_term = torch.zeros((), device=log_probs.device, dtype=log_probs.dtype)
            if self._ref_logprobs is not None:
                ref_lp = self._ref_logprobs[gen_idx].to(log_probs.device, non_blocking=True)

                # KL penalty keeps the current policy close to a reference policy.
                # That helps stop training from drifting too far too quickly.
                kl_term = compute_kl_penalty(
                    log_probs.detach(), ref_lp, g_mask
                ).to(log_probs.dtype)

            # Final loss = policy-gradient part + optional KL penalty.
            return pg_term + (self.config.kl_coeff * kl_term)

        def loss_fn_batch(
            batch_log_probs: Tensor,
            batch_mask: Tensor,
            hidden_comp: Optional[Tensor] = None,
        ) -> Tensor:
            del hidden_comp
            device = batch_log_probs.device
            mask = batch_mask.to(device, non_blocking=True).float()

            # `adv` becomes shape [G, 1] so it can be broadcast across tokens.
            adv = advantages.to(device, non_blocking=True).view(-1, 1)

            # Number of real tokens in each rollout.
            lengths = mask.sum(dim=1).clamp(min=1.0)

            if self._old_logprobs is not None:
                old_lp = self._old_logprobs.to(device, non_blocking=True)
                ratio = torch.exp(batch_log_probs - old_lp)
                adv_full = adv.expand_as(ratio)

                # Same PPO logic as above, but done for the whole batch at once.
                clipped_ratio = apply_ppo_clip(
                    ratio=ratio,
                    advantage=adv_full,
                    clip_low=self.config.clip_ratio_low,
                    clip_high=self.config.clip_ratio_high,
                )
                surrogate = torch.minimum(ratio * adv_full, clipped_ratio * adv_full)
                pg_term = -((surrogate * mask).sum(dim=1) / lengths)
            else:
                # Same simple fallback loss, but vectorized over the batch.
                pg_term = -((adv * batch_log_probs * mask).sum(dim=1) / lengths)

            total = pg_term
            if self._ref_logprobs is not None:
                ref_lp = self._ref_logprobs.to(device, non_blocking=True)

                # Add KL per rollout, then average the final losses.
                kl = ((batch_log_probs.detach() - ref_lp) * mask).sum(dim=1) / lengths
                total = total + (float(self.config.kl_coeff) * kl)
            return total.mean()

        # Some callers compute one rollout at a time, others compute a whole batch.
        # We attach the batch version here so both entry points live together.
        setattr(loss_fn, "loss_fn_batch", loss_fn_batch)
        return loss_fn
