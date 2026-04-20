from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, List, Optional

import torch
import torch.nn.functional as F
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
    from .utils import masked_mean, normalize_advantages
except ImportError:  # pragma: no cover - direct script execution fallback
    from base import AlgoConfig, AlgoOutput, BaseAlgo
    from utils import masked_mean, normalize_advantages


@dataclass
class SDPOConfig:
    # Credit assignment mode for fallback sampled-token objective.
    credit_assignment: str = "logit"  # "logit" | "token" | "sequence"

    # Teacher regularization.
    teacher_reg: str = "ema"  # "ema" | "trust_region" | "none"
    teacher_alpha: float = 0.05

    # Hybrid SDPO + GRPO weight.
    lambda_grpo: float = 0.0

    # Fallback sampled-token advantage clipping.
    clip_advantages: float = 5.0

    # Official SDPO self-distillation knobs.
    full_logit_distillation: bool = True
    distillation_topk: Optional[int] = 100
    distillation_add_tail: bool = True
    alpha: float = 0.5  # 0.0=forward KL, 1.0=reverse KL, middle=JSD-like interpolation
    is_clip: Optional[float] = None

    # Optional adapter routing names (single-base multi-LoRA setup).
    student_adapter: Optional[str] = None
    teacher_adapter: Optional[str] = None
    ref_adapter: Optional[str] = None

    # Backward-compatible aliases from older config shape.
    top_k: Optional[int] = None
    divergence: Optional[str] = None

    def __post_init__(self) -> None:
        if self.top_k is not None and self.distillation_topk is None:
            self.distillation_topk = int(self.top_k)
        if self.divergence:
            d = str(self.divergence).lower()
            if d == "forward_kl":
                self.alpha = 0.0
            elif d == "reverse_kl":
                self.alpha = 1.0
            elif d == "js":
                self.alpha = 0.5


class SDPOAlgo(BaseAlgo):
    def __init__(
        self,
        config: Optional[AlgoConfig],
        sdpo_config: Optional[SDPOConfig],
        model: Any,
        *,
        tokenizer: Optional[Any] = None,
    ) -> None:
        super().__init__(config or AlgoConfig())
        self.sdpo_config = sdpo_config or SDPOConfig()
        self.model = model
        self.tokenizer = tokenizer

        self._old_logprobs: Optional[Tensor] = None
        self._ema_model: Optional[Any] = None

        # Keep EMA full-model path for non-adapter setups.
        if (
            self.sdpo_config.teacher_reg == "ema"
            and self.sdpo_config.teacher_adapter is None
        ):
            try:
                self._ema_model = copy.deepcopy(model).eval()
                if hasattr(self._ema_model, "parameters"):
                    for p in self._ema_model.parameters():
                        p.requires_grad_(False)
            except Exception:
                self._ema_model = None

    def requires_rich_feedback(self) -> bool:
        return True

    @property
    def needs_hidden_states(self) -> bool:
        return bool(
            self.sdpo_config.full_logit_distillation
            or self.sdpo_config.credit_assignment == "logit"
        )

    def bind_old_logprobs(self, old_logprobs: Optional[Tensor]) -> None:
        self._old_logprobs = old_logprobs

    def bind_adapter_names(
        self,
        *,
        student_adapter: Optional[str] = None,
        teacher_adapter: Optional[str] = None,
        ref_adapter: Optional[str] = None,
    ) -> None:
        if student_adapter is not None:
            self.sdpo_config.student_adapter = student_adapter
        if teacher_adapter is not None:
            self.sdpo_config.teacher_adapter = teacher_adapter
        if ref_adapter is not None:
            self.sdpo_config.ref_adapter = ref_adapter

    def _decode(self, token_ids: Tensor) -> str:
        if self.tokenizer is None or not hasattr(self.tokenizer, "decode"):
            return " ".join(str(int(x)) for x in token_ids.tolist())
        return self.tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)

    def _encode(self, text: str, device: torch.device) -> Tensor:
        if self.tokenizer is None:
            raise RuntimeError(
                "SDPOAlgo requires a tokenizer to build teacher contexts."
            )

        if hasattr(self.tokenizer, "encode"):
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            return torch.tensor([ids], dtype=torch.long, device=device)

        encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        return encoded.input_ids.to(device)

    def _model_device(self, model_obj: Any) -> torch.device:
        if hasattr(model_obj, "parameters"):
            return next(model_obj.parameters()).device
        if hasattr(model_obj, "model") and hasattr(model_obj.model, "parameters"):
            return next(model_obj.model.parameters()).device
        raise RuntimeError(
            "SDPOAlgo requires model.parameters() or model.model.parameters()."
        )

    def _base_model_obj(self, model_obj: Any) -> Any:
        return model_obj.model if hasattr(model_obj, "model") else model_obj

    def _get_active_adapter(self) -> Optional[str]:
        base = self._base_model_obj(self.model)
        active = getattr(base, "active_adapter", None)
        if isinstance(active, str):
            return active
        if isinstance(active, list) and active:
            return str(active[0])
        if isinstance(active, tuple) and active:
            return str(active[0])
        return None

    def _set_adapter(self, adapter_name: Optional[str]) -> None:
        # None means "base / adapters disabled" when supported.
        if adapter_name is None:
            base = self._base_model_obj(self.model)
            if hasattr(base, "disable_adapter_layers"):
                base.disable_adapter_layers()
            return

        base = self._base_model_obj(self.model)
        if hasattr(base, "enable_adapter_layers"):
            base.enable_adapter_layers()

        if hasattr(self.model, "set_active_lora_adapter"):
            self.model.set_active_lora_adapter(adapter_name)
            return
        if hasattr(base, "set_adapter"):
            base.set_adapter(adapter_name)
            return
        raise RuntimeError("Model does not support LoRA adapter switching.")

    @contextmanager
    def _using_adapter(self, adapter_name: Optional[str]) -> Iterator[None]:
        prev = self._get_active_adapter()
        try:
            self._set_adapter(adapter_name)
            yield
        finally:
            # Restore previous adapter state.
            if prev is None:
                self._set_adapter(None)
            else:
                self._set_adapter(prev)

    def build_teacher_context(
        self,
        prompt_ids: Tensor,
        completion_text: str,
        feedback_text: str,
        peer_solution: Optional[str],
    ) -> Tensor:
        model_device = self._model_device(self.model)
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
            # Completion text intentionally omitted, consistent with SDPO ablations.
            _ = completion_text
        teacher_prompt = "\n".join(lines)
        return self._encode(teacher_prompt, model_device)

    def _teacher_forward_targets(
        self,
        teacher_model: Any,
        teacher_prompt_ids: Tensor,
        completion_ids: Tensor,
    ) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        if completion_ids.ndim == 1:
            completion_ids = completion_ids.unsqueeze(0)
        if completion_ids.ndim != 2:
            raise ValueError(
                f"completion_ids must be 1D or 2D, got shape={tuple(completion_ids.shape)}"
            )

        device = self._model_device(teacher_model)
        completion_ids = completion_ids.to(device, non_blocking=True)
        prompt_ids = teacher_prompt_ids.to(device, non_blocking=True)
        if prompt_ids.shape[0] == 1 and completion_ids.shape[0] > 1:
            prompt_ids = prompt_ids.expand(completion_ids.shape[0], prompt_ids.shape[1])

        prompt_mask = torch.ones_like(prompt_ids, dtype=torch.float32)
        completion_mask = torch.ones_like(completion_ids, dtype=torch.float32)
        full_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        full_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        completion_len = completion_ids.shape[1]

        with torch.inference_mode():
            hidden_prefix, pos_ids = teacher_model._forward_prefix(full_ids, full_mask)
            hidden_suffix = teacher_model._forward_suffix(
                hidden_prefix, pos_ids, full_mask
            )
            hidden_comp = hidden_suffix[:, -(completion_len + 1) : -1, :]

            sampled_lp = (
                teacher_model._token_logprobs_chunked(hidden_comp, completion_ids)
                .detach()
                .float()
                .cpu()
            )

            if not self.sdpo_config.full_logit_distillation:
                return sampled_lp, None, None

            k = self.sdpo_config.distillation_topk
            if k is None or int(k) <= 0:
                # In this codebase we only use top-k full-logit distillation for memory safety.
                return sampled_lp, None, None

            lm_head = teacher_model._lm_head
            seq_len = hidden_comp.shape[1]
            chunk = max(
                1, int(getattr(teacher_model, "chunk_size", seq_len) or seq_len)
            )
            topk_logprob_chunks: List[Tensor] = []
            topk_index_chunks: List[Tensor] = []

            for start in range(0, seq_len, chunk):
                end = min(start + chunk, seq_len)
                h_slice = hidden_comp[:, start:end, :]
                logits = lm_head(h_slice)
                log_probs = torch.log_softmax(logits.float(), dim=-1)
                k_eff = min(int(k), int(log_probs.shape[-1]))
                topk_lp, topk_idx = torch.topk(log_probs, k=k_eff, dim=-1)
                topk_logprob_chunks.append(topk_lp.detach().cpu())
                topk_index_chunks.append(topk_idx.detach().cpu())

            teacher_topk_lp = torch.cat(topk_logprob_chunks, dim=1)
            teacher_topk_idx = torch.cat(topk_index_chunks, dim=1)
            return sampled_lp, teacher_topk_lp, teacher_topk_idx

    @staticmethod
    def _add_tail(log_probs: Tensor) -> Tensor:
        # log(1 - sum(p_i)) with numerical stability in log-space.
        log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
        log_s = torch.clamp(log_s, max=-1e-7)
        tail_log = torch.log(-torch.expm1(log_s))
        return torch.cat([log_probs, tail_log], dim=-1)

    @staticmethod
    def _renorm_topk_log_probs(log_probs: Tensor) -> Tensor:
        log_z = torch.logsumexp(log_probs, dim=-1, keepdim=True)
        return log_probs - log_z

    def _student_topk_logprobs(
        self,
        hidden_comp: Tensor,
        teacher_topk_idx: Tensor,
    ) -> Tensor:
        # hidden_comp: [T, H] or [1, T, H], teacher_topk_idx: [T, K]
        if hidden_comp.ndim == 2:
            hidden_comp = hidden_comp.unsqueeze(0)
        if hidden_comp.ndim != 3:
            raise ValueError(
                f"hidden_comp must be shape [T,H] or [B,T,H], got {tuple(hidden_comp.shape)}"
            )

        teacher_topk_idx = teacher_topk_idx.to(hidden_comp.device, non_blocking=True)
        lm_head = self.model._lm_head
        seq_len = hidden_comp.shape[1]
        chunk = max(1, int(getattr(self.model, "chunk_size", seq_len) or seq_len))
        gathered_chunks: List[Tensor] = []

        for start in range(0, seq_len, chunk):
            end = min(start + chunk, seq_len)
            h_slice = hidden_comp[:, start:end, :]
            idx_slice = teacher_topk_idx[start:end].unsqueeze(0)
            logits = lm_head(h_slice)
            student_log_probs = torch.log_softmax(logits.float(), dim=-1)
            gathered = student_log_probs.gather(dim=-1, index=idx_slice)
            gathered_chunks.append(gathered)

        return torch.cat(gathered_chunks, dim=1).squeeze(0)

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

        model_device = self._model_device(self.model)

        teacher_model = (
            self._ema_model
            if self.sdpo_config.teacher_reg == "ema" and self._ema_model is not None
            else self.model
        )
        teacher_adapter = self.sdpo_config.teacher_adapter
        student_adapter = self.sdpo_config.student_adapter
        ref_adapter = self.sdpo_config.ref_adapter

        teacher_logprobs: List[Tensor] = []
        teacher_topk_logprobs: List[Optional[Tensor]] = []
        teacher_topk_indices: List[Optional[Tensor]] = []
        distill_mask = torch.zeros(g, dtype=torch.float32)

        n_success = 0
        used_peer = 0

        for i in range(g):
            sc = scores[i]
            success = bool(sc.total > 0 and sc.passed == sc.total)
            n_success += int(success)
            feedback_text = feedback[i] or (sc.error or "")

            completion_text = self._decode(completion_ids[i].detach().cpu())
            if success:
                context_peer_solution = completion_text
            else:
                context_peer_solution = peer_solution
                used_peer += int(context_peer_solution is not None)

            # Match official self_distillation_mask semantics:
            # distill only when reprompt context contains solution or feedback.
            has_distill_signal = bool(context_peer_solution) or bool(
                str(feedback_text).strip()
            )
            distill_mask[i] = 1.0 if has_distill_signal else 0.0

            teacher_prompt_ids = self.build_teacher_context(
                prompt_ids=prompt_ids.detach().cpu(),
                completion_text=completion_text,
                feedback_text=feedback_text,
                peer_solution=context_peer_solution,
            )

            comp_i = completion_ids[i].to(model_device, non_blocking=True)

            if teacher_model is self.model:
                with self._using_adapter(teacher_adapter):
                    t_lp, t_topk_lp, t_topk_idx = self._teacher_forward_targets(
                        teacher_model=self.model,
                        teacher_prompt_ids=teacher_prompt_ids,
                        completion_ids=comp_i,
                    )
            else:
                t_lp, t_topk_lp, t_topk_idx = self._teacher_forward_targets(
                    teacher_model=teacher_model,
                    teacher_prompt_ids=teacher_prompt_ids,
                    completion_ids=comp_i,
                )

            if self.sdpo_config.teacher_reg == "trust_region":
                with self._using_adapter(ref_adapter):
                    ref_lp, _, _ = self._teacher_forward_targets(
                        teacher_model=self.model,
                        teacher_prompt_ids=teacher_prompt_ids,
                        completion_ids=comp_i,
                    )
                a = float(self.sdpo_config.teacher_alpha)
                t_lp = (1.0 - a) * ref_lp + a * t_lp

            if t_lp.numel() != t_c:
                raise RuntimeError(
                    f"Teacher logprob length mismatch: got {t_lp.numel()} expected {t_c}"
                )

            teacher_logprobs.append(t_lp)
            teacher_topk_logprobs.append(
                t_topk_lp[0] if t_topk_lp is not None else None
            )
            teacher_topk_indices.append(
                t_topk_idx[0] if t_topk_idx is not None else None
            )

        # Ensure training path runs student adapter.
        with self._using_adapter(student_adapter):
            pass

        grpo_advantages: Optional[Tensor] = None
        if self.sdpo_config.lambda_grpo > 0.0:
            grpo_advantages = normalize_advantages(
                torch.tensor(rewards, dtype=torch.float32)
            )

        loss_fn = self._make_loss_fn(
            teacher_logprobs=teacher_logprobs,
            teacher_topk_logprobs=teacher_topk_logprobs,
            teacher_topk_indices=teacher_topk_indices,
            distill_mask=distill_mask,
            completion_mask=completion_mask.float(),
            credit_assignment=self.sdpo_config.credit_assignment,
            lambda_grpo=self.sdpo_config.lambda_grpo,
            grpo_advantages=grpo_advantages,
        )

        stacked_teacher = (
            torch.stack(teacher_logprobs) if teacher_logprobs else torch.empty(0)
        )
        stats = {
            "mean_reward": (
                float(torch.tensor(rewards).mean().item()) if rewards else 0.0
            ),
            "mean_teacher_logprob": (
                float(stacked_teacher.mean().item()) if stacked_teacher.numel() else 0.0
            ),
            "success_ratio": float(n_success / max(1, g)),
            "used_peer_ratio": float(used_peer / max(1, g)),
            "lambda_grpo": float(self.sdpo_config.lambda_grpo),
            "distill_mask_fraction": (
                float(distill_mask.mean().item()) if distill_mask.numel() else 0.0
            ),
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
        teacher_logprobs: List[Tensor],
        teacher_topk_logprobs: List[Optional[Tensor]],
        teacher_topk_indices: List[Optional[Tensor]],
        distill_mask: Tensor,
        completion_mask: Tensor,
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

        def _full_logit_distill(
            log_probs: Tensor, idx: int, hidden_comp: Optional[Tensor], mask: Tensor
        ) -> Optional[Tensor]:
            if not self.sdpo_config.full_logit_distillation:
                return None
            if hidden_comp is None:
                return None
            t_topk_lp = teacher_topk_logprobs[idx]
            t_topk_idx = teacher_topk_indices[idx]
            if t_topk_lp is None or t_topk_idx is None:
                return None

            teacher_distill_log_probs = t_topk_lp.to(
                log_probs.device, non_blocking=True
            )
            teacher_topk_idx_dev = t_topk_idx.to(log_probs.device, non_blocking=True)
            student_distill_log_probs = self._student_topk_logprobs(
                hidden_comp, teacher_topk_idx_dev
            )

            if self.sdpo_config.distillation_add_tail:
                student_distill_log_probs = self._add_tail(student_distill_log_probs)
                teacher_distill_log_probs = self._add_tail(teacher_distill_log_probs)
            else:
                student_distill_log_probs = self._renorm_topk_log_probs(
                    student_distill_log_probs
                )
                teacher_distill_log_probs = self._renorm_topk_log_probs(
                    teacher_distill_log_probs
                )

            alpha = float(self.sdpo_config.alpha)
            if alpha <= 0.0:
                kl_loss = F.kl_div(
                    student_distill_log_probs,
                    teacher_distill_log_probs,
                    reduction="none",
                    log_target=True,
                )
            elif alpha >= 1.0:
                kl_loss = F.kl_div(
                    teacher_distill_log_probs,
                    student_distill_log_probs,
                    reduction="none",
                    log_target=True,
                )
            else:
                alpha_t = torch.tensor(
                    alpha,
                    dtype=student_distill_log_probs.dtype,
                    device=student_distill_log_probs.device,
                )
                mixture_log_probs = torch.logsumexp(
                    torch.stack(
                        [
                            student_distill_log_probs + torch.log(1.0 - alpha_t),
                            teacher_distill_log_probs + torch.log(alpha_t),
                        ]
                    ),
                    dim=0,
                )
                kl_teacher = F.kl_div(
                    mixture_log_probs,
                    teacher_distill_log_probs,
                    reduction="none",
                    log_target=True,
                )
                kl_student = F.kl_div(
                    mixture_log_probs,
                    student_distill_log_probs,
                    reduction="none",
                    log_target=True,
                )
                kl_loss = torch.lerp(kl_student, kl_teacher, alpha_t)

            per_token_loss = kl_loss.sum(-1)

            # Optional IS clip on distillation term, as in official implementation.
            is_clip = self.sdpo_config.is_clip
            if is_clip is not None and self._old_logprobs is not None:
                old_lp = self._old_logprobs[idx].to(log_probs.device, non_blocking=True)
                ratio = torch.exp(
                    torch.clamp((log_probs - old_lp).detach(), min=-20.0, max=20.0)
                ).clamp(max=float(is_clip))
                per_token_loss = per_token_loss * ratio

            return masked_mean(per_token_loss, mask)

        def loss_fn(
            log_probs: Tensor, gen_idx: int, hidden_comp: Optional[Tensor] = None
        ) -> Tensor:
            mask = completion_mask[gen_idx].to(log_probs.device, non_blocking=True)
            sample_distill_mask = distill_mask[gen_idx].to(
                log_probs.device, non_blocking=True
            )
            mask = mask * sample_distill_mask

            full_logit_loss = _full_logit_distill(log_probs, gen_idx, hidden_comp, mask)
            if full_logit_loss is not None:
                sdpo_loss = full_logit_loss
            else:
                teacher_lp = teacher_logprobs[gen_idx].to(
                    log_probs.device, non_blocking=True
                )
                token_adv = (teacher_lp - log_probs.detach()).clamp(
                    min=-max_adv, max=max_adv
                ) * mask
                if credit_assignment == "sequence":
                    seq_adv = masked_mean(token_adv, mask)
                    sdpo_loss = -(seq_adv * log_probs * mask).sum() / mask.sum().clamp(
                        min=1.0
                    )
                else:
                    sdpo_loss = -masked_mean(token_adv * log_probs, mask)

            if lambda_grpo <= 0.0:
                return sdpo_loss
            # GRPO term uses original completion mask, independent of distillation gating.
            grpo_mask = completion_mask[gen_idx].to(log_probs.device, non_blocking=True)
            grpo_loss = _grpo_term(log_probs, gen_idx, grpo_mask)
            return lambda_grpo * grpo_loss + (1.0 - lambda_grpo) * sdpo_loss

        return loss_fn

    def update_ema_teacher(self, alpha: float = 0.01) -> None:
        a = float(alpha)
        if (
            self._ema_model is not None
            and hasattr(self._ema_model, "parameters")
            and hasattr(self.model, "parameters")
        ):
            with torch.inference_mode():
                for p_ema, p_cur in zip(
                    self._ema_model.parameters(), self.model.parameters()
                ):
                    p_ema.mul_(1.0 - a).add_(p_cur.detach(), alpha=a)
            return

        # Adapter-only EMA path (single base model with student/teacher adapters).
        student = self.sdpo_config.student_adapter
        teacher = self.sdpo_config.teacher_adapter
        if not student or not teacher or student == teacher:
            return

        base = self._base_model_obj(self.model)
        if not hasattr(base, "named_parameters"):
            return

        params = dict(base.named_parameters())
        with torch.inference_mode():
            for name, p_student in list(params.items()):
                if "lora_" not in name:
                    continue
                tag = f".{student}."
                if tag not in name:
                    continue
                t_name = name.replace(tag, f".{teacher}.")
                p_teacher = params.get(t_name)
                if p_teacher is None:
                    continue
                p_teacher.mul_(1.0 - a).add_(p_student.detach(), alpha=a)
