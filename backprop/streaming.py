from __future__ import annotations

from typing import Callable, Dict, Optional

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from tqdm.auto import tqdm

from .base import BackpropConfig, BaseBackprop
from .models.auto import get_adapter
from .models.base_adapter import BaseModelAdapter

try:
    from peft import set_peft_model_state_dict
    from peft.utils.save_and_load import load_peft_weights
except Exception:  # pragma: no cover - optional dependency at runtime
    set_peft_model_state_dict = None
    load_peft_weights = None


def _chunk_lse_no_bias(hidden_chunk_fp32: Tensor, lm_head_weight_fp32: Tensor) -> Tensor:
    logits = hidden_chunk_fp32 @ lm_head_weight_fp32.T
    return torch.logsumexp(logits, dim=-1)


def _chunk_lse_with_bias(
    hidden_chunk_fp32: Tensor,
    lm_head_weight_fp32: Tensor,
    lm_head_bias_fp32: Tensor,
) -> Tensor:
    logits = hidden_chunk_fp32 @ lm_head_weight_fp32.T
    logits = logits + lm_head_bias_fp32
    return torch.logsumexp(logits, dim=-1)


class StreamingBackprop(BaseBackprop):
    def __init__(
        self,
        model: nn.Module,
        adapter: Optional[BaseModelAdapter] = None,
        config: Optional[BackpropConfig] = None,
    ) -> None:
        adapter = adapter or get_adapter(model)
        config = config or BackpropConfig()
        super().__init__(model=model, adapter=adapter, config=config)

        self.layer_types: Dict[int, str] = {}
        self.split_layer: int = 0
        self._current_lora_path: Optional[str] = None
        self.chunk_profiler = None
        self._compiled_chunk_lse_no_bias = None
        self._compiled_chunk_lse_with_bias = None
        self._init_compiled_kernels()
        self.setup(model)

    def _init_compiled_kernels(self) -> None:
        if not bool(getattr(self.config, "use_torch_compile", False)):
            return
        if not hasattr(torch, "compile"):
            return
        try:
            self._compiled_chunk_lse_no_bias = torch.compile(_chunk_lse_no_bias, dynamic=True)
            self._compiled_chunk_lse_with_bias = torch.compile(_chunk_lse_with_bias, dynamic=True)
        except Exception:
            # Fall back silently; functional path remains identical.
            self._compiled_chunk_lse_no_bias = None
            self._compiled_chunk_lse_with_bias = None

    def setup(self, model: nn.Module) -> None:
        self.layer_types = self.adapter.identify_layer_types(model)
        self.split_layer = self.adapter.get_split_index(model, self.config.top_frac)

        _, inner = self.adapter.unwrap(model)
        layers = self.adapter.get_layers(inner)
        n_layers = len(layers)
        if self.split_layer < 0 or self.split_layer > n_layers:
            raise ValueError(f"Invalid split_layer={self.split_layer} for {n_layers} layers")

        trainable = [
            name
            for name, p in model.named_parameters()
            if p.requires_grad and ("lora" in name.lower() or "adapter" in name.lower())
        ]
        if not trainable:
            raise ValueError("No trainable LoRA parameters found. Ensure PEFT LoRA is attached.")

    def _run_frozen_prefix(self, model: nn.Module, input_ids: Tensor) -> tuple[Tensor, Tensor]:
        # input_ids: (B, T_total)
        # Prefix path is intentionally frozen (no grad) so we only backprop
        # through the trainable suffix layers.
        _, inner = self.adapter.unwrap(model)
        layers = self.adapter.get_layers(inner)
        device = input_ids.device

        with torch.inference_mode():
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
            hidden_states = self.adapter.get_embed_tokens(inner)(input_ids)

            if self.split_layer <= 0:
                return hidden_states, position_ids

            # Compute positional embeddings once and reuse across prefix layers.
            position_embeddings = self.adapter.get_position_embeddings(inner, hidden_states, position_ids)
            for layer_idx in range(self.split_layer):
                layer_type = self.layer_types.get(layer_idx, "full_attention")
                hidden_states = self.adapter.layer_forward(
                    layers[layer_idx],
                    hidden_states,
                    position_embeddings,
                    layer_type,
                )

        # hidden_states: (B, T_total, H), position_ids: (B, T_total)
        return hidden_states, position_ids

    def _run_lora_suffix(
        self,
        model: nn.Module,
        hidden_prefix: Tensor,
        pos_ids: Tensor,
        completion_len: int,
    ) -> Tensor:
        # hidden_prefix: (B, T_total, H), pos_ids: (B, T_total)
        _, inner = self.adapter.unwrap(model)
        layers = self.adapter.get_layers(inner)
        device = next(model.parameters()).device

        # Prefix is produced under inference_mode. Clone into regular tensors so
        # autograd/checkpoint can safely retain what backward needs.
        # `.to(device)` is mostly a no-op now, but keeps this robust if callers
        # ever provide tensors on a different device.
        hidden_states = hidden_prefix.to(device).clone()
        position_ids = pos_ids.to(device).clone()

        # Build positional embeddings once for the full suffix pass.
        position_embeddings = self.adapter.get_position_embeddings(inner, hidden_states, position_ids)

        for layer_idx in range(self.split_layer, len(layers)):
            layer = layers[layer_idx]
            layer_type = self.layer_types.get(layer_idx, "full_attention")

            if self.config.use_grad_checkpoint and torch.is_grad_enabled():
                # Per-layer checkpointing keeps activation memory bounded in the
                # trainable suffix at the cost of recomputation.
                def _fn(h: Tensor, cos: Tensor, sin: Tensor, _layer=layer, _lt=layer_type) -> Tensor:
                    return self.adapter.layer_forward(_layer, h, (cos, sin), _lt)

                hidden_states = torch_checkpoint(
                    _fn,
                    hidden_states,
                    position_embeddings[0],
                    position_embeddings[1],
                    use_reentrant=False,
                )
            else:
                hidden_states = self.adapter.layer_forward(
                    layer,
                    hidden_states,
                    position_embeddings,
                    layer_type,
                )

        # Return only completion-token hidden states used for token logprobs.
        # Shape: (B, T_completion, H)
        normed_hidden = self.adapter.get_final_norm(inner)(hidden_states)
        return normed_hidden[:, -completion_len:]

    @staticmethod
    def _normalize_prompt_batch(prompt_ids: Tensor, batch_size: int) -> Tensor:
        # prompt_ids expected shape: (1, T_prompt) or (B, T_prompt)
        if prompt_ids.ndim != 2:
            raise ValueError("prompt_ids must be shape (B, T_prompt) or (1, T_prompt)")
        if prompt_ids.shape[0] == batch_size:
            return prompt_ids
        if prompt_ids.shape[0] == 1:
            return prompt_ids.expand(batch_size, prompt_ids.shape[1])
        raise ValueError(
            f"prompt_ids batch ({prompt_ids.shape[0]}) must be 1 or completion batch ({batch_size})"
        )

    def _normalize_completion_batch(self, completion_ids: Tensor) -> tuple[Tensor, bool]:
        # completion_ids accepted shapes:
        # - (T_completion,) for single sample
        # - (B, T_completion) for batched path
        if completion_ids.ndim == 1:
            return completion_ids.unsqueeze(0), True
        if completion_ids.ndim == 2:
            return completion_ids, False
        raise ValueError("completion_ids must be shape (T_completion,) or (B, T_completion)")

    def _chunked_lm_head_logprobs(
        self,
        lm_head: nn.Module,
        hidden_comp: Tensor,
        token_ids: Tensor,
    ) -> Tensor:
        # Goal: compute token-level log-probabilities without materializing the
        # full (B, T_completion, vocab_size) logits tensor at once.
        #
        # accepted:
        # - hidden_comp: (T_completion, H), token_ids: (T_completion,)
        # - hidden_comp: (B, T_completion, H), token_ids: (B, T_completion)
        squeeze_output = False
        if hidden_comp.ndim == 2 and token_ids.ndim == 1:
            hidden_comp = hidden_comp.unsqueeze(0)
            token_ids = token_ids.unsqueeze(0)
            squeeze_output = True
        elif hidden_comp.ndim != 3 or token_ids.ndim != 2:
            raise ValueError(
                "hidden_comp/token_ids must be (T_completion,H)/(T_completion,) "
                "or (B,T_completion,H)/(B,T_completion)"
            )

        token_ids = token_ids.to(hidden_comp.device, non_blocking=True)
        batch_size = hidden_comp.shape[0]
        completion_len = hidden_comp.shape[1]
        num_completion_tokens = batch_size * completion_len
        chunk_size = max(1, self.config.logit_chunk)
        if self.chunk_profiler is not None:
            try:
                chunk_size = max(
                    1,
                    int(
                        self.chunk_profiler.get_chunk_size(
                            int(completion_len),
                            batch_size=int(batch_size),
                        )
                    ),
                )
            except Exception:
                chunk_size = max(1, self.config.logit_chunk)

        # Flatten (B, T_completion, H) -> (B*T_completion, H) so chunking logic
        # stays simple and readable.
        flat_hidden = hidden_comp.reshape(num_completion_tokens, hidden_comp.shape[-1])
        flat_token_ids = token_ids.reshape(num_completion_tokens)

        # logsumexp over vocabulary for each completion token.
        logsumexp_per_token = torch.empty(
            num_completion_tokens,
            device=flat_hidden.device,
            dtype=torch.float32,
        )
        lm_head_weight = lm_head.weight
        lm_head_weight_fp32 = lm_head_weight.detach().float()
        lm_head_bias = getattr(lm_head, "bias", None)
        lm_head_bias_fp32 = lm_head_bias.detach().float() if lm_head_bias is not None else None

        lse_no_bias = self._compiled_chunk_lse_no_bias or _chunk_lse_no_bias
        lse_with_bias = self._compiled_chunk_lse_with_bias or _chunk_lse_with_bias
        with torch.no_grad():
            for token_start in range(0, num_completion_tokens, chunk_size):
                token_end = min(token_start + chunk_size, num_completion_tokens)
                hidden_chunk_fp32 = flat_hidden[token_start:token_end].detach().float()
                if lm_head_bias_fp32 is None:
                    logsumexp_per_token[token_start:token_end] = lse_no_bias(
                        hidden_chunk_fp32,
                        lm_head_weight_fp32,
                    )
                else:
                    logsumexp_per_token[token_start:token_end] = lse_with_bias(
                        hidden_chunk_fp32,
                        lm_head_weight_fp32,
                        lm_head_bias_fp32,
                    )

        # Gather only target-token weights so this path still carries gradients
        # from loss -> hidden states without constructing full logits.
        selected_token_weights = lm_head_weight[flat_token_ids]
        selected_token_logits = (flat_hidden * selected_token_weights).sum(dim=-1)
        if lm_head_bias is not None:
            selected_token_logits = selected_token_logits + lm_head_bias[flat_token_ids]

        token_log_probs = selected_token_logits.float() - logsumexp_per_token
        batch_token_log_probs = token_log_probs.to(hidden_comp.dtype).reshape(batch_size, completion_len)
        if squeeze_output:
            return batch_token_log_probs[0]
        return batch_token_log_probs

    def compute_logprobs(
        self,
        model: nn.Module,
        prompt_ids: Tensor,
        completion_ids: Tensor,
        lora_path: Optional[str] = None,
    ) -> Tensor:
        # prompt_ids: (1, T_prompt) or (B, T_prompt)
        # completion_ids: (T_completion,) or (B, T_completion)
        if lora_path is not None:
            self.load_lora(lora_path)
        completion_batch, squeeze_output = self._normalize_completion_batch(completion_ids)
        batch_size = completion_batch.shape[0]
        prompt_batch = self._normalize_prompt_batch(prompt_ids, batch_size=batch_size)

        # Concat prompt and completion for one model forward path.
        full_ids = torch.cat([prompt_batch, completion_batch], dim=1)
        completion_len = completion_batch.shape[1]
        base, _ = self.adapter.unwrap(model)
        lm_head = self.adapter.get_lm_head(base)

        hidden_prefix, pos_ids = self._run_frozen_prefix(model, full_ids)
        hidden_comp = self._run_lora_suffix(model, hidden_prefix, pos_ids, completion_len)
        batch_log_probs = self._chunked_lm_head_logprobs(lm_head, hidden_comp, completion_batch)
        if squeeze_output:
            return batch_log_probs[0]
        return batch_log_probs

    def compute_ref_logprobs(
        self,
        model: nn.Module,
        prompt_ids: Tensor,
        completion_ids: Tensor,
        lora_path: Optional[str] = None,
    ) -> Tensor:
        # Case 1: reference under base weights (disable adapters, no disk IO)
        if lora_path is None:
            if not hasattr(model, "disable_adapter_layers") or not hasattr(model, "enable_adapter_layers"):
                raise RuntimeError(
                    "Model does not expose disable_adapter_layers/enable_adapter_layers, "
                    "cannot compute base-model reference without LoRA."
                )
            model.disable_adapter_layers()
            try:
                # Base-model reference: adapters disabled, no grad required.
                with torch.inference_mode():
                    out = self.compute_logprobs(
                        model=model,
                        prompt_ids=prompt_ids,
                        completion_ids=completion_ids,
                        lora_path=None,
                    )
                return out.detach()
            finally:
                model.enable_adapter_layers()

        # Case 2: requested path is already active
        if self._current_lora_path == lora_path:
            # Requested reference is already loaded; just run no-grad scoring.
            with torch.inference_mode():
                out = self.compute_logprobs(
                    model=model,
                    prompt_ids=prompt_ids,
                    completion_ids=completion_ids,
                    lora_path=None,
                )
            return out.detach()

        # Case 3: temporary swap to a different checkpoint
        # Save current LoRA tensors, load requested reference adapter, score,
        # then restore original adapter tensors.
        state = self.get_current_lora_state()
        prev_path = self._current_lora_path
        self.load_lora(lora_path)
        try:
            with torch.inference_mode():
                out = self.compute_logprobs(
                    model=model,
                    prompt_ids=prompt_ids,
                    completion_ids=completion_ids,
                    lora_path=None,
                )
            out = out.detach()
        finally:
            self.restore_lora_state(state)
            self._current_lora_path = prev_path
        return out

    def backward_on_batch(
        self,
        model: nn.Module,
        prompt_ids: Tensor,
        completion_ids: Tensor,
        completion_mask: Tensor,
        loss_fn: Callable,
        loss_scale: float = 1.0,
        lora_path: Optional[str] = None,
    ) -> Dict[str, float]:
        if lora_path is not None:
            self.load_lora(lora_path)

        completion_batch, _ = self._normalize_completion_batch(completion_ids)
        if completion_mask.ndim == 1:
            completion_mask_batch = completion_mask.unsqueeze(0)
        elif completion_mask.ndim == 2:
            completion_mask_batch = completion_mask
        else:
            raise ValueError("completion_mask must be shape (T_completion,) or (B, T_completion)")

        if completion_batch.shape != completion_mask_batch.shape:
            raise ValueError(
                f"completion_ids shape {tuple(completion_batch.shape)} "
                f"must match completion_mask shape {tuple(completion_mask_batch.shape)}"
            )

        g = completion_batch.shape[0]
        prompt_batch = self._normalize_prompt_batch(prompt_ids, batch_size=g)
        total_loss = 0.0
        total_logp = 0.0
        total_tokens = 0.0

        # Compute batched log-probabilities once (no per-sample recompute).
        batch_log_probs = self.compute_logprobs(
            model=model,
            prompt_ids=prompt_batch,
            completion_ids=completion_batch,
            lora_path=None,
        )

        progress_cb = getattr(self, "progress_callback", None)
        explicit_batch_loss_fn = getattr(loss_fn, "loss_fn_batch", None)
        batch_loss_callable = explicit_batch_loss_fn if callable(explicit_batch_loss_fn) else None
        if batch_loss_callable is None and getattr(loss_fn, "__name__", "") == "loss_fn_batch":
            batch_loss_callable = loss_fn

        if batch_loss_callable is not None:
            batch_loss = batch_loss_callable(batch_log_probs, completion_mask_batch, None)
            (batch_loss * loss_scale).backward()
            total_loss = float(batch_loss.item())
            if callable(progress_cb):
                try:
                    progress_cb(g, g)
                except Exception:
                    pass
        else:
            iterator = range(g)
            if bool(getattr(self, "enable_tqdm", False)):
                iterator = tqdm(iterator, total=g, desc="backprop", leave=False)
            for idx in iterator:
                sample_log_probs = batch_log_probs[idx]
                sample_loss = loss_fn(sample_log_probs, idx)
                # Legacy compatibility path: we reuse one batched forward graph
                # and run per-sample loss backward. Retain graph until last step.
                keep_graph = idx < (g - 1)
                (sample_loss * loss_scale).backward(retain_graph=keep_graph)
                total_loss += float(sample_loss.item())
                if callable(progress_cb):
                    try:
                        progress_cb(idx + 1, g)
                    except Exception:
                        pass

        with torch.no_grad():
            valid_token_mask = completion_mask_batch.to(batch_log_probs.device)
            token_count = float(valid_token_mask.sum().item())
            if token_count > 0:
                total_logp = float((batch_log_probs * valid_token_mask).sum().item())
                total_tokens = token_count

        mean_loss = total_loss if batch_loss_callable is not None else (total_loss / max(1, g))
        return {
            "loss": mean_loss,
            "mean_logp": total_logp / max(1.0, total_tokens),
        }

    def load_lora(self, path: str) -> None:
        if load_peft_weights is None or set_peft_model_state_dict is None:
            # Fallback for environments where PEFT utility import is unavailable.
            if hasattr(self.model, "load_adapter"):
                self.model.load_adapter(path, adapter_name="default", is_trainable=True)
                if hasattr(self.model, "set_adapter"):
                    self.model.set_adapter("default")
            else:
                raise RuntimeError("PEFT utilities unavailable and model.load_adapter not supported.")
        else:
            adapter_state = load_peft_weights(path, device="cpu")
            set_peft_model_state_dict(self.model, adapter_state)

        self._current_lora_path = path

    def save_lora(self, path: str) -> None:
        if not hasattr(self.model, "save_pretrained"):
            raise RuntimeError("Model does not support save_pretrained for adapter export.")
        self.model.save_pretrained(path)
        self._current_lora_path = path
