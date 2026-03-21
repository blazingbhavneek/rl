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
        self.setup(model)

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
        base, inner = self.adapter.unwrap(model)
        layers = self.adapter.get_layers(inner)
        device = input_ids.device

        with torch.inference_mode():
            t = input_ids.shape[1]
            position_ids = torch.arange(t, device=device).unsqueeze(0)
            hidden = self.adapter.get_embed_tokens(inner)(input_ids)

            if self.split_layer > 0:
                pos_emb = self.adapter.get_position_embeddings(inner, hidden, position_ids)
                for idx in range(self.split_layer):
                    lt = self.layer_types.get(idx, "full_attention")
                    hidden = self.adapter.layer_forward(layers[idx], hidden, pos_emb, lt)
                    if idx % 4 == 0 and hidden.is_cuda:
                        torch.cuda.empty_cache()

        if self.config.offload_prefix_cpu:
            return hidden.cpu(), position_ids.cpu()
        return hidden, position_ids

    def _run_lora_suffix(
        self,
        model: nn.Module,
        hidden_prefix: Tensor,
        pos_ids: Tensor,
        completion_len: int,
    ) -> Tensor:
        _, inner = self.adapter.unwrap(model)
        layers = self.adapter.get_layers(inner)
        device = next(model.parameters()).device

        # Prefix is produced under inference_mode; clone to regular tensors so
        # checkpoint/autograd can save them for backward safely.
        hidden = hidden_prefix.to(device).clone()
        pos_ids = pos_ids.to(device).clone()

        pos_emb = self.adapter.get_position_embeddings(inner, hidden, pos_ids)

        for i in range(self.split_layer, len(layers)):
            layer = layers[i]
            lt = self.layer_types.get(i, "full_attention")

            if self.config.use_grad_checkpoint and torch.is_grad_enabled():
                def _fn(h: Tensor, cos: Tensor, sin: Tensor, _layer=layer, _lt=lt) -> Tensor:
                    return self.adapter.layer_forward(_layer, h, (cos, sin), _lt)

                hidden = torch_checkpoint(_fn, hidden, pos_emb[0], pos_emb[1], use_reentrant=False)
            else:
                hidden = self.adapter.layer_forward(layer, hidden, pos_emb, lt)

        normed = self.adapter.get_final_norm(inner)(hidden)
        return normed[0, -completion_len:]

    def _chunked_lm_head_logprobs(
        self,
        lm_head: nn.Module,
        hidden_comp: Tensor,
        token_ids: Tensor,
    ) -> Tensor:
        token_ids = token_ids.to(hidden_comp.device, non_blocking=True)
        t_c = hidden_comp.shape[0]
        chunk = max(1, self.config.logit_chunk)
        if self.chunk_profiler is not None:
            try:
                chunk = max(1, int(self.chunk_profiler.get_chunk_size(int(t_c))))
            except Exception:
                chunk = max(1, self.config.logit_chunk)

        lse = torch.empty(t_c, device=hidden_comp.device, dtype=torch.float32)
        weight_detached = lm_head.weight.detach()

        with torch.no_grad():
            for t0 in range(0, t_c, chunk):
                t1 = min(t0 + chunk, t_c)
                h = hidden_comp[t0:t1].detach().float()
                logits = h @ weight_detached.float().T
                if getattr(lm_head, "bias", None) is not None:
                    logits = logits + lm_head.bias.detach().float()
                lse[t0:t1] = torch.logsumexp(logits, dim=-1)
                del h, logits

        w_sel = lm_head.weight[token_ids]
        tok_logits = (hidden_comp * w_sel).sum(dim=-1)
        if getattr(lm_head, "bias", None) is not None:
            tok_logits = tok_logits + lm_head.bias[token_ids]

        log_probs = tok_logits.float() - lse
        return log_probs.to(hidden_comp.dtype)

    def _compute_logprobs_impl(
        self,
        model: nn.Module,
        prompt_ids: Tensor,
        completion_ids: Tensor,
        with_grad: bool,
    ) -> Tensor:
        if completion_ids.ndim != 1:
            raise ValueError("completion_ids must be shape (T_c,)")
        full_ids = torch.cat([prompt_ids, completion_ids.unsqueeze(0)], dim=1)
        completion_len = completion_ids.shape[0]

        base, _ = self.adapter.unwrap(model)
        lm_head = self.adapter.get_lm_head(base)

        if with_grad:
            hidden_prefix, pos_ids = self._run_frozen_prefix(model, full_ids)
            hidden_comp = self._run_lora_suffix(model, hidden_prefix, pos_ids, completion_len)
            return self._chunked_lm_head_logprobs(lm_head, hidden_comp, completion_ids)

        with torch.inference_mode():
            hidden_prefix, pos_ids = self._run_frozen_prefix(model, full_ids)
            hidden_comp = self._run_lora_suffix(model, hidden_prefix, pos_ids, completion_len)
            return self._chunked_lm_head_logprobs(lm_head, hidden_comp, completion_ids).detach().cpu()

    def compute_logprobs(
        self,
        model: nn.Module,
        prompt_ids: Tensor,
        completion_ids: Tensor,
        lora_path: Optional[str] = None,
    ) -> Tensor:
        if lora_path is not None:
            self.load_lora(lora_path)
        return self._compute_logprobs_impl(model, prompt_ids, completion_ids, with_grad=True)

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
                return self._compute_logprobs_impl(model, prompt_ids, completion_ids, with_grad=False)
            finally:
                model.enable_adapter_layers()

        # Case 2: requested path is already active
        if self._current_lora_path == lora_path:
            return self._compute_logprobs_impl(model, prompt_ids, completion_ids, with_grad=False)

        # Case 3: temporary swap to a different checkpoint
        state = self.get_current_lora_state()
        prev_path = self._current_lora_path
        self.load_lora(lora_path)
        try:
            out = self._compute_logprobs_impl(model, prompt_ids, completion_ids, with_grad=False)
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

        g = completion_ids.shape[0]
        total_loss = 0.0
        total_logp = 0.0
        total_tokens = 0.0

        iterator = range(g)
        if bool(getattr(self, "enable_tqdm", False)):
            iterator = tqdm(iterator, total=g, desc="backprop", leave=False)
        for idx in iterator:
            log_probs = self.compute_logprobs(model, prompt_ids, completion_ids[idx], lora_path=None)
            loss = loss_fn(log_probs, idx)
            (loss * loss_scale).backward()
            progress_cb = getattr(self, "progress_callback", None)
            if callable(progress_cb):
                try:
                    progress_cb(idx + 1, g)
                except Exception:
                    pass

            with torch.no_grad():
                mask = completion_mask[idx].to(log_probs.device)
                tok = mask.sum().item()
                if tok > 0:
                    total_logp += (log_probs * mask).sum().item()
                    total_tokens += tok
                total_loss += float(loss.item())

            del log_probs, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return {
            "loss": total_loss / max(1, g),
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
