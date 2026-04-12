import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

from .base import BaseModel


@dataclass
class _PrefixBundle:
    hidden_prefix: torch.Tensor
    position_ids: torch.Tensor
    shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]]
    per_layer_inputs: torch.Tensor | None


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (_rotate_half(x) * sin)


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class _StreamCheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        run_function: Callable,
        chunk_size: int,
        offload_to_cpu: bool,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        ctx.run_function = run_function
        ctx.chunk_size = max(1, int(chunk_size))
        saved_hidden = hidden_states.detach()
        if offload_to_cpu and saved_hidden.device.type != "cpu":
            saved_hidden = saved_hidden.to("cpu")
        ctx.save_for_backward(saved_hidden)

        seq_len = hidden_states.shape[1]
        with torch.no_grad():
            if hasattr(ctx.run_function, "prepare_for_replay"):
                ctx.run_function.prepare_for_replay(hidden_states, requires_grad=False)
            outputs = []
            for start in range(0, seq_len, ctx.chunk_size):
                end = min(start + ctx.chunk_size, seq_len)
                outputs.append(run_function(hidden_states, chunk_range=(start, end)))
            if hasattr(ctx.run_function, "clear_replay_cache"):
                ctx.run_function.clear_replay_cache()
        return torch.cat(outputs, dim=1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (hidden_states_saved,) = ctx.saved_tensors
        hidden_states_detached = hidden_states_saved.to(
            device=grad_output.device,
            non_blocking=True,
        ).detach().requires_grad_(True)
        seq_len = grad_output.shape[1]
        shared_graph = False
        if hasattr(ctx.run_function, "prepare_for_replay"):
            with torch.enable_grad():
                ctx.run_function.prepare_for_replay(hidden_states_detached, requires_grad=True)
            shared_graph = True

        try:
            for start in range(0, seq_len, ctx.chunk_size):
                end = min(start + ctx.chunk_size, seq_len)
                with torch.enable_grad():
                    output_chunk = ctx.run_function(
                        hidden_states_detached,
                        chunk_range=(start, end),
                    )
                    torch.autograd.backward(
                        output_chunk,
                        grad_tensors=grad_output[:, start:end, :].detach(),
                        retain_graph=shared_graph and end < seq_len,
                    )
        finally:
            if hasattr(ctx.run_function, "clear_replay_cache"):
                ctx.run_function.clear_replay_cache()

        return None, None, None, hidden_states_detached.grad


def _stream_attention_forward(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    key_states = _repeat_kv(key, module.num_key_value_groups)
    value_states = _repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * module.scaling
    softcap = getattr(module.config, "attn_logit_softcapping", None)
    if softcap is not None:
        attn_weights = torch.tanh(attn_weights / softcap) * softcap
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    if module.training and module.attention_dropout:
        attn_weights = nn.functional.dropout(attn_weights, p=module.attention_dropout, training=True)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class _StreamGemmaAttention(nn.Module):
    def __init__(self, base_attn: nn.Module):
        super().__init__()
        self.base_attn = base_attn
        self._replay_key_states: torch.Tensor | None = None
        self._replay_value_states: torch.Tensor | None = None

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_attn, name)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None,
        shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]],
        past_key_values=None,
        chunk_range: tuple[int, int] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if chunk_range is None:
            return self.base_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                shared_kv_states=shared_kv_states,
                past_key_values=past_key_values,
                **kwargs,
            )

        batch_size, seq_len = hidden_states.shape[:2]
        start, end = chunk_range
        chunk_len = end - start
        cos, sin = position_embeddings

        query_states = self.q_proj(hidden_states).view(batch_size, chunk_len, -1, self.head_dim)
        query_states = self.q_norm(query_states)
        query_states = _apply_rotary_pos_emb(
            query_states,
            cos[:, start:end, :],
            sin[:, start:end, :],
            unsqueeze_dim=2,
        ).transpose(1, 2)

        if self._replay_key_states is not None and self._replay_value_states is not None:
            key_states = self._replay_key_states[:, :, :end, :]
            value_states = self._replay_value_states[:, :, :end, :]
        elif self.is_kv_shared_layer:
            key_states, value_states = shared_kv_states[self.kv_shared_layer_index]
            key_states = key_states.to(query_states.device, non_blocking=True)[:, :, :end, :]
            value_states = value_states.to(query_states.device, non_blocking=True)[:, :, :end, :]
        else:
            hidden_shape = (batch_size, -1, self.head_dim)
            kv_hidden = hidden_states[:, :end, :]
            key_states = self.k_proj(kv_hidden).view(batch_size, end, -1, self.head_dim)
            value_states = self.v_proj(kv_hidden).view(batch_size, end, -1, self.head_dim) if self.v_proj is not None else key_states
            key_states = self.k_norm(key_states)
            key_states = _apply_rotary_pos_emb(
                key_states,
                cos[:, :end, :],
                sin[:, :end, :],
                unsqueeze_dim=2,
            ).transpose(1, 2)
            value_states = self.v_norm(value_states).transpose(1, 2)

        if self.store_full_length_kv and not self.is_kv_shared_layer and end == seq_len:
            shared_kv_states[self.layer_idx] = (key_states, value_states)

        chunk_mask = None
        if attention_mask is not None:
            chunk_mask = attention_mask[:, :, start:end, :end]

        attn_output, attn_weights = _stream_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            chunk_mask,
        )
        attn_output = attn_output.reshape(batch_size, chunk_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    def prepare_replay_kv(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        cos, sin = position_embeddings
        if self.is_kv_shared_layer:
            key_states, value_states = shared_kv_states[self.kv_shared_layer_index]
            self._replay_key_states = key_states
            self._replay_value_states = value_states
            return

        batch_size, seq_len = hidden_states.shape[:2]
        key_states = self.k_proj(hidden_states).view(batch_size, seq_len, -1, self.head_dim)
        value_states = self.v_proj(hidden_states).view(batch_size, seq_len, -1, self.head_dim) if self.v_proj is not None else key_states
        key_states = self.k_norm(key_states)
        key_states = _apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2).transpose(1, 2)
        value_states = self.v_norm(value_states).transpose(1, 2)
        self._replay_key_states = key_states
        self._replay_value_states = value_states
        if self.store_full_length_kv:
            shared_kv_states[self.layer_idx] = (key_states, value_states)

    def clear_replay_cache(self) -> None:
        self._replay_key_states = None
        self._replay_value_states = None


class _StreamGemmaDecoderLayer(nn.Module):
    def __init__(self, base_layer: nn.Module):
        super().__init__()
        self.base_layer = base_layer
        if not isinstance(self.base_layer.self_attn, _StreamGemmaAttention):
            self.base_layer.self_attn = _StreamGemmaAttention(self.base_layer.self_attn)
        self._replay_shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_layer, name)

    def forward(
        self,
        hidden_states: torch.Tensor,
        per_layer_input: torch.Tensor = None,
        shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
        position_embeddings: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        chunk_range: tuple[int, int] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if chunk_range is None:
            return self.base_layer(
                hidden_states=hidden_states,
                per_layer_input=per_layer_input,
                shared_kv_states=shared_kv_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

        start, end = chunk_range
        chunk_len = end - start
        residual = hidden_states[:, start:end, :]
        use_replay_cache = self._replay_shared_kv_states is not None
        if use_replay_cache:
            hidden_states_norm = self.base_layer.input_layernorm(hidden_states[:, start:end, :])
        else:
            hidden_states_norm = self.base_layer.input_layernorm(hidden_states)
        attn_out, _ = self.base_layer.self_attn(
            hidden_states=hidden_states_norm,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            shared_kv_states=self._replay_shared_kv_states or shared_kv_states,
            position_ids=position_ids,
            past_key_values=past_key_values,
            chunk_range=chunk_range,
            **kwargs,
        )
        hidden_states_chunk = self.base_layer.post_attention_layernorm(attn_out)
        hidden_states_chunk = residual + hidden_states_chunk

        residual = hidden_states_chunk
        hidden_states_chunk = self.base_layer.pre_feedforward_layernorm(hidden_states_chunk)
        hidden_states_chunk = self.base_layer.mlp(hidden_states_chunk)

        if getattr(self.base_layer, "enable_moe_block", False):
            hidden_states_1 = self.base_layer.post_feedforward_layernorm_1(hidden_states_chunk)
            hidden_states_flat = residual.reshape(-1, residual.shape[-1])
            _, top_k_weights, top_k_index = self.base_layer.router(hidden_states_flat)
            hidden_states_2 = self.base_layer.pre_feedforward_layernorm_2(hidden_states_flat)
            hidden_states_2 = self.base_layer.experts(hidden_states_2, top_k_index, top_k_weights)
            hidden_states_2 = hidden_states_2.reshape(residual.shape)
            hidden_states_2 = self.base_layer.post_feedforward_layernorm_2(hidden_states_2)
            hidden_states_chunk = hidden_states_1 + hidden_states_2

        hidden_states_chunk = self.base_layer.post_feedforward_layernorm(hidden_states_chunk)
        hidden_states_chunk = residual + hidden_states_chunk

        if getattr(self.base_layer, "hidden_size_per_layer_input", 0):
            residual = hidden_states_chunk
            if per_layer_input is None:
                raise RuntimeError("per_layer_input is required for chunked Gemma4 replay")
            if per_layer_input.shape[1] != chunk_len:
                per_layer_input = per_layer_input[:, start:end, :]
            hidden_states_chunk = self.base_layer.per_layer_input_gate(hidden_states_chunk)
            hidden_states_chunk = self.base_layer.act_fn(hidden_states_chunk)
            hidden_states_chunk = hidden_states_chunk * per_layer_input
            hidden_states_chunk = self.base_layer.per_layer_projection(hidden_states_chunk)
            hidden_states_chunk = self.base_layer.post_per_layer_input_norm(hidden_states_chunk)
            hidden_states_chunk = residual + hidden_states_chunk

        return hidden_states_chunk * self.base_layer.layer_scalar

    def prepare_for_replay(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        self._replay_shared_kv_states = shared_kv_states
        hidden_states_norm = self.base_layer.input_layernorm(hidden_states)
        self.base_layer.self_attn.prepare_replay_kv(hidden_states_norm, position_embeddings, shared_kv_states)

    def clear_replay_cache(self) -> None:
        self._replay_shared_kv_states = None
        self.base_layer.self_attn.clear_replay_cache()


def _get_layer_types(inner_model) -> list[str]:
    cfg = inner_model.config
    if hasattr(cfg, "layer_types"):
        return list(cfg.layer_types)
    return ["full_attention"] * len(inner_model.layers)


def _build_mask_mapping(inner_model, hidden_states, attention_mask, position_ids):
    cfg = inner_model.config
    base_kwargs = {
        "config": cfg,
        "inputs_embeds": hidden_states,
        "attention_mask": attention_mask,
        "cache_position": None,
        "past_key_values": None,
        "position_ids": position_ids,
    }
    try:
        full_mask = create_causal_mask(**base_kwargs)
        sliding_mask = create_sliding_window_causal_mask(**base_kwargs)
    except TypeError:
        legacy_kwargs = {
            "config": cfg,
            "input_embeds": hidden_states,
            "attention_mask": attention_mask,
            "past_key_values": None,
            "position_ids": position_ids,
        }
        full_mask = create_causal_mask(**legacy_kwargs)
        sliding_mask = create_sliding_window_causal_mask(**legacy_kwargs)
    return {
        "full_attention": full_mask,
        "sliding_attention": sliding_mask,
    }


def _to_normal_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    with torch.inference_mode(False):
        return tensor.clone().detach()


def _run_layer(
    layer,
    hidden_states: torch.Tensor,
    mask_mapping: dict,
    position_embeddings: dict,
    position_ids: torch.Tensor,
    shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]],
    layer_type: str,
    per_layer_input: torch.Tensor | None = None,
    chunk_range: tuple[int, int] | None = None,
) -> torch.Tensor:
    output = layer(
        hidden_states=hidden_states,
        per_layer_input=per_layer_input,
        shared_kv_states=shared_kv_states,
        position_embeddings=position_embeddings.get(layer_type),
        attention_mask=mask_mapping.get(layer_type, mask_mapping["full_attention"]),
        position_ids=position_ids,
        past_key_values=None,
        chunk_range=chunk_range,
    )
    return output if not isinstance(output, tuple) else output[0]


class Gemma4Model(BaseModel):
    def setup(self) -> None:
        device = f"cuda:{self.cuda_device_index}"
        attn_impl = getattr(self.config, "attn_implementation", "eager")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map={"": device},
            dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        )
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False

        self.lora = getattr(self.config, "lora")
        self.lora_fraction = float(getattr(self.config, "lora_fraction", 0.5))
        self._use_grad_checkpoint = bool(getattr(self.config, "use_grad_checkpoint", False))
        self._offload_prefix_to_cpu = bool(getattr(self.config, "offload_prefix_to_cpu", False))
        token_chunk_size = int(getattr(self.config, "token_chunk_size", 0) or 0)
        self._token_chunk_size = token_chunk_size if token_chunk_size > 0 else 0

        self._inner_model, self._lm_head = self._resolve_text_stack(self.model)
        self._layers = self._inner_model.layers
        self._layer_types = _get_layer_types(self._inner_model)
        self._prefix_split_layer = self._compute_prefix_end(len(self._layers), self.lora_fraction)

        lora_targets = (
            self.lora
            if isinstance(self.lora, list)
            else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        target_modules = self._suffix_lora_targets(lora_targets)

        lora_rank = int(getattr(self.config, "lora_rank", 128))
        lora_alpha = int(getattr(self.config, "lora_alpha", lora_rank * 2))
        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        self.model = get_peft_model(self.model, lora_cfg)

        self._inner_model, self._lm_head = self._resolve_text_stack(self.model)
        self._layers = self._inner_model.layers
        self._layer_types = _get_layer_types(self._inner_model)
        self._wrap_suffix_layers_for_streaming()

        for param in self.model.parameters():
            param.requires_grad_(False)
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                param.requires_grad_(True)

    def _wrap_suffix_layers_for_streaming(self) -> None:
        for idx in range(self._prefix_split_layer, len(self._layers)):
            layer = self._layers[idx]
            if isinstance(layer, _StreamGemmaDecoderLayer):
                continue
            self._inner_model.layers[idx] = _StreamGemmaDecoderLayer(layer)
        self._layers = self._inner_model.layers

    def _resolve_text_stack(self, model):
        seen = set()
        queue = [model]
        ordered_candidates = []

        while queue:
            candidate = queue.pop(0)
            if candidate is None or id(candidate) in seen:
                continue
            seen.add(id(candidate))
            ordered_candidates.append(candidate)
            for attr in ("base_model", "model"):
                if hasattr(candidate, attr):
                    queue.append(getattr(candidate, attr))

        inner_model = None
        for candidate in ordered_candidates:
            if hasattr(candidate, "language_model"):
                inner_model = candidate.language_model
                break
            if hasattr(candidate, "layers") and hasattr(candidate, "embed_tokens"):
                inner_model = candidate
                break
        if inner_model is None:
            raise RuntimeError(
                "Unable to resolve Gemma 4 text stack from loaded model. "
                f"candidate_types={[type(c).__name__ for c in ordered_candidates]}"
            )

        lm_head = None
        for candidate in ordered_candidates:
            if hasattr(candidate, "lm_head"):
                lm_head = candidate.lm_head
                break
        if lm_head is None:
            raise RuntimeError(
                "Unable to resolve lm_head from loaded Gemma 4 model. "
                f"candidate_types={[type(c).__name__ for c in ordered_candidates]}"
            )

        return inner_model, lm_head

    def _compute_prefix_end(self, n_layers: int, frac: float) -> int:
        if frac <= 0.0:
            return n_layers
        if frac >= 1.0:
            return 0
        return int(math.floor(n_layers * (1.0 - frac)))

    def _suffix_lora_targets(self, requested_targets: list[str]) -> list[str]:
        target_leaves = set(requested_targets)
        targets: list[str] = []
        for name, module in self.model.named_modules():
            if "language_model.layers." not in name:
                continue
            parts = name.split(".")
            layer_idx = None
            for i in range(len(parts) - 1):
                if parts[i] == "layers" and parts[i + 1].isdigit():
                    layer_idx = int(parts[i + 1])
                    break
            if layer_idx is None or layer_idx < self._prefix_split_layer:
                continue
            if parts[-1] in target_leaves and hasattr(module, "weight"):
                targets.append(name)
        return sorted(set(targets))

    def _get_per_layer_inputs(self, input_ids: torch.Tensor) -> torch.Tensor | None:
        if not getattr(self._inner_model.config, "hidden_size_per_layer_input", None):
            return None
        inputs_embeds = self._inner_model.embed_tokens(input_ids)
        return self._inner_model.get_per_layer_inputs(input_ids, inputs_embeds)

    def _project_per_layer_inputs(
        self,
        hidden_states: torch.Tensor,
        per_layer_inputs: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if per_layer_inputs is None:
            return None
        return self._inner_model.project_per_layer_inputs(hidden_states, per_layer_inputs)

    def _build_runtime(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[dict, dict]:
        mask_mapping = _build_mask_mapping(
            self._inner_model,
            hidden_states.detach(),
            attention_mask,
            position_ids,
        )
        position_embeddings = {}
        for layer_type in set(self._layer_types):
            position_embeddings[layer_type] = self._inner_model.rotary_emb(
                hidden_states.detach(),
                position_ids,
                layer_type=layer_type,
            )
        return mask_mapping, position_embeddings

    def _detach_shared_kv_states(
        self,
        shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]],
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        return {
            key: (_to_normal_tensor(key_states), _to_normal_tensor(value_states))
            for key, (key_states, value_states) in shared_kv_states.items()
        }

    def _move_tensor(self, tensor: torch.Tensor | None, device: torch.device) -> torch.Tensor | None:
        if tensor is None:
            return None
        if tensor.device == device:
            return tensor
        return tensor.to(device=device, non_blocking=True)

    def _move_tensor_slice(
        self,
        tensor: torch.Tensor | None,
        device: torch.device,
        seq_end: int | None = None,
    ) -> torch.Tensor | None:
        if tensor is None:
            return None
        if seq_end is not None:
            tensor = tensor[:, :seq_end, ...]
        return self._move_tensor(tensor, device)

    def _materialize_shared_kv_states(
        self,
        shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]],
        device: torch.device,
        seq_end: int | None = None,
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        materialized = {}
        for key, (key_states, value_states) in shared_kv_states.items():
            if seq_end is not None:
                key_states = key_states[:, :, :seq_end, :]
                value_states = value_states[:, :, :seq_end, :]
            materialized[key] = (
                self._move_tensor(key_states, device),
                self._move_tensor(value_states, device),
            )
        return materialized

    def _offload_prefix_bundle_to_cpu(self, bundle: _PrefixBundle) -> _PrefixBundle:
        return _PrefixBundle(
            hidden_prefix=bundle.hidden_prefix.to("cpu"),
            position_ids=bundle.position_ids.to("cpu"),
            shared_kv_states={
                key: (key_states.to("cpu"), value_states.to("cpu"))
                for key, (key_states, value_states) in bundle.shared_kv_states.items()
            },
            per_layer_inputs=bundle.per_layer_inputs.to("cpu") if bundle.per_layer_inputs is not None else None,
        )

    def _build_prefix_bundle(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> _PrefixBundle:
        hidden_prefix, position_ids, shared_kv_states, per_layer_inputs = self._forward_prefix(
            input_ids,
            attention_mask,
        )
        bundle = _PrefixBundle(
            hidden_prefix=_to_normal_tensor(hidden_prefix),
            position_ids=_to_normal_tensor(position_ids),
            shared_kv_states=self._detach_shared_kv_states(shared_kv_states),
            per_layer_inputs=_to_normal_tensor(per_layer_inputs),
        )
        if self._offload_prefix_to_cpu:
            bundle = self._offload_prefix_bundle_to_cpu(bundle)
        return bundle

    def _prepare_suffix_inputs_from_bundle(
        self,
        bundle: _PrefixBundle,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        dict[int, tuple[torch.Tensor, torch.Tensor]],
        torch.Tensor | None,
    ]:
        model_device = next(self.model.parameters()).device
        hidden_prefix = self._move_tensor(bundle.hidden_prefix, model_device)
        if hidden_prefix is None:
            raise RuntimeError("prefix bundle did not contain hidden_prefix")
        position_ids = self._move_tensor(bundle.position_ids, model_device)
        if position_ids is None:
            raise RuntimeError("prefix bundle did not contain position_ids")
        return (
            hidden_prefix.detach().requires_grad_(torch.is_grad_enabled()),
            position_ids,
            bundle.shared_kv_states,
            bundle.per_layer_inputs,
        )

    def _run_suffix_from_prefix_bundle(
        self,
        bundle: _PrefixBundle,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden_for_suffix, position_ids, shared_kv_states, per_layer_inputs = self._prepare_suffix_inputs_from_bundle(
            bundle
        )
        return self._forward_suffix(
            hidden_for_suffix,
            position_ids,
            attention_mask,
            shared_kv_states,
            per_layer_inputs,
        )

    def _forward_prefix(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        dict[int, tuple[torch.Tensor, torch.Tensor]],
        torch.Tensor | None,
    ]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        hidden_states = self._inner_model.embed_tokens(input_ids)
        position_ids = (
            torch.arange(seq_len, device=device, dtype=torch.long)
            .unsqueeze(0)
            .expand(batch_size, seq_len)
        )

        mask_mapping, position_embeddings = self._build_runtime(
            hidden_states,
            attention_mask,
            position_ids,
        )

        shared_kv_states = {}
        raw_per_layer_inputs = self._get_per_layer_inputs(input_ids)
        per_layer_inputs = self._project_per_layer_inputs(hidden_states, raw_per_layer_inputs)

        for idx, layer in enumerate(self._layers[: self._prefix_split_layer]):
            per_layer_input = per_layer_inputs[:, :, idx, :] if per_layer_inputs is not None else None
            hidden_states = _run_layer(
                layer,
                hidden_states,
                mask_mapping,
                position_embeddings,
                position_ids,
                shared_kv_states,
                self._layer_types[idx],
                per_layer_input=per_layer_input,
            )

        return hidden_states, position_ids, shared_kv_states, per_layer_inputs

    def _forward_suffix(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]],
        per_layer_inputs: torch.Tensor | None,
    ) -> torch.Tensor:
        mask_mapping, position_embeddings = self._build_runtime(
            hidden_states,
            attention_mask,
            position_ids,
        )

        for offset, layer in enumerate(self._layers[self._prefix_split_layer :]):
            abs_idx = self._prefix_split_layer + offset
            current_per_layer_input_source = None
            if per_layer_inputs is not None:
                current_per_layer_input_source = per_layer_inputs[:, :, abs_idx, :]

            class _LayerReplay:
                def __init__(self):
                    self.shared_kv_local: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None

                def prepare_for_replay(self, h: torch.Tensor, requires_grad: bool) -> None:
                    self.shared_kv_local = self_outer._materialize_shared_kv_states(shared_kv_states, h.device, None)
                    if isinstance(layer, _StreamGemmaDecoderLayer):
                        layer.prepare_for_replay(
                            h,
                            position_embeddings[self_outer._layer_types[abs_idx]],
                            self.shared_kv_local,
                        )

                def clear_replay_cache(self) -> None:
                    if isinstance(layer, _StreamGemmaDecoderLayer):
                        layer.clear_replay_cache()
                    self.shared_kv_local = None

                def __call__(
                    self,
                    h: torch.Tensor,
                    chunk_range: tuple[int, int] | None = None,
                ) -> torch.Tensor:
                    seq_end = None if chunk_range is None else chunk_range[1]
                    per_layer_input_local = None
                    if current_per_layer_input_source is not None:
                        if chunk_range is None:
                            per_layer_input_local = self_outer._move_tensor(current_per_layer_input_source, h.device)
                        else:
                            per_layer_input_local = self_outer._move_tensor(
                                current_per_layer_input_source[:, chunk_range[0]:chunk_range[1], :],
                                h.device,
                            )

                    shared_kv_local = self.shared_kv_local
                    if shared_kv_local is None:
                        shared_kv_local = self_outer._materialize_shared_kv_states(shared_kv_states, h.device, seq_end)

                    output = _run_layer(
                        layer,
                        h,
                        mask_mapping,
                        position_embeddings,
                        position_ids,
                        shared_kv_local,
                        self_outer._layer_types[abs_idx],
                        per_layer_input=per_layer_input_local,
                        chunk_range=chunk_range,
                    )
                    if seq_end is None or seq_end == h.shape[1]:
                        for key, value in shared_kv_local.items():
                            shared_kv_states[key] = value
                    return output

            self_outer = self
            layer_replay = _LayerReplay()

            if self._use_grad_checkpoint and torch.is_grad_enabled() and self._token_chunk_size > 0:
                hidden_states = _StreamCheckpointFunction.apply(
                    layer_replay,
                    self._token_chunk_size,
                    self._offload_prefix_to_cpu,
                    hidden_states,
                )
            elif self._use_grad_checkpoint and torch.is_grad_enabled():
                hidden_states = torch.utils.checkpoint.checkpoint(layer_replay, hidden_states, use_reentrant=False)
            else:
                hidden_states = layer_replay(hidden_states)

        return self._inner_model.norm(hidden_states)

    def _lm_head_logits_chunked(self, hidden_states: torch.Tensor) -> torch.Tensor:
        seq_len = hidden_states.shape[1]
        chunk = seq_len if self.chunk_size is None else max(1, int(self.chunk_size))
        logits_parts = []
        softcap = getattr(self._inner_model.config, "final_logit_softcapping", None)

        for start in range(0, seq_len, chunk):
            end = min(start + chunk, seq_len)
            logits = self._lm_head(hidden_states[:, start:end, :])
            if softcap is not None:
                logits = torch.tanh(logits / softcap) * softcap
            logits_parts.append(logits)

        return torch.cat(logits_parts, dim=1)

    def _token_logprobs_chunked(
        self,
        hidden_comp: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        completion_len = hidden_comp.shape[1]
        chunk = completion_len if self.chunk_size is None else max(1, int(self.chunk_size))
        softcap = getattr(self._inner_model.config, "final_logit_softcapping", None)
        token_logprob_chunks = []

        for start in range(0, completion_len, chunk):
            end = min(start + chunk, completion_len)
            logits = self._lm_head(hidden_comp[:, start:end, :])
            if softcap is not None:
                logits = torch.tanh(logits / softcap) * softcap
            target_ids = token_ids[:, start:end].to(hidden_comp.device, non_blocking=True)
            token_logprob_chunks.append(
                torch.log_softmax(logits.float(), dim=-1)
                .gather(dim=-1, index=target_ids.unsqueeze(-1))
                .squeeze(-1)
            )

        return torch.cat(token_logprob_chunks, dim=1)

    def forward(self, messages: list[list[dict]]):
        prompts = [
            self.tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=True,
            )
            for convo in messages
        ]
        tokenized = self.tokenizer(prompts, return_tensors="pt", padding=True)
        model_device = next(self.model.parameters()).device
        input_ids = tokenized["input_ids"].to(model_device, non_blocking=True)
        attention_mask = tokenized["attention_mask"].to(model_device, non_blocking=True)

        with torch.inference_mode():
            prefix_bundle = self._build_prefix_bundle(input_ids, attention_mask)
            hidden_suffix = self._run_suffix_from_prefix_bundle(prefix_bundle, attention_mask)
            return self._lm_head_logits_chunked(hidden_suffix)

    def backward(
        self,
        messages: list[list[dict]],
        completion_texts: list[str],
        loss_fn: Callable,
        loss_scale: float = 1.0,
    ) -> dict[str, float]:
        if not completion_texts:
            raise ValueError("completion_texts cannot be empty")

        prompts = [
            self.tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=True,
            )
            for convo in messages
        ]
        tokenized = self.tokenizer(prompts, return_tensors="pt", padding=True)
        model_device = next(self.model.parameters()).device
        prompt_ids = tokenized["input_ids"].to(model_device, non_blocking=True)
        prompt_mask = tokenized["attention_mask"].to(model_device, non_blocking=True)

        completion_tok = self.tokenizer(
            completion_texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        completion_ids = completion_tok["input_ids"].to(model_device, non_blocking=True)
        completion_mask = completion_tok["attention_mask"].to(
            model_device,
            dtype=torch.float32,
            non_blocking=True,
        )

        num_generations = completion_ids.shape[0]
        if prompt_ids.shape[0] == 1 and num_generations > 1:
            prompt_ids = prompt_ids.expand(num_generations, prompt_ids.shape[1])
            prompt_mask = prompt_mask.expand(num_generations, prompt_mask.shape[1])
        if prompt_ids.shape[0] != num_generations:
            raise ValueError(
                f"prompt batch ({prompt_ids.shape[0]}) must be 1 or match completion batch ({num_generations})"
            )

        full_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        full_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        completion_len = completion_ids.shape[1]

        with torch.inference_mode():
            prefix_bundle = self._build_prefix_bundle(full_ids, full_mask)

        with torch.enable_grad():
            hidden_suffix = self._run_suffix_from_prefix_bundle(prefix_bundle, full_mask)
            hidden_comp = hidden_suffix[:, -(completion_len + 1) : -1, :]
            batch_logprobs = self._token_logprobs_chunked(hidden_comp, completion_ids)

            batch_loss_callable = None
            explicit_batch = getattr(loss_fn, "loss_fn_batch", None)
            if callable(explicit_batch):
                batch_loss_callable = explicit_batch
            elif getattr(loss_fn, "__name__", "") == "loss_fn_batch":
                batch_loss_callable = loss_fn

            if batch_loss_callable is not None:
                total_loss_t = batch_loss_callable(batch_logprobs, completion_mask, hidden_comp)
                (total_loss_t * float(loss_scale)).backward()
                total_loss = float(total_loss_t.item())
            else:
                total_loss = 0.0
                for idx in range(num_generations):
                    sample_loss = loss_fn(batch_logprobs[idx], idx, hidden_comp[idx])
                    retain_graph = idx < (num_generations - 1)
                    (sample_loss * float(loss_scale)).backward(retain_graph=retain_graph)
                    total_loss += float(sample_loss.item())
                total_loss = total_loss / max(1, num_generations)

        with torch.no_grad():
            valid_tokens = completion_mask.sum().clamp(min=1.0)
            mean_logp = float((batch_logprobs * completion_mask).sum().item() / valid_tokens.item())

        return {
            "loss": float(total_loss),
            "mean_logp": mean_logp,
            "batch_size": float(num_generations),
            "valid_tokens": float(completion_mask.sum().item()),
        }
