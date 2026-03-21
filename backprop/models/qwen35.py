from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn

from .base_adapter import BaseModelAdapter


class Qwen35Adapter(BaseModelAdapter):
    def unwrap(self, model: nn.Module) -> Tuple[nn.Module, nn.Module]:
        base = model.base_model.model if hasattr(model, "base_model") else model
        # Qwen3_5ForCausalLM has .model = Qwen3_5TextModel
        inner = base.model if hasattr(base, "model") else base
        return base, inner

    def get_layers(self, inner: nn.Module) -> nn.ModuleList:
        return inner.layers

    def get_embed_tokens(self, inner: nn.Module) -> nn.Module:
        return inner.embed_tokens

    def get_final_norm(self, inner: nn.Module) -> nn.Module:
        return inner.norm

    def get_lm_head(self, base: nn.Module) -> nn.Module:
        return base.lm_head

    def get_position_embeddings(
        self,
        inner: nn.Module,
        hidden: Tensor,
        position_ids: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if position_ids.ndim == 2:
            # Qwen3.5 text rotary expects (3, B, T) for text/h/w branches
            position_ids = position_ids.unsqueeze(0).expand(3, position_ids.shape[0], position_ids.shape[1])
        cos, sin = inner.rotary_emb(hidden, position_ids)
        return cos, sin

    def layer_forward(
        self,
        layer: nn.Module,
        hidden_states: Tensor,
        position_embeddings: Tuple[Tensor, Tensor],
        layer_type: str,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        residual = hidden_states
        x = layer.input_layernorm(hidden_states)

        if layer_type == "linear_attention":
            mixed = layer.linear_attn(
                hidden_states=x,
                cache_params=None,
                attention_mask=attention_mask,
            )
        else:
            mixed, _ = layer.self_attn(
                hidden_states=x,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=None,
            )

        hidden_states = residual + mixed
        residual = hidden_states

        x = layer.post_attention_layernorm(hidden_states)
        mlp_out = layer.mlp(x)
        hidden_states = residual + mlp_out
        return hidden_states

    def identify_layer_types(self, model: nn.Module) -> Dict[int, str]:
        _, inner = self.unwrap(model)
        if hasattr(inner, "config") and hasattr(inner.config, "layer_types"):
            return {i: t for i, t in enumerate(inner.config.layer_types)}
        # Safe fallback if config is absent
        return {i: "full_attention" for i, _ in enumerate(self.get_layers(inner))}

    def get_split_index(self, model: nn.Module, top_frac: float) -> int:
        return self._default_split_from_lora_names(model, top_frac)

    def supports_layer_type(self, layer_type: str) -> bool:
        return layer_type in ("full_attention", "linear_attention", "sliding_attention")
