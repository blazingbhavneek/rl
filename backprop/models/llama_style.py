from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn

from .base_adapter import BaseModelAdapter


class LlamaStyleAdapter(BaseModelAdapter):
    def unwrap(self, model: nn.Module) -> Tuple[nn.Module, nn.Module]:
        base = model.base_model.model if hasattr(model, "base_model") else model
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
        if hasattr(inner, "rotary_emb"):
            cos, sin = inner.rotary_emb(hidden, position_ids)
            return cos, sin
        raise RuntimeError("Unsupported Llama-style model: missing rotary_emb")

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

        try:
            attn_out = layer.self_attn(
                hidden_states=x,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )
        except TypeError:
            # Compatibility fallback for model variants that still use position_ids.
            attn_out = layer.self_attn(
                hidden_states=x,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]

        hidden_states = residual + attn_out
        residual = hidden_states

        x = layer.post_attention_layernorm(hidden_states)
        mlp_out = layer.mlp(x)
        if isinstance(mlp_out, tuple):
            mlp_out = mlp_out[0]
        hidden_states = residual + mlp_out
        return hidden_states

    def identify_layer_types(self, model: nn.Module) -> Dict[int, str]:
        _, inner = self.unwrap(model)
        types: Dict[int, str] = {}
        for i, layer in enumerate(self.get_layers(inner)):
            attn = getattr(layer, "self_attn", None)
            if attn is not None and getattr(attn, "sliding_window", None) is not None:
                types[i] = "sliding_attention"
            else:
                types[i] = "full_attention"
        return types

    def get_split_index(self, model: nn.Module, top_frac: float) -> int:
        return self._default_split_from_lora_names(model, top_frac)
