from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .base_adapter import BaseModelAdapter


def _apply_rotary_emb(states: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    if cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    first, second = torch.chunk(states, 2, dim=-1)
    return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)


def _repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    if n_rep == 1:
        return hidden_states
    bsz, heads, seq, dim = hidden_states.shape
    return hidden_states[:, :, None, :, :].expand(bsz, heads, n_rep, seq, dim).reshape(
        bsz, heads * n_rep, seq, dim
    )


class GptOssAdapter(BaseModelAdapter):
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
        cfg = inner.config
        head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        t = position_ids.shape[1]
        dummy = hidden.new_zeros(1, t, head_dim)
        cos, sin = inner.rotary_emb(dummy, position_ids)
        return cos, sin

    def _full_attention_forward(
        self,
        self_attn: nn.Module,
        hidden_states: Tensor,
        position_embeddings: Tuple[Tensor, Tensor],
    ) -> Tensor:
        bsz, seq, _ = hidden_states.shape
        cos, sin = position_embeddings

        cfg = self_attn.config
        num_heads = cfg.num_attention_heads
        num_kv_heads = cfg.num_key_value_heads
        num_kv_grps = num_heads // num_kv_heads
        head_dim = self_attn.head_dim

        q = self_attn.q_proj(hidden_states).view(bsz, seq, num_heads, head_dim).transpose(1, 2)
        k = self_attn.k_proj(hidden_states).view(bsz, seq, num_kv_heads, head_dim).transpose(1, 2)
        v = self_attn.v_proj(hidden_states).view(bsz, seq, num_kv_heads, head_dim).transpose(1, 2)

        q = _apply_rotary_emb(q, cos, sin)
        k = _apply_rotary_emb(k, cos, sin)
        k = _repeat_kv(k, num_kv_grps)
        v = _repeat_kv(v, num_kv_grps)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            scale=head_dim ** -0.5,
            dropout_p=0.0,
        )
        out = out.transpose(1, 2).contiguous().view(bsz, seq, -1)
        return self_attn.o_proj(out)

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

        if layer_type == "full_attention":
            attn_out = self._full_attention_forward(layer.self_attn, x, position_embeddings)
        else:
            attn_out, _ = layer.self_attn(
                x,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
            )

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
        if hasattr(inner, "config") and hasattr(inner.config, "layer_types"):
            return {i: t for i, t in enumerate(inner.config.layer_types)}

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
