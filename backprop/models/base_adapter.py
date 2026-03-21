from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn


class BaseModelAdapter(ABC):
    @abstractmethod
    def unwrap(self, model: nn.Module) -> Tuple[nn.Module, nn.Module]:
        """Return (base_model, inner_transformer)."""

    @abstractmethod
    def get_layers(self, inner: nn.Module) -> nn.ModuleList:
        ...

    @abstractmethod
    def get_embed_tokens(self, inner: nn.Module) -> nn.Module:
        ...

    @abstractmethod
    def get_final_norm(self, inner: nn.Module) -> nn.Module:
        ...

    @abstractmethod
    def get_lm_head(self, base: nn.Module) -> nn.Module:
        ...

    @abstractmethod
    def get_position_embeddings(
        self,
        inner: nn.Module,
        hidden: Tensor,
        position_ids: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Return model-specific positional embeddings (typically RoPE cos/sin)."""

    @abstractmethod
    def layer_forward(
        self,
        layer: nn.Module,
        hidden_states: Tensor,
        position_embeddings: Tuple[Tensor, Tensor],
        layer_type: str,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        ...

    @abstractmethod
    def identify_layer_types(self, model: nn.Module) -> Dict[int, str]:
        ...

    @abstractmethod
    def get_split_index(self, model: nn.Module, top_frac: float) -> int:
        ...

    def supports_layer_type(self, layer_type: str) -> bool:
        return layer_type in ("full_attention", "sliding_attention")

    @staticmethod
    def _discover_layer_indices(model: nn.Module) -> set[int]:
        indices: set[int] = set()
        for name, _ in model.named_modules():
            parts = name.split(".")
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                    indices.add(int(parts[i + 1]))
        return indices

    def _default_split_from_lora_names(self, model: nn.Module, top_frac: float) -> int:
        if top_frac >= 1.0:
            return 0
        layer_indices = self._discover_layer_indices(model)
        if not layer_indices:
            return 0
        max_layer = max(layer_indices)
        return int(torch.floor(torch.tensor(max_layer * (1.0 - top_frac))).item())
