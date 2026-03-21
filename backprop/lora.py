from __future__ import annotations

from dataclasses import replace
from typing import Optional

from torch import nn

from .base import BackpropConfig
from .models.base_adapter import BaseModelAdapter
from .streaming import StreamingBackprop


class LoRABackprop(StreamingBackprop):
    """Vanilla LoRA backprop: all layers trainable path (split_layer=0)."""

    def __init__(
        self,
        model: nn.Module,
        adapter: Optional[BaseModelAdapter] = None,
        config: Optional[BackpropConfig] = None,
    ) -> None:
        config = replace(config or BackpropConfig(), top_frac=1.0)
        super().__init__(model=model, adapter=adapter, config=config)
