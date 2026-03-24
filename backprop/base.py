from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch
from torch import Tensor, nn

from .models.base_adapter import BaseModelAdapter

try:
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
except Exception:  # pragma: no cover - optional dependency at runtime
    get_peft_model_state_dict = None
    set_peft_model_state_dict = None


@dataclass
class BackpropConfig:
    top_frac: float = 0.25
    logit_chunk: int = 64
    use_grad_checkpoint: bool = True
    use_torch_compile: bool = False


class BaseBackprop(ABC):
    def __init__(
        self,
        model: nn.Module,
        adapter: BaseModelAdapter,
        config: BackpropConfig,
    ) -> None:
        self.model = model
        self.adapter = adapter
        self.config = config

    @abstractmethod
    def compute_logprobs(
        self,
        model: nn.Module,
        prompt_ids: Tensor,
        completion_ids: Tensor,
        lora_path: Optional[str] = None,
    ) -> Tensor:
        ...

    @abstractmethod
    def compute_ref_logprobs(
        self,
        model: nn.Module,
        prompt_ids: Tensor,
        completion_ids: Tensor,
        lora_path: Optional[str] = None,
    ) -> Tensor:
        ...

    @abstractmethod
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
        ...

    @abstractmethod
    def load_lora(self, path: str) -> None:
        ...

    @abstractmethod
    def save_lora(self, path: str) -> None:
        ...

    def get_current_lora_state(self) -> Dict[str, Tensor]:
        if get_peft_model_state_dict is not None:
            state = get_peft_model_state_dict(self.model)
            return {k: v.detach().clone() for k, v in state.items()}

        state = {}
        for name, tensor in self.model.state_dict().items():
            if "lora" in name.lower() or "adapter" in name.lower():
                state[name] = tensor.detach().clone()
        return state

    def restore_lora_state(self, state: Dict[str, Tensor]) -> None:
        if set_peft_model_state_dict is not None:
            set_peft_model_state_dict(self.model, state)
            return

        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if len(unexpected) > 0:
            raise RuntimeError(f"Unexpected keys while restoring LoRA state: {unexpected}")
