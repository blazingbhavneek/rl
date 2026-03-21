from __future__ import annotations

from typing import Dict, Type

from torch import nn

from .base_adapter import BaseModelAdapter
from .gptoss import GptOssAdapter
from .llama_style import LlamaStyleAdapter
from .qwen35 import Qwen35Adapter

ADAPTER_REGISTRY: Dict[str, Type[BaseModelAdapter]] = {
    "GptOssForCausalLM": GptOssAdapter,
    "Qwen3_5ForCausalLM": Qwen35Adapter,
    "Qwen3_5ForConditionalGeneration": Qwen35Adapter,
    "LlamaForCausalLM": LlamaStyleAdapter,
    "Qwen2ForCausalLM": LlamaStyleAdapter,
    "MistralForCausalLM": LlamaStyleAdapter,
    "MixtralForCausalLM": LlamaStyleAdapter,
}


def register_adapter(model_class_name: str, adapter_class: Type[BaseModelAdapter]) -> None:
    ADAPTER_REGISTRY[model_class_name] = adapter_class


def _candidate_class_names(model: nn.Module) -> list[str]:
    names = [model.__class__.__name__]

    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        base = model.base_model.model
        names.append(base.__class__.__name__)
        if hasattr(base, "model"):
            names.append(base.model.__class__.__name__)

    if hasattr(model, "model"):
        names.append(model.model.__class__.__name__)

    # preserve order and dedupe
    seen = set()
    out = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def get_adapter(model: nn.Module) -> BaseModelAdapter:
    for cls_name in _candidate_class_names(model):
        adapter_cls = ADAPTER_REGISTRY.get(cls_name)
        if adapter_cls is not None:
            return adapter_cls()

    known = ", ".join(sorted(ADAPTER_REGISTRY))
    tried = ", ".join(_candidate_class_names(model))
    raise ValueError(
        "No adapter registered for model class. "
        f"Tried: [{tried}]. Known: [{known}]. "
        "Register one with backprop.models.auto.register_adapter('<ClassName>', AdapterClass)."
    )
