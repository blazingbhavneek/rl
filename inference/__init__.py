from .base import BaseEngine

try:
    from .vllm_engine import VLLMEngine
except Exception:  # pragma: no cover - optional runtime dependency
    VLLMEngine = None  # type: ignore

__all__ = ["BaseEngine", "VLLMEngine"]
