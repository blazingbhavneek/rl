from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List


class WeightSwapMode(str, Enum):
    LORA = "lora"
    FULL = "full"
    COLD_START = "cold_start"


@dataclass
class SamplingParams:
    max_new_tokens: int
    temperature: float
    n: int = 1
    top_p: float = 1.0
    stop: List[str] = field(default_factory=list)


@dataclass
class GenerationOutput:
    text: str
    token_ids: List[int]
    prompt_tokens: int


class BaseEngine(ABC):
    @abstractmethod
    def generate(self, prompt: str, params: SamplingParams) -> List[GenerationOutput]:
        """Single prompt, returns n completions."""

    @abstractmethod
    def generate_batch(
        self, prompts: List[str], params: SamplingParams
    ) -> List[List[GenerationOutput]]:
        """Multiple prompts in one engine call. Returns one List[GenerationOutput] per prompt."""

    @abstractmethod
    def swap_weights(self, checkpoint_path: str, mode: WeightSwapMode) -> None:
        """Swap weights. Mode is LORA, FULL, or COLD_START."""

    @abstractmethod
    def is_healthy(self) -> bool:
        """Quick health check before a rollout batch."""

    @abstractmethod
    def shutdown(self) -> None:
        """Release engine resources."""


if __name__ == "__main__":
    params = SamplingParams(max_new_tokens=32, temperature=0.7)
    assert params.n == 1
    assert params.top_p == 1.0
    assert params.stop == []

    out = GenerationOutput(text="hello world", token_ids=[1, 2], prompt_tokens=3)
    assert out.text == "hello world"
    assert len(out.token_ids) == 2
    print("base.py self-test passed")
