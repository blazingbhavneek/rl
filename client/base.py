from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from inference.base import BaseEngine
from tasksets.base import Problem


@dataclass
class ClientContext:
    pass_number: int = 1
    error_context: Optional[str] = None
    best_code: Optional[str] = None
    lora_path: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClientResult:
    completions: List[str]
    token_ids: List[List[int]]
    prompt_text: str
    prompt_token_ids: List[int]
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseClient(ABC):
    def __init__(self, engine: BaseEngine, tokenizer) -> None:
        self.engine = engine
        self.tokenizer = tokenizer

    @abstractmethod
    def run(
        self,
        problem: Problem,
        context: ClientContext,
        n: int,
    ) -> ClientResult:
        ...

    @abstractmethod
    def build_messages(
        self,
        problem: Problem,
        context: ClientContext,
    ) -> List[Dict[str, str]]:
        ...

    def apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        # HF tokenizers usually expose apply_chat_template.
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        # Conservative fallback.
        lines: List[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            lines.append(f"[{role}]\n{content}")
        lines.append("[assistant]\n")
        return "\n\n".join(lines)
