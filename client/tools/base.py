from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ToolResult:
    success: bool
    output: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseTool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...

    @property
    @abstractmethod
    def parameters_schema(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        ...

    def to_openai_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }
