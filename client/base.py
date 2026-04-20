from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Type

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

OutputModel = Optional[Type[BaseModel]]
RunOutput = tuple[str, str] | tuple[str, BaseModel]


class BaseClient(ABC):
    @abstractmethod
    def __init__(
        self,
        base_url: str,
        api_key: str,
        temperature: float,
        max_output_tokens: int,
        system_prompt: str,
        model: str,
    ) -> None: ...

    @abstractmethod
    def reset_history(self, system_prompt: Optional[str] = None) -> None: ...

    @abstractmethod
    def build_messages(self, prompt: str) -> list[BaseMessage]: ...

    @abstractmethod
    async def run(
        self,
        prompt: str,
        output_model: OutputModel = None,
    ) -> RunOutput: ...
