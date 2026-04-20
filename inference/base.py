from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseEngine(ABC):
    def __init__(
        self, model_path: str, engine_kwargs: Optional[dict[str, Any]] = None
    ) -> None:
        self.model_path = model_path
        self.engine_kwargs = dict(engine_kwargs or {})
        self.is_awake = False
        self._closed = False
        self._load_model()

    @abstractmethod
    def _load_model(self) -> None: ...

    @abstractmethod
    async def init(self) -> None: ...

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def sleep(self, level: int = 1) -> None: ...

    @abstractmethod
    async def wake(self) -> None: ...

    @abstractmethod
    async def is_sleeping(self) -> bool: ...

    @abstractmethod
    async def kill(self) -> None: ...

    @abstractmethod
    async def shutdown(self) -> None: ...
