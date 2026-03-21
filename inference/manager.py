from __future__ import annotations

import time
from typing import Callable, Optional

try:
    from .base import BaseEngine, WeightSwapMode
except ImportError:  # pragma: no cover - direct script execution fallback
    from base import BaseEngine, WeightSwapMode


class EngineManager:
    def __init__(
        self,
        engine: BaseEngine,
        poll_interval: float,
        *,
        engine_factory: Optional[Callable[[], BaseEngine]] = None,
    ) -> None:
        self.engine = engine
        self.poll_interval = float(max(0.0, poll_interval))
        self.engine_factory = engine_factory
        self._last_poll_ts = 0.0
        self._last_checkpoint_path: Optional[str] = None
        self._last_swap_mode: WeightSwapMode = WeightSwapMode.COLD_START

    def start(self, checkpoint_path: Optional[str]) -> None:
        if checkpoint_path:
            self.engine.swap_weights(checkpoint_path, WeightSwapMode.COLD_START)
            self._last_checkpoint_path = checkpoint_path
            self._last_swap_mode = WeightSwapMode.COLD_START
        if not self.engine.is_healthy():
            self._restart()

    def sync_weights(self, checkpoint_path: str, mode: WeightSwapMode) -> None:
        self.engine.swap_weights(checkpoint_path, mode)
        if not self.engine.is_healthy():
            self._restart()
            if not self.engine.is_healthy():
                raise RuntimeError("Engine is unhealthy after weight sync and restart.")
        self._last_checkpoint_path = checkpoint_path
        self._last_swap_mode = mode
        self._last_poll_ts = time.monotonic()

    def get_engine(self) -> BaseEngine:
        now = time.monotonic()
        if (now - self._last_poll_ts) >= self.poll_interval:
            if not self.engine.is_healthy():
                self._restart()
            self._last_poll_ts = now
        return self.engine

    def _restart(self) -> None:
        self.engine.shutdown()
        if self.engine_factory is None:
            raise RuntimeError("Engine restart requested but no engine_factory was provided.")
        self.engine = self.engine_factory()
        if self._last_checkpoint_path:
            self.engine.swap_weights(self._last_checkpoint_path, self._last_swap_mode)
        if not self.engine.is_healthy():
            raise RuntimeError("Engine restart failed health check.")

    def shutdown(self) -> None:
        self.engine.shutdown()


if __name__ == "__main__":
    class _MockEngine(BaseEngine):
        def __init__(self) -> None:
            self.health_calls = 0
            self.swap_calls = []
            self.alive = True

        def generate(self, prompt, params):
            return []

        def generate_batch(self, prompts, params):
            return [[] for _ in prompts]

        def swap_weights(self, checkpoint_path, mode):
            self.swap_calls.append((checkpoint_path, mode))

        def is_healthy(self):
            self.health_calls += 1
            if self.health_calls > 2:
                self.alive = False
            return self.alive

        def shutdown(self):
            self.alive = False

    def _factory():
        return _MockEngine()

    mgr = EngineManager(engine=_factory(), poll_interval=0.0, engine_factory=_factory)
    mgr.start(checkpoint_path=None)
    _ = mgr.get_engine()
    _ = mgr.get_engine()  # triggers restart
    mgr.sync_weights("/tmp/ckpt", WeightSwapMode.FULL)
    print("manager.py self-test passed")
