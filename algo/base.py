from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from torch import Tensor

try:
    from taskset.base import Score
except ImportError:  # pragma: no cover - direct script execution fallback
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from taskset.base import Score


@dataclass
class AlgoConfig:
    kl_coeff: float = 0.0
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.28
    norm_advantages: bool = True
    loss_agg: str = "token_mean"  # "token_mean" | "seq_mean"


@dataclass
class AlgoOutput:
    """
    Everything algo produces after processing one problem's rollouts.
    Pipeline takes this and hands loss_fn to backprop.
    """

    loss_fn: Callable[[Tensor, int, Optional[Tensor]], Tensor]
    needs_hidden_states: bool
    stats: Dict[str, float]
    ref_prompt_ids: Optional[Tensor]


class BaseAlgo(ABC):
    def __init__(self, config: AlgoConfig) -> None:
        self.config = config

    @abstractmethod
    def process_rollouts(
        self,
        prompt_ids: Tensor,  # (1, T_p)
        completion_ids: Tensor,  # (G, T_c)
        completion_mask: Tensor,  # (G, T_c)
        scores: List[Score],
        rewards: List[float],
        feedback: Optional[List[str]] = None,
        peer_solution: Optional[str] = None,
    ) -> AlgoOutput:
        ...

    @abstractmethod
    def requires_rich_feedback(self) -> bool:
        ...

    @property
    @abstractmethod
    def needs_hidden_states(self) -> bool:
        ...
