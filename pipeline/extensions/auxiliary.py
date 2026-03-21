from __future__ import annotations

from typing import Callable, List, Optional

from torch import Tensor, nn

from pipeline.base import ProblemRollouts

from .base import BaseExtension


class AuxiliaryLossExtension(BaseExtension):
    """Adds an auxiliary loss term computed from model + rollout batch."""

    def __init__(self, loss_fn: Callable[[nn.Module, List[ProblemRollouts]], Optional[Tensor]]) -> None:
        self.loss_fn = loss_fn

    def extra_loss(self, model: nn.Module, problem_rollouts: List[ProblemRollouts]) -> Optional[Tensor]:
        return self.loss_fn(model, problem_rollouts)
