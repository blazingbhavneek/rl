from __future__ import annotations

from abc import ABC
from typing import Dict, List, Optional

from torch import Tensor, nn

from pipeline.base import ProblemRollouts


class BaseExtension(ABC):
    def on_rollouts_complete(self, problem_rollouts: List[ProblemRollouts]) -> List[ProblemRollouts]:
        return problem_rollouts

    def extra_loss(self, model: nn.Module, problem_rollouts: List[ProblemRollouts]) -> Optional[Tensor]:
        del model, problem_rollouts
        return None

    def on_step_end(self, step: int, stats: Dict) -> None:
        del step, stats
        return None
