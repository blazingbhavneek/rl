from __future__ import annotations

from typing import Callable, List, Optional

from pipeline.base import ProblemRollouts

from .base import BaseExtension


class ProcessRewardExtension(BaseExtension):
    """
    Injects additional process-level signal into rollout rewards.

    score_fn should return a delta reward for one rollout.
    Signature: score_fn(problem_rollouts, rollout) -> float
    """

    def __init__(self, score_fn: Callable) -> None:
        self.score_fn = score_fn

    def on_rollouts_complete(self, problem_rollouts: List[ProblemRollouts]) -> List[ProblemRollouts]:
        for pr in problem_rollouts:
            for r in pr.rollouts:
                r.reward = float(r.reward + float(self.score_fn(pr, r)))
            if pr.rollouts:
                pr.best_rollout = max(pr.rollouts, key=lambda x: x.reward)
                passed = [x for x in pr.rollouts if x.passed]
                pr.any_passed = bool(passed)
                pr.peer_solution = max(passed, key=lambda x: x.reward).completion_text if passed else None
        return problem_rollouts
