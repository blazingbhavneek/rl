from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Problem:
    id: str
    statement: str
    bucket: int
    difficulty_label: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class Score:
    # Raw verifier output. Reward shaping belongs to pipeline code.
    compiled: bool
    passed: int
    total: int
    error: Optional[str] = None
    details: Dict = field(default_factory=dict)


@dataclass
class ProblemState:
    id: str
    bucket: int
    total_attempts: int = 0
    solve_rate: float = 0.0
    consecutive_solves: int = 0
    last_sampled_step: int = -1
    promoted: bool = False


class BaseVerifier(ABC):
    """Verifier interface shared across tasksets."""

    @abstractmethod
    def verify(self, problem: Problem, completion: str) -> Score:
        """Verify a single completion."""

    @abstractmethod
    def verify_batch(self, problem: Problem, completions: List[str]) -> List[Score]:
        """Verify multiple completions for one problem."""

    @abstractmethod
    def check_dependencies(self) -> None:
        """
        Validate runtime dependencies (compiler/libs/tools).
        Must raise SystemExit on hard failure.
        """
