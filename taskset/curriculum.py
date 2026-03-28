import math
from dataclasses import asdict
from typing import Dict, Iterable, List, Sequence, Tuple

from .base import ProblemState


class BucketDistribution:
    def __init__(
        self,
        n_buckets: int,
        initial_mean: float,
        std: float,
        shift_success_ratio: float = 0.7,
        exhaustion_ratio: float = 0.8,
        exhaustion_threshold: float = 0.8,
    ) -> None:
        if n_buckets <= 0:
            raise ValueError("n_buckets must be positive")
        self.n_buckets = n_buckets
        self.mean = float(max(0.0, min(initial_mean, n_buckets - 1)))
        self.std = float(std)
        self.shift_success_ratio = float(shift_success_ratio)
        self.exhaustion_ratio = float(exhaustion_ratio)
        self.exhaustion_threshold = float(exhaustion_threshold)

    def get_probs(self) -> List[float]:
        if self.std <= 0:
            idx = int(round(self.mean))
            return [1.0 if i == idx else 0.0 for i in range(self.n_buckets)]

        weights = []
        for i in range(self.n_buckets):
            z = (i - self.mean) / self.std
            weights.append(math.exp(-0.5 * z * z))
        s = sum(weights)
        if s <= 0:
            return [1.0 / self.n_buckets] * self.n_buckets
        return [w / s for w in weights]

    def shift_right(self, delta: float) -> None:
        self.mean = min(self.n_buckets - 1, self.mean + float(delta))

    def should_shift(
        self,
        states: Iterable[ProblemState],
        window: Tuple[int, int],
        threshold: float,
        consecutive_required: int,
        min_evaluated: int,
    ) -> bool:
        lo, hi = window
        in_window = [
            s for s in states
            if lo <= s.bucket <= hi and not s.promoted and s.total_attempts > 0
        ]
        evaluated = len(in_window)
        if evaluated < min_evaluated:
            return False

        mastered = [
            s for s in in_window
            if s.solve_rate >= threshold and s.consecutive_solves >= consecutive_required
        ]
        return (len(mastered) / evaluated) >= self.shift_success_ratio

    def is_exhausted(self, states: Iterable[ProblemState]) -> bool:
        probs = self.get_probs()
        mode = max(range(self.n_buckets), key=lambda i: probs[i])
        if mode != self.n_buckets - 1:
            return False

        right_states = [s for s in states if s.bucket == self.n_buckets - 1]
        if not right_states:
            return False

        solved = 0
        for s in right_states:
            if s.promoted or s.solve_rate >= self.exhaustion_threshold:
                solved += 1
        return (solved / len(right_states)) >= self.exhaustion_ratio

    def export(self) -> Dict:
        return {
            "n_buckets": self.n_buckets,
            "mean": self.mean,
            "std": self.std,
            "shift_success_ratio": self.shift_success_ratio,
            "exhaustion_ratio": self.exhaustion_ratio,
            "exhaustion_threshold": self.exhaustion_threshold,
        }

    def load(self, state: Dict) -> None:
        self.n_buckets = int(state["n_buckets"])
        self.mean = float(state["mean"])
        self.std = float(state["std"])
        self.shift_success_ratio = float(state.get("shift_success_ratio", 0.7))
        self.exhaustion_ratio = float(state.get("exhaustion_ratio", 0.8))
        self.exhaustion_threshold = float(state.get("exhaustion_threshold", 0.8))


if __name__ == "__main__":
    from .base import ProblemState

    dist = BucketDistribution(n_buckets=9, initial_mean=0.6, std=1.0)
    p0 = dist.get_probs()
    print("Initial probs:", [round(x, 4) for x in p0])
    print("Sum:", round(sum(p0), 8))
    print("Mode bucket:", max(range(len(p0)), key=lambda i: p0[i]))

    dist.shift_right(1.0)
    dist.shift_right(1.0)
    p1 = dist.get_probs()
    print("After shift probs:", [round(x, 4) for x in p1])
    print("Mode bucket:", max(range(len(p1)), key=lambda i: p1[i]))

    fake_states = [
        ProblemState(id=f"b1_{i}", bucket=1, total_attempts=4, solve_rate=0.9, consecutive_solves=2)
        for i in range(20)
    ]
    print(
        "should_shift True case:",
        dist.should_shift(
            fake_states,
            window=(1, 1),
            threshold=0.8,
            consecutive_required=2,
            min_evaluated=8,
        ),
    )

    tiny_states = fake_states[:2]
    print(
        "should_shift False (min_evaluated guard):",
        dist.should_shift(
            tiny_states,
            window=(1, 1),
            threshold=0.8,
            consecutive_required=2,
            min_evaluated=8,
        ),
    )
