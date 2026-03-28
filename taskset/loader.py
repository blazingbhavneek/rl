import json
import random
from collections import defaultdict, deque
from dataclasses import asdict
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

from .base import Problem, ProblemState, Score
from .curriculum import BucketDistribution
from .stats import StatsWriter
from .codeforces.dataset import CodeforcesDataset


class CurriculumLoader:
    def __init__(
        self,
        dataset_dir: str,
        x: int,
        solve_threshold: float,
        consecutive_required: int,
        max_steps: int,
        distribution: BucketDistribution,
        min_evaluated: int = 8,
        shift_delta: float = 1.0,
        shift_window_radius: int = 0,
        rolling_window: int = 20,
        require_full_bucket_coverage: bool = True,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.x = int(x)
        self.solve_threshold = float(solve_threshold)
        self.consecutive_required = int(consecutive_required)
        self.max_steps = int(max_steps)
        self.distribution = distribution
        self.min_evaluated = int(min_evaluated)
        self.shift_delta = float(shift_delta)
        self.shift_window_radius = int(shift_window_radius)
        self.rolling_window = int(rolling_window)
        self.require_full_bucket_coverage = bool(require_full_bucket_coverage)

        self.dataset = CodeforcesDataset(data_dir=str(self.dataset_dir))
        if self.distribution.n_buckets != self.dataset.n_buckets():
            raise ValueError(
                f"distribution n_buckets ({self.distribution.n_buckets}) != dataset n_buckets ({self.dataset.n_buckets()})"
            )

        self.checkpoint_dir = self.dataset_dir.parent / "checkpoint"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.checkpoint_dir / "problem_states.json"

        self.stats = StatsWriter(str(self.checkpoint_dir))

        self.problem_states: Dict[str, ProblemState] = {}
        self._history: Dict[str, Deque[int]] = {}
        self._stop_exhausted = False

        self.load_checkpoint()

    def _window(self) -> Tuple[int, int]:
        center = int(round(self.distribution.mean))
        lo = max(0, center - self.shift_window_radius)
        hi = min(self.distribution.n_buckets - 1, center + self.shift_window_radius)
        return lo, hi

    def _get_or_create_state(self, p: Problem, step: int) -> ProblemState:
        if p.id not in self.problem_states:
            self.problem_states[p.id] = ProblemState(
                id=p.id,
                bucket=p.bucket,
                last_sampled_step=step,
            )
            self._history[p.id] = deque(maxlen=self.rolling_window)
        return self.problem_states[p.id]

    def _eligible_in_bucket(self, bucket_idx: int) -> List[Problem]:
        bucket = self.dataset.get_bucket(bucket_idx)
        out = []
        for p in bucket:
            s = self.problem_states.get(p.id)
            if s is None or not s.promoted:
                out.append(p)
        return out

    def _fair_pick_in_bucket(self, bucket_idx: int) -> Optional[Problem]:
        """
        Fairness policy inside a bucket:
        pick uniformly among the least-sampled problems.
        This guarantees unsampled tasks are consumed before repeated sampling,
        and keeps per-bucket exposure approximately equal over time.
        """
        candidates = self._eligible_in_bucket(bucket_idx)
        if not candidates:
            return None

        min_attempts = None
        least_sampled: List[Problem] = []
        for p in candidates:
            attempts = self.problem_states.get(p.id).total_attempts if p.id in self.problem_states else 0
            if min_attempts is None or attempts < min_attempts:
                min_attempts = attempts
                least_sampled = [p]
            elif attempts == min_attempts:
                least_sampled.append(p)
        return random.choice(least_sampled)

    def _resolve_bucket_sample(self) -> Optional[Problem]:
        probs = self.distribution.get_probs()
        bucket_idx = random.choices(range(len(probs)), weights=probs, k=1)[0]

        picked = self._fair_pick_in_bucket(bucket_idx)
        if picked is not None:
            return picked

        for offset in range(1, self.dataset.n_buckets()):
            left = bucket_idx - offset
            right = bucket_idx + offset
            if left >= 0:
                c = self._fair_pick_in_bucket(left)
                if c is not None:
                    return c
            if right < self.dataset.n_buckets():
                c = self._fair_pick_in_bucket(right)
                if c is not None:
                    return c
        return None

    def _promote_mastered(self) -> int:
        lo, hi = self._window()
        promoted = 0
        for s in self.problem_states.values():
            if s.promoted:
                continue
            if not (lo <= s.bucket <= hi):
                continue
            if s.solve_rate >= self.solve_threshold and s.consecutive_solves >= self.consecutive_required:
                s.promoted = True
                promoted += 1
        return promoted

    def _window_fully_covered(self) -> bool:
        """
        Coverage gate: all problems in the active curriculum window must be
        sampled at least once before the distribution can shift right.
        """
        lo, hi = self._window()
        for bucket_idx in range(lo, hi + 1):
            problems = self.dataset.get_bucket(bucket_idx)
            if not problems:
                continue
            for p in problems:
                st = self.problem_states.get(p.id)
                if st is None or st.total_attempts <= 0:
                    return False
        return True

    def _maybe_shift(self) -> int:
        promoted = 0
        if self.distribution.should_shift(
            states=self.problem_states.values(),
            window=self._window(),
            threshold=self.solve_threshold,
            consecutive_required=self.consecutive_required,
            min_evaluated=self.min_evaluated,
        ) and (
            (not self.require_full_bucket_coverage) or self._window_fully_covered()
        ):
            promoted = self._promote_mastered()
            old_mean = self.distribution.mean
            self.distribution.shift_right(self.shift_delta)
            if self.distribution.is_exhausted(self.problem_states.values()):
                self._stop_exhausted = True
        return promoted

    def sample(self, step: int) -> List[Problem]:
        sampled = []
        for _ in range(self.x):
            self._maybe_shift()
            p = self._resolve_bucket_sample()
            if p is None:
                self._stop_exhausted = True
                break
            st = self._get_or_create_state(p, step)
            st.last_sampled_step = step
            sampled.append(p)
        return sampled

    def update(self, problem_ids: List[str], scores: List[Score], step: int) -> None:
        if len(problem_ids) != len(scores):
            raise ValueError("problem_ids and scores must have equal length")

        for pid, sc in zip(problem_ids, scores):
            bucket = self.dataset.get_problem_bucket(pid)
            if bucket is None:
                continue

            if pid not in self.problem_states:
                self.problem_states[pid] = ProblemState(id=pid, bucket=bucket)
                self._history[pid] = deque(maxlen=self.rolling_window)

            st = self.problem_states[pid]
            solved = int(sc.total > 0 and (sc.passed / sc.total) >= self.solve_threshold)
            hist = self._history.setdefault(pid, deque(maxlen=self.rolling_window))
            hist.append(solved)

            st.total_attempts += 1
            st.solve_rate = sum(hist) / len(hist)
            st.consecutive_solves = st.consecutive_solves + 1 if solved else 0
            st.last_sampled_step = step

        promoted_this_step = self._maybe_shift()
        self.save_checkpoint()

        per_bucket_sum = defaultdict(float)
        per_bucket_n = defaultdict(int)
        for s in self.problem_states.values():
            per_bucket_sum[s.bucket] += s.solve_rate
            per_bucket_n[s.bucket] += 1
        per_bucket_rate = {
            b: (per_bucket_sum[b] / per_bucket_n[b]) if per_bucket_n[b] else 0.0
            for b in range(self.dataset.n_buckets())
        }

        mean_score = 0.0
        if scores:
            mean_score = sum((sc.passed / sc.total) if sc.total else 0.0 for sc in scores) / len(scores)

        self.stats.write(
            step=step,
            distribution=self.distribution.get_probs(),
            solve_rates=per_bucket_rate,
            mean_score=mean_score,
            promoted_this_step=promoted_this_step,
            total_problems_seen=len(self.problem_states),
        )

    def should_stop(self, step: int) -> bool:
        if step >= self.max_steps:
            return True
        if self._stop_exhausted:
            return True
        if self.distribution.is_exhausted(self.problem_states.values()):
            self._stop_exhausted = True
            return True
        return False

    def save_checkpoint(self) -> None:
        data = {
            "distribution": self.distribution.export(),
            "problem_states": [asdict(s) for s in self.problem_states.values()],
            "history": {k: list(v) for k, v in self._history.items()},
            "stop_exhausted": self._stop_exhausted,
            "rolling_window": self.rolling_window,
        }
        with self.checkpoint_path.open("w", encoding="utf-8") as f:
            json.dump(data, f)

    def load_checkpoint(self) -> None:
        if not self.checkpoint_path.exists():
            return
        with self.checkpoint_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if "distribution" in data:
            self.distribution.load(data["distribution"])

        self.problem_states = {}
        for row in data.get("problem_states", []):
            ps = ProblemState(**row)
            self.problem_states[ps.id] = ps

        self._history = {}
        for pid, arr in data.get("history", {}).items():
            dq = deque(maxlen=self.rolling_window)
            for v in arr:
                dq.append(int(v))
            self._history[pid] = dq

        self._stop_exhausted = bool(data.get("stop_exhausted", False))

    def get_stats(self) -> Dict:
        return {
            "n_states": len(self.problem_states),
            "distribution": self.distribution.get_probs(),
            "mean": self.distribution.mean,
            "stop_exhausted": self._stop_exhausted,
            "checkpoint_path": str(self.checkpoint_path),
        }


if __name__ == "__main__":
    from .base import Score
    from .curriculum import BucketDistribution

    data_dir = Path(__file__).parent / "codeforces" / "data"
    if not data_dir.exists():
        print(f"missing data dir: {data_dir}")
        raise SystemExit(0)

    loader = CurriculumLoader(
        dataset_dir=str(data_dir),
        x=10,
        solve_threshold=0.8,
        consecutive_required=2,
        max_steps=50,
        distribution=BucketDistribution(n_buckets=9, initial_mean=0.6, std=1.0),
        min_evaluated=3,
    )

    s0 = loader.sample(step=0)
    bucket_counts_0 = defaultdict(int)
    for p in s0:
        bucket_counts_0[p.bucket] += 1
    print("sample0 bucket counts:", dict(bucket_counts_0))

    for step in range(1, 4):
        if not s0:
            break
        ids = [p.id for p in s0]
        scores = [Score(compiled=True, passed=10, total=10, error=None) for _ in ids]
        loader.update(ids, scores, step=step)
        s0 = loader.sample(step=step)

    bucket_counts_1 = defaultdict(int)
    for p in s0:
        bucket_counts_1[p.bucket] += 1
    print("sample_after_updates bucket counts:", dict(bucket_counts_1))
    print("current mean:", loader.distribution.mean)
    print("should_stop(max_steps):", loader.should_stop(step=loader.max_steps))
