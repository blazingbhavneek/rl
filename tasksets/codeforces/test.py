import argparse
import random
from collections import Counter
from collections import deque
from pathlib import Path
from typing import Dict, List

from tasksets.base import Score
from tasksets.curriculum import BucketDistribution
from tasksets.loader import CurriculumLoader

BUCKET_FILES = ["b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b9"]


def count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def print_data_distribution(data_dir: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    total = 0
    print("Data bucket distribution:")
    for b in BUCKET_FILES:
        n = count_jsonl_lines(data_dir / f"{b}.jsonl")
        counts[b] = n
        total += n
        print(f"  {b}: {n}")
    print(f"  total: {total}")
    return counts


def skill_from_phase(step: int, top_reached_step: int, scenario: str) -> float:
    if top_reached_step < 0:
        # Realistic gradual improvement.
        return min(0.90, 0.48 + 0.008 * step)
    if scenario == "rise_degrade":
        # After reaching top, degrade steadily.
        return max(0.35, 0.90 - 0.010 * (step - top_reached_step))
    # For rise_stop, this value is not used after top is reached.
    return 0.90


def simulate_rewards(batch, skill: float, rng: random.Random) -> List[Score]:
    scores: List[Score] = []
    for p in batch:
        solve_prob = max(0.01, min(0.99, skill - 0.055 * p.bucket + rng.uniform(-0.05, 0.05)))
        solved = rng.random() < solve_prob

        if solved:
            passed = 8 + int(rng.random() * 3)  # 8-10
            error = None
        else:
            passed = int(rng.random() * 6)      # 0-5
            error = "simulated failure"

        scores.append(
            Score(
                compiled=True,
                passed=passed,
                total=10,
                error=error,
                details={"synthetic_solve_prob": round(solve_prob, 4)},
            )
        )
    return scores


def run_simulation(data_dir: Path, steps: int, sample_size: int, seed: int, scenario: str) -> None:
    rng = random.Random(seed)
    random.seed(seed)

    shift_success_ratio = 0.32
    shift_delta = 0.25
    min_evaluated = 12
    initial_mean = 0.60
    std = 1.0

    loader = CurriculumLoader(
        dataset_dir=str(data_dir),
        x=sample_size,
        solve_threshold=0.78,
        consecutive_required=1,
        max_steps=max(steps + 20, 120),
        distribution=BucketDistribution(
            n_buckets=9,
            initial_mean=initial_mean,
            std=std,
            shift_success_ratio=0.32,
        ),
        min_evaluated=min_evaluated,
        shift_delta=shift_delta,
        shift_window_radius=0,
        rolling_window=20,
    )
    # Smoke tests should be read-only with no persisted side effects.
    loader.save_checkpoint = lambda: None
    loader.stats.write = lambda **_: None
    # Ignore any loaded checkpoint state and start deterministic from config.
    loader.distribution = BucketDistribution(
        n_buckets=9,
        initial_mean=initial_mean,
        std=std,
        shift_success_ratio=shift_success_ratio,
    )
    loader.problem_states = {}
    loader._history = {}
    loader._stop_exhausted = False

    print("\nSimulation:")
    print(f"  scenario: {scenario}")
    print(
        "step skill mean mode avg_reward eval/mastered ratio shift "
        "sampled[b1..b9] probs[b1..b9]"
    )
    top_reached_step = -1

    hold_streak = 0
    top_reward_window = deque(maxlen=30)
    top_best_reward = -1.0
    top_steps_since_best = 0
    for step in range(1, steps + 1):
        batch = loader.sample(step)
        if not batch:
            print(f"{step:>4}  ---  {loader.distribution.mean:>4.2f}   -   --- no samples")
            break

        if top_reached_step < 0 and loader.distribution.mean >= 7.8 and mode == 9:
            top_reached_step = step
            if scenario == "rise_stop":
                print(f"{step:>4}  ---  {loader.distribution.mean:>4.2f}   -   --- TOP_REACHED_STOP")
                break
            print(f"{step:>4}  ---  {loader.distribution.mean:>4.2f}   -   --- TOP_REACHED_DEGRADE_START")

        skill = skill_from_phase(step, top_reached_step, scenario)
        mean_before = loader.distribution.mean
        scores = simulate_rewards(batch, skill=skill, rng=rng)
        loader.update([p.id for p in batch], scores, step)
        avg_reward = sum(sc.passed / sc.total for sc in scores) / len(scores)
        shifted = loader.distribution.mean > mean_before
        shift_reason = "loader"
        probs = loader.distribution.get_probs()
        mode = max(range(len(probs)), key=lambda i: probs[i]) + 1  # print as b1..b9

        lo, hi = loader._window()
        window_states = [
            s for s in loader.problem_states.values()
            if lo <= s.bucket <= hi and not s.promoted and s.total_attempts > 0
        ]
        evaluated = len(window_states)
        mastered = sum(
            1 for s in window_states
            if s.solve_rate >= loader.solve_threshold and s.consecutive_solves >= loader.consecutive_required
        )
        ratio = (mastered / evaluated) if evaluated else 0.0
        if shifted:
            hold_streak = 0
        else:
            hold_streak += 1
            # Plateau trigger for smoke runs:
            # if we hold too long with saturated reward in the same window, nudge right.
            if hold_streak >= 8 and avg_reward >= 0.72 and loader.distribution.mean < 8.0:
                loader.distribution.shift_right(0.25)
                shifted = True
                shift_reason = "plateau"
                hold_streak = 0
                probs = loader.distribution.get_probs()
                mode = max(range(len(probs)), key=lambda i: probs[i]) + 1

        sampled = Counter(p.bucket for p in batch)
        sampled_arr = [sampled.get(i, 0) for i in range(9)]
        probs_arr = [round(probs[i], 2) for i in range(9)]

        print(
            f"{step:>4} {skill:>4.2f} {loader.distribution.mean:>4.2f} b{mode} "
            f"{avg_reward:>5.3f} {evaluated:>3}/{mastered:<3} {ratio:>4.2f} "
            f"{('SHIFT(' + shift_reason + ')') if shifted else 'hold         '} "
            f"{sampled_arr} {probs_arr}"
        )

        # Smoke-stop: when top bucket is sustained and rewards plateau, stop early.
        at_top = loader.distribution.mean >= 7.8 and mode == 9
        if at_top:
            top_reward_window.append(avg_reward)
            if avg_reward > top_best_reward + 0.02:
                top_best_reward = avg_reward
                top_steps_since_best = 0
            else:
                top_steps_since_best += 1

            if top_steps_since_best >= 120:
                print(
                    f"{step + 1:>4}  ---  {loader.distribution.mean:>4.2f}   b9  "
                    f"--- STOP_TOP_PLATEAU (no reward improvement for {top_steps_since_best} top steps; best={top_best_reward:.3f})"
                )
                break

            if len(top_reward_window) == top_reward_window.maxlen:
                spread = max(top_reward_window) - min(top_reward_window)
                if spread < 0.08:
                    print(
                        f"{step + 1:>4}  ---  {loader.distribution.mean:>4.2f}   b9  "
                        f"--- STOP_TOP_PLATEAU (reward spread {spread:.3f} over {top_reward_window.maxlen} steps)"
                    )
                    break
        else:
            top_reward_window.clear()
            top_best_reward = -1.0
            top_steps_since_best = 0

    print("\nFinal loader stats:")
    stats = loader.get_stats()
    print(f"  mean: {stats['mean']:.3f}")
    print(f"  tracked problem states: {stats['n_states']}")
    print(f"  should_stop({steps}): {loader.should_stop(steps)}")
    print(f"  top_reached_step: {top_reached_step if top_reached_step >= 0 else 'not reached'}")
    print("  persisted checkpoint/stats writes: disabled for smoke test")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tasksets Codeforces all-in-one smoke test")
    parser.add_argument("--data-dir", type=str, default=str(Path(__file__).parent / "data"))
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--sample-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument(
        "--scenario",
        type=str,
        default="rise_degrade",
        choices=["rise_stop", "rise_degrade"],
        help="rise_stop: stop when top bucket reached; rise_degrade: after top, degrade skill and continue",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Missing data dir: {data_dir}")
        print("Run dataset build first, e.g. python -m tasksets.codeforces.make_dataset --max-cf 500 --no-leetcode --out-dir tasksets/codeforces/data")
        raise SystemExit(1)

    counts = print_data_distribution(data_dir)
    if sum(counts.values()) == 0:
        print("No bucket data found. Build dataset first.")
        raise SystemExit(1)

    run_simulation(
        data_dir,
        steps=args.steps,
        sample_size=args.sample_size,
        seed=args.seed,
        scenario=args.scenario,
    )


if __name__ == "__main__":
    main()
