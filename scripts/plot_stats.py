#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def _get_nested(d: Dict, path: str) -> Optional[float]:
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    if cur is None:
        return None
    try:
        return float(cur)
    except Exception:
        return None


def _first_present(d: Dict, paths: Iterable[str]) -> Optional[float]:
    for p in paths:
        v = _get_nested(d, p)
        if v is not None:
            return v
    return None


def _moving_average(values: List[float], window: int) -> List[float]:
    if window <= 1 or not values:
        return values
    out: List[float] = []
    running = 0.0
    q: List[float] = []
    for v in values:
        q.append(v)
        running += v
        if len(q) > window:
            running -= q.pop(0)
        out.append(running / len(q))
    return out


def _load_rows(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _plot_line(
    x: List[int],
    y: List[float],
    title: str,
    ylabel: str,
    out_path: Path,
    smooth: int = 1,
) -> None:
    import matplotlib.pyplot as plt

    if not x or not y:
        return
    ys = _moving_average(y, smooth)
    plt.figure(figsize=(10, 4))
    plt.plot(x, ys, linewidth=1.6)
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required. Install with: pip install matplotlib"
        ) from exc

    parser = argparse.ArgumentParser(description="Plot RL training stats from steps.jsonl")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run directory containing stats/steps.jsonl",
    )
    parser.add_argument(
        "--steps-file",
        type=str,
        default=None,
        help="Direct path to steps.jsonl (overrides --run-dir)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for PNG files (default: <run>/stats/plots)",
    )
    parser.add_argument("--smooth", type=int, default=1, help="Moving average window size")
    args = parser.parse_args()

    if args.steps_file:
        steps_path = Path(args.steps_file)
    else:
        if args.run_dir:
            run_dir = Path(args.run_dir)
        else:
            import config as cfg

            run_dir = Path(cfg.OUTPUT_DIR)
        steps_path = run_dir / "stats" / "steps.jsonl"

    if not steps_path.exists():
        raise FileNotFoundError(f"steps file not found: {steps_path}")

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = steps_path.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(steps_path)
    if not rows:
        raise RuntimeError(f"no rows found in {steps_path}")

    steps = [int(r.get("step", i)) for i, r in enumerate(rows)]

    mean_reward = [
        _first_present(r, ("rl_stats.mean_reward",))
        for r in rows
    ]
    loss = [
        _first_present(r, ("rl_stats.bp_loss", "rl_stats.loss"))
        for r in rows
    ]
    kl = [
        _first_present(r, ("rl_stats.bp_kl_loss", "rl_stats.kl_loss"))
        for r in rows
    ]
    grad_norm = [
        _first_present(r, ("rl_stats.grad_norm",))
        for r in rows
    ]
    lr = [
        _first_present(r, ("rl_stats.lr",))
        for r in rows
    ]
    success_ratio = [
        _first_present(r, ("rl_stats.success_ratio",))
        for r in rows
    ]
    passed_rl = [float(r.get("n_passed_rl", 0)) for r in rows]
    passed_hint = [float(r.get("n_passed_after_hint", 0)) for r in rows]
    sft_pairs = [
        _first_present(r, ("sft_stats.n_pairs_saved",))
        for r in rows
    ]

    def _valid_xy(vals: List[Optional[float]]) -> tuple[List[int], List[float]]:
        x: List[int] = []
        y: List[float] = []
        for step, v in zip(steps, vals):
            if v is None:
                continue
            x.append(step)
            y.append(v)
        return x, y

    plots = [
        ("mean_reward", mean_reward, "Mean Reward", "reward"),
        ("loss", loss, "Loss", "loss"),
        ("kl", kl, "KL Divergence", "kl"),
        ("grad_norm", grad_norm, "Gradient Norm", "grad_norm"),
        ("lr", lr, "Learning Rate", "lr"),
        ("success_ratio", success_ratio, "Success Ratio", "ratio"),
        ("passed_rl", passed_rl, "Passed in RL Phase", "count"),
        ("passed_after_hint", passed_hint, "Passed After Hint", "count"),
        ("sft_pairs", sft_pairs, "SFT Pairs Saved", "count"),
    ]

    for name, vals, title, ylabel in plots:
        x, y = _valid_xy(vals)  # type: ignore[arg-type]
        _plot_line(x, y, title, ylabel, out_dir / f"{name}.png", smooth=max(1, args.smooth))

    # Bucket sampling chart.
    bucket_counts: Dict[str, List[float]] = {}
    for r in rows:
        sb = r.get("sampled_buckets", {}) or {}
        for k in sb.keys():
            bucket_counts.setdefault(str(k), [])
    for k in bucket_counts.keys():
        bucket_counts[k] = [float((r.get("sampled_buckets", {}) or {}).get(k, 0)) for r in rows]
    if bucket_counts:
        plt.figure(figsize=(12, 5))
        for k in sorted(bucket_counts.keys(), key=lambda x: int(x)):
            plt.plot(steps, _moving_average(bucket_counts[k], max(1, args.smooth)), label=f"bucket {k}")
        plt.title("Sampled Buckets Per Step")
        plt.xlabel("step")
        plt.ylabel("count")
        plt.grid(alpha=0.25)
        plt.legend(ncol=3, fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / "sampled_buckets.png", dpi=160)
        plt.close()

    print(f"saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
