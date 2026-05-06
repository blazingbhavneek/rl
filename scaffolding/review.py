"""CLI to browse scaffolded task output.

Usage:
    python -m scaffolding.review                        # summary: all functions
    python -m scaffolding.review --function <name>      # all tasks for one function
    python -m scaffolding.review --function <name> --task 2   # one task's files
    python -m scaffolding.review --output path/to/out
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _task_status(task_dir: Path) -> str:
    log = task_dir / "run_log.txt"
    if not log.exists():
        return "no run"
    text = log.read_text()
    if "PIPELINE ERROR" in text:
        return "ERROR"
    if "TIMEOUT" in text:
        return "TIMEOUT"
    return "PASS" if "exit: 0" in text else "FAIL"


def _task_tag(task_dir: Path) -> str:
    return "2p" if (task_dir / "process2.py").exists() else "1p"


def _task_desc(task_dir: Path) -> str:
    tj = task_dir / "task.json"
    if not tj.exists():
        return ""
    try:
        return json.loads(tj.read_text()).get("description", "")[:80]
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------


def summary(output_dir: Path) -> None:
    """One line per function showing pass/fail counts."""
    fns = sorted(p for p in output_dir.iterdir() if p.is_dir())
    if not fns:
        print(f"No functions found in {output_dir}/")
        return

    total_pass = total_fail = total_other = 0
    print(f"\n{'FUNCTION':<40} {'PASS':>5} {'FAIL':>5} {'OTHER':>6}")
    print("─" * 60)
    for fn_dir in fns:
        tasks = sorted(fn_dir.glob("task_*"))
        counts = {"PASS": 0, "FAIL": 0, "other": 0}
        for t in tasks:
            s = _task_status(t)
            if s == "PASS":
                counts["PASS"] += 1
            elif s == "FAIL":
                counts["FAIL"] += 1
            else:
                counts["other"] += 1
        total_pass += counts["PASS"]
        total_fail += counts["FAIL"]
        total_other += counts["other"]
        print(f"  {fn_dir.name:<38} {counts['PASS']:>5} {counts['FAIL']:>5} {counts['other']:>6}")
    print("─" * 60)
    print(f"  {'TOTAL':<38} {total_pass:>5} {total_fail:>5} {total_other:>6}")


def show_function(fn_dir: Path) -> None:
    """One line per task for a function."""
    tasks = sorted(fn_dir.glob("task_*"), key=lambda p: int(p.name.split("_")[1]))
    if not tasks:
        print(f"No tasks found in {fn_dir}/")
        return

    print(f"\n  {fn_dir.name}")
    print(f"  {'TASK':<8} {'STATUS':<8} {'TYPE':<5} DESCRIPTION")
    print("  " + "─" * 78)
    for t in tasks:
        status = _task_status(t)
        tag = _task_tag(t)
        desc = _task_desc(t)
        print(f"  {t.name:<8} {status:<8} {tag:<5} {desc}")


def show_task(task_dir: Path) -> None:
    """Dump all files for a single task."""
    print(f"\n{'=' * 70}")
    tj = task_dir / "task.json"
    if tj.exists():
        print(f"  task.json")
        print(f"  {json.loads(tj.read_text()).get('description', '')}")
    print("=" * 70)

    for fname in ["process1.py", "process2.py", "harness.py", "run_log.txt"]:
        fpath = task_dir / fname
        if not fpath.exists():
            continue
        print(f"\n{'─' * 4} {fname} {'─' * (64 - len(fname))}")
        print(fpath.read_text())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Review scaffolded SCADA output")
    parser.add_argument("--output", default="output", help="Output directory (default: output/)")
    parser.add_argument("--function", help="Show tasks for one function")
    parser.add_argument("--task", type=int, help="Show all files for task N (requires --function)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    if not output_dir.exists():
        print(f"Output directory '{output_dir}' does not exist.")
        return

    if args.function:
        fn_dir = output_dir / args.function
        if not fn_dir.exists():
            print(f"Function '{args.function}' not found in {output_dir}/")
            return
        if args.task is not None:
            task_dir = fn_dir / f"task_{args.task}"
            if not task_dir.exists():
                print(f"task_{args.task} not found in {fn_dir}/")
                return
            show_task(task_dir)
        else:
            show_function(fn_dir)
    else:
        summary(output_dir)


if __name__ == "__main__":
    main()
