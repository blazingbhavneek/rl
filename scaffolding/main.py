"""Entry point for the SCADA scaffolding pipeline.

Full concurrent pipeline for 100s of functions × N tasks each:

  Layer 1  parse_md        → function entries from scada.md
  Layer 0.5 generate_tasks → LLM invents N diverse task descriptions per function
  Layer 2  generate_code   → LLM classifies + generates process*.py per task
  Layer 3  generate_harness → templated harness.py per task
  Layer 4  run_sandbox     → subprocess verify, writes run_log.txt

All (function × task) pairs run concurrently under a bounded asyncio.Semaphore.

Usage:
    python -m scaffolding.main --input scada.md
    python -m scaffolding.main --input scada.md --function RingBufferInsert
    python -m scaffolding.main --input scada.md --n-tasks 5 --concurrency 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import traceback
from pathlib import Path

from langchain_openai import ChatOpenAI

from .generate_code import generate_code
from .generate_harness import generate_harness
from .generate_tasks import generate_tasks
from .parse_md import parse_scada_md
from .run_sandbox import run_sandbox

# ---------------------------------------------------------------------------
# Per-task worker
# ---------------------------------------------------------------------------


async def _run_task(
    task: dict,
    task_idx: int,
    config: dict,
    fn_dir: Path,
    sem: asyncio.Semaphore,
) -> dict:
    """Run the full pipeline for one (function, task) pair under the semaphore."""
    fn_name = task["function_name"]
    label = f"{fn_name}/task_{task_idx}"

    async with sem:
        task_dir = fn_dir / f"task_{task_idx}"
        if (task_dir / "run_log.txt").exists():
            return {"label": label, "skipped": True}

        task_dir.mkdir(parents=True, exist_ok=True)

        # Save the task description so review.py and future layers can read it
        (task_dir / "task.json").write_text(
            json.dumps(
                {
                    "function_name": fn_name,
                    "task_id": task_idx,
                    "description": task["description"],
                    "pattern": task.get("pattern", ""),
                    "primary_type": task.get("primary_type", "i"),
                    "two_process": task.get("two_process", False),
                    "process_count": 2 if task.get("two_process") else 1,
                    "two_process_reason": task.get("two_process_reason", ""),
                },
                indent=2,
            )
        )

        try:
            await generate_code(task, config, task_dir)
            generate_harness(task_dir, task["two_process"])
            success, _ = run_sandbox(task_dir)
        except Exception as exc:
            (task_dir / "run_log.txt").write_text(
                f"=== PIPELINE ERROR ===\n{traceback.format_exc()}\n"
            )
            success = False

        status = "PASS" if success else "FAIL"
        tag = "2p" if task.get("two_process") else "1p"
        print(f"  [{status}] [{tag}] {label}", flush=True)
        return {"label": label, "success": success}


# ---------------------------------------------------------------------------
# Per-function worker
# ---------------------------------------------------------------------------


async def _run_function(
    entry: dict,
    llm: ChatOpenAI,
    config: dict,
    output_dir: Path,
    n_tasks: int,
    sem: asyncio.Semaphore,
    skip_existing: bool,
) -> list[dict]:
    fn_name = entry["name"]
    fn_dir = output_dir / fn_name

    if skip_existing and fn_dir.exists():
        existing = list(fn_dir.glob("task_*/run_log.txt"))
        if len(existing) >= n_tasks:
            print(f"  [skip] {fn_name} ({len(existing)} tasks already done)", flush=True)
            return []

    print(f"  [design] {fn_name} — generating {n_tasks} tasks…", flush=True)
    try:
        tasks = await generate_tasks(entry, llm, n=n_tasks)
    except Exception as exc:
        print(f"  [ERROR] {fn_name} task design failed: {exc}", flush=True)
        return []

    workers = [
        _run_task(task, i, config, fn_dir, sem)
        for i, task in enumerate(tasks)
    ]
    return await asyncio.gather(*workers)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def _run(args: argparse.Namespace) -> None:
    config = json.loads(Path(args.config).read_text())
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = parse_scada_md(args.input)
    if not entries:
        print(f"No functions parsed from '{args.input}'. Check the file format.")
        return

    if args.function:
        entries = [e for e in entries if e["name"] == args.function]
        if not entries:
            print(f"Function '{args.function}' not found in '{args.input}'.")
            return

    # One shared LLM client for task design (cheap, stateless calls)
    llm = ChatOpenAI(
        model=config["model"],
        api_key=config["api_key"],
        base_url=config["base_url"],
        temperature=0.4,          # slightly higher for task variety
        max_tokens=config.get("max_tokens", 2048),
    )

    sem = asyncio.Semaphore(args.concurrency)

    print(
        f"Scaffolding {len(entries)} function(s) × {args.n_tasks} tasks"
        f" (concurrency={args.concurrency}) → '{output_dir}/'"
    )

    fn_results = await asyncio.gather(
        *[
            _run_function(entry, llm, config, output_dir, args.n_tasks, sem, args.skip_existing)
            for entry in entries
        ]
    )

    all_results = [r for fn in fn_results for r in fn]
    ran = [r for r in all_results if not r.get("skipped")]
    passed = sum(1 for r in ran if r.get("success"))
    print(f"\nDone: {passed}/{len(ran)} passed  ({len(all_results) - len(ran)} skipped)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SCADA scaffolding pipeline: scada.md → output/{fn}/task_N/"
    )
    parser.add_argument("--input", default="scada.md", help="Path to scada.md")
    parser.add_argument("--output", default="output", help="Output directory (default: output/)")
    parser.add_argument(
        "--config",
        default="scaffolding/config.json",
        help="LLM config JSON (model, base_url, api_key)",
    )
    parser.add_argument("--function", help="Process only this function by name")
    parser.add_argument("--n-tasks", type=int, default=5, help="Task variations per function (default: 5)")
    parser.add_argument("--concurrency", type=int, default=10, help="Max concurrent (fn×task) workers (default: 10)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip functions already fully generated")
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
