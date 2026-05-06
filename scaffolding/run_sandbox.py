"""Layer 4 — Sandbox runner.

Executes harness.py in a subprocess from its own directory,
captures stdout/stderr, writes run_log.txt, returns (success, log).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

DEFAULT_TIMEOUT = 10


def run_sandbox(task_dir: Path, timeout: int = DEFAULT_TIMEOUT) -> tuple[bool, str]:
    """Run harness.py in task_dir as a subprocess.

    Returns:
        (success, log_text) — success is True iff exit code is 0.
    Writes run_log.txt into task_dir.
    """
    harness = task_dir.resolve() / "harness.py"
    if not harness.exists():
        raise FileNotFoundError(f"harness.py not found in {task_dir}")

    try:
        result = subprocess.run(
            [sys.executable, str(harness)],
            cwd=str(task_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        success = result.returncode == 0
        log = (
            f"=== stdout ===\n{result.stdout}"
            f"\n=== stderr ===\n{result.stderr}"
            f"\n=== exit: {result.returncode} ===\n"
        )
    except subprocess.TimeoutExpired:
        success = False
        log = f"=== TIMEOUT ({timeout}s) ===\n"

    (task_dir / "run_log.txt").write_text(log)
    return success, log
