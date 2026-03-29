import os
import re
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..base import BaseVerifier, Problem, Score

_GLOBAL_EXECUTOR: Optional[ProcessPoolExecutor] = None
_FALLBACK_THREAD_EXECUTOR: Optional[ThreadPoolExecutor] = None


def _extract_code(text: str) -> Optional[str]:
    for lang in ("c", "cpp", ""):
        m = re.search(rf"```{lang}\s*\n(.*?)```", text, re.DOTALL)
        if m:
            return m.group(1).strip()
    return None


def _compile_code(code: str, tmp_dir: str) -> Tuple[bool, str, Optional[str]]:
    src = Path(tmp_dir) / "sol.c"
    exe = Path(tmp_dir) / "sol"
    src.write_text(code, encoding="utf-8")

    try:
        cp = subprocess.run(
            ["gcc", "-std=c11", "-O2", str(src), "-o", str(exe), "-lm", "-lgmp"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return False, "compilation timed out", None

    if cp.returncode != 0:
        return False, (cp.stderr.strip() or "compilation failed"), None
    return True, "", str(exe)


def _run_one_test(exe_path: str, test_case: Dict, timeout: float) -> Tuple[bool, str, float]:
    try:
        rp = subprocess.run(
            [exe_path],
            input=test_case.get("input", ""),
            capture_output=True,
            text=True,
            timeout=timeout,
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout}s", timeout

    if rp.returncode != 0:
        return False, f"runtime error (exit {rp.returncode})", 0.0

    actual = "\n".join(l.rstrip() for l in rp.stdout.rstrip("\n").split("\n"))
    expected = "\n".join(l.rstrip() for l in test_case.get("output", "").rstrip("\n").split("\n"))
    if actual != expected:
        return False, f"wrong answer: expected '{expected[:120]}', got '{actual[:120]}'", 0.0
    return True, "", 0.0


def _verify_worker(completion: str, test_cases: List[Dict], timeout: float) -> Dict:
    code = _extract_code(completion)
    total = len(test_cases)

    if code is None:
        return {
            "compiled": False,
            "passed": 0,
            "total": total,
            "error": "no code block found",
            "details": {"reason": "extract_code_failed"},
        }

    with tempfile.TemporaryDirectory(prefix="tasksets_v_") as tmp:
        compiled, err, exe_path = _compile_code(code, tmp)
        if not compiled:
            return {
                "compiled": False,
                "passed": 0,
                "total": total,
                "error": err,
                "details": {"stage": "compile"},
            }

        passed = 0
        first_error = None
        for i, tc in enumerate(test_cases):
            ok, msg, _ = _run_one_test(exe_path, tc, timeout)
            if ok:
                passed += 1
            elif first_error is None:
                first_error = f"test {i + 1}: {msg}"

    return {
        "compiled": True,
        "passed": passed,
        "total": total,
        "error": first_error,
        "details": {"stage": "run" if first_error else "pass"},
    }


class CodeforcesVerifier(BaseVerifier):
    def __init__(self, timeout: float = 5.0, n_workers: int = 32) -> None:
        self.timeout = float(timeout)
        self.n_workers = int(n_workers)

    def _executor(self):
        global _GLOBAL_EXECUTOR
        if _GLOBAL_EXECUTOR is None:
            try:
                _GLOBAL_EXECUTOR = ProcessPoolExecutor(max_workers=self.n_workers)
            except Exception:
                global _FALLBACK_THREAD_EXECUTOR
                if _FALLBACK_THREAD_EXECUTOR is None:
                    _FALLBACK_THREAD_EXECUTOR = ThreadPoolExecutor(max_workers=self.n_workers)
                return _FALLBACK_THREAD_EXECUTOR
        return _GLOBAL_EXECUTOR

    def extract_code(self, completion: str) -> Optional[str]:
        return _extract_code(completion)

    def verify(self, problem: Problem, completion: str) -> Score:
        tcs = problem.metadata.get("test_cases", [])
        out = _verify_worker(completion, tcs, self.timeout)
        return Score(
            compiled=bool(out["compiled"]),
            passed=int(out["passed"]),
            total=int(out["total"]),
            error=out.get("error"),
            details=out.get("details", {}),
        )

    def verify_batch(self, problem: Problem, completions: List[str]) -> List[Score]:
        tcs = problem.metadata.get("test_cases", [])
        executor = self._executor()
        futures = [executor.submit(_verify_worker, c, tcs, self.timeout) for c in completions]

        results: List[Score] = []
        max_wait = self.timeout * max(len(tcs), 1) + 60.0
        for fut in futures:
            try:
                out = fut.result(timeout=max_wait)
            except Exception as e:
                out = {
                    "compiled": False,
                    "passed": 0,
                    "total": len(tcs),
                    "error": str(e),
                    "details": {"stage": "exception"},
                }
            results.append(
                Score(
                    compiled=bool(out["compiled"]),
                    passed=int(out["passed"]),
                    total=int(out["total"]),
                    error=out.get("error"),
                    details=out.get("details", {}),
                )
            )
        return results

    def _compile(self, code: str, tmp_dir: str) -> Tuple[bool, str]:
        ok, err, _ = _compile_code(code, tmp_dir)
        return ok, err

    def _run_testcase(self, exe: str, test_case: Dict) -> Tuple[bool, str]:
        ok, msg, _ = _run_one_test(exe, test_case, self.timeout)
        return ok, msg

    def check_dependencies(self) -> None:
        errors = []

        if subprocess.run(["which", "gcc"], capture_output=True).returncode != 0:
            errors.append("gcc not found")

        gmp_src = "#include <gmp.h>\nint main(){mpz_t x;mpz_init(x);mpz_clear(x);return 0;}\n"
        with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as f:
            f.write(gmp_src)
            fname = f.name
        try:
            r = subprocess.run(
                ["gcc", "-std=c11", fname, "-o", "/dev/null", "-lgmp"],
                capture_output=True,
                text=True,
            )
            if r.returncode != 0:
                errors.append("libgmp not found")
        finally:
            os.unlink(fname)

        uthash_ok = any(
            p.exists()
            for p in [
                Path("uthash.h"),
                Path("/usr/include/uthash.h"),
                Path("/usr/local/include/uthash.h"),
            ]
        )
        if not uthash_ok:
            errors.append("uthash.h not found")

        if errors:
            raise SystemExit("Dependency check FAILED: " + "; ".join(errors))


if __name__ == "__main__":
    test_problem = Problem(
        id="ab_demo",
        statement="Given two integers a and b, print a+b.",
        bucket=0,
        difficulty_label="b1",
        metadata={
            "test_cases": [
                {"input": "1 2\n", "output": "3\n"},
                {"input": "10 -7\n", "output": "3\n"},
            ]
        },
    )

    good = """```c
#include <stdio.h>
int main(){ long long a,b; if(scanf(\"%lld %lld\", &a, &b)!=2) return 0; printf(\"%lld\\n\", a+b); return 0; }
```"""

    wrong = """```c
#include <stdio.h>
int main(){ long long a,b; if(scanf(\"%lld %lld\", &a, &b)!=2) return 0; printf(\"%lld\\n\", a-b); return 0; }
```"""

    broken = """```c
#include <stdio.h>
int main( { return 0; }
```"""

    v = CodeforcesVerifier(timeout=2.0, n_workers=4)

    s_good = v.verify(test_problem, good)
    s_wrong = v.verify(test_problem, wrong)
    s_broken = v.verify(test_problem, broken)

    print("good:", s_good)
    print("wrong:", s_wrong)
    print("broken:", s_broken)

    assert s_good.compiled and s_good.passed == s_good.total
    assert s_wrong.compiled and s_wrong.passed < s_wrong.total
    assert (not s_broken.compiled) and s_broken.passed == 0

    batch = v.verify_batch(test_problem, [good, wrong, broken])
    print("batch:", batch)
    assert batch[0].passed == s_good.passed
    assert batch[1].passed == s_wrong.passed
    assert batch[2].compiled == s_broken.compiled

    try:
        v.check_dependencies()
        print("dependency check: OK")
    except SystemExit as e:
        print("dependency check failed:", e)
