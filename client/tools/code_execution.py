from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from .base import BaseTool, ToolResult


class CodeExecutionTool(BaseTool):
    def __init__(
        self,
        language: str = "python",
        timeout: float = 10.0,
        max_output_chars: int = 2000,
        working_dir: Optional[str] = None,
    ) -> None:
        self.language = language
        self.timeout = float(timeout)
        self.max_output_chars = int(max_output_chars)
        self.working_dir = working_dir

    @property
    def name(self) -> str:
        return "execute_code"

    @property
    def description(self) -> str:
        return "Execute code and return stdout/stderr."

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "stdin": {"type": "string"},
            },
            "required": ["code"],
        }

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_output_chars:
            return text
        return text[: self.max_output_chars] + "\n...[truncated]"

    def _run_python(self, src: Path, stdin: str) -> ToolResult:
        try:
            cp = subprocess.run(
                ["python", str(src)],
                input=stdin,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.working_dir,
            )
            output = (cp.stdout or "") + (cp.stderr or "")
            success = cp.returncode == 0
            return ToolResult(
                success=success,
                output=self._truncate(output),
                metadata={"returncode": cp.returncode, "language": "python"},
            )
        except subprocess.TimeoutExpired:
            return ToolResult(False, f"Execution timed out after {self.timeout}s", {"language": "python"})
        except Exception as exc:
            return ToolResult(False, f"Execution failed: {exc}", {"language": "python"})

    def _run_c_family(self, src: Path, stdin: str, is_cpp: bool) -> ToolResult:
        exe = src.parent / "a.out"
        compiler = "g++" if is_cpp else "gcc"
        std_flag = "-std=c++17" if is_cpp else "-std=c11"

        try:
            cp = subprocess.run(
                [compiler, std_flag, "-O2", str(src), "-o", str(exe)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.working_dir,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(False, f"Compilation timed out after {self.timeout}s", {"language": self.language})
        except Exception as exc:
            return ToolResult(False, f"Compilation failed: {exc}", {"language": self.language})

        if cp.returncode != 0:
            output = (cp.stdout or "") + (cp.stderr or "")
            return ToolResult(
                success=False,
                output=self._truncate(output or "Compilation failed."),
                metadata={"returncode": cp.returncode, "language": self.language, "stage": "compile"},
            )

        try:
            rp = subprocess.run(
                [str(exe)],
                input=stdin,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.working_dir,
            )
            output = (rp.stdout or "") + (rp.stderr or "")
            return ToolResult(
                success=rp.returncode == 0,
                output=self._truncate(output),
                metadata={"returncode": rp.returncode, "language": self.language, "stage": "run"},
            )
        except subprocess.TimeoutExpired:
            return ToolResult(False, f"Execution timed out after {self.timeout}s", {"language": self.language})
        except Exception as exc:
            return ToolResult(False, f"Execution failed: {exc}", {"language": self.language})

    def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        code = arguments.get("code")
        stdin = arguments.get("stdin", "")

        if not isinstance(code, str) or not code.strip():
            return ToolResult(False, "Missing required non-empty 'code' string", {})
        if not isinstance(stdin, str):
            return ToolResult(False, "'stdin' must be a string", {})

        suffix = {
            "python": ".py",
            "c": ".c",
            "cpp": ".cpp",
        }.get(self.language)
        if suffix is None:
            return ToolResult(False, f"Unsupported language: {self.language}", {})

        try:
            with tempfile.TemporaryDirectory(prefix="client_exec_") as tmp:
                src = Path(tmp) / f"main{suffix}"
                src.write_text(code, encoding="utf-8")

                if self.language == "python":
                    return self._run_python(src, stdin)
                return self._run_c_family(src, stdin, is_cpp=(self.language == "cpp"))
        except Exception as exc:
            return ToolResult(False, f"Tool error: {exc}", {})
