from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

from langchain_core.tools import tool

from .schemas import VerifyResult

DOCKER_IMAGE = os.getenv("COMPILE_DOCKER_IMAGE", "your-library-image")
DOCKER_CONTAINER = os.getenv("COMPILE_DOCKER_CONTAINER", "")
INCLUDE_PATH = os.getenv("LIBRARY_INCLUDE_PATH", "/usr/local/include")
LIB_PATH = os.getenv("LIBRARY_LIB_PATH", "/usr/local/lib")
LIB_NAME = os.getenv("LIBRARY_LIB_NAME", "yourlib")


async def compile_code(code: str, timeout: int = 30) -> VerifyResult:
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "test.c"
        src_path.write_text(code)

        if DOCKER_CONTAINER:
            cp_cmd = f"docker cp {src_path} {DOCKER_CONTAINER}:/tmp/test.c"
            await _run(cp_cmd, timeout=10)
            compile_cmd = (
                f"docker exec {DOCKER_CONTAINER} "
                f"gcc -Wall -Wextra -I{INCLUDE_PATH} -L{LIB_PATH} "
                f"/tmp/test.c -l{LIB_NAME} -o /tmp/test_out 2>&1"
            )
        else:
            compile_cmd = (
                f"docker run --rm -v {tmpdir}:/src {DOCKER_IMAGE} "
                f"gcc -Wall -Wextra -I{INCLUDE_PATH} -L{LIB_PATH} "
                f"/src/test.c -l{LIB_NAME} -o /src/test_out 2>&1"
            )

        returncode, output = await _run(compile_cmd, timeout=timeout)

        warnings = [l for l in output.splitlines() if "warning:" in l.lower()]
        errors = [l for l in output.splitlines() if "error:" in l.lower()]

        return VerifyResult(
            success=(returncode == 0),
            compiler_output=output,
            warnings=warnings,
            errors=errors,
        )


async def _run(cmd: str, timeout: int = 30) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return proc.returncode or 0, stdout.decode(errors="replace")
    except asyncio.TimeoutError:
        proc.kill()
        return -1, "TIMEOUT: compilation exceeded time limit"


@tool
async def verify_code(code: str) -> str:
    """Compile C code in Docker. Returns JSON with success flag and compiler output.

    Args:
        code: Complete C source code to compile.
    """
    result = await compile_code(code)
    return result.model_dump_json()
