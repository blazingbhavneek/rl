"""Layer 3 — Harness template generator (no LLM).

Always the same shape; only the gather vs single-run branch changes.
The asyncio.Event is created inside the async context to avoid
deprecation warnings on Python 3.10+.
"""

from __future__ import annotations

from pathlib import Path

_COMMON_HEADER = '''\
import asyncio
import importlib.util


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


p1 = _load("process1.py", "process1")
'''

_TWO_PROCESS_LOAD = 'p2 = _load("process2.py", "process2")\n'

_COMMON_IO = '''

def _write(state, type1, val1, type2, val2):
    state["2001"] = int(val1) if type1 == "i" else str(val1)[:4]
    state["2002"] = int(val2) if type2 == "i" else str(val2)[:4]


def _read(state, type1, type2):
    t1 = str(int(state["2001"])) if type1 == "i" else str(state["2001"])[:4]
    t2 = str(int(state["2002"])) if type2 == "i" else str(state["2002"])[:4]
    return f"{t1}\\n{t2}"

'''

_SINGLE_RUN = '''\
async def _run(state):
    await p1.main(state)


def run(type1, val1, type2, val2):
    state = {"2001": None, "2002": None, "_event": None}
    _write(state, type1, val1, type2, val2)
    asyncio.run(_run(state))
    return _read(state, type1, type2)
'''

_TWO_RUN = '''\
async def _run(state):
    state["_event"] = asyncio.Event()
    await asyncio.gather(p1.main(state), p2.main(state))


def run(type1, val1, type2, val2):
    state = {"2001": None, "2002": None}
    _write(state, type1, val1, type2, val2)
    asyncio.run(_run(state))
    return _read(state, type1, type2)
'''

_SMOKE_TEST = '''
if __name__ == "__main__":
    print(run("i", "42", "i", "7"))
'''


def generate_harness(task_dir: Path, two_process: bool) -> None:
    parts = [_COMMON_HEADER]
    if two_process:
        parts.append(_TWO_PROCESS_LOAD)
    parts.append(_COMMON_IO)
    parts.append(_TWO_RUN if two_process else _SINGLE_RUN)
    parts.append(_SMOKE_TEST)
    (task_dir / "harness.py").write_text("".join(parts))
