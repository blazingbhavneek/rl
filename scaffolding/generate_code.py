"""Layer 2 — LLM code generator.

Reads task["two_process"] (decided in generate_tasks) and generates
process1.py (always) and process2.py (two-process tasks only).

No classification happens here — that is a task design decision made in Layer 0.5.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_CODEGEN_SYSTEM = (
    "You are generating Python async functions for a SCADA IO harness. "
    "Output ONLY code inside a ```python block. No explanation."
)

_IO_CONTRACT = """\
IO contract — enforced strictly by the harness:
- state["2001"] and state["2002"] are the ONLY IO. No files, no network, nothing else.
- Each value is either an int (signed 32-bit) or a str (≤4 chars). Never mixed in one slot.
- Read BOTH values at the very start of main().
- Before returning, assign the final value of each register back to state["2001"] and \
state["2002"] — even if one is unchanged, re-assign it so the harness can read it.
- TYPE MUST BE PRESERVED: if a value came in as int, write an int back. If str, write a str.
  Use isinstance(v, int) to branch on type.
- INT RANGE: clamp any int result to [-2147483648, 2147483647].
- STR LENGTH: any str result must be at most 4 chars (truncate if needed).
- TRANSFORMATION REQUIRED: at least one of state["2001"] or state["2002"] must differ \
from its input value for typical inputs.

Use only Python asyncio and stdlib — no external packages.
No SCADA naming, no fake APIs — just the plain logic. Keep it short.\
"""

_SINGLE_PROMPT = """\
Task specification:
  {description}

Implement this as:

    async def main(state: dict) -> None

{io_contract}

Output ONLY the function inside a ```python block.
"""

_P1_PROMPT = """\
Task specification (TWO-PROCESS — this is PROCESS 1, the listener/responder):
  {description}

Implement PROCESS 1:

    async def main(state: dict) -> None

- Call await state["_event"].wait() before reading or transforming anything.
- After the event fires, read state, apply the transformation, write results back.

{io_contract}

Output ONLY the function inside a ```python block.
"""

_P2_PROMPT = """\
Task specification (TWO-PROCESS — this is PROCESS 2, the emitter/trigger):
  {description}

Implement PROCESS 2:

    async def main(state: dict) -> None

- Write output data to state["2001"] and/or state["2002"].
- Then call state["_event"].set(). Do NOT await it.

{io_contract}

Output ONLY the function inside a ```python block.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_python(text: str) -> str:
    m = re.search(r"```python\s*\n(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*\n(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    if "async def main" in text:
        return text.strip()
    return text.strip()


def _valid_syntax(code: str) -> bool:
    try:
        compile(code, "<string>", "exec")
        return True
    except SyntaxError:
        return False


async def _call(llm: ChatOpenAI, prompt: str) -> str:
    msgs = [SystemMessage(content=_CODEGEN_SYSTEM), HumanMessage(content=prompt)]
    return str((await llm.ainvoke(msgs)).content)


async def _gen_process(llm: ChatOpenAI, prompt: str) -> str:
    raw = await _call(llm, prompt)
    code = _extract_python(raw)
    if not _valid_syntax(code):
        raw = await _call(
            llm,
            prompt + "\n\nYour previous output had a syntax error. Output ONLY valid Python.",
        )
        code = _extract_python(raw)
    return code


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def generate_code(task: dict, config: dict, task_dir: Path) -> None:
    """Generate process files for a task dict.

    Reads task["two_process"] — set by generate_tasks, not decided here.
    Writes process1.py (always) and process2.py (two-process tasks only).
    """
    llm = ChatOpenAI(
        model=config["model"],
        api_key=config["api_key"],
        base_url=config["base_url"],
        temperature=0.2,
        max_tokens=config.get("max_tokens", 1024),
    )

    desc = task["description"]
    two_process = task["two_process"]

    if two_process:
        p1, p2 = await asyncio.gather(
            _gen_process(llm, _P1_PROMPT.format(description=desc, io_contract=_IO_CONTRACT)),
            _gen_process(llm, _P2_PROMPT.format(description=desc, io_contract=_IO_CONTRACT)),
        )
        (task_dir / "process1.py").write_text(p1 + "\n")
        (task_dir / "process2.py").write_text(p2 + "\n")
    else:
        code = await _gen_process(
            llm, _SINGLE_PROMPT.format(description=desc, io_contract=_IO_CONTRACT)
        )
        (task_dir / "process1.py").write_text(code + "\n")
