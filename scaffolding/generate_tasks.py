"""Layer 0.5 — Task Designer.

Given a raw function entry (name, description, interacts_with) from parse_md,
asks the LLM to design N diverse, precise task specifications.

Each task includes:
  - A detailed IO spec (precise enough to serve as a future C-code prompt)
  - The process count decision (one vs two) with reasoning
  - Pattern and type metadata

The two_process flag is decided HERE, not in generate_code, because it is a
task design choice that depends on whether the behavior requires event-driven
inter-process communication — and that is determined by the task logic, not
by the implementation language.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are a SCADA test engineer writing precise IO task specifications. "
    "Output only valid JSON, no prose outside the JSON."
)

_PROMPT = """\
A SCADA C library function has this behavior:

  Name: {name}
  Description: {description}
  Interacts with: {interacts_with}

Design {n} distinct IO task specifications that could plausibly represent
different behavioral aspects of this function, tested through two 4-byte
sensor files.

--- IO rules every task must obey ---
- File 2001 and file 2002 are the ONLY IO channels.
- Each file holds exactly one value: either a signed 32-bit integer
  (-2147483648 to 2147483647) or a string of at most 4 characters.
- A file's type does not change within a task (int in → int out, str in → str out).
- At least one file MUST change value for typical inputs.

--- Task description requirements ---
Each description must be a precise, unambiguous specification:
  1. Which file(s) are read at the start, and what type they hold.
  2. The exact transformation: arithmetic (add/subtract/negate/abs/clamp),
     comparison (threshold/flag/sign check), string op (truncate/reverse/
     uppercase/append char), or cross-register (copy/swap/sum/diff).
  3. Which file(s) are written back, and exactly what value.
  4. Which file(s) are left unchanged (explicitly stated).
  5. Any boundary behaviour (overflow clamp, empty string, zero, negative).

These descriptions will later be matched against real SCADA library docs to
generate C code — they must be specific enough to act as a functional spec.

--- Two-process decision ---
A task needs TWO processes only when the behavior inherently requires one
process to BLOCK and WAIT for another to write data and signal it — i.e. a
listener/emitter pattern where sequential execution cannot model it.
For two-process tasks the description must specify:
  - What process 2 (emitter) writes and then signals.
  - What process 1 (listener) does after receiving the signal.

--- Pattern and type diversity ---
Across the {n} tasks, vary:
  - Pattern: arithmetic, comparison, string, cross_register, two_process
  - Primary type: some tasks with integers, some with strings
  - Scope: some touch only file 2001, some only 2002, some both

Output a JSON array of exactly {n} objects. Each object:
{{
  "description": "<precise functional spec — 2 to 4 sentences>",
  "pattern": "<arithmetic | comparison | string | cross_register | two_process>",
  "primary_type": "<'i' for integer tasks, 's' for string tasks>",
  "two_process": <true if listener/emitter pattern required, false otherwise>,
  "two_process_reason": "<one sentence: why two processes are/aren't needed>"
}}
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_JSON_RE = re.compile(r"\[.*?\]", re.DOTALL)


def _parse(raw: str, n: int) -> list[dict]:
    m = _JSON_RE.search(raw)
    if not m:
        raise ValueError(f"No JSON array in response:\n{raw[:400]}")
    tasks = json.loads(m.group())
    if not isinstance(tasks, list):
        raise ValueError("Parsed JSON is not a list")
    required = {"description", "pattern", "primary_type", "two_process", "two_process_reason"}
    for t in tasks:
        missing = required - t.keys()
        if missing:
            raise ValueError(f"Task missing keys {missing}: {t}")
        if not isinstance(t["two_process"], bool):
            raise ValueError(f"two_process must be bool, got: {t['two_process']}")
    return tasks[:n]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def generate_tasks(entry: dict, llm: ChatOpenAI, n: int = 5) -> list[dict]:
    """Return n task dicts for the given function entry.

    Each task dict contains:
        function_name, description, pattern, primary_type,
        two_process (bool), two_process_reason (str)
    """
    prompt = _PROMPT.format(
        name=entry["name"],
        description=entry["description"],
        interacts_with=", ".join(entry.get("interacts_with", [])) or "N/A",
        n=n,
    )
    messages = [SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)]

    for attempt in range(3):
        response = await llm.ainvoke(messages)
        raw = str(response.content)
        try:
            tasks = _parse(raw, n)
            for t in tasks:
                t["function_name"] = entry["name"]
            return tasks
        except (ValueError, json.JSONDecodeError) as exc:
            if attempt == 2:
                raise RuntimeError(
                    f"generate_tasks failed for {entry['name']} after 3 attempts: {exc}"
                ) from exc
            messages += [
                response,
                HumanMessage(
                    content=(
                        f"Invalid output: {exc}. "
                        f"Output ONLY a valid JSON array of {n} task objects "
                        f"with all required fields including two_process as a boolean."
                    )
                ),
            ]
    return []
