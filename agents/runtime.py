from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Imports to be provided by caller
# ---------------------------------------------------------------------------
# from your_module import compile_c_code   # LangChain tool: (code: str) -> JSON str {success, logs}
# from your_module import run_code         # async fn: (code, io_pairs, code2=None) -> RunCodeResult
# from your_module import rag_tool         # LangChain tool for SCADA library RAG

# Placeholder imports — replace with real ones
compile_c_code = None   # type: ignore
run_code = None         # type: ignore
rag_tool = None         # type: ignore

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "x")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-coder:32b")

MAX_CONCURRENT_LLM = 100
MAX_PHASE2_RETRIES = 3
MAX_COMPILE_RETRIES = 10
MAX_STAGE_A_RETRIES = 3
MAX_STAGE_B_RETRIES = 3
MASTER_AGENT_MAX_TURNS = 10
SPECIALIST_AGENT_MAX_TURNS = 5

INPUT_JSONL = Path(os.getenv("INPUT_JSONL", "data.jsonl"))
OUTPUT_JSONL = Path(os.getenv("OUTPUT_JSONL", "data_cleaned.jsonl"))
QUARANTINE_JSONL = Path(os.getenv("QUARANTINE_JSONL", "data_quarantine.jsonl"))

INT32_MIN = -2_147_483_648
INT32_MAX = 2_147_483_647

# ---------------------------------------------------------------------------
# Pydantic models — structured LLM output
# ---------------------------------------------------------------------------

class TestCase(BaseModel):
    inputs: list[str] = Field(description="[type1, val1, type2, val2]")
    expected_output: str = Field(description="Two tokens separated by newline")


class FixedTestCases(BaseModel):
    test_cases: list[TestCase]
    reason: str


class FixedProcessDescriptions(BaseModel):
    descriptions: list[str] = Field(description="One description per process, same order")
    reason: str


class FixedIO(BaseModel):
    test_cases: list[TestCase]
    reason: str


# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------

def _make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        model=LLM_MODEL,
    )


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _task_id(source_function: str, task_index: int) -> str:
    h = hashlib.sha256(source_function.encode()).hexdigest()[:8]
    return f"{h}_{task_index}"


def _load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if "_task_id" in row:
                    done.add(row["_task_id"])
            except json.JSONDecodeError:
                pass
    return done


def _append_jsonl(path: Path, row: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# Phase 1 — structural cleaning
# ---------------------------------------------------------------------------

def _validate_test_case(tc: dict) -> list[str]:
    flags = []
    inputs = tc.get("inputs", tc.get("input", []))

    if len(inputs) != 4:
        flags.append(f"input_length:{len(inputs)}")
        return flags  # can't check further without 4 fields

    type1, val1, type2, val2 = inputs

    for idx, (t, v) in enumerate([(type1, val1), (type2, val2)], start=1):
        if t not in ("i", "s"):
            flags.append(f"bad_type_{idx}:{t}")
        elif t == "s" and len(str(v)) > 4:
            flags.append(f"string_too_long_{idx}:{v}")
        elif t == "i":
            try:
                iv = int(v)
                if not (INT32_MIN <= iv <= INT32_MAX):
                    flags.append(f"int_out_of_range_{idx}:{v}")
            except (ValueError, TypeError):
                flags.append(f"int_not_parseable_{idx}:{v}")

    expected: str = tc.get("expected_output", "")
    tokens = expected.split("\n")
    if len(tokens) != 2:
        flags.append(f"output_token_count:{len(tokens)}")
    else:
        for idx, (t, tok) in enumerate([(type1, tokens[0]), (type2, tokens[1])], start=1):
            if t == "i":
                try:
                    int(tok.strip())
                except (ValueError, TypeError):
                    flags.append(f"output_type_mismatch_{idx}:expected_int_got:{tok!r}")
            elif t == "s" and len(tok.strip()) > 4:
                flags.append(f"output_string_too_long_{idx}:{tok}")

    # I == O check
    if not flags and len(tokens) == 2:
        if tokens[0].strip() == str(val1).strip() and tokens[1].strip() == str(val2).strip():
            flags.append("io_identical")

    return flags


def _normalise_tc(tc: dict) -> dict:
    """Normalise 'input' key to 'inputs' for consistency."""
    if "input" in tc and "inputs" not in tc:
        tc = dict(tc)
        tc["inputs"] = tc.pop("input")
    return tc


def structural_clean(task: dict) -> tuple[Optional[dict], dict[str, list[str]]]:
    """Return (cleaned_task, {tc_index: [flags]}) or (None, {}) if task dropped."""
    processes = task.get("processes", [])

    if len(processes) > 2:
        return None, {}

    orders = [p.get("execution_order") for p in processes]
    if len(set(orders)) != len(orders):
        return None, {}
    if set(orders) not in ({1}, {1, 2}):
        return None, {}

    # Normalise and deduplicate test cases
    seen = set()
    unique_tcs = []
    for tc in task.get("test_cases", []):
        tc = _normalise_tc(tc)
        key = json.dumps(tc, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique_tcs.append(tc)

    task = {**task, "test_cases": unique_tcs}

    flags: dict[str, list[str]] = {}
    for i, tc in enumerate(unique_tcs):
        f = _validate_test_case(tc)
        if f:
            flags[str(i)] = f

    return task, flags


# ---------------------------------------------------------------------------
# Phase 2 — LLM test case fix
# ---------------------------------------------------------------------------

async def llm_fix_test_cases(
    task: dict,
    flags: dict[str, list[str]],
    sem: asyncio.Semaphore,
) -> Optional[dict]:
    llm = _make_llm().with_structured_output(FixedTestCases)
    test_cases = list(task["test_cases"])

    for idx_str, reasons in flags.items():
        idx = int(idx_str)
        broken_tc = test_cases[idx]

        for attempt in range(MAX_PHASE2_RETRIES):
            prompt = (
                f"Source function:\n```c\n{task['source_function']}\n```\n\n"
                f"Task: {task['task']}\n\n"
                f"Processes:\n"
                + "\n".join(
                    f"  [{p['execution_order']}] {p['description']}"
                    for p in task["processes"]
                )
                + f"\n\nBroken test case:\n{json.dumps(broken_tc, indent=2)}\n\n"
                f"Issues: {', '.join(reasons)}\n\n"
                "Generate a replacement test case that:\n"
                "- Has inputs=[type1,val1,type2,val2] where types are 'i' or 's'\n"
                "- String values max 4 chars, integers within signed 32-bit range\n"
                "- expected_output has exactly 2 newline-separated tokens matching input types\n"
                "- Input != Output (transformation must be visible)\n"
                "- Reflects what the process actually does to files 2001 and 2002"
            )

            async with sem:
                try:
                    result: FixedTestCases = await llm.ainvoke([HumanMessage(content=prompt)])
                    if result.test_cases:
                        fixed_tc = result.test_cases[0].model_dump()
                        new_flags = _validate_test_case(fixed_tc)
                        if not new_flags:
                            test_cases[idx] = fixed_tc
                            break
                except Exception:
                    pass
        else:
            return None  # exhausted retries for this test case

    # Update process descriptions to reflect any changed test cases
    if test_cases != task["test_cases"]:
        desc_llm = _make_llm().with_structured_output(FixedProcessDescriptions)
        prompt = (
            f"Source function:\n```c\n{task['source_function']}\n```\n\n"
            f"Task: {task['task']}\n\n"
            f"Current process descriptions:\n"
            + "\n".join(
                f"  [{p['execution_order']}] {p['description']}"
                for p in task["processes"]
            )
            + f"\n\nUpdated test cases:\n{json.dumps(test_cases, indent=2)}\n\n"
            "Update process descriptions so they accurately describe what the code "
            "must do to produce these IO results. Keep one description per process."
        )
        async with sem:
            try:
                desc_result: FixedProcessDescriptions = await desc_llm.ainvoke(
                    [HumanMessage(content=prompt)]
                )
                processes = sorted(task["processes"], key=lambda p: p["execution_order"])
                for i, p in enumerate(processes):
                    if i < len(desc_result.descriptions):
                        p = dict(p)
                        p["description"] = desc_result.descriptions[i]
                        processes[i] = p
                task = {**task, "processes": processes}
            except Exception:
                pass

    return {**task, "test_cases": test_cases}


# ---------------------------------------------------------------------------
# Phase 3 — code generation agents
# ---------------------------------------------------------------------------

def _make_specialist_agent():
    llm = _make_llm()
    return create_react_agent(
        llm,
        tools=[rag_tool],
        state_modifier=SystemMessage(content=(
            "You are a SCADA library specialist. Answer questions about the library "
            "by querying the RAG tool. Be precise and code-focused. If your first "
            "query is insufficient, do follow-up queries."
        )),
    )


def _make_ask_specialist_tool(specialist_agent):
    from langchain_core.tools import tool as lc_tool

    @lc_tool
    async def ask_specialist(question: str) -> str:
        """Ask the SCADA library specialist a question. Use for library API details.

        Args:
            question: Specific question about the SCADA library API or usage.
        """
        result = await specialist_agent.ainvoke(
            {"messages": [HumanMessage(content=question)]},
            {"recursion_limit": SPECIALIST_AGENT_MAX_TURNS * 2},
        )
        return result["messages"][-1].content

    return ask_specialist


def _extract_c_code(text: str) -> Optional[str]:
    import re
    match = re.search(r"```c\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


async def _run_master_agent(
    process: dict,
    task: dict,
    sem: asyncio.Semaphore,
) -> Optional[str]:
    specialist_agent = _make_specialist_agent()
    ask_specialist = _make_ask_specialist_tool(specialist_agent)

    # Wrap compile tool with semaphore
    from langchain_core.tools import tool as lc_tool

    @lc_tool
    async def compile_with_sem(code: str) -> str:
        """Compile C code. Returns JSON with success and logs.

        Args:
            code: Complete C source code to compile.
        """
        async with sem:
            return await compile_c_code.ainvoke({"code": code})

    llm = _make_llm()
    agent = create_react_agent(
        llm,
        tools=[compile_with_sem, ask_specialist],
        state_modifier=SystemMessage(content=(
            "You are a C code generator for a SCADA library. "
            "Given a process description, write a complete C program with main(). "
            "Use compile_with_sem to check your code compiles. Fix errors until it compiles. "
            "Use ask_specialist for any SCADA library API questions. "
            "When done, output the final code in a ```c``` block."
        )),
    )

    prompt = (
        f"Source function:\n```c\n{task['source_function']}\n```\n\n"
        f"Task: {task['task']}\n\n"
        f"Process to implement (execution_order={process['execution_order']}):\n"
        f"{process['description']}\n\n"
        f"Test cases for reference:\n{json.dumps(task['test_cases'], indent=2)}\n\n"
        "Write a complete C program that implements this process. "
        "Compile and fix until it compiles successfully. "
        "Output final code in a ```c``` block."
    )

    async with sem:
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=prompt)]},
            {"recursion_limit": MASTER_AGENT_MAX_TURNS * 2},
        )

    final_message = result["messages"][-1].content
    return _extract_c_code(final_message)


async def _regenerate_codes(task: dict, sem: asyncio.Semaphore) -> Optional[list[str]]:
    processes = sorted(task["processes"], key=lambda p: p["execution_order"])
    codes = []
    for process in processes:
        code = await _run_master_agent(process, task, sem)
        if code is None:
            return None
        codes.append(code)
    return codes


async def _fix_descriptions(
    task: dict,
    runtime_result,
    sem: asyncio.Semaphore,
) -> Optional[dict]:
    llm = _make_llm().with_structured_output(FixedProcessDescriptions)
    actual_outputs = [
        f"test {i}: stdout={r.stdout!r} stderr={r.stderr!r}"
        for i, r in enumerate(runtime_result.test_results)
    ]
    prompt = (
        f"Source function:\n```c\n{task['source_function']}\n```\n\n"
        f"Task: {task['task']}\n\n"
        f"Current process descriptions:\n"
        + "\n".join(
            f"  [{p['execution_order']}] {p['description']}"
            for p in sorted(task["processes"], key=lambda p: p["execution_order"])
        )
        + f"\n\nTest cases:\n{json.dumps(task['test_cases'], indent=2)}\n\n"
        f"Runtime output (all failed):\n" + "\n".join(actual_outputs) + "\n\n"
        "Rewrite the process descriptions so that a C implementation would "
        "correctly produce the expected outputs. Keep one description per process."
    )
    async with sem:
        try:
            result: FixedProcessDescriptions = await llm.ainvoke([HumanMessage(content=prompt)])
            processes = sorted(task["processes"], key=lambda p: p["execution_order"])
            for i, p in enumerate(processes):
                if i < len(result.descriptions):
                    p = dict(p)
                    p["description"] = result.descriptions[i]
                    processes[i] = p
            return {**task, "processes": processes}
        except Exception:
            return None


async def _fix_io(
    task: dict,
    runtime_result,
    sem: asyncio.Semaphore,
) -> Optional[dict]:
    llm = _make_llm().with_structured_output(FixedIO)
    actual_outputs = [
        f"test {i}: stdout={r.stdout!r} stderr={r.stderr!r}"
        for i, r in enumerate(runtime_result.test_results)
    ]
    prompt = (
        f"Source function:\n```c\n{task['source_function']}\n```\n\n"
        f"Task: {task['task']}\n\n"
        f"Process descriptions:\n"
        + "\n".join(
            f"  [{p['execution_order']}] {p['description']}"
            for p in sorted(task["processes"], key=lambda p: p["execution_order"])
        )
        + f"\n\nCurrent test cases:\n{json.dumps(task['test_cases'], indent=2)}\n\n"
        f"Runtime output:\n" + "\n".join(actual_outputs) + "\n\n"
        "The process descriptions are correct. Rewrite the test cases so their "
        "expected_output matches what these processes actually do. "
        "Keep input != output, types consistent, values in range."
    )
    async with sem:
        try:
            result: FixedIO = await llm.ainvoke([HumanMessage(content=prompt)])
            new_tcs = [tc.model_dump() for tc in result.test_cases]
            # Validate new test cases
            for tc in new_tcs:
                if _validate_test_case(tc):
                    return None
            return {**task, "test_cases": new_tcs}
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Per-task pipeline
# ---------------------------------------------------------------------------

async def process_task(
    task: dict,
    sem: asyncio.Semaphore,
    output_path: Path,
    quarantine_path: Path,
) -> None:
    task_id = task["_task_id"]

    # Phase 1
    cleaned, flags = structural_clean(task)
    if cleaned is None:
        _append_jsonl(quarantine_path, {**task, "failure_reason": "structural_drop", "phase": 1})
        return
    task = cleaned

    # Phase 2
    if flags:
        fixed = await llm_fix_test_cases(task, flags, sem)
        if fixed is None:
            _append_jsonl(quarantine_path, {**task, "failure_reason": "phase2_exhausted", "phase": 2})
            return
        task = fixed

    # Phase 3 — generate code
    processes = sorted(task["processes"], key=lambda p: p["execution_order"])
    codes: list[str] = []
    for process in processes:
        code = await _run_master_agent(process, task, sem)
        if code is None:
            _append_jsonl(quarantine_path, {**task, "failure_reason": "compile_exhausted", "phase": 3})
            return
        codes.append(code)

    code1 = codes[0]
    code2 = codes[1] if len(codes) > 1 else None
    io_pairs = [{"inputs": tc["inputs"], "expected_output": tc["expected_output"]} for tc in task["test_cases"]]

    result = await run_code(code1, io_pairs, code2)

    if all(r.passed for r in result.test_results):
        _append_jsonl(output_path, {k: v for k, v in task.items() if k != "_task_id"})
        return

    # Stage A — fix descriptions
    for _ in range(MAX_STAGE_A_RETRIES):
        fixed_task = await _fix_descriptions(task, result, sem)
        if fixed_task is None:
            continue
        task = fixed_task
        codes = await _regenerate_codes(task, sem)
        if codes is None:
            continue
        code1, code2 = codes[0], (codes[1] if len(codes) > 1 else None)
        result = await run_code(code1, io_pairs, code2)
        if all(r.passed for r in result.test_results):
            _append_jsonl(output_path, {k: v for k, v in task.items() if k != "_task_id"})
            return

    # Stage B — fix IO
    for _ in range(MAX_STAGE_B_RETRIES):
        fixed_task = await _fix_io(task, result, sem)
        if fixed_task is None:
            continue
        task = fixed_task
        io_pairs = [{"inputs": tc["inputs"], "expected_output": tc["expected_output"]} for tc in task["test_cases"]]
        codes = await _regenerate_codes(task, sem)
        if codes is None:
            continue
        code1, code2 = codes[0], (codes[1] if len(codes) > 1 else None)
        result = await run_code(code1, io_pairs, code2)
        if all(r.passed for r in result.test_results):
            _append_jsonl(output_path, {k: v for k, v in task.items() if k != "_task_id"})
            return

    _append_jsonl(quarantine_path, {**task, "failure_reason": "runtime_exhausted", "phase": 3})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    done_ids = _load_done_ids(OUTPUT_JSONL) | _load_done_ids(QUARANTINE_JSONL)

    tasks: list[dict] = []
    with INPUT_JSONL.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("refused"):
                continue
            source_fn = row.get("source_function", "")
            for idx, task in enumerate(row.get("tasks", [])):
                tid = _task_id(source_fn, idx)
                if tid in done_ids:
                    continue
                tasks.append({
                    **task,
                    "source_function": source_fn,
                    "original_description": row.get("original_description", ""),
                    "_task_id": tid,
                })

    print(f"Tasks to process: {len(tasks)}")

    sem = asyncio.Semaphore(MAX_CONCURRENT_LLM)
    coros = [
        process_task(task, sem, OUTPUT_JSONL, QUARANTINE_JSONL)
        for task in tasks
    ]

    done = 0
    total = len(coros)
    for future in asyncio.as_completed(coros):
        await future
        done += 1
        if done % 10 == 0:
            print(f"  {done}/{total}")

    print(f"Done. Cleaned: {OUTPUT_JSONL}, Quarantine: {QUARANTINE_JSONL}")


if __name__ == "__main__":
    asyncio.run(main())
