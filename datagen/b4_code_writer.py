"""
b4 Stage 2: Code Writer master agent.
Reads task spec files produced by b4_task_designer.py.
For each task variant:
  - Writes code following the spec
  - Uses ask_specialists for any gaps
  - Uses verify_code to compile and self-correct
  - Produces TWO SFT pairs: (detailed_prompt, code) and (vague_prompt, code)
"""
from __future__ import annotations
import asyncio
import argparse
import importlib
import json
import os
from pathlib import Path

from agents.master_agent import MasterAgent
from agents.verify import verify_code, compile_code
from agents.prompts import CODE_WRITER_PROMPT
from agents.schemas import SFTPair, MultiTaskOutput, TaskDesignerOutput
from agents.utils import extract_code


def load_task_file(path: str) -> MultiTaskOutput:
    with open(path) as f:
        return MultiTaskOutput.model_validate(json.load(f))


async def process_task(
    function_name: str,
    task_idx: int,
    task: TaskDesignerOutput,
    config: dict,
    specialist_tools: list,
    output_dir: str,
) -> list[SFTPair]:
    spec = task.spec
    detailed_prompt = task.detailed_prompt
    vague_prompt    = task.vague_prompt

    agent = MasterAgent(
        system_prompt=CODE_WRITER_PROMPT,
        model=config["model"],
        base_url=config["base_url"],
        api_key=config["api_key"],
        temperature=0.3,
        max_turns=28,
        specialist_tools=specialist_tools,
        specialist_model=config.get("specialist_model", config["model"]),
        specialist_base_url=config.get("specialist_base_url", config["base_url"]),
        specialist_api_key=config.get("specialist_api_key", config["api_key"]),
        extra_tools=[verify_code],
        enable_summarization=True,
        summarize_token_limit=40000,
        summarize_target=14000,
    )

    # Build writer prompt from spec — code writer sees the full structured spec
    steps_block = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(spec.detailed_steps))
    pre_block   = "\n".join(f"  - {p}" for p in spec.preconditions)   or "  (none)"
    clean_block = "\n".join(f"  - {c}" for c in spec.cleanup_steps)   or "  (none)"
    req_block   = ", ".join(spec.required_functions)                   or "(see steps)"

    writer_prompt = f"""\
TASK: {spec.task_description}
TARGET FUNCTION: {spec.target_function}

REQUIRED FUNCTIONS: {req_block}

PRECONDITIONS:
{pre_block}

STEP-BY-STEP IMPLEMENTATION:
{steps_block}

CLEANUP (ALL PATHS):
{clean_block}

EXPECTED BEHAVIOR: {spec.expected_behavior}

IO PLACEHOLDER (include verbatim as a comment block at end of main):
{spec.io_placeholder}

TEST CASE SKELETON (include verbatim as a comment block after the IO placeholder):
{spec.test_case_commented}

INSTRUCTIONS:
- If any function signature, type, flag value, or include file is unclear,
  use ask_specialists BEFORE writing code.
- Write the full program, then call verify_code.
- Fix all errors. If a fix requires library details, use ask_specialists.
- Max 5 verify_code attempts.
- Output ONLY the final compiling C code.
"""

    raw, _ = await agent.run(writer_prompt)
    code_str = extract_code(str(raw))

    # Final authoritative compile check
    result = await compile_code(code_str)
    status = "✓ COMPILED" if result.success else "✗ FAILED"
    print(f"[b4-code] {function_name} v{task_idx}: {status}")

    tag = f"b4_{function_name}_v{task_idx}"

    # Pair 1: detailed prompt → code
    detailed_pair = SFTPair(
        function_name=function_name,
        difficulty="b4",
        prompt=detailed_prompt,
        code=code_str,
        compiled=result.success,
        compiler_output=result.compiler_output,
        vague_prompt=None,
    )

    # Pair 2: vague prompt → same code
    vague_pair = SFTPair(
        function_name=function_name,
        difficulty="b4_vague",
        prompt=vague_prompt,
        code=code_str,
        compiled=result.success,
        compiler_output=result.compiler_output,
        vague_prompt=vague_prompt,
    )

    (Path(output_dir) / f"{tag}_detailed.json").write_text(detailed_pair.model_dump_json(indent=2))
    (Path(output_dir) / f"{tag}_vague.json").write_text(vague_pair.model_dump_json(indent=2))

    return [detailed_pair, vague_pair]


async def main():
    parser = argparse.ArgumentParser(description="b4: Code Writer")
    parser.add_argument("--tasks-dir",        required=True, help="Folder with b4_tasks_*.json from designer")
    parser.add_argument("--output-dir",       required=True, help="Folder for SFT pair outputs")
    parser.add_argument("--config",           required=True)
    parser.add_argument("--mcp-tools-module", required=True)
    parser.add_argument("--concurrency",      type=int, default=3,
                        help="b4 code writers are heavy — keep low")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config) as f:
        config = json.load(f)

    tools_mod = importlib.import_module(args.mcp_tools_module)
    specialist_tools = tools_mod.tools

    task_files = sorted(
        p for p in Path(args.tasks_dir).glob("b4_tasks_*.json")
        if "_RAW" not in p.name
    )
    print(f"[b4-code] Found {len(task_files)} task design files")

    # Flatten to (fn_name, idx, task) tuples
    jobs = []
    for tf in task_files:
        try:
            task_input = load_task_file(str(tf))
            for idx, task in enumerate(task_input.tasks):
                jobs.append((task_input.function_name, idx, task))
        except Exception as e:
            print(f"[b4-code] SKIP {tf.name}: {e}")

    print(f"[b4-code] {len(jobs)} total tasks to write code for "
          f"({len(jobs) * 2} SFT pairs expected)")

    semaphore = asyncio.Semaphore(args.concurrency)

    async def bounded(fn_name, idx, task):
        async with semaphore:
            try:
                return await process_task(
                    fn_name, idx, task, config, specialist_tools, args.output_dir
                )
            except Exception as e:
                print(f"[b4-code] ERROR {fn_name} v{idx}: {e}")
                return []

    results = await asyncio.gather(*[bounded(fn, i, t) for fn, i, t in jobs])

    all_pairs = [p for batch in results for p in batch]
    compiled  = sum(1 for p in all_pairs if p.compiled)
    total     = len(all_pairs)
    print(f"\n{'='*50}")
    print(f"b4-code SUMMARY: {compiled}/{total} pairs compiled")
    print(f"{'='*50}")


if __name__ == "__main__":
    asyncio.run(main())
