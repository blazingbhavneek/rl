"""
b4 Stage 1: Task Designer master agent.
Reads raw function JSON, spawns specialists to deeply understand the function
and its ecosystem, then designs N distinct task variants.

Each variant has:
  - A full TaskSpec (step-by-step instructions for the code writer)
  - A detailed prompt (for SFT pair 1)
  - A vague prompt (for SFT pair 2)

Output files are consumed by b4_code_writer.py
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
from pathlib import Path

from agents.master_agent import MasterAgent
from agents.prompt_builder import (
    extract_function_name,
    load_raw_function,
    raw_to_string,
)
from agents.prompts import TASK_DESIGNER_PROMPT
from agents.schemas import MultiTaskOutput

VARIANT_INSTRUCTION = """\
Design {n} DIFFERENT coding tasks that use `{fn_name}` as the target function.

Each task must be meaningfully distinct:
  - Variant 0: basic happy-path usage
  - Variant 1: focuses on error handling (wrong params, boundary conditions)
  - Variant 2+: combines target function with different related functions,
    or exercises a less common but valid use case

For EACH variant produce:
1. spec — a TaskSpec with ALL fields filled in detail:
   - detailed_steps: specific enough that someone with zero library knowledge
     can write the code (exact param values, exact flag names, exact error checks)
   - cleanup_steps: must cover ALL paths including early error exits
   - test_case_commented: a commented-out skeleton that opens two files
     (input.bin, output.bin), writes something via the function, reads back,
     asserts correctness — placeholder for future IO harness
2. detailed_prompt — a natural developer instruction that includes all info:
   function names, types, error codes, init/cleanup steps, expected behavior
3. vague_prompt — only the goal, no specific function names (except the target),
   no types, no error handling details; e.g. "write a program that opens a
   session and records some data, then closes cleanly"

Output valid JSON matching this schema:
{schema}
"""


async def process_function(
    json_path: str,
    config: dict,
    specialist_tools: list,
    output_dir: str,
    tasks_per_function: int = 3,
) -> MultiTaskOutput | None:
    fn = load_raw_function(json_path)
    fn_name = extract_function_name(fn)
    full_json = raw_to_string(fn)

    agent = MasterAgent(
        system_prompt=TASK_DESIGNER_PROMPT,
        model=config["model"],
        base_url=config["base_url"],
        api_key=config["api_key"],
        temperature=0.7,
        max_turns=35,
        specialist_tools=specialist_tools,
        specialist_model=config.get("specialist_model", config["model"]),
        specialist_base_url=config.get("specialist_base_url", config["base_url"]),
        specialist_api_key=config.get("specialist_api_key", config["api_key"]),
        extra_tools=[],  # task designer does not compile
        enable_summarization=True,
        summarize_token_limit=48000,
        summarize_target=16000,
    )

    schema_str = json.dumps(MultiTaskOutput.model_json_schema(), indent=2)
    variant_block = VARIANT_INSTRUCTION.format(
        n=tasks_per_function,
        fn_name=fn_name,
        schema=schema_str,
    )

    prompt = (
        f"Here is EVERYTHING we know about the target function from our documentation:\n\n"
        f"{full_json}\n\n"
        f"This is a raw JSON dump — trust all fields present.\n"
        f"If you need info about RELATED functions, deeper theory, call ordering, "
        f"or anything not covered above, use ask_specialists.\n\n"
        f"Do NOT design any tasks until you have asked specialists about:\n"
        f"  1. All functions that must be called before {fn_name} (init chain)\n"
        f"  2. All functions that must be called after (cleanup chain, all paths)\n"
        f"  3. Every related function you might use in a task variant\n"
        f"  4. All error codes with their exact trigger conditions\n"
        f"  5. Any threading, timing, or ordering constraints\n\n"
        f"{variant_block}"
    )

    raw, result = await agent.run(prompt, output_model=MultiTaskOutput)

    if isinstance(result, MultiTaskOutput):
        result.function_name = fn_name
        out_path = Path(output_dir) / f"b4_tasks_{fn_name}.json"
        out_path.write_text(result.model_dump_json(indent=2))
        print(f"[b4-design] {fn_name}: {len(result.tasks)} tasks designed")
        return result
    else:
        # Save raw for debugging
        out_path = Path(output_dir) / f"b4_tasks_{fn_name}_RAW.txt"
        out_path.write_text(str(raw))
        print(f"[b4-design] {fn_name}: FAILED — raw output saved")
        return None


async def main():
    parser = argparse.ArgumentParser(description="b4: Task Designer")
    parser.add_argument(
        "--input-dir", required=True, help="Folder with per-function JSON files"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Folder for task spec outputs"
    )
    parser.add_argument("--config", required=True, help="JSON file with LLM config")
    parser.add_argument("--mcp-tools-module", required=True)
    parser.add_argument("--tasks-per-function", type=int, default=3)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="b4 designers are heavy — keep this low",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config) as f:
        config = json.load(f)

    tools_mod = importlib.import_module(args.mcp_tools_module)
    specialist_tools = tools_mod.tools

    json_files = sorted(Path(args.input_dir).glob("*.json"))
    print(
        f"[b4-design] Found {len(json_files)} functions, "
        f"{args.tasks_per_function} tasks each = "
        f"{len(json_files) * args.tasks_per_function} total tasks"
    )

    semaphore = asyncio.Semaphore(args.concurrency)

    async def bounded(path):
        async with semaphore:
            try:
                return await process_function(
                    str(path),
                    config,
                    specialist_tools,
                    args.output_dir,
                    args.tasks_per_function,
                )
            except Exception as e:
                print(f"[b4-design] ERROR {path.name}: {e}")
                return None

    results = await asyncio.gather(*[bounded(p) for p in json_files])

    designed = sum(1 for r in results if isinstance(r, MultiTaskOutput))
    total_tasks = sum(len(r.tasks) for r in results if isinstance(r, MultiTaskOutput))
    errors = sum(1 for r in results if r is None)
    print(f"\n{'='*50}")
    print(
        f"b4-design SUMMARY: {designed} functions | {total_tasks} tasks | {errors} errors"
    )
    print(f"{'='*50}")


if __name__ == "__main__":
    asyncio.run(main())
