"""
b3: Only function name given. Agent must use ask_specialists for everything.
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
    build_b3_input,
    extract_function_name,
    load_raw_function,
)
from agents.prompts import b3_prompt
from agents.schemas import SFTPair
from agents.utils import extract_code
from agents.verify import compile_code, verify_code


async def process_function(
    json_path: str,
    config: dict,
    specialist_tools: list,
    output_dir: str,
) -> SFTPair | None:
    fn = load_raw_function(json_path)
    fn_name = extract_function_name(fn)
    _, sft_prompt = build_b3_input(fn)

    agent = MasterAgent(
        system_prompt=b3_prompt(fn_name),
        model=config["model"],
        base_url=config["base_url"],
        api_key=config["api_key"],
        temperature=0.5,
        max_turns=25,
        specialist_tools=specialist_tools,
        specialist_model=config.get("specialist_model", config["model"]),
        specialist_base_url=config.get("specialist_base_url", config["base_url"]),
        specialist_api_key=config.get("specialist_api_key", config["api_key"]),
        extra_tools=[verify_code],
        enable_summarization=True,
        summarize_token_limit=48000,
        summarize_target=16000,
    )

    raw, _ = await agent.run(
        f"Write a complete C program using `{fn_name}`. "
        f"You only have the function name — use ask_specialists first to get "
        f"the full signature, types, error codes, init and cleanup functions. "
        f"Then write and compile with verify_code. "
        f"Output ONLY the final compiling C code."
    )

    code_str = extract_code(str(raw))
    result = await compile_code(code_str)

    pair = SFTPair(
        function_name=fn_name,
        difficulty="b3",
        prompt=sft_prompt,
        code=code_str,
        compiled=result.success,
        compiler_output=result.compiler_output,
    )

    out_path = Path(output_dir) / f"b3_{fn_name}.json"
    out_path.write_text(pair.model_dump_json(indent=2))
    status = "✓ COMPILED" if result.success else "✗ FAILED"
    print(f"[b3] {fn_name}: {status}")
    return pair


async def main():
    parser = argparse.ArgumentParser(description="b3: Function name only")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mcp-tools-module", required=True)
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config) as f:
        config = json.load(f)

    tools_mod = importlib.import_module(args.mcp_tools_module)
    specialist_tools = tools_mod.tools

    json_files = sorted(Path(args.input_dir).glob("*.json"))
    print(f"[b3] Found {len(json_files)} functions")

    semaphore = asyncio.Semaphore(args.concurrency)

    async def bounded(path):
        async with semaphore:
            try:
                return await process_function(
                    str(path), config, specialist_tools, args.output_dir
                )
            except Exception as e:
                print(f"[b3] ERROR {path.name}: {e}")
                return None

    results = await asyncio.gather(*[bounded(p) for p in json_files])

    compiled = sum(1 for r in results if r and r.compiled)
    failed = sum(1 for r in results if r and not r.compiled)
    errors = sum(1 for r in results if r is None)
    print(f"\nb3 SUMMARY: {compiled} compiled | {failed} failed | {errors} errors")


if __name__ == "__main__":
    asyncio.run(main())
