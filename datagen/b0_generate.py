"""
b0: Full function JSON given directly.
Single MasterAgent with verify_code + ask_specialists.
Generates SFT pairs: (detailed_prompt_with_full_json, compiling_code)
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
from agents.prompts import b0_prompt
from agents.prompt_builder import load_raw_function, extract_function_name, build_b0_input
from agents.schemas import SFTPair
from agents.utils import extract_code


async def process_function(
    json_path: str,
    config: dict,
    specialist_tools: list,
    output_dir: str,
) -> SFTPair | None:
    fn = load_raw_function(json_path)
    fn_name = extract_function_name(fn)
    full_json, sft_prompt = build_b0_input(fn)

    agent = MasterAgent(
        system_prompt=b0_prompt(full_json),
        model=config["model"],
        base_url=config["base_url"],
        api_key=config["api_key"],
        temperature=0.4,
        max_turns=15,
        specialist_tools=specialist_tools,
        specialist_model=config.get("specialist_model", config["model"]),
        specialist_base_url=config.get("specialist_base_url", config["base_url"]),
        specialist_api_key=config.get("specialist_api_key", config["api_key"]),
        extra_tools=[verify_code],
        enable_summarization=False,
    )

    raw, _ = await agent.run(
        f"Write a complete C program that uses `{fn_name}` correctly. "
        f"Use verify_code to ensure it compiles cleanly. "
        f"Output ONLY the final compiling C code."
    )

    code_str = extract_code(str(raw))
    result = await compile_code(code_str)

    pair = SFTPair(
        function_name=fn_name,
        difficulty="b0",
        prompt=sft_prompt,
        code=code_str,
        compiled=result.success,
        compiler_output=result.compiler_output,
    )

    out_path = Path(output_dir) / f"b0_{fn_name}.json"
    out_path.write_text(pair.model_dump_json(indent=2))
    status = "✓ COMPILED" if result.success else "✗ FAILED"
    print(f"[b0] {fn_name}: {status}")
    return pair


async def main():
    parser = argparse.ArgumentParser(description="b0: Full function info given")
    parser.add_argument("--input-dir",       required=True, help="Folder with per-function JSON files")
    parser.add_argument("--output-dir",      required=True, help="Folder for SFT pair outputs")
    parser.add_argument("--config",          required=True, help="JSON file with LLM config")
    parser.add_argument("--mcp-tools-module",required=True, help="Python module path exposing tools list")
    parser.add_argument("--concurrency",     type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config) as f:
        config = json.load(f)

    tools_mod = importlib.import_module(args.mcp_tools_module)
    specialist_tools = tools_mod.tools

    json_files = sorted(Path(args.input_dir).glob("*.json"))
    print(f"[b0] Found {len(json_files)} functions")

    semaphore = asyncio.Semaphore(args.concurrency)

    async def bounded(path):
        async with semaphore:
            try:
                return await process_function(str(path), config, specialist_tools, args.output_dir)
            except Exception as e:
                print(f"[b0] ERROR {path.name}: {e}")
                return None

    results = await asyncio.gather(*[bounded(p) for p in json_files])

    compiled = sum(1 for r in results if r and r.compiled)
    failed   = sum(1 for r in results if r and not r.compiled)
    errors   = sum(1 for r in results if r is None)
    print(f"\n{'='*50}")
    print(f"b0 SUMMARY: {compiled} compiled | {failed} failed | {errors} errors")
    print(f"{'='*50}")


if __name__ == "__main__":
    asyncio.run(main())
