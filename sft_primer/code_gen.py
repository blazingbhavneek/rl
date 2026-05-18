"""
codegen_fast.py — generate C code from a task JSONL using a simple LLM + fix agent loop.

Usage:
    python codegen_fast.py \
        --input tasks.jsonl --output results.jsonl \
        --config config.json --functions-dir /path/to/function_json_docs
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Optional

import requests
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from tree_sitter import Parser, Language
from tree_sitter_c import language as c_language


# -- prompts ------------------------------------------------------------------

def _gen_system(all_functions: set[str]) -> str:
    fn_list = ", ".join(sorted(all_functions))
    return (
        "You are an expert C developer. Write complete, compilable C code for the given task.
"
        "- NEVER define, stub, reimplement, or forward/extern declare any library function. Only #include and call.
"
        f"- The following are all known library functions — do not implement or declare any of them: {fn_list}
"
        "- Use C99/C11. No C++ comments (//).
"
        "- Output ONLY the full code wrapped in ```c\n...\n```. No placeholders or ellipsis."
    )


def _fix_system(all_functions: set[str]) -> str:
    fn_list = ", ".join(sorted(all_functions))
    return (
        "You are an expert C compiler error fixer.
"
        "Use the provided tools (RAG, MCP) to look up signatures, headers, or usage if needed. At most 2 tool calls.
"
        "Then output ONLY a JSON array of minimal edits:
"
        '[{"old": "<exact verbatim substring>", "new": "<replacement>"}]
'
        "- 'old' must match the code character-for-character.
"
        "- Smallest change that fixes the error. Don't rewrite whole functions.
"
        "- Missing #include? Add it as an edit at the top of the file.
"
        f"- The following are all known library functions — never redefine, stub, or declare any of them: {fn_list}
"
        "- Raw JSON array only. No explanation. No markdown fences."
    )


# -- config & docs ------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_function_docs(functions_dir: Path) -> dict[str, str]:
    docs = {}
    for file in functions_dir.glob("*.json"):
        try:
            docs[file.stem] = file.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    return docs


def make_llm(config: dict, temperature: float = 0.1) -> ChatOpenAI:
    return ChatOpenAI(
        model=config["model"],
        base_url=config["base_url"],
        api_key=config["api_key"],
        temperature=temperature,
        max_tokens=config.get("max_tokens", 2048),
    )


# -- helpers ------------------------------------------------------------------

def extract_code(text: str) -> str:
    m = re.search(r"```(?:c|cpp)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else text.strip()


def preflight(code: str) -> tuple[bool, str]:
    lines = [l for l in code.strip().splitlines() if l.strip()]
    if len(lines) < 8:      return False, f"too short ({len(lines)} lines)"
    if "#include" not in code: return False, "no #include"
    if "{" not in code:     return False, "no function body"
    return True, ""


def apply_edits(code: str, edits: list[dict]) -> tuple[str, list[str]]:
    result, skipped = code, []
    for i, edit in enumerate(edits):
        old, new = edit.get("old", ""), edit.get("new", "")
        if not old:
            skipped.append(f"edit {i}: empty old"); continue
        if old not in result:
            skipped.append(f"edit {i}: not found: {old[:50]!r}"); continue
        result = result.replace(old, new, 1)
    return result, skipped


def extract_mentioned(task: str, all_functions: set[str]) -> set[str]:
    if not task or not all_functions:
        return set()
    pattern = r'\b(' + '|'.join(map(re.escape, all_functions)) + r')\b'
    return set(re.findall(pattern, task))


# -- LLM calls ----------------------------------------------------------------

def _call_llm_sync(prompt: str, config: dict, system: str) -> str:
    messages = ([{"role": "system", "content": system}] if system else [])
    messages += [{"role": "user", "content": prompt}]
    r = requests.post(
        f"{config['base_url']}/chat/completions",
        headers={"Authorization": f"Bearer {config['api_key']}"},
        json={"model": config["model"], "temperature": config.get("temperature", 0.3),
              "max_tokens": config.get("max_tokens", 4096), "messages": messages},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


async def call_llm(prompt: str, config: dict, system: str = "") -> str:
    return await asyncio.to_thread(_call_llm_sync, prompt, config, system)


# -- semantic analyzer --------------------------------------------------------

class CCodeAnalyzer:
    def __init__(self):
        self.parser = Parser(Language(c_language()))

    def analyze(self, src: str) -> dict[str, set[str]]:
        try:
            tree = self.parser.parse(src.encode("utf8"))
        except Exception:
            return {"called": set(), "defined": set(), "declared": set()}

        called, defined, declared = set(), set(), set()

        def visit(node):
            if node.type == "call_expression":
                func = node.child_by_field_name("function")
                if func and (name := self._get_id(func)):
                    called.add(name)
            elif node.type == "function_definition":
                decl = node.child_by_field_name("declarator")
                if decl and (name := self._get_decl_name(decl)):
                    defined.add(name)
            elif node.type == "declaration":
                decl = node.child_by_field_name("declarator")
                if decl and decl.type in ("function_declarator", "parenthesized_declarator"):
                    if (name := self._get_decl_name(decl)) and not self._in_func_body(node):
                        declared.add(name)
            for c in node.children:
                visit(c)

        visit(tree.root_node)
        return {"called": called, "defined": defined, "declared": declared}

    def _get_id(self, n) -> str | None:
        if n.type == "identifier":
            return n.text.decode("utf8")
        if n.type == "field_expression":
            attr = n.child_by_field_name("field")
            return attr.text.decode("utf8") if attr else None
        return None

    def _get_decl_name(self, n) -> str | None:
        while n.type == "parenthesized_declarator":
            n = n.child_by_field_name("declarator")
            if not n:
                return None
        if n.type == "function_declarator":
            d = n.child_by_field_name("declarator")
            return d.text.decode("utf8") if d and d.type == "identifier" else None
        return n.text.decode("utf8") if n.type == "identifier" else None

    def _in_func_body(self, node) -> bool:
        p = node.parent
        while p:
            if p.type == "function_definition":
                b = p.child_by_field_name("body")
                if b and b.start_byte <= node.start_byte < b.end_byte:
                    return True
            p = p.parent
        return False

_analyzer = CCodeAnalyzer()


# -- verify -------------------------------------------------------------------

def _verify_sync(code: str, target_fn: str, server_url: str, apl_variant: int) -> dict:
    try:
        r = requests.post(f"{server_url}/compile",
                          json={"code": code, "apl_variant": apl_variant}, timeout=60)
    except Exception as e:
        return {"compiled": False, "logs": str(e), "error_count": 99, "issues": []}

    if r.status_code != 200:
        return {"compiled": False, "logs": r.text, "error_count": 99, "issues": []}

    data = r.json()
    compiled = bool(data.get("compiled"))
    logs = (data.get("logs") or "").strip()

    issues = []
    if target_fn:
        analysis = _analyzer.analyze(code)
        called, defined, declared = analysis["called"], analysis["defined"], analysis["declared"]
        if target_fn in defined:
            issues.append(f"CHEATING: '{target_fn}' is locally defined — remove it and #include the proper header")
        elif target_fn in declared:
            issues.append(f"CHEATING: '{target_fn}' is forward-declared — remove the declaration and #include the proper header")
        elif target_fn not in called:
            issues.append(f"MISSING: '{target_fn}' required by task but never called")

    return {"compiled": compiled, "logs": logs,
            "error_count": logs.lower().count("error:"), "issues": issues}


async def verify(code: str, target_fn: str, server_url: str, apl_variant: int) -> dict:
    return await asyncio.to_thread(_verify_sync, code, target_fn, server_url, apl_variant)


# -- fix agent ----------------------------------------------------------------

class FixAgent:
    def __init__(self, config: dict, tools: list, all_functions: set[str]):
        self.llm = make_llm(config, temperature=0.1)
        self.tools = tools
        self.all_functions = all_functions

    async def get_edits(self, code: str, error: str, task: str) -> list[dict]:
        agent = create_react_agent(model=self.llm, tools=self.tools, prompt=_fix_system(self.all_functions))
        prompt = (f"Task (reference only):\n{task}\n\n"
                  f"Code:\n```c\n{code}\n```\n\nError:\n```\n{error}\n```\n\n"
                  "Output the JSON edit array now.")
        try:
            result = await agent.ainvoke({"messages": [{"role": "user", "content": prompt}]})
            raw = result["messages"][-1].content.strip()
        except Exception as e:
            print(f"    [fix_agent] error: {e}")
            return []

        raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw.strip())
        try:
            edits = json.loads(raw)
            return edits if isinstance(edits, list) else []
        except json.JSONDecodeError as e:
            print(f"    [fix_agent] JSON parse error: {e} | got: {raw[:120]!r}")
            return []


# -- core loop ----------------------------------------------------------------

async def generate_for_task(
    row: dict,
    all_functions: set[str],
    function_docs: dict[str, str],
    llm_config: dict,
    fix_agent: FixAgent,
    verify_server: str,
    apl_variant: int,
    max_gen_attempts: int,
    max_fix_attempts: int,
) -> dict:
    fn = row["target_func"]
    task = row["task"]
    p = lambda msg: print(f"  [{fn}] {msg}")

    mentioned = extract_mentioned(task, all_functions) | {fn}
    doc_block = "\n\n".join(function_docs[f] for f in sorted(mentioned) if f in function_docs)
    gen_prompt = (f"Task:\n{task}\n\n"
                  f"LIBRARY DEFINITIONS (JSON):\n{doc_block}\n\n"
                  "Write the complete C code. Format: ```c\\n<code>\\n```")

    best_compiled: Optional[str] = None

    for gen in range(1, max_gen_attempts + 1):
        p(f"gen {gen}/{max_gen_attempts}")
        try:
            code = extract_code(await call_llm(gen_prompt, llm_config, system=_gen_system(all_functions)))
        except Exception as e:
            p(f"LLM error: {e}"); continue

        ok, reason = preflight(code)
        if not ok:
            p(f"preflight failed: {reason}"); continue

        current = code

        for fix in range(1, max_fix_attempts + 1):
            result = await verify(current, fn, verify_server, apl_variant)
            p(f"fix {fix}/{max_fix_attempts} | compiled={result['compiled']} "
              f"errors={result['error_count']} issues={len(result['issues'])}")

            if result["compiled"] and best_compiled is None:
                best_compiled = current

            if result["compiled"] and not result["issues"]:
                p(f"✔ done (gen={gen} fix={fix})")
                return {**row, "code": current, "failed": False, "task_desc_change": None}

            error_str = result["logs"]
            if result["issues"]:
                error_str += "\n\nSemantic issues:\n" + "\n".join(f"- {i}" for i in result["issues"])

            edits = await fix_agent.get_edits(current, error_str, task)
            if not edits:
                p("no edits returned"); continue

            new_code, skipped = apply_edits(current, edits)
            if skipped:
                p(f"{len(skipped)} edit(s) skipped: {skipped[0]}")
            if new_code == current:
                p("patch changed nothing"); continue

            new_result = await verify(new_code, fn, verify_server, apl_variant)
            if new_result["error_count"] > result["error_count"]:
                p(f"patch made things worse ({result['error_count']} → {new_result['error_count']}), reverting")
                continue

            current = new_code

    p("✗ all attempts exhausted")
    return {**row, "code": best_compiled or "", "failed": True,
            "task_desc_change": f"failed after {max_gen_attempts}×{max_fix_attempts} attempts"}


# -- orchestrator -------------------------------------------------------------

async def run(
    input_jsonl: Path,
    output_jsonl: Path,
    llm_config: dict,
    functions_dir: Path,
    verify_server: str,
    apl_variant: int,
    concurrency: int,
    specialist_tools: list,
    max_gen_attempts: int,
    max_fix_attempts: int,
):
    rows = []
    with open(input_jsonl) as f:
        for line in f:
            if line.strip():
                try:
                    row = json.loads(line)
                    row["code"] = None
                    rows.append(row)
                except json.JSONDecodeError:
                    pass

    print(f"[codegen] {len(rows)} tasks")

    function_docs = load_function_docs(functions_dir)
    all_functions = set(function_docs)
    print(f"[codegen] {len(function_docs)} function docs")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    write_lock = asyncio.Lock()

    async def save(result: dict):
        async with write_lock:
            with open(output_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    agent = FixAgent(config=llm_config, tools=specialist_tools, all_functions=all_functions)
    sem = asyncio.Semaphore(concurrency)
    done = 0

    async def worker(row: dict):
        nonlocal done
        async with sem:
            result = await generate_for_task(
                row=row, all_functions=all_functions, function_docs=function_docs,
                llm_config=llm_config, fix_agent=agent, verify_server=verify_server,
                apl_variant=apl_variant, max_gen_attempts=max_gen_attempts,
                max_fix_attempts=max_fix_attempts,
            )
            await save(result)
            done += 1
            print(f"[{done}/{len(rows)}] {'✔' if not result['failed'] else '✗'} {row['target_func']}")

    await asyncio.gather(*[worker(r) for r in rows])
    print(f"\n[codegen] done → {output_jsonl}")


# -- main ---------------------------------------------------------------------

async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",         required=True)
    p.add_argument("--output",        required=True)
    p.add_argument("--config",        required=True)
    p.add_argument("--functions-dir", required=True)
    p.add_argument("--verify-server", default="http://10.160.152.38:10000")
    p.add_argument("--apl-variant",   type=int, default=1)
    p.add_argument("--concurrency",   type=int, default=10)
    p.add_argument("--max-gen",       type=int, default=3,  dest="max_gen_attempts")
    p.add_argument("--max-fix",       type=int, default=6,  dest="max_fix_attempts")
    p.add_argument("--rag-folder",    default="/home/seigyo/rl/sft_primer/input/moove")
    p.add_argument("--rag-persist",   default="logs/chroma_moove_rag_2")
    p.add_argument("--mcp-url",       default="http://10.160.152.38:9001/mcp")
    args = p.parse_args()

    config = load_config(args.config)

    from client.tools.rag import build_markdown_rag_tool
    rag_tool = build_markdown_rag_tool(
        docs_folder=args.rag_folder, persist_directory=args.rag_persist,
        embedding_backend="server", embedding_model="cl-nagoya/ruri-v3-310m",
        embedding_base_url="http://10.160.144.101:51025/v1",
    )
    mcp_tools = await MultiServerMCPClient(
        {"moove": {"transport": "http", "url": args.mcp_url}}
    ).get_tools()

    await run(
        input_jsonl=Path(args.input), output_jsonl=Path(args.output),
        llm_config=config, functions_dir=Path(args.functions_dir),
        verify_server=args.verify_server, apl_variant=args.apl_variant,
        concurrency=args.concurrency, specialist_tools=[rag_tool, *mcp_tools],
        max_gen_attempts=args.max_gen_attempts, max_fix_attempts=args.max_fix_attempts,
    )


if __name__ == "__main__":
    asyncio.run(main())
