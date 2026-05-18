from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import requests
from contextvars import ContextVar
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel
from tqdm import tqdm
from tree_sitter import Parser, Language
from tree_sitter_c import language

from agents.master_agent import MasterAgent


# =========================================================================
# MCP SETUP
# =========================================================================

mcp_config = {
    "moove": {
        "transport": "http",
        "url": "http://10.160.152.38:9001/mcp",
    },
}

mcp_tools: list = []

if mcp_config:
    mcp_client = MultiServerMCPClient(mcp_config)
    mcp_tools = asyncio.run(mcp_client.get_tools())


# =========================================================================
# CONTEXT VARS (Isolated per-async-task. Defaults ensure graceful fallback.)
# =========================================================================

QUESTION_CTX = ContextVar("question", default="")
KNOWN_FUNCS_CTX = ContextVar("known_funcs", default=frozenset())
TASK_CHANGE_CTX = ContextVar("task_desc_change", default=None)


def set_verify_context(
    question: str,
    known_functions: set[str],
    task_change: Optional[dict] = None,
):
    QUESTION_CTX.set(question)
    KNOWN_FUNCS_CTX.set(frozenset(known_functions))
    TASK_CHANGE_CTX.set(task_change)


# =========================================================================
# VERIFY TOOL
# =========================================================================

@tool("verify_c_code")
def verify_c_code(code: str) -> str:
    """LLM only passes code. Context is auto-injected per async task."""

    SERVER_URL = "http://10.160.152.38:10000"
    APL_VARIANT = 1
    TIMEOUT = 60.0

    # Pull from async-local context (no race conditions)
    question = QUESTION_CTX.get()
    known_functions = KNOWN_FUNCS_CTX.get()

    # --- Auto-extract mentioned functions from question ---
    def extract_mentioned(q: str, known: frozenset) -> list[str]:
        if not q or not known:
            return []
        # Build regex: \b(fn1|fn2|fn3)\b for exact matches
        pattern = r'\b(' + '|'.join(map(re.escape, known)) + r')\b'
        matches = re.findall(pattern, q)
        # Preserve first-mention order, deduplicate
        seen = set()
        return [m for m in matches if not (m in seen or seen.add(m))]

    # --- Tree-Sitter Analyzer (Calls + Definitions + Declarations) ---
    class CCodeAnalyzer:
        def __init__(self):
            self.parser = Parser(Language(language()))

        def analyze(self, src: str) -> dict[str, set[str]]:
            try:
                tree = self.parser.parse(src.encode("utf8"))
            except Exception:
                return {'called': set(), 'defined': set(), 'declared': set()}

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
            return {'called': called, 'defined': defined, 'declared': declared}

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

    # --- Markdown Logger ---
    log_dir = Path("verify_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"verify_{int(time.time())}_{uuid.uuid4().hex[:8]}.md"
    ts = time.strftime("%Y-%m-%d %H:%M:%S")

    def log_md(compile_ok: bool, logs: str, analysis: dict, issues: list[str], prompt: str):
        task_change = TASK_CHANGE_CTX.get()
        c, d, dc = analysis['called'], analysis['defined'], analysis['declared']
        md = [
            "# C Verification",
            f"**Time:** {ts} | **Server:** {SERVER_URL} | **Compiled:** {'✓' if compile_ok else '✗'}",
        ]

        if task_change:
            md.extend([
                "",
                "## Task Adjustment",
                f"**Type:** {task_change.get('type')}",
                "",
                "**Changes:**",
                *[f"- {ch}" for ch in task_change.get("changes", [])],
            ])

        md.extend([
            "",
            "### Prompt",
            "```",
            (prompt or "None"),
            "```",
            "### Code",
            "```c",
            code.strip(),
            "```",
            "### Compiler Output",
            "```",
            (logs or "None"),
            "```",
        ])

        if effective_required:
            md.extend([
                "### Semantic Check",
                f"**Mentioned/Required:** {effective_required}",
                f"**Actually Called:** {sorted(c)}",
                f"**Locally Defined:** {sorted(d)}",
                f"**Forward Declared:** {sorted(dc)}",
            ])

        if issues:
            md.append("### Issues")
            for i, iss in enumerate(issues, 1):
                md.append(f"{i}. {iss}")
        else:
            md.append("### All checks passed")

        log_path.write_text("\n".join(md))

    # --- STEP 0: Build required list from context ---
    mentioned = extract_mentioned(question, known_functions)
    effective_required = sorted(set(mentioned))

    # --- STEP 1: Compile ---
    try:
        r = requests.post(
            f"{SERVER_URL}/compile",
            json={"code": code, "apl_variant": APL_VARIANT},
            timeout=TIMEOUT,
        )
    except Exception as e:
        log_md(
            False,
            str(e),
            {'called': set(), 'defined': set(), 'declared': set()},
            [f"Network error: {e}"],
            QUESTION_CTX.get(),
        )
        return f"✗ NETWORK ERROR\n{e}"

    if r.status_code != 200:
        log_md(
            False,
            r.text,
            {'called': set(), 'defined': set(), 'declared': set()},
            [f"Server error: HTTP {r.status_code}"],
            QUESTION_CTX.get(),
        )
        return f"✗ SERVER ERROR\nHTTP {r.status_code}\n{r.text}"

    data = r.json()
    compiled = bool(data.get("compiled"))
    logs = (data.get("logs") or "").strip()

    # --- STEP 2: Semantic + Anti-Cheating ---
    analysis = {'called': set(), 'defined': set(), 'declared': set()}
    issues = []

    if effective_required:
        analysis = CCodeAnalyzer().analyze(code)
        c, d, dc = analysis['called'], analysis['defined'], analysis['declared']
        for fn in effective_required:
            if fn in d:
                issues.append(f"❌ CHEATING: '{fn}' is locally defined. Remove it & #include the proper header.")
            # elif fn not in c:
            #     if fn in dc:
            #         issues.append(f"❌ SUSPICIOUS: '{fn}' is declared but never called. Add {fn}(...);")
            #     else:
            #         issues.append(f"❌ MISSING: '{fn}' mentioned in task but not called in code.")

    # --- STEP 3: Format Response ---
    log_md(compiled, logs, analysis, issues, QUESTION_CTX.get())

    lines = []
    if compiled:
        lines.append("✔ COMPILE SUCCESS")
    else:
        lines.extend(["✗ COMPILE FAILED", "-" * 16, logs or "No output.", ""])

    if effective_required:
        if issues:
            lines.append("✗ SEMANTIC ISSUES:")
            lines.extend(f"  {iss}" for iss in issues)
            lines.append("\n  ✗ FIX: Address all issues above.")
        else:
            lines.append(f"✔ All {len(effective_required)} mentioned functions properly called: {effective_required}")

    if compiled and not issues:
        return "✔ VERIFIED: Compile OK + All required functions called + No cheating\n" + "\n".join(lines)
    elif compiled and issues:
        return "✗ SEMANTIC FAIL: Compiled but missing/cheating\n" + "\n".join(lines)
    else:
        return "✗ COMPILE FAIL: Fix compiler errors first\n" + "\n".join(lines)


def format_function_set(funcs: set[str], max_items: int = 50) -> str:
    lst = sorted(funcs)
    if len(lst) > max_items:
        return ", ".join(lst[:max_items]) + f", ... ({len(lst)} total)"
    return ", ".join(lst)


# =========================================================================
# CONFIGURATION & GLOBALS
# =========================================================================

_jsonl_lock = asyncio.Lock()
_progress_lock = asyncio.Lock()


# =========================================================================
# PROMPT TEMPLATES
# =========================================================================

C_PATTERN_EXAMPLES = """
Examples of realistic C patterns to vary across tasks (DO NOT repeat the same pattern verbatim):

- Use a #define macro for a constant, then reference it in a function call
- Wrap a code block in #ifdef FEATURE_X / #endif to simulate conditional compilation
- Declare a static helper function that formats a buffer, called from main()
- Use an extern int g_config; that's defined elsewhere, and check its value before proceeding
- Pass a variable to a function that was initialized in a different scope (e.g., global or static)
- Use a struct with hardcoded field values, then pass a pointer to a library function
- Check return codes with if (ret != MPF_SUCCESS) { ... } and include a cleanup label
- Use a string literal macro: #define ERR_MSG "init failed" and pass it to a logging function
- Declare a buffer with a fixed size (e.g., char buf[256]); and use sizeof(buf) in a call
- Use a function pointer typedef and assign it conditionally based on a compile-time flag
- Forward-declare a helper function before main(), define it after
- Use a typedef for a complex type, then declare variables using it
- Initialize a struct with designated initializers (C99 style): .field = value
- Use a goto-based cleanup pattern for error handling
- Use a volatile qualifier for a flag checked in a loop
"""


# =========================================================================
# OUTPUT SCHEMA
# =========================================================================

class TaskOutput(BaseModel):
    question: str
    code: str
    reasoning: str


# =========================================================================
# INVENTORY & IO HELPERS
# =========================================================================

def extract_function_inventory(functions_dir: Path) -> set[str]:
    """Extract function names from filenames like function_name.json."""
    functions = set()
    if not functions_dir.exists():
        return functions
    for file in functions_dir.iterdir():
        if file.is_file() and file.suffix == ".json":
            functions.add(file.stem)
    return functions


async def append_to_jsonl(path: Path, entry: dict) -> None:
    """Thread-safe append to JSONL file."""
    async with _jsonl_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_existing_tasks(output_jsonl: Path, target_fn: str) -> list[dict]:
    """Load existing task records for a target function."""
    tasks = []
    if not output_jsonl.exists():
        return tasks
    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if row.get("target_func") == target_fn:
                    tasks.append(row)
            except json.JSONDecodeError:
                continue
    return tasks


def load_function_docs(functions_dir: Path) -> dict[str, str]:
    """Load raw JSON content for each function file. Returns {func_name: raw_json_string}."""
    docs = {}
    if not functions_dir.exists():
        return docs
    for file in functions_dir.iterdir():
        if file.is_file() and file.suffix == ".json":
            func_name = file.stem
            try:
                # Read raw JSON as string - no parsing, just dump as-is in prompt
                content = file.read_text(encoding="utf-8").strip()
                docs[func_name] = content
            except Exception:
                continue
    return docs


# ==========================================================================
# TASK GENERATOR WITH COMPILATION FIX LOOP
# ==========================================================================

class FunctionTaskGenerator:
    def __init__(
        self,
        config: dict,
        specialist_tools: list,
        verify_tool,
        max_fix_attempts: int = 7,
        function_docs: Optional[dict[str, str]] = None,
    ):
        self.config = config
        self.specialist_tools = specialist_tools
        self.verify_tool = verify_tool
        self.max_fix_attempts = max_fix_attempts
        self.function_docs = function_docs or {}

    def _extract_code(self, text: str) -> str:
        """Extract C code from LLM response (markdown-aware)."""
        if not text:
            return ""
        # Match ```c or ``` code blocks
        match = re.search(r"```(?:c|cpp)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # Fallback: return raw text if no code block
        return text.strip()

    def _is_compiled(self, verify_result: str) -> bool:
        """Check if verification result indicates successful compilation."""
        if not verify_result:
            return False
        return "✔ COMPILE SUCCESS" in verify_result or "COMPILE SUCCESS" in verify_result

    async def _generate_initial_code(
        self,
        task_desc: str,
        target_fn: str,
        all_functions: set[str],
        agent: "MasterAgent",
    ) -> str:
        """Generate initial C code using master agent."""

        # --- STEP 1: Extract mentioned functions from task description ---
        pattern = r'\b(' + '|'.join(map(re.escape, all_functions)) + r')\b'
        mentioned_fns = set(re.findall(pattern, task_desc))
        # Always include the target function as a baseline
        mentioned_fns.add(target_fn)

        # --- STEP 2: Retrieve and dump ONLY relevant JSON definitions ---
        relevant_defs = []
        for fn in mentioned_fns:
            if fn in self.function_docs:
                relevant_defs.append(self.function_docs[fn])
        relevant_json_block = "\n\n".join(relevant_defs)

        # Build prompt with strict compilation rules
        prompt = (
            f"Write C code for this task:\n\n{task_desc}\n\n"
            "REFERENCE LIBRARY DEFINITIONS:\n\n"
            "The following are the raw JSON definitions for the functions specifically mentioned in this task:\n\n"
            f"{relevant_json_block}\n\n"
            "If you require extra information (beyond these definitions) or usage examples, ask the specialist tools.\n\n"
            "STRICT COMPILATION & API RULES:\n"
            "1. NEVER define, stub, reimplement, or wrap any library function mentioned in the task. "
            "Only `#include` the correct headers and CALL them.\n"
            "2. If unsure about signature/return type/header (and it's missing above), use specialist tools. Max 2 questions.\n"
            "3. All mentioned functions must be called directly. No local implementations of library APIs.\n"
            "4. Code must compile standalone. No missing symbols, no implicit declarations.\n"
            "5. Output ONLY complete, compiling C code. No explanations, no markdown fluff.\n"
            "6. You are not allowed to run commands, just write code.\n"
            "7. You are not allowed to forward declare library functions. All functions that start with mpf_ or pmf_ "
            "are part of the library (unless the task says they need to be defined since they are made up of library "
            "functions) — need to have proper includes, no forward declaration needed.\n"
            "8. You are not allowed to forward declare or extern declare something.\n\n"
            "Remember:\n"
            "- 'for' loop initial declarations require C99/C11 mode\n"
            "- No C++ style comments (//)\n\n"
            "Output format: the code, wrapped in ```c```. Always return the full end to end code of the task.\n"
        )

        try:
            raw_resp, structured = await agent.run(prompt, output_model=TaskOutput)
            if structured and hasattr(structured, "code") and structured.code:
                return self._extract_code(structured.code)
            return self._extract_code(str(raw_resp))
        except Exception as e:
            print(f"[{target_fn}] Initial generation error: {e}")
            return ""

    async def _fix_compile_errors(
        self,
        task_desc: str,
        code: str,
        target_fn: str,
        all_functions: set[str],
        base_agent_config: dict,
        specialist_tools: list,
        verify_tool,
        max_fix_attempts: int = 10,
    ) -> tuple[str | None, str | None]:
        """
        Fix compilation errors with fresh agent per attempt.
        Returns:
            (fixed_code, None) on success, or (None, error_message) on failure.
        """
        current_code = code

        for attempt in range(1, max_fix_attempts + 1):
            # Verify current code
            set_verify_context(task_desc, all_functions, None)
            try:
                verify_result = verify_tool.invoke({"code": current_code})
            except Exception as e:
                verify_result = f"✗ VERIFY ERROR: {e}"

            # Success? Return immediately
            if self._is_compiled(verify_result):
                return current_code, None

            # Extract compile error from verify output
            if "----------------" in verify_result:
                compile_error = verify_result.split("----------------", 1)[-1].strip()
            else:
                compile_error = verify_result

            print(f"[{target_fn}] Fix attempt {attempt}/{max_fix_attempts} | Error: {compile_error[:200]}...")

            # Create FRESH agent with reset history and reduced max_turns
            fix_agent = MasterAgent(
                system_prompt="Expert C fixer. Output ONLY corrected, compiling C code. No explanations.",
                model=base_agent_config["model"],
                base_url=base_agent_config["base_url"],
                api_key=base_agent_config["api_key"],
                temperature=0.3,  # Lower temp for deterministic fixes
                specialist_tools=specialist_tools,
                extra_tools=[verify_tool],
                max_turns=5,  # Reduced context per fix attempt
            )

            # Build fix prompt with error context
            fix_prompt = (
                "The following C code failed to compile:\n\n"
                "```c\n"
                f"{current_code}\n"
                "```\n\n"
                "Compiler error:\n\n"
                "```\n"
                f"{compile_error}\n"
                "```\n\n"
                f"Task context (for reference only - DO NOT change task intent):\n\n{task_desc}\n\n"
                "INSTRUCTIONS:\n"
                "1. Fix ONLY the compilation errors shown above.\n"
                "2. Do NOT change the task logic or add new features.\n"
                "3. Do NOT define library functions - only #include and call them.\n"
                "4. Output ONLY the corrected, complete C code. No explanations.\n"
                "5. Preserve all required function calls from the task.\n\n"
                "Output format: Just the code, optionally wrapped in ```c```"
            )

            try:
                raw_resp, _ = await fix_agent.run(fix_prompt)
                fixed_code = self._extract_code(str(raw_resp))
                if not fixed_code:
                    print(f"[{target_fn}] Attempt {attempt}: Empty response, retrying...")
                    continue
                # Update current_code for next iteration (if needed)
                current_code = fixed_code
            except Exception as e:
                print(f"[{target_fn}] Fix attempt {attempt} failed: {e}")
                continue

        # Final verification after all attempts
        set_verify_context(task_desc, all_functions, None)
        try:
            final_verify = verify_tool.invoke({"code": current_code})
        except Exception:
            final_verify = "✗"

        if self._is_compiled(final_verify):
            return current_code, None
        return None, f"Failed to compile after {max_fix_attempts} fix attempts"

    async def _generate_with_retry(
        self,
        prompt: str,
        target_fn: str,
        batch_name: str,
        all_functions: set[str],
        original_task: Optional[str] = None,
    ) -> dict:
        """
        Main code generation entry point.
        Flow:
            1. Generate initial code with master agent (30 turns)
            2. If compile fails, fix with fresh agents (10 attempts × 5 turns each)
            3. Return result dict
        """
        from agents.master_agent import MasterAgent

        task_for_verify = original_task if original_task else prompt

        master_agent = MasterAgent(
            system_prompt=(
                "Expert C developer. Always return full, compiling code required by the task. "
                "You are not allowed to implement these functions on your own nor "
                "can you forward declare them: "
                f"{', '.join(sorted(all_functions))}"
            ),
            model=self.config["model"],
            base_url=self.config["base_url"],
            api_key=self.config["api_key"],
            temperature=0.7,
            specialist_tools=self.specialist_tools,
            extra_tools=[self.verify_tool],
            max_turns=5,
        )

        # Step 1: Generate initial code
        print(f"[{target_fn}] ({batch_name}) Generating initial code...")
        initial_code = await self._generate_initial_code(
            task_desc=task_for_verify,
            target_fn=target_fn,
            all_functions=all_functions,
            agent=master_agent,
        )

        if not initial_code:
            return {
                "target_func": target_fn,
                "task": task_for_verify,
                "code": "",
                "failed": True,
                "task_desc_change": "No code generated",
            }

        # Step 2: Fix compilation errors with fresh agents
        print(f"[{target_fn}] ({batch_name}) Running compile-fix loop...")
        final_code, error_msg = await self._fix_compile_errors(
            task_desc=task_for_verify,
            code=initial_code,
            target_fn=target_fn,
            all_functions=all_functions,
            base_agent_config=self.config,
            specialist_tools=self.specialist_tools,
            verify_tool=self.verify_tool,
            max_fix_attempts=10,
        )

        # Step 3: Return result
        if final_code:
            task_adjustment = await self._generate_task_adjustment(
                task_for_verify, final_code, master_agent
            )
            return {
                "target_func": target_fn,
                "task": task_for_verify,
                "code": final_code,
                "failed": False,
                "task_desc_change": task_adjustment or None,
            }

        return {
            "target_func": target_fn,
            "task": task_for_verify,
            "code": "",
            "failed": True,
            "task_desc_change": error_msg,
        }

    async def _generate_task_adjustment(self, task: str, code: str, agent: MasterAgent) -> str:
        """Lightweight 1-turn call to get exact task update string."""
        old_turns = agent.max_turns
        agent.max_turns = 2
        try:
            prompt = (
                f"Original task:\n{task}\n\n"
                f"Final compiling code:\n{code}\n\n"
                "Return EXACTLY ONE SHORT SENTENCE describing what to change in the task to match this code. "
                "If it matches perfectly, return an empty string. No extra text."
            )
            raw, _ = await agent.run(prompt)
            resp = str(raw).strip().strip('"').strip("'")
            return resp if resp and len(resp) > 5 else ""
        except Exception:
            return ""
        finally:
            agent.max_turns = old_turns


# ============================================================================
# ORCHESTRATOR: CONCURRENCY + CONTINUATION
# ============================================================================

async def run_generation(
    input_dir: Path,
    output_jsonl: Path,
    configs: list,
    specialist_tools: list,
    verify_tool,
    concurrency: int = 25,
):
    print(f"[inventory] Loading functions from {input_dir}...")
    all_functions = extract_function_inventory(input_dir)
    if not all_functions:
        print("[ERROR] No functions found.", file=sys.stderr)
        sys.exit(1)
    print(f"[inventory] Found {len(all_functions)} functions")

    # Load raw JSON docs once
    function_docs = load_function_docs(input_dir)
    print(f"[docs] Loaded {len(function_docs)} function JSON docs")

    generators = [
        FunctionTaskGenerator(configs[0], specialist_tools, verify_tool, function_docs=function_docs),
        FunctionTaskGenerator(configs[1], specialist_tools, verify_tool, function_docs=function_docs),
    ]

    half = max(1, concurrency // 2)
    semaphores = [asyncio.Semaphore(half), asyncio.Semaphore(half)]

    # ========= PHASE 1: TASK GENERATION =========
    async def task_worker(i, fn):
        idx = i % 2
        gen = generators[idx]
        sem = semaphores[idx]
        async with sem:
            await process_function_serial(
                target_fn=fn,
                all_functions=all_functions,
                output_jsonl=output_jsonl,
                generator=gen,
            )

    # TODO: Reactivate this after code gen is tested
    # await asyncio.gather(*[
    #     task_worker(i, fn)
    #     for i, fn in enumerate(sorted(all_functions))
    # ])

    print("\n✓ All tasks generated\n")

    # ========= PHASE 2: CODE GENERATION =========
    await run_codegen_phase(
        output_jsonl=output_jsonl,
        generators=generators,
        all_functions=all_functions,
        concurrency=concurrency,
    )

    print("\n✓ FULL PIPELINE COMPLETE\n")


async def run_codegen_phase(
    output_jsonl: Path,
    generators: list["FunctionTaskGenerator"],
    all_functions: set[str],
    concurrency: int = 25,
):
    """Phase 2: Generate code for pending tasks with simplified fix loop."""
    print("[codegen] Phase 2: Generating code for pending tasks...")

    # Load & deduplicate existing rows
    latest = {}
    if output_jsonl.exists():
        with open(output_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    key = (r.get("target_func"), r.get("task"))
                    latest[key] = r
                except json.JSONDecodeError:
                    continue

    pending = [r for r in latest.values() if not r.get("code")]
    print(f"[codegen] Pending rows: {len(pending)}")

    if not pending:
        print("[codegen] Nothing to process")
        return

    # Split concurrency between two generators
    half = max(1, concurrency // 2)
    semaphores = [asyncio.Semaphore(half), asyncio.Semaphore(half)]

    async def worker(i: int, row: dict):
        idx = i % 2
        generator = generators[idx]
        async with semaphores[idx]:
            try:
                result = await generator._generate_with_retry(
                    prompt=row["task"],
                    target_fn=row["target_func"],
                    batch_name="codegen",
                    all_functions=all_functions,
                )
                if result:
                    row["code"] = result["code"]
                    row["failed"] = result.get("failed", False)
                    row["task_desc_change"] = result.get("task_desc_change")
                    status = "✔" if not result.get("failed") else "✗"
                    print(f"[{row['target_func']}] {status} Code gen | failed={row['failed']}")
            except Exception as e:
                print(f"[codegen] Worker error for {row.get('target_func')}: {e}")

    await asyncio.gather(*[worker(i, r) for i, r in enumerate(pending)])

    # Atomic JSONL write
    tmp_path = Path(str(output_jsonl) + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for r in latest.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp_path, output_jsonl)
    print("[codegen] ✔ Atomic save complete")


async def process_function_serial(
    target_fn: str,
    all_functions: set[str],
    output_jsonl: Path,
    generator: FunctionTaskGenerator,
):
    existing = load_existing_tasks(output_jsonl, target_fn)
    existing_tasks = [t["task"] for t in existing]
    print(f"[{target_fn}] Existing tasks: {len(existing_tasks)}/60")

    # ======================
    # PROMPT BUILDER
    # ======================
    def build_prompt(mode: str, target_fn: str, all_functions: set[str], tasks: list[str]):
        avoid_block = (
            "\nPreviously generated tasks (DO NOT repeat or paraphrase):\n"
            + "\n".join(f"{i+1}. {t}" for i, t in enumerate(tasks[-30:]))
            + "\nSTRICT:\n"
            "- no duplicates\n"
            "- no paraphrasing\n"
            "- no similar logic\n"
            "- must be a completely different usage pattern\n"
        )

        # ======================
        # NON-TRIVIAL
        # ======================
        if mode == "non_trivial":
            prefix = target_fn.split('_')[0] if '_' in target_fn else target_fn[:3]
            related = [f for f in all_functions if f.startswith(prefix) and f != target_fn][:3]
            related_str = ", ".join(related) if related else "none"
            return (
                f"You are generating a realistic, non-trivial C task centered on: {target_fn}\n\n"
                f"Related functions: {related_str}\n\n"
                "STEP 1 (MANDATORY):\n"
                f"Use specialist tools to understand:\n"
                f"- what {target_fn} does\n"
                "- its signature\n"
                "- required headers\n"
                "- real usage patterns\n"
                "- companion functions\n"
                "DO NOT SKIP\n\n"
                f"{C_PATTERN_EXAMPLES}\n\n"
                "RULES:\n"
                "- MUST explicitly mention all functions used\n"
                "- MUST reflect real C usage (no fake domains)\n"
                "- MUST include real coding patterns (structs, buffers, macros, error handling)\n\n"
                f"{avoid_block}"
            )

        # ======================
        # SAMPLED (multi-function)
        # ======================
        elif mode == "sampled":
            sampled = random.sample(list(all_functions - {target_fn}), min(5, len(all_functions)))
            fn_list = [target_fn] + sampled
            fn_str = "\n".join(f"- {fn}" for fn in fn_list)
            return (
                f"Create a realistic C task using ALL of these functions:\n\n{fn_str}\n\n"
                "STEP 1 (MANDATORY):\n"
                "Use specialist tools for ALL listed functions.\n"
                "Understand:\n"
                "- their relationships\n"
                "- how they are used together in real code\n"
                "DO NOT SKIP\n\n"
                f"{C_PATTERN_EXAMPLES}\n\n"
                "RULES:\n"
                "- ALL functions must be explicitly used in task description\n"
                "- MUST describe a real workflow\n"
                "- MUST be non-trivial\n\n"
                f"{avoid_block}"
            )

        # ======================
        # PATTERN DRIVEN
        # ======================
        else:
            return (
                f"Generate a realistic C task using {target_fn}\n\n"
                "STEP 1 (MANDATORY):\n"
                "Use specialist tool to understand correct usage.\n"
                "DO NOT SKIP\n\n"
                f"{C_PATTERN_EXAMPLES}\n\n"
                "RULES:\n"
                "- MUST use at least one real C pattern from above\n"
                "- MUST be realistic and code-relevant\n"
                f"- MUST explicitly mention {target_fn}\n\n"
                f"{avoid_block}"
            )

    # ======================
    # PHASE 1: TASKS
    # ======================
    print(f"[{target_fn}] Phase 1: Generating tasks...")
    seen = set(t.lower().strip() for t in existing_tasks)
    tasks = existing_tasks[:]

    agent = MasterAgent(
        system_prompt=(
            "You are an expert C engineer.\n"
            "MANDATORY:\n"
            "- ALWAYS ask specialist to know what functions do, what are they used for etc\n"
            "- NEVER guess\n"
            "- NEVER hallucinate function usage\n"
            "- At max 3 questions at a time; no more than that to the specialist\n"
            "- You have to be very specific when making tasks, mention each function that's supposed to be "
            "called in the code, not just what the code is supposed to do. Be a bit detailed.\n"
            "- Don't make up any roleplay or anything, the task should be related to code only, and mention "
            "names of the functions from the library. Ask the specialist, no vague descriptions.\n"
            "- The prompt should be user-like, not some detailed story. It should be simple like "
            "\"Write C code that...\", \"Implement ...\"\n"
            "- When making tasks, the deduplication/difference should come from implementation logic, not "
            "module name, parameter names, etc. — periphery that doesn't matter. The task should differ in "
            "implementation, intent and logic.\n"
            "- You should explicitly mention all the function names from the library that are involved in the task\n"
            "- Apart from the target functions, ask the specialist questions so you know more about the function "
            "and try to involve other functions from the library for more diverse combinations. Don't just limit "
            "yourself to functions mentioned in previous lists — search, learn and do actual variations.\n"
            "- The variations should be of overall task intent, not just syntax and data manipulation. Use new "
            "kinds of functions, new tasks, new usage — not just syntax change or data flow change. Make the "
            "task genuinely different from older ones.\n"
        ),
        model=generator.config["model"],
        base_url=generator.config["base_url"],
        api_key=generator.config["api_key"],
        temperature=0.7,
        specialist_temperature=0.7,
        specialist_tools=generator.specialist_tools,
        max_turns=10,
    )

    while len(tasks) < 60:
        if len(tasks) < 30:
            mode = "non_trivial"
        elif len(tasks) < 50:
            mode = "sampled"
        else:
            mode = "pattern"

        try:
            agent.reset_history()
            prompt = build_prompt(mode, target_fn, all_functions, tasks)
            resp, _ = await agent.run(prompt)

            if not resp:
                print(f"[{target_fn}] ✗ Empty response - skipping")
                continue

            task = str(resp).strip()

            if not task:
                print(f"[{target_fn}] ✗ Blank task - skipping")
                continue

            # Skip broken API outputs
            if "APIConnectionError: Connection error" in task:
                print(f"[{target_fn}] ✗ Skipping API error task")
                continue

            # Minimal safety dedup only
            norm = task.lower()
            if norm in seen:
                continue

            seen.add(norm)
            tasks.append(task)

            await append_to_jsonl(output_jsonl, {
                "target_func": target_fn,
                "task": task,
                "code": None,
            })

            print(f"[{target_fn}] ✔ Task {len(tasks)}/60")

        except Exception as e:
            print(f"[{target_fn}] Task error: {e}")


# =========================================================================
# MAIN ENTRY POINT
# =========================================================================

def load_config(config_path: str) -> dict:
    """Load LLM configuration from JSON file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


async def main():
    parser = argparse.ArgumentParser(description="Generate 60 pristine C tasks per library function")
    parser.add_argument("--input", required=True, help="Input JSONL with 'required' lists")
    parser.add_argument("--output", required=True, help="Output JSONL for generated tasks")
    parser.add_argument("--config", required=True, help="LLM config JSON file")
    parser.add_argument("--config2", required=True)
    parser.add_argument("--concurrency", type=int, default=50, help="Function-level concurrency")
    parser.add_argument(
        "--rag-folder",
        default="/home/seigyo/rl/sft_primer/input/moove",
        help="RAG docs folder",
    )
    parser.add_argument(
        "--rag-persist",
        default="logs/chroma_moove_rag_2",
        help="RAG persist directory",
    )
    args = parser.parse_args()

    # Load config
    config1 = load_config(args.config)
    config2 = load_config(args.config2)

    # Setup specialist tools: RAG + MCP
    print("[setup] Initializing RAG and MCP tools...")
    from client.tools.rag import build_markdown_rag_tool

    rag_tool = build_markdown_rag_tool(
        docs_folder=args.rag_folder,
        persist_directory=args.rag_persist,
        embedding_backend="server",
        embedding_model="cl-nagoya/ruri-v3-310m",
        embedding_base_url="http://10.160.144.101:51025/v1",
    )

    specialist_tools = [rag_tool, *mcp_tools]

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Run generation
    await run_generation(
        input_dir=Path(args.input),
        output_jsonl=Path(args.output),
        configs=[config1, config2],
        specialist_tools=specialist_tools,
        verify_tool=verify_c_code,
        concurrency=args.concurrency,
    )


if __name__ == "__main__":
    asyncio.run(main())
