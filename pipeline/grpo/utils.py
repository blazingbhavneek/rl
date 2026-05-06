import asyncio
import json
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import count
from pathlib import Path
from typing import Optional

from tqdm.auto import tqdm
from tree_sitter import Language, Parser
from tree_sitter_c import language as c_language


def _build_prompt_text(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


async def _generate(
    messages: list[dict],
    *,
    engine,
    sem: asyncio.Semaphore,
    use_lora: bool,
    active_adapter: str,
    model_path: str,
    temperature: float,
    max_tokens: int,
    gen_extra_payload: dict,
) -> Optional[tuple[str, str, str]]:
    model_name = active_adapter if use_lora else model_path
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    payload.update(gen_extra_payload)
    async with sem:
        resp = await engine._request_json("POST", "/chat/completions", payload)
    choices = resp.get("choices") or []
    if not choices:
        return None
    raw_msg = choices[0].get("message") or {}
    reasoning = str(raw_msg.get("reasoning") or "").strip()
    content = str(raw_msg.get("content") or "").strip()
    if reasoning:
        text = f"<|channel|>thought\n{reasoning}\n<channel|>\n{content}"
    else:
        text = f"{content}<turn|>"
    return text, reasoning, content


def _is_clean_logs(compile_logs: str, stdout: str = "") -> bool:
    combined = f"{compile_logs or ''} {stdout or ''}".lower()
    if not combined.strip():
        return True
    return not any(
        kw in combined
        for kw in [
            "warning:",
            "error:",
            "note:",
            "implicit",
            "addresssanitizer",
            "undefined behavior",
            "segmentation fault",
            "segfault",
            "abort",
        ]
    )


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def _as_c_block(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    text = _strip_think(text)
    m = re.search(r"```c\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    code = m.group(1).strip() if m else text.strip()
    if not code:
        return None
    return f"```c\n{code}\n```"

def _log_rollout_stats(tag: str, rows: list[dict], extra: str = "") -> None:
    if not rows:
        return
    ratios = [
        (
            (float(r["score"].passed) / float(r["score"].total))
            if r["score"].total > 0
            else 0.0
        )
        for r in rows
    ]
    n_pass = sum(1 for r in rows if r["passed"])
    parts = [
        f"[{tag}]",
        f"n={len(rows)}",
        f"mean_score={sum(ratios) / max(1, len(ratios)):.3f}",
        f"passed={n_pass}/{len(rows)}",
    ]
    if extra:
        parts.append(extra)
    tqdm.write(" ".join(parts))


def write_generation_log(
    pipeline,
    *,
    step: int,
    model_name: str,
    messages: list[dict],
    output: Optional[str],
    score=None,
    finish_reason: Optional[str] = None,
    error: Optional[str] = None,
    reasoning: Optional[str] = None,
    content: Optional[str] = None,
) -> None:
    log_dir = Path("gen_logs") / f"step_{step}"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    idx = next(pipeline._gen_log_counter)
    path = log_dir / f"gen_{ts}_{idx:06d}.md"

    with path.open("w", encoding="utf-8") as f:
        f.write("# Student Generation\n\n")
        f.write(f"- step: `{step}`\n")
        f.write(f"- model: `{model_name}`\n")
        if finish_reason is not None:
            f.write(f"- finish_reason: `{finish_reason}`\n")
        if error is not None:
            f.write(f"- error: `{error}`\n")
        f.write("\n")

        f.write("## Message History\n\n")
        for i, m in enumerate(messages, 1):
            role = m.get("role", "unknown")
            f.write(f"### {i}. {role}\n\n")
            msg_content = m.get("content", "")
            if isinstance(msg_content, str):
                f.write(f"{msg_content}\n\n")
            else:
                f.write("```json\n")
                f.write(json.dumps(msg_content, ensure_ascii=False, indent=2))
                f.write("\n```\n\n")

        f.write("## Full Prompt\n\n")
        prompt_text = _build_prompt_text(pipeline.tokenizer, messages)
        f.write(f"```text\n{prompt_text}\n```\n\n")

        f.write("## Response\n\n")
        if output is not None:
            f.write("### Reasoning\n\n")
            f.write(f"```text\n{reasoning or ''}\n```\n\n")
            f.write("### Content\n\n")
            f.write(f"```text\n{content or output}\n```\n\n")
        else:
            f.write("_No output returned._\n")

        # Add compile logs if available
        if score is not None and hasattr(score, "details") and score.details:
            details = score.details or {}
            compile_logs = details.get("compile_logs", "") or ""
            stdout = details.get("stdout", "") or ""
            if compile_logs or stdout:
                f.write("## Compile Logs\n\n")
                f.write("```\n")
                if compile_logs:
                    f.write(compile_logs)
                if stdout:
                    if compile_logs:
                        f.write("\n")
                    f.write(stdout)
                f.write("\n```\n")


# region Utils


# This parses C code to check which functions were called in code, written to check whether required function were called or not
# otherwise model learns to game the system by either calling no functions or calling random other functions that are easy to compile
class FunctionCallAnalyzer:

    def __init__(self):

        # Initialize parser
        self.parser = Parser(Language(c_language()))

    # Make a set of functions that were called in the given code
    def extract_called_functions(self, code: str) -> set[str]:
        tree = self.parser.parse(
            code.encode("utf8")
        )  # encode it incase of euc_jp encoding?
        called = set()

        # Recursive function that goes through all entities
        def visit(node):
            if (
                node.type == "call_expression"
            ):  # call expression is node for when a function is called
                fn = node.child_by_field_name("function")
                if fn:
                    name = self._extract_name(
                        fn
                    )  # for handling multiple types of function calls
                    if name:
                        called.add(name)
            for c in node.children:
                visit(c)

        visit(tree.root_node)
        return called

    # Extracting name of function called, either direct calls (ret = foo(x)) which have identifiers, or calling methods (ret = obj.foo(x))
    def _extract_name(self, node):
        if node.type == "identifier":
            return (
                node.text.decode()
            )  # nodes have encoded code, we need to decode it to get string
        if node.type == "field_expression":
            f = node.child_by_field_name("field")
            if f:
                return f.text.decode()
        return None


analyzer = FunctionCallAnalyzer()

# Final correct response made with teacher's help needs to have its own Chain of thought
COT_SYSTEM_PROMPT = """\
You are generating chain-of-thought data for training a reasoning model.
Write the internal reasoning an expert C programmer would have when solving
the task, based on programming theory, API knowledge, and implementation planning.

Rules:
- Do NOT mention tools, verification, or compilation.
- Do NOT refer to any external process.
- Do NOT invent behavior not present in the code.
- Keep the reasoning technical, neutral, and educational.
- The output should be long and informative
- Use the specialist to ask questions properly first, dont include any information
  in the output that wasnt retrieved or checked by specialist
- Research properly, about the code and theory to make a correct and technically
  sound reasoning
- Answer should be simple text, no formatting, no codeblocks, no markdown or code
- Answer like a person thinking, not announcements

Bad output:
- An expert C programmer analyzing the task to write a minimal program using...
- When approaching the task of writing a minimal C program that uses mpf_mfs_stuprqbf,
  an expert C programmer would begin by examining the function...
- I need to first understand what this function does and how it fits into the PMF
  library architecture ...

Good output:
- I need to first understand what users want. The user wants to write C code ...
  know that this function ...

Dont let the reasoning show that you dont know, the model knows it already. The
reasoning should always sound confident.
Give long reasoning traces rich with information.
"""


@dataclass(frozen=True)
class _ModelProfile:
    module: str  # which class from the model module we need to import for this, which are subclasses of BaseModel
    cls_name: str  # Name of the subclass
    reasoning_parser: Optional[str]  # reasoning parser used by vllm
    tool_call_parser: str  # tool call parser used by vllm
    teacher_extra_body: (
        dict  # some calls may need extra kwargs to enable thinking from server side
    )
    gen_extra_payload: dict


_MODEL_PROFILES: dict[str, _ModelProfile] = {
    "qwen3": _ModelProfile(
        module="model.qwen3",
        cls_name="Qwen3Model",
        reasoning_parser="qwen3",
        tool_call_parser="hermes",
        teacher_extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        gen_extra_payload={"chat_template_kwargs": {"enable_thinking": True}},
    ),
    "qwen3_5": _ModelProfile(
        module="model.qwen3_5",
        cls_name="Qwen3_5Model",
        reasoning_parser="qwen3",
        tool_call_parser="qwen3_coder",
        teacher_extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        gen_extra_payload={"chat_template_kwargs": {"enable_thinking": True}},
    ),
    "gptoss": _ModelProfile(
        module="model.gptoss",
        cls_name="GptOssModel",
        reasoning_parser=None,
        tool_call_parser="openai",
        teacher_extra_body={},
        gen_extra_payload={"include_reasoning": True},
    ),
    "gemma": _ModelProfile(
        module="model.gemma4",
        cls_name="Gemma4Model",
        # match vLLM launch flags
        reasoning_parser="gemma4",
        tool_call_parser="gemma4",
        teacher_extra_body={},  # do NOT force enable_thinking
        gen_extra_payload={},  # let reasoning be server-controlled
    ),
}


# TODO: clean it up
def _get_profile(model_type: str) -> _ModelProfile:
    if model_type not in _MODEL_PROFILES:
        raise ValueError(
            f"Unknown model_type {model_type!r}. Choose from: {list(_MODEL_PROFILES)}"
        )
    return _MODEL_PROFILES[model_type]


# Get one valid answer if any, so teacher can have a reference when suggesting fix for wrong ones?
# TODO: clean it up, called only once
def _sample_ref_answer(problem) -> Optional[str]:
    raw = (problem.metadata or {}).get("answer", None)
    if not raw:
        return None
    if isinstance(raw, list):
        valid = [s for s in raw if s and isinstance(s, str)]
        return random.choice(valid) if valid else None
    if isinstance(raw, str):
        return raw.strip() or None
    return None


# endregion Utils
