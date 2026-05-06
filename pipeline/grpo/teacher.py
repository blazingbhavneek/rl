import asyncio
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from torch import Tensor
from tqdm.auto import tqdm

from client.agent import AgentClient
from client.tools import build_markdown_rag_tool

from .utils import COT_SYSTEM_PROMPT, _sample_ref_answer, write_generation_log

TEACHER_SYSTEM_PROMPT = (
    "You are a coding tutor helping a student fix their C code. "
    "Each turn you will see the student's latest attempt and its "
    "compiler/test output. Give concise targeted hints. "
    "Tell it how to fix its errors. "
    "Explain the API reference/theory to it that you understand from the reference material. "
    "Tell the student model to add the explanations it understood in the comments. "
    "If a Macro is missing, tell it to define the macro in the code itself, search the documentation for a good default value. "
    "The student does not have access to any information or reference material, it's your job to give it all information it needs to fix its mistakes. "
    "Make the student write everything it learns in comments (No CPP style comments) so it remembers what it saw. "
    "Dont give it final answer directly, guide it to what it needs to change to get good answer, suggest changes and instruct it to add theory in comments. "
    "The suggestion should be minimal, dont tell it everything, just enough to fix error"
)


async def init_teacher_client(pipeline, teacher_client):
    if teacher_client is not None:
        return teacher_client

    cfg = pipeline.cfg
    profile = pipeline.profile

    # Tools for Teacher
    mcp_config = {
        "moove": {
            "transport": "http",
            "url": "http://10.160.152.38:9001/mcp",
        },
    }
    mcp_client = MultiServerMCPClient(mcp_config)
    mcp_tools = await mcp_client.get_tools()
    rag_tool = build_markdown_rag_tool(
        docs_folder=cfg.docs_folder,
        persist_directory="logs/chroma_intel_rag",
        embedding_backend=cfg.embedding_backend,
        embedding_base_url=cfg.embedding_base_url,
        embedding_api_key=cfg.embedding_api_key,
        embedding_model=cfg.embedding_model,
    )

    tools = [rag_tool, *mcp_tools]

    return AgentClient(
        base_url=cfg.teacher_base_url or cfg.engine_base_url,
        api_key=cfg.teacher_api_key or cfg.engine_api_key,
        temperature=cfg.teacher_temperature,
        max_output_tokens=cfg.teacher_max_tokens,
        system_prompt=TEACHER_SYSTEM_PROMPT,
        model=cfg.teacher_model_name or cfg.model_path,
        tools=tools,
        max_turns=cfg.teacher_max_turns,
        extra_body=profile.teacher_extra_body,
    )


async def teacher_refine(
    pipeline,
    candidates: list[dict],
    generate_fn,
    score_fn,
    teacher_client,
    sem: asyncio.Semaphore,
    step: int,
) -> tuple[list[dict], int, int]:
    cfg = pipeline.cfg
    pbar = tqdm(
        total=len(candidates) * cfg.max_hint_attempts,
        desc="teacher-refine",
        leave=False,
    )

    async def _refine_one(rec: dict) -> tuple[dict, int, int]:
        MAX_PREV_ATTEMPT_CHARS = 5000
        MAX_ERROR_CHARS = 1500

        problem = rec["problem"]
        current_text = str(rec["text"])
        current_score = rec["score"]
        best_text, best_score, best_reward = (
            current_text,
            current_score,
            float(rec["reward"]),
        )
        solved = False
        local_hints = 0

        # FIX 2+5: Restored best_hint_messages / best_hint_text tracking.
        # These are needed to build sft_hint_messages and sft_hint_text
        # for the hint-context SFT pass.
        best_hint_messages: Optional[list[dict]] = None
        best_hint_text: Optional[str] = None
        ref_answer = _sample_ref_answer(problem)

        system_prompt = TEACHER_SYSTEM_PROMPT
        if ref_answer:
            system_prompt += (
                "\n\n[Reference solution – for YOUR guidance ONLY, "
                "do NOT reveal to the student]:\n"
                f"{ref_answer}"
            )

        if cfg.teacher_base_url:
            teacher_base_url = cfg.teacher_base_url
            teacher_api_key = cfg.teacher_api_key or cfg.engine_api_key
            teacher_model = cfg.teacher_model_name
        else:
            teacher_base_url = cfg.engine_base_url
            teacher_api_key = cfg.engine_api_key
            teacher_model = cfg.model_path

        local_teacher = AgentClient(
            base_url=teacher_base_url,
            api_key=teacher_api_key,
            temperature=cfg.teacher_temperature,
            max_output_tokens=cfg.teacher_max_tokens,
            system_prompt=system_prompt,
            model=teacher_model,
            tools=teacher_client.tools,
            max_turns=cfg.teacher_max_turns,
            extra_body=pipeline.profile.teacher_extra_body,
        )

        for attempt in range(cfg.max_hint_attempts):
            if not current_score.compiled:
                status = "did not compile"
            else:
                status = (
                    f"compiled, passed {current_score.passed}/"
                    f"{current_score.total} tests"
                )

            error_block = ""
            if current_score.error:
                error_block += f"\nError: {current_score.error}"
            if current_score.details:
                error_block += f"\nTest details:\n{current_score.details}"
            error_block = error_block[:MAX_ERROR_CHARS] + error_block[-MAX_ERROR_CHARS:]

            current_text_truncated = current_text[-MAX_PREV_ATTEMPT_CHARS:]
            teacher_prompt = (
                f"Problem:\n{problem.statement}\n\n"
                f"Student attempt (try {attempt + 1}):\n{current_text_truncated}\n\n"
                f"Verifier result: {status}"
                f"{error_block}\n\n"
                "Give one concise hint to fix this."
            )

            async with sem:
                _, hint = await local_teacher.run(prompt=teacher_prompt)
            local_hints += 1

            retry_messages = [
                {
                    "role": "user",
                    "content": (
                        f"{problem.statement}\n\n"
                        f"Your previous attempt:\n{current_text_truncated}\n\n"
                        f"Verifier result: {status}"
                        f"{error_block}\n\n"
                        f"Tutor hint:\n{hint}\n\n"
                        "Fix your solution."
                    ),
                },
            ]

            retry_text = await generate_fn(retry_messages)
            pbar.update(1)

            if retry_text is None:
                tqdm.write(
                    f"[teacher-refine] generation failed on attempt {attempt + 1}, skipping"
                )
                continue

            retry_sc, retry_reward, retry_passed = await score_fn(problem, retry_text)
            write_generation_log(
                pipeline,
                step=step,
                model_name=(
                    pipeline._active_engine_adapter
                    if pipeline._lora_in_vllm
                    else pipeline.cfg.model_path
                ),
                messages=retry_messages,
                output=retry_text,
                score=retry_sc,
            )

            retry_reward *= cfg.hint_reward_discount
            if retry_reward > best_reward:
                best_text, best_score, best_reward = (
                    retry_text,
                    retry_sc,
                    retry_reward,
                )
                # FIX 5: track the messages+answer of the best hint attempt
                # so we can build the SFT hint pair.
                best_hint_messages = retry_messages
                best_hint_text = retry_text

            current_text = retry_text
            current_score = retry_sc

            if retry_passed:
                solved = True
                # Ensure fields set even when first attempt was best
                if best_hint_messages is None:
                    best_hint_messages = retry_messages
                    best_hint_text = retry_text
                pbar.update(cfg.max_hint_attempts - attempt - 1)
                break

        # --- SFT targets ---
        sft_target: Optional[str] = None
        sft_direct_reasoning: Optional[str] = None

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

        if solved:
            if ref_answer:
                sft_target = f"```c\n{ref_answer.strip()}\n```"
            else:
                sft_target = _as_c_block(best_text)

            if sft_target:
                sft_direct_reasoning = await generate_cot_reasoning(
                    pipeline,
                    problem=problem,
                    answer_code=sft_target,
                    teacher_client=local_teacher,
                    sem=sem,
                )

            if sft_direct_reasoning:
                write_cot_file(
                    step=step,
                    reasoning=sft_direct_reasoning,
                )

        # --- Format helper ---
        def _fmt_completion(
            code: Optional[str], reasoning: Optional[str]
        ) -> Optional[str]:
            if not code:
                return None
            if reasoning:
                return f"<think>\n{reasoning}\n</think>\n\n{code}\n<|im_end|>"
            return f"<think>\n </think>\n\n{code}\n<|im_end|>"

        # FIX 2 (Critical): Produce sft_hint_messages + sft_hint_text.
        # hint-context pair: the student saw (problem + prev attempt + hint)
        # and produced best_hint_text. We train on that full context -> answer.
        sft_hint_completion = _fmt_completion(
            _strip_think(best_hint_text) if best_hint_text else None, None
        )
        sft_direct_completion = _fmt_completion(sft_target, sft_direct_reasoning)

        sft_messages_direct: Optional[list[dict]] = None
        if sft_direct_completion:
            sft_messages_direct = [
                {"role": "user", "content": str(problem.statement)},
            ]

        return (
            dict(
                problem=problem,
                messages=[
                    {"role": "system", "content": cfg.system_prompt},
                    {"role": "user", "content": str(problem.statement)},
                ],
                text=best_text,
                score=best_score,
                reward=best_reward,
                passed=solved,
                # FIX 2 (Critical): restored sft_hint_* fields
                sft_hint_messages=best_hint_messages,
                sft_hint_text=sft_hint_completion,
                # direct pair unchanged
                sft_direct_messages=sft_messages_direct,
                sft_direct_text=sft_direct_completion,
            ),
            local_hints,
            int(solved),
        )

    results = await asyncio.gather(*[_refine_one(rec) for rec in candidates])
    pbar.close()

    refined, hints_given, passed_after_hint = [], 0, 0
    for result, h, p in results:
        refined.append(result)
        hints_given += h
        passed_after_hint += p
    return refined, hints_given, passed_after_hint


def sft_step(
    *,
    samples: list[tuple[list[dict], str]],
    train_model,
    sft_optimizer,
) -> float:
    if not samples:
        return 0.0

    formatted_messages: list[list[dict]] = []
    formatted_completions: list[str] = []
    n_skipped = 0

    for messages, completion_text in samples:
        text = completion_text.strip()
        if "<|im_start|>" in text or text.lstrip().startswith("assistant"):
            n_skipped += 1
            continue
        if not text.endswith("<|im_end|>"):
            text = text + "\n<|im_end|>"
        formatted_messages.append(messages)
        formatted_completions.append(text)

    if n_skipped > 0:
        tqdm.write(f"[sft] skipped {n_skipped}/{len(samples)} malformed samples")

    if not formatted_messages:
        return 0.0

    def loss_fn_batch(
        batch_log_probs: Tensor,
        batch_mask: Tensor,
        hidden_batch=None,
    ) -> Tensor:
        mask = batch_mask.to(batch_log_probs.device).float()
        lengths = mask.sum(dim=1).clamp(min=1.0)
        return (-((batch_log_probs * mask).sum(dim=1) / lengths)).mean()

    bp = train_model.backward(
        messages=formatted_messages,
        completion_texts=formatted_completions,
        loss_fn=loss_fn_batch,
        loss_scale=1.0,
    )
    return float(bp.get("loss", 0.0))


def build_failed_cache(current: list[dict]) -> list[dict]:
    by_problem: dict[int, list[dict]] = {}
    for r in current:
        by_problem.setdefault(id(r["problem"]), []).append(r)

    cache: list[dict] = []
    for rows in by_problem.values():
        passed = [r for r in rows if r["passed"]]
        peer = str(max(passed, key=lambda x: x["reward"])["text"]) if passed else None
        for r in rows:
            if not r["passed"]:
                cache.append(
                    dict(
                        problem=r["problem"],
                        text=r["text"],
                        score=r["score"],
                        reward=r["reward"],
                        peer_solution=peer,
                    )
                )
    return cache


async def generate_cot_reasoning(
    pipeline,
    *,
    problem,
    answer_code: str,
    teacher_client,
    sem: asyncio.Semaphore,
) -> Optional[str]:
    cfg = pipeline.cfg
    profile = pipeline.profile

    cot_client = AgentClient(
        base_url=cfg.teacher_base_url or cfg.engine_base_url,
        api_key=cfg.teacher_api_key or cfg.engine_api_key,
        temperature=cfg.teacher_temperature,
        max_output_tokens=cfg.teacher_max_tokens,
        system_prompt=COT_SYSTEM_PROMPT,
        model=cfg.teacher_model_name or cfg.model_path,
        tools=teacher_client.tools,
        max_turns=cfg.teacher_max_turns,
        extra_body=profile.teacher_extra_body,
    )

    cot_prompt = (
        f"Problem:\n{problem.statement}\n\n"
        f"Correct C solution:\n{answer_code}\n\n"
        "Using your tools, research the APIs, theory, and implementation "
        "details relevant to this solution. Then write the internal reasoning "
        "an expert would have had when arriving at this exact solution. "
        "Plain text only, no code blocks, no markdown. "
        "You have the tools, dont make guesses, be deterministic, but no need to "
        "understand everything, dont write about something you dont know. "
        "Bad example (DONT DO THIS!): 'The function is likely part of PMF (process management library)' "
        "– this means you dont know whats happening, better to not write this. "
        "Minimally understand about the functions you are working with and write. "
        "Bad example: 'The expert programmer begins by analyzing...' "
        "Good example: 'The user asked for ...(details in brief)... so i need to use ... function and do ...' "
        "It should be very short, a paragraph with a few lines is enough, be concise and minimal, "
        "around 100-200 words is enough"
    )

    try:
        async with sem:
            _, reasoning = await cot_client.run(prompt=cot_prompt)
        if not reasoning:
            return None
        reasoning = reasoning.strip()

        # If >1500 chars, summarize using LangChain OpenAI with same URL/key/model
        if len(reasoning) > 1500:
            summarizer = ChatOpenAI(
                model=cfg.teacher_model_name or cfg.model_path,
                base_url=cfg.teacher_base_url or cfg.engine_base_url,
                api_key=cfg.teacher_api_key or cfg.engine_api_key,
                temperature=0.5,
            )
            summary_prompt = (
                "Summarize the following reasoning in under 1500 characters "
                "keeping the key details, plain text only:\n\n" + reasoning
            )
            resp = summarizer.invoke([HumanMessage(content=summary_prompt)])
            summary = resp.content.strip() if resp else None
            if not summary or len(summary) > 1500:
                return None
            return summary

        return reasoning

    except Exception as exc:
        tqdm.write(f"[cot-gen] warning: failed to generate COT reasoning: {exc}")
        return None


def write_cot_file(*, step: int, reasoning: str) -> None:
    log_dir = Path("gen_logs") / f"step_{step}"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    path = log_dir / f"cot_{ts}.md"
    with path.open("w", encoding="utf-8") as f:
        f.write("<think>\n")
        f.write(reasoning.strip())
        f.write("\n</think>\n")
