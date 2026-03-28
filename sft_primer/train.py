from __future__ import annotations

import asyncio
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List

import torch
from client.chat import ChatClient
from inference.vllm_engine import VLLMEngine
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from tqdm import tqdm

from torch.optim import AdamW
from model.config import ModelConfig
from model.gptoss import GptOssModel
from model.qwen3 import Qwen3Model
from model.qwen3_5 import Qwen3_5Model

# -----------------------------------------------------------------------------
# QA generation settings and schemas
# -----------------------------------------------------------------------------

# Global concurrency limit for QA requests.
GLOBAL_LLM_SEMAPHORE = asyncio.Semaphore(50)


class QuestionList(BaseModel):
    questions: List[str] = Field(
        description="本文情報だけで答えられる、自然で多様な質問リスト"
    )


class AnswerItem(BaseModel):
    answer: str = Field(
        description="与えられたコンテキストのみに基づく回答。厳密な表現があれば優先して使う。"
    )
    reasoning: str = Field(
        description="回答に至る思考過程。根拠抽出→照合→結論の順で自然文で記述する。"
    )


# Input markdown -> overlapped chunk JSON export.
def process_input_folder_to_chunks(
    input_folder: str,
    input_root: str = "sft_primer/input",
    output_root: str = "sft_primer/output",
    chunk_size: int = 2048,
    chunk_overlap: int = 512,
) -> Path:
    input_dir = Path(input_root) / input_folder
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks: List[Dict[str, str]] = []
    for md_path in sorted(input_dir.rglob("*.md")):
        source = str(md_path.relative_to(input_dir))
        text = md_path.read_text(encoding="utf-8", errors="ignore")
        for i, part in enumerate(splitter.split_text(text)):
            part = part.strip()
            if not part:
                continue
            chunks.append({"source": source, "chunk_index": i, "text": part})

    output_dir = Path(output_root) / input_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "chunks.json"
    out_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out_path


async def generate_qa_for_chunk_with_lookbehind(
    chunks: List[str],
    chunk_index: int,
    lookbehind_chunks: int,
    num_questions: int,
    model_path: str,
    port: int,
) -> List[Dict[str, str]]:
    """
    Generate QA samples for a single chunk.

    Flow:
    1. Build lookbehind context window.
    2. Ask question model for diverse questions.
    3. Ask answer model for {answer, reasoning} per question.
    """
    chunk_text = chunks[chunk_index]
    context_start_idx = max(0, chunk_index - max(0, lookbehind_chunks))
    context_window = chunks[context_start_idx : chunk_index + 1]
    context_text = "\n\n".join(context_window)

    base_url = f"http://127.0.0.1:{port}/v1"
    api_key = os.environ.get("OPENAI_API_KEY", "")

    # A lightweight question generator with structured output.
    question_generator = ChatClient(
        base_url=base_url,
        api_key=api_key,
        temperature=0.7,
        max_output_tokens=None,
        system_prompt=(
            "あなたは技術文書QAデータ作成アシスタントです。"
            "与えられた本文だけで完全に答えられる質問を作成してください。"
            "質問は、実利用のユーザー質問として自然な文にしてください。"
            "質問文に「チャンク」「文脈」「上記本文」などのメタ表現を入れてはいけません。"
            "質問の種類は限定せず、定義、手順、比較、制約、注意点、例外、背景をバランスよく含めてください。"
            "必ず JSON で {\"questions\": [\"...\"]} 形式のみを返してください。"
        ),
        model=model_path,
    )

    # Answer generator that also returns a reasoning trace for SFT.
    answer_generator = ChatClient(
        base_url=base_url,
        api_key=api_key,
        temperature=0.1,
        max_output_tokens=None,
        system_prompt=(
            "あなたは技術文書QAアシスタントです。"
            "回答は必ず与えられた本文のみを根拠にしてください。"
            "本文に厳密な表現がある場合はその表現を優先してください。"
            "reasoning は次の思考パターンに厳密に従ってください: "
            "1) 質問の意図整理 2) 本文からの根拠抽出 3) 根拠同士の照合 4) 最終結論。"
            "reasoning は自然な説明文で、簡潔だが論理のつながりが分かるように書いてください。"
            "reasoning や answer に「チャンク」「文脈」「上記」などのメタ参照語を入れてはいけません。"
            "本文に無い情報は推測せず、その旨を明示してください。"
            "必ず JSON で {\"answer\": \"...\", \"reasoning\": \"...\"} 形式のみを返してください。"
        ),
        model=model_path,
    )

    question_generator.reset_history()
    question_prompt = (
        f"次の本文に基づいて、{max(1, int(num_questions))} 個の質問を作成してください。\n\n"
        "要件:\n"
        "- 質問は実利用のユーザーがそのまま尋ねる自然な文にする\n"
        "- 質問文にメタ表現（例: チャンク、本文、上記）は入れない\n"
        "- 同型の言い換えを避け、観点を分散させる\n\n"
        f"本文:\n{chunk_text}"
    )
    async with GLOBAL_LLM_SEMAPHORE:
        _, question_obj = await question_generator.run(
            question_prompt,
            output_model=QuestionList,
        )
    questions = [q.strip() for q in question_obj.questions if q and q.strip()][
        : max(1, int(num_questions))
    ]

    async def _generate_answer_row(question: str) -> Dict[str, str]:
        answer_generator.reset_history()
        answer_prompt = (
            f"本文:\n{context_text}\n\n"
            f"質問:\n{question}\n\n"
            "上の本文だけを使って日本語で回答してください。"
            "reasoning は「意図整理→根拠抽出→照合→結論」の流れを明確に示してください。"
            "reasoning と answer にメタ参照語（チャンク、文脈、上記）を含めないでください。"
        )
        async with GLOBAL_LLM_SEMAPHORE:
            _, answer_obj = await answer_generator.run(
                answer_prompt,
                output_model=AnswerItem,
            )
        return {
            "question": question,
            "answer": answer_obj.answer.strip(),
            "reasoning": answer_obj.reasoning.strip(),
        }

    return await asyncio.gather(*[_generate_answer_row(q) for q in questions])


def train_model_on_qa_pairs(
    qa_pairs: List[Dict[str, str]],
    train_model_path: str,
    output_dir: str,
    epochs: int = 1,
    lr: float = 2e-4,
    weight_decay: float = 0.01,
    grad_accum_steps: int = 1,
    max_grad_norm: float = 1.0,
    lora_rank: int = 128,
    lora_alpha: int = 256,
    lora_target: List[str] = None,
    lora_layers_frac: float = 0.25,
    seed: int = 42,
    train_batch_size: int = 8,
) -> str:
    """
    Train LoRA adapters using the model wrapper's native `backward(...)` API.

    Notes:
    - Each sample trains on completion text that includes both `reasoning` and `answer`.
    - This function intentionally keeps logic explicit for easier debugging/resume extension.
    """

    if not qa_pairs:
        raise ValueError("qa_pairs is empty.")

    model_key = (
        train_model_path.strip("/")
        .replace("\\", "_")
        .replace("/", "_")
        .replace(":", "_")
    )
    output_root = Path(output_dir)
    lora_model_dir = output_root / "lora" / model_key
    ckpt_dir = output_root / "checkpoint" / model_key
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    completed_epoch = 0
    for p in lora_model_dir.glob("epoch_*"):
        if not p.is_dir():
            continue
        name = p.name
        if not name.startswith("epoch_"):
            continue
        suffix = name.split("_", 1)[1]
        if suffix.isdigit():
            completed_epoch = max(completed_epoch, int(suffix))
    if completed_epoch >= int(epochs):
        final_dir = lora_model_dir / "final"
        if final_dir.exists():
            return str(final_dir)

    rng = random.Random(int(seed))
    rows = list(qa_pairs)
    rng.shuffle(rows)

    # Select wrapper + default LoRA targets from model family.
    lower_path = train_model_path.lower()
    if "gpt-oss" in lower_path:
        model_cls = GptOssModel
        if lora_target is None:
            lora_target = ["q_proj", "k_proj", "v_proj", "o_proj"]
    elif "qwen3.5" in lower_path:
        model_cls = Qwen3_5Model
        if lora_target is None:
            lora_target = ["gate_proj", "up_proj", "down_proj"]
    else:
        model_cls = Qwen3Model
        if lora_target is None:
            lora_target = ["gate_proj", "up_proj", "down_proj"]

    train_cfg = ModelConfig(
        lora=list(lora_target),
        lora_fraction=float(lora_layers_frac),
        lora_rank=int(lora_rank),
        lora_alpha=int(lora_alpha),
        chunk_size=256,
        cuda_device_index=0,
        use_grad_checkpoint=True,
    )
    trainer = model_cls(model_path=train_model_path, config=train_cfg)
    if completed_epoch > 0:
        from peft import PeftModel

        resume_dir = lora_model_dir / f"epoch_{completed_epoch}"
        if resume_dir.exists():
            trainer.model = PeftModel.from_pretrained(
                trainer.model,
                str(resume_dir),
                is_trainable=True,
            )
    trainer.model.train()

    # Convert raw QA rows into supervised (messages, completion_text) pairs.
    train_samples: List[Dict[str, object]] = []
    for row in rows:
        if (
            "messages" in row
            and isinstance(row["messages"], list)
            and len(row["messages"]) >= 2
        ):
            prompt_messages = row["messages"][:-1]
            completion_text = str(row["messages"][-1].get("content", "")).strip()
        else:
            question_text = str(row.get("question", "")).strip()
            answer_text = str(row.get("answer", "")).strip()
            reasoning_text = str(row.get("reasoning", "")).strip()
            prompt_messages = [{"role": "user", "content": question_text}]
            completion_parts: List[str] = []
            if reasoning_text:
                completion_parts.append(f"Reasoning:\n{reasoning_text}")
            if answer_text:
                completion_parts.append(f"Answer:\n{answer_text}")
            completion_text = "\n\n".join(completion_parts).strip()

        if not completion_text:
            continue
        train_samples.append({"messages": prompt_messages, "completion_text": completion_text})

    if not train_samples:
        raise ValueError("No valid train samples after QA conversion.")

    optimizer = AdamW(
        [p for p in trainer.model.parameters() if p.requires_grad],
        lr=float(lr),
        weight_decay=float(weight_decay),
    )

    num_samples = len(train_samples)
    train_batch_size = max(1, int(train_batch_size))
    grad_accum_steps = max(1, int(grad_accum_steps))
    num_micro_batches = max(1, math.ceil(num_samples / train_batch_size))
    total_updates = max(
        1,
        math.ceil((num_micro_batches * int(epochs)) / grad_accum_steps),
    )

    def lr_lambda(step: int) -> float:
        warmup = max(1, int(0.05 * total_updates))
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_updates - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    step_count = 0
    start_epoch = completed_epoch + 1
    for epoch in range(start_epoch, int(epochs) + 1):
        rng.shuffle(train_samples)
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        seen_batches = 0

        for batch_start in range(0, len(train_samples), train_batch_size):
            batch_rows = train_samples[batch_start : batch_start + train_batch_size]
            batch_size = len(batch_rows)
            if batch_size == 0:
                continue

            # Build one micro-batch for wrapper.backward().
            batch_messages = [row["messages"] for row in batch_rows]
            batch_completions = [str(row["completion_text"]) for row in batch_rows]

            # Token-level NLL over valid completion tokens.
            def loss_fn_batch(batch_log_probs, batch_mask, hidden_batch=None):
                del hidden_batch
                denom = batch_mask.sum().clamp(min=1.0)
                return -((batch_log_probs * batch_mask).sum() / denom)

            stats = trainer.backward(
                messages=batch_messages,
                completion_texts=batch_completions,
                loss_fn=loss_fn_batch,
                loss_scale=1.0 / float(grad_accum_steps),
            )
            running_loss += float(stats.get("loss", 0.0))
            seen_batches += 1

            should_step = (seen_batches % grad_accum_steps == 0) or (
                batch_start + train_batch_size >= len(train_samples)
            )
            if should_step:
                torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), float(max_grad_norm))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step_count += 1

        mean_loss = running_loss / max(1, seen_batches)
        print(
            f"[sft_primer] epoch={epoch} "
            f"mean_batch_loss={mean_loss:.4f} "
            f"optimizer_steps={step_count} "
            f"batches={seen_batches} "
            f"samples={len(train_samples)}",
            flush=True,
        )
        epoch_dir = lora_model_dir / f"epoch_{epoch}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(str(epoch_dir))

    final_dir = lora_model_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(final_dir))
    return str(final_dir)
