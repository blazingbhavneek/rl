from __future__ import annotations

import asyncio
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Literal

import torch
from client.chat import ChatClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer

from model.config import ModelConfig
from model.gptoss import GptOssModel
from model.qwen3 import Qwen3Model
from model.qwen3_5 import Qwen3_5Model

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GLOBAL_LLM_SEMAPHORE = asyncio.Semaphore(50)

ModelFamily = Literal["qwen3", "qwen3_5", "gpt-oss"]


# ---------------------------------------------------------------------------
# QA generation schemas
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# QA generation
# ---------------------------------------------------------------------------

async def generate_qa_for_chunk_with_lookbehind(
    chunks: List[str],
    chunk_index: int,
    lookbehind_chunks: int,
    num_questions: int,
    model_path: str,
    port: int,
) -> List[Dict[str, str]]:
    chunk_text = chunks[chunk_index]
    context_start_idx = max(0, chunk_index - max(0, lookbehind_chunks))
    context_text = "\n\n".join(chunks[context_start_idx: chunk_index + 1])

    base_url = f"http://127.0.0.1:{port}/v1"

    def _make_client(temperature: float, system_prompt: str) -> ChatClient:
        return ChatClient(
            base_url=base_url,
            api_key="",
            temperature=temperature,
            max_output_tokens=None,
            system_prompt=system_prompt,
            model=model_path,
        )

    question_generator = _make_client(
        temperature=0.7,
        system_prompt=(
            "あなたは技術文書QAデータ作成アシスタントです。"
            "与えられた本文だけで完全に答えられる質問を作成してください。"
            "質問は、実利用のユーザー質問として自然な文にしてください。"
            "質問文に「チャンク」「文脈」「上記本文」などのメタ表現を入れてはいけません。"
            "質問の種類は限定せず、定義、手順、比較、制約、注意点、例外、背景をバランスよく含めてください。"
            "必ず JSON で {\"questions\": [\"...\"]} 形式のみを返してください。"
        ),
    )

    async with GLOBAL_LLM_SEMAPHORE:
        _, question_obj = await question_generator.run(
            f"次の本文に基づいて、{max(1, int(num_questions))} 個の質問を作成してください。\n\n"
            "要件:\n"
            "- 質問は実利用のユーザーがそのまま尋ねる自然な文にする\n"
            "- 質問文にメタ表現（例: チャンク、本文、上記）は入れない\n"
            "- 同型の言い換えを避け、観点を分散させる\n\n"
            f"本文:\n{chunk_text}",
            output_model=QuestionList,
        )
    questions = [q.strip() for q in question_obj.questions if q and q.strip()][
        : max(1, int(num_questions))
    ]

    answer_system_prompt = (
        "あなたは技術文書QAアシスタントです。"
        "回答は必ず与えられた本文のみを根拠にしてください。"
        "本文に厳密な表現がある場合はその表現を優先してください。"
        "reasoning は次の思考パターンに厳密に従ってください: "
        "1) 質問の意図整理 2) 本文からの根拠抽出 3) 根拠同士の照合 4) 最終結論。"
        "reasoning は自然な説明文で、簡潔だが論理のつながりが分かるように書いてください。"
        "reasoning や answer に「チャンク」「文脈」「上記」などのメタ参照語を入れてはいけません。"
        "本文に無い情報は推測せず、その旨を明示してください。"
        "必ず JSON で {\"answer\": \"...\", \"reasoning\": \"...\"} 形式のみを返してください。"
    )

    async def _generate_answer_row(question: str) -> Dict[str, str]:
        # Each answer gets its own client so concurrent gather calls don't
        # stomp on each other's history via reset_history().
        client = _make_client(temperature=0.1, system_prompt=answer_system_prompt)
        async with GLOBAL_LLM_SEMAPHORE:
            _, answer_obj = await client.run(
                f"本文:\n{context_text}\n\n"
                f"質問:\n{question}\n\n"
                "上の本文だけを使って日本語で回答してください。"
                "reasoning は「意図整理→根拠抽出→照合→結論」の流れを明確に示してください。"
                "reasoning と answer にメタ参照語（チャンク、文脈、上記）を含めないでください。",
                output_model=AnswerItem,
            )
        return {
            "question": question,
            "answer": answer_obj.answer.strip(),
            "reasoning": answer_obj.reasoning.strip(),
        }

    # All answers for this chunk run concurrently.
    return await asyncio.gather(*[_generate_answer_row(q) for q in questions])


# ---------------------------------------------------------------------------
# Completion formatting
# ---------------------------------------------------------------------------

def detect_model_family(model_path: str) -> ModelFamily:
    lower = model_path.lower()
    if "gpt-oss" in lower:
        return "gpt-oss"
    if "qwen3.5" in lower or "qwen3_5" in lower:
        return "qwen3_5"
    return "qwen3"


def build_completion_text(
    row: Dict[str, str],
    tokenizer: AutoTokenizer,
    train_on_reasoning: bool = True,
) -> str:
    """
    Format the assistant completion using apply_chat_template for all model families.

    Passes only the assistant turn — no user message — so nothing from the prompt
    leaks into the completion string. The tokenizer inserts the correct special
    tokens (<think>...</think> for Qwen3, Harmony channels for gpt-oss) natively.

    train_on_reasoning=False omits the reasoning_content field entirely.
    """
    reasoning = row.get("reasoning", "").strip()
    answer = row.get("answer", "").strip()

    assistant_msg = {"role": "assistant", "content": answer}
    if train_on_reasoning and reasoning:
        assistant_msg["reasoning_content"] = reasoning

    try:
        return tokenizer.apply_chat_template(
            [assistant_msg],
            tokenize=False,
            add_generation_prompt=False,
        ).strip()
    except Exception:
        # Fallback for tokenizers that don't support reasoning_content yet.
        if train_on_reasoning and reasoning:
            return f"<think>\n{reasoning}\n</think>\n{answer}"
        return answer


def build_train_samples(
    qa_pairs: List[Dict[str, str]],
    tokenizer: AutoTokenizer,
    train_on_reasoning: bool = True,
) -> List[Dict]:
    """
    Convert raw QA rows into {messages, completion_text} dicts ready for
    trainer.backward(). Rows that already carry a 'messages' key (pre-built
    multi-turn conversations) are passed through with only the completion
    reformatted.
    """
    samples = []
    for row in qa_pairs:
        if "messages" in row and isinstance(row["messages"], list) and len(row["messages"]) >= 2:
            prompt_messages = row["messages"][:-1]
            last = row["messages"][-1]
            answer_row = {"answer": str(last.get("content", "")).strip(), "reasoning": ""}
            completion = build_completion_text(answer_row, tokenizer, train_on_reasoning=False)
        else:
            question = str(row.get("question", "")).strip()
            if not question:
                continue
            prompt_messages = [{"role": "user", "content": question}]
            completion = build_completion_text(row, tokenizer, train_on_reasoning)

        if not completion:
            continue
        samples.append({"messages": prompt_messages, "completion_text": completion})

    return samples


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

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
    lora_target: List[str] | None = None,
    lora_layers_frac: float = 0.25,
    seed: int = 42,
    train_batch_size: int = 8,
    train_on_reasoning: bool = True,
) -> str:
    if not qa_pairs:
        raise ValueError("qa_pairs is empty.")

    model_family = detect_model_family(train_model_path)

    # --- model class + default LoRA targets ---
    if model_family == "gpt-oss":
        model_cls = GptOssModel
        lora_target = lora_target or ["q_proj", "k_proj", "v_proj", "o_proj"]
    elif model_family == "qwen3_5":
        model_cls = Qwen3_5Model
        lora_target = lora_target or ["gate_proj", "up_proj", "down_proj"]
    else:
        model_cls = Qwen3Model
        lora_target = lora_target or ["gate_proj", "up_proj", "down_proj"]

    # --- checkpoint resume ---
    model_key = (
        train_model_path.strip("/")
        .replace("\\", "_").replace("/", "_").replace(":", "_")
    )
    output_root = Path(output_dir)
    lora_model_dir = output_root / "lora" / model_key
    lora_model_dir.mkdir(parents=True, exist_ok=True)

    completed_epoch = 0
    for p in lora_model_dir.glob("epoch_*"):
        if p.is_dir() and p.name.split("_", 1)[1].isdigit():
            completed_epoch = max(completed_epoch, int(p.name.split("_", 1)[1]))
    if completed_epoch >= int(epochs):
        final_dir = lora_model_dir / "final"
        if final_dir.exists():
            return str(final_dir)

    # --- build samples ---
    tokenizer = AutoTokenizer.from_pretrained(train_model_path)

    rng = random.Random(int(seed))
    rows = list(qa_pairs)
    rng.shuffle(rows)

    train_samples = build_train_samples(rows, tokenizer, train_on_reasoning)
    if not train_samples:
        raise ValueError("No valid train samples after QA conversion.")

    # --- model ---
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
                trainer.model, str(resume_dir), is_trainable=True,
            )

    trainer.model.train()

    # --- optimizer + scheduler ---
    optimizer = AdamW(
        [p for p in trainer.model.parameters() if p.requires_grad],
        lr=float(lr),
        weight_decay=float(weight_decay),
    )

    train_batch_size = max(1, int(train_batch_size))
    grad_accum_steps = max(1, int(grad_accum_steps))
    num_micro_batches = max(1, math.ceil(len(train_samples) / train_batch_size))
    total_updates = max(1, math.ceil((num_micro_batches * int(epochs)) / grad_accum_steps))

    def lr_lambda(step: int) -> float:
        warmup = max(20, int(0.1 * total_updates))
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_updates - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    def loss_fn_batch(batch_log_probs, batch_mask, hidden_batch=None):
        del hidden_batch
        return -((batch_log_probs * batch_mask).sum() / batch_mask.sum().clamp(min=1.0))

    # --- training loop ---
    step_count = 0
    for epoch in range(completed_epoch + 1, int(epochs) + 1):
        rng.shuffle(train_samples)
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        seen_batches = 0

        batch_starts = list(range(0, len(train_samples), train_batch_size))
        pbar = tqdm(
            batch_starts,
            desc=f"Epoch {epoch}/{epochs}",
            unit="batch",
            dynamic_ncols=True,
        )
        for batch_start in pbar:
            batch = train_samples[batch_start: batch_start + train_batch_size]
            if not batch:
                continue

            stats = trainer.backward(
                messages=[r["messages"] for r in batch],
                completion_texts=[r["completion_text"] for r in batch],
                loss_fn=loss_fn_batch,
                loss_scale=1.0 / float(grad_accum_steps),
            )
            batch_loss = float(stats.get("loss", 0.0))
            running_loss += batch_loss
            seen_batches += 1

            is_last_batch = (batch_start + train_batch_size >= len(train_samples))
            if seen_batches % grad_accum_steps == 0 or is_last_batch:
                torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), float(max_grad_norm))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step_count += 1

            current_lr = scheduler.get_last_lr()[0]
            mean_loss = running_loss / max(1, seen_batches)
            pbar.set_postfix(
                loss=f"{batch_loss:.4f}",
                mean=f"{mean_loss:.4f}",
                lr=f"{current_lr:.2e}",
                step=step_count,
            )

        pbar.close()
        print(
            f"[sft_primer] epoch={epoch} "
            f"mean_batch_loss={running_loss / max(1, seen_batches):.4f} "
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
