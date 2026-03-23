from __future__ import annotations

import asyncio
import json
import math
import os
import random
import signal
import subprocess
from pathlib import Path
from typing import Dict, List

import torch
from backprop import BackpropConfig, ChunkSizeProfiler, StreamingBackprop
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

GLOBAL_LLM_SEMAPHORE = asyncio.Semaphore(50)


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


def start_sglang_openai_server(model_path: str, port: int) -> int:
    host = "0.0.0.0"
    cmd = [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--host",
        host,
        "--port",
        str(port),
    ]

    env = os.environ.copy()
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return process.pid


async def generate_qa_for_chunk_with_lookbehind(
    chunks: List[str],
    chunk_index: int,
    lookbehind_chunks: int,
    num_questions: int,
    model_path: str,
    port: int,
) -> List[Dict[str, str]]:
    class QuestionList(BaseModel):
        questions: List[str] = Field(
            description="チャンク内の情報だけで答えられる、あらゆる種類の質問リスト"
        )

    class AnswerItem(BaseModel):
        answer: str = Field(
            description="与えられたコンテキストのみに基づく回答。厳密な表現があれば優先して使う。"
        )

    chunk_text = chunks[chunk_index]
    start_idx = max(0, chunk_index - max(0, lookbehind_chunks))
    context_parts = chunks[start_idx : chunk_index + 1]
    context_text = "\n\n".join(context_parts)

    base_url = f"http://127.0.0.1:{port}/v1"
    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    question_llm = ChatOpenAI(
        model=model_path,
        base_url=base_url,
        api_key=api_key,
        temperature=0.7,
    ).with_structured_output(QuestionList)
    answer_llm = ChatOpenAI(
        model=model_path,
        base_url=base_url,
        api_key=api_key,
        temperature=0.1,
    ).with_structured_output(AnswerItem)

    question_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "あなたは与えられた文章から、本文だけで完全に答えられる質問を網羅的に作るアシスタントです。"
                "質問の種類は限定しません。事実確認、定義、比較、手順、制約、注意点、例外、背景など多様に作ってください。"
                "出力は必ず構造化形式に従ってください。",
            ),
            (
                "human",
                "次のチャンクに基づいて、{num_questions} 個の質問を作成してください。\n\n"
                "チャンク:\n{chunk_text}",
            ),
        ]
    )
    q_messages = question_prompt.format_messages(
        num_questions=max(1, int(num_questions)),
        chunk_text=chunk_text,
    )
    async with GLOBAL_LLM_SEMAPHORE:
        q_obj = await question_llm.ainvoke(q_messages)
    questions = [q.strip() for q in q_obj.questions if q and q.strip()][
        : max(1, int(num_questions))
    ]

    async def _answer_one(question: str) -> Dict[str, str]:
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたはドキュメントQAアシスタントです。"
                    "回答は必ず与えられたコンテキストのみを根拠にしてください。"
                    "コンテキストに厳密な表現がある場合はその表現を優先して使ってください。",
                ),
                (
                    "human",
                    "コンテキスト:\n{context}\n\n質問:\n{question}\n\n"
                    "上のコンテキストだけを使って日本語で回答してください。",
                ),
            ]
        )
        a_messages = answer_prompt.format_messages(
            context=context_text, question=question
        )
        async with GLOBAL_LLM_SEMAPHORE:
            answer_obj = await answer_llm.ainvoke(a_messages)
        answer = answer_obj.answer.strip()
        return {"question": question, "context": context_text, "answer": answer}

    return await asyncio.gather(*[_answer_one(q) for q in questions])


def train_model_on_qa_pairs(
    qa_pairs: List[Dict[str, str]],
    train_model_path: str,
    output_dir: str,
    epochs: int = 3,
    lr: float = 2e-4,
    weight_decay: float = 0.01,
    grad_accum_steps: int = 8,
    max_grad_norm: float = 1.0,
    lora_rank: int = 128,
    lora_alpha: int = 256,
    lora_dropout: float = 0.0,
    lora_target: List[str] = None,
    lora_layers_frac: float = 0.25,
    seed: int = 42,
) -> str:
    from peft import LoraConfig, TaskType, get_peft_model
    from torch.optim import AdamW
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not qa_pairs:
        raise ValueError("qa_pairs is empty.")

    if lora_target is None:
        lora_target = ["q_proj", "k_proj", "v_proj", "o_proj"]

    rng = random.Random(int(seed))
    rows = list(qa_pairs)
    rng.shuffle(rows)

    tokenizer = AutoTokenizer.from_pretrained(train_model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        train_model_path,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=int(lora_rank),
        lora_alpha=int(lora_alpha),
        lora_dropout=float(lora_dropout),
        target_modules=list(lora_target),
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    bp_cfg = BackpropConfig(
        top_frac=float(lora_layers_frac),
        use_grad_checkpoint=True,
        offload_prefix_cpu=True,
    )
    backprop = StreamingBackprop(model, config=bp_cfg)

    base, _ = backprop.adapter.unwrap(model)
    lm_head = backprop.adapter.get_lm_head(base)
    model_cfg = getattr(base, "config", getattr(model, "config", None))
    hidden_size = int(getattr(model_cfg, "hidden_size"))
    vocab_size = int(getattr(model_cfg, "vocab_size"))
    profile_dir = str(Path(output_dir) / "chunk_profiles")
    max_completion_len = 1
    for row in rows:
        if "messages" in row and isinstance(row["messages"], list) and len(row["messages"]) >= 2:
            answer_text = row["messages"][-1].get("content", "")
        else:
            answer_text = str(row.get("answer", "")).strip()
        if not answer_text:
            continue
        ln = len(tokenizer.encode(answer_text, add_special_tokens=False))
        if ln > max_completion_len:
            max_completion_len = ln

    profiler = ChunkSizeProfiler(
        lm_head=lm_head,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        device=device,
        model_path=train_model_path,
        sglang_mem_frac=0.0,
        top_frac=float(lora_layers_frac),
        cache_dir=profile_dir,
        dtype=dtype,
    )
    profiler.load_or_profile()
    backprop.chunk_profiler = profiler
    backprop.config.logit_chunk = profiler.get_chunk_size(int(max_completion_len))

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(lr),
        weight_decay=float(weight_decay),
    )

    total_updates = max(1, math.ceil((len(rows) * int(epochs)) / max(1, int(grad_accum_steps))))

    def lr_lambda(step: int) -> float:
        warmup = max(1, int(0.05 * total_updates))
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_updates - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    step_count = 0
    for epoch in range(1, int(epochs) + 1):
        rng.shuffle(rows)
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0

        for i, row in enumerate(rows):
            if "messages" in row and isinstance(row["messages"], list) and len(row["messages"]) >= 2:
                prompt_messages = row["messages"][:-1]
                answer_text = row["messages"][-1].get("content", "")
            else:
                q = str(row.get("question", "")).strip()
                a = str(row.get("answer", "")).strip()
                prompt_messages = [{"role": "user", "content": q}]
                answer_text = a

            if not answer_text.strip():
                continue

            if hasattr(tokenizer, "apply_chat_template"):
                prompt_text = tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt_text = "\n".join([m.get("content", "") for m in prompt_messages]) + "\nAssistant:"

            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            completion_ids_1d = tokenizer.encode(answer_text, add_special_tokens=False)
            if not completion_ids_1d:
                continue

            prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            completion_tensor = torch.tensor([completion_ids_1d], dtype=torch.long, device=device)
            completion_mask = torch.ones_like(completion_tensor, dtype=torch.float32, device=device)

            def sft_loss_fn(log_probs, gen_idx: int, hidden_comp=None):
                del hidden_comp
                mask = completion_mask[gen_idx].to(log_probs.device, non_blocking=True)
                return -(log_probs * mask).sum() / mask.sum().clamp(min=1.0)

            stats = backprop.backward_on_batch(
                model=model,
                prompt_ids=prompt_tensor,
                completion_ids=completion_tensor,
                completion_mask=completion_mask,
                loss_fn=sft_loss_fn,
                loss_scale=1.0 / max(1, int(grad_accum_steps)),
                lora_path=None,
            )
            running_loss += float(stats.get("loss", 0.0))

            if ((i + 1) % max(1, int(grad_accum_steps)) == 0) or (i + 1 == len(rows)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step_count += 1

        mean_loss = running_loss / max(1, len(rows))
        print(f"[sft_primer] epoch={epoch} mean_loss={mean_loss:.4f} optimizer_steps={step_count}", flush=True)

    final_dir = Path(output_dir) / "lora" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    backprop.save_lora(str(final_dir))
    return str(final_dir)


def main() -> None:
    INPUT_FOLDER = "mylib"
    QA_MODEL_PATH = "openai/gpt-oss-20b"
    QA_PORT = 30000
    LOOKBEHIND_CHUNKS = 2
    NUM_QUESTIONS_PER_CHUNK = 20
    TRAIN_MODEL_PATH = "openai/gpt-oss-20b"
    TRAIN_OUTPUT_DIR = "sft_primer/output/mylib_train"

    chunks_path = process_input_folder_to_chunks(INPUT_FOLDER)
    output_dir = Path("sft_primer/output") / INPUT_FOLDER
    qa_pairs_path = output_dir / "qa_pair.json"

    chunk_rows = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks = [str(row.get("text", "")) for row in chunk_rows if str(row.get("text", "")).strip()]

    server_pid = start_sglang_openai_server(QA_MODEL_PATH, QA_PORT)
    qa_pairs: List[Dict[str, str]] = []

    async def run_all_chunks() -> None:
        tasks = [
            generate_qa_for_chunk_with_lookbehind(
                chunks,
                i,
                LOOKBEHIND_CHUNKS,
                NUM_QUESTIONS_PER_CHUNK,
                QA_MODEL_PATH,
                QA_PORT,
            )
            for i in range(len(chunks))
        ]
        qa_pairs_path.parent.mkdir(parents=True, exist_ok=True)
        with qa_pairs_path.open("a", encoding="utf-8") as f:
            for fut in asyncio.as_completed(tasks):
                chunk_pairs = await fut
                qa_pairs.extend(chunk_pairs)
                for pair in chunk_pairs:
                    f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                f.flush()

    try:
        asyncio.run(run_all_chunks())
    finally:
        try:
            os.kill(server_pid, signal.SIGTERM)
        except OSError:
            pass

    train_model_on_qa_pairs(qa_pairs, TRAIN_MODEL_PATH, TRAIN_OUTPUT_DIR)


if __name__ == "__main__":
    main()
