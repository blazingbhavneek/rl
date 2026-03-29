from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import torch
from peft import PeftModel
from tqdm.asyncio import tqdm as async_tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from inference.vllm_engine import VLLMEngine
from sft_primer.train import (
    generate_qa_for_chunk_with_lookbehind,
    process_input_folder_to_chunks,
    train_model_on_qa_pairs,
)


async def _run() -> None:
    md_path = Path("/media/blazingbhavneek/Common/Code/rl/sft_primer/input/intel/test.md")
    input_root = str(md_path.parents[1])
    output_root = str(md_path.parents[2] / "output")
    input_folder = md_path.parent.name
    qa_port = int(os.environ.get("QA_PORT", "30000"))
    chunk_concurrency = int(os.environ.get("CHUNK_CONCURRENCY", "10"))
    train_epochs = int(os.environ.get("TRAIN_EPOCHS", "4"))
    train_lr = float(os.environ.get("TRAIN_LR", "1e-5"))
    infer_max_new_tokens = int(os.environ.get("INFER_MAX_NEW_TOKENS", "8192"))

    print("=== SFT Primer Test ===")
    print(f"md_path: {md_path}")
    print(f"input_root: {input_root}")
    print(f"output_root: {output_root}")
    print(f"input_folder: {input_folder}")
    print(f"qa_port: {qa_port}")
    print(f"chunk_concurrency: {chunk_concurrency}")
    print(f"train_epochs: {train_epochs}")
    print(f"train_lr: {train_lr}")
    print(f"infer_max_new_tokens: {infer_max_new_tokens}")

    if not md_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {md_path}")

    out_path = Path(output_root) / input_folder / "chunks.json"
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"chunks_checkpoint_found: {out_path}")
    else:
        out_path = process_input_folder_to_chunks(
            input_folder=input_folder,
            input_root=input_root,
            output_root=output_root,
        )
    print(f"chunks_file: {out_path}")

    chunks = json.loads(Path(out_path).read_text(encoding="utf-8"))
    print(f"chunks_count: {len(chunks)}")

    sources = sorted({str(row.get("source", "")) for row in chunks if isinstance(row, dict)})
    print(f"sources: {sources}")

    if chunks:
        first = chunks[0]
        text_preview = str(first.get("text", ""))[:300]
        print("first_chunk_preview:")
        print(text_preview)

    chunk_texts = [str(row.get("text", "")).strip() for row in chunks if str(row.get("text", "")).strip()]
    if not chunk_texts:
        raise ValueError("No non-empty chunks found for QA generation.")

    train_model_path = os.environ.get(
        "MODEL_PATH",
        "/media/blazingbhavneek/Common/Code/sglangServer/Infer/Qwen/Qwen3-1.7B",
    )
    engine = VLLMEngine(
        model_path=train_model_path,
        engine_kwargs={
            "base_url": f"http://127.0.0.1:{qa_port}/v1",
            "model_name": train_model_path,
            "api_key": os.environ.get("OPENAI_API_KEY", ""),
            "save_vllm_logs": True,
            "gpu_memory_utilization": 0.90,
            "reasoning_parser": "qwen3",
            "enable_auto_tool_choice": True,
            "tool_call_parser": "hermes",
        },
    )

    qa_pairs_path = Path(output_root) / input_folder / "qa_pair.json"
    qa_pairs = []
    if qa_pairs_path.exists() and qa_pairs_path.stat().st_size > 0:
        print(f"qa_checkpoint_found: {qa_pairs_path}")
        with qa_pairs_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (
                    isinstance(row, dict)
                    and str(row.get("question", "")).strip()
                    and str(row.get("answer", "")).strip()
                    and str(row.get("reasoning", "")).strip()
                ):
                    qa_pairs.append(row)
    else:
        try:
            print("starting_qa_engine...")
            await engine.start()
            await engine.init()
            print("qa_engine_ready")

            sem = asyncio.Semaphore(max(1, chunk_concurrency))

            async def _run_chunk(i: int) -> tuple[int, list[dict[str, str]]]:
                async with sem:
                    try:
                        rows = await generate_qa_for_chunk_with_lookbehind(
                            chunks=chunk_texts,
                            chunk_index=i,
                            lookbehind_chunks=2,
                            num_questions=5,
                            model_path=train_model_path,
                            port=qa_port,
                        )
                    except Exception:
                        return i, []
                    clean_rows = []
                    for row in rows:
                        if not isinstance(row, dict):
                            continue
                        question = str(row.get("question", "")).strip()
                        answer = str(row.get("answer", "")).strip()
                        reasoning = str(row.get("reasoning", "")).strip()
                        if not (question and answer and reasoning):
                            continue
                        clean_rows.append(
                            {
                                "question": question,
                                "answer": answer,
                                "reasoning": reasoning,
                            }
                        )
                    return i, clean_rows

            tasks = [asyncio.create_task(_run_chunk(i)) for i in range(len(chunk_texts))]
            total = len(tasks)
            qa_pairs_path.parent.mkdir(parents=True, exist_ok=True)
            with qa_pairs_path.open("w", encoding="utf-8") as f:
                with async_tqdm(total=total, desc="Generating QA", unit="chunk") as pbar:
                    for fut in asyncio.as_completed(tasks):
                        try:
                            _, rows = await fut
                        except Exception:
                            rows = []
                        for row in rows:
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        f.flush()
                        pbar.update(1)
        finally:
            print("shutting_down_qa_engine...")
            await engine.shutdown()
            print("qa_engine_shutdown_complete")

    if not qa_pairs and qa_pairs_path.exists():
        with qa_pairs_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (
                    isinstance(row, dict)
                    and str(row.get("question", "")).strip()
                    and str(row.get("answer", "")).strip()
                    and str(row.get("reasoning", "")).strip()
                ):
                    qa_pairs.append(row)

    if not qa_pairs:
        raise ValueError("No QA pairs could be created from chunks.")
    print(f"chunks_for_qa: {len(chunk_texts)}")
    print(f"qa_pairs_count: {len(qa_pairs)}")
    if qa_pairs_path.exists() and qa_pairs_path.stat().st_size > 0:
        print(f"qa_pairs_file: {qa_pairs_path}")

    print(f"train_model_path: {train_model_path}")
    print(f"starting_lora_training: epochs={train_epochs}")
    train_output_dir = str(Path(output_root) / input_folder)
    print(f"train_output_dir: {train_output_dir}")
    adapter_dir = train_model_on_qa_pairs(
        qa_pairs=qa_pairs,
        train_model_path=train_model_path,
        output_dir=train_output_dir,
        epochs=train_epochs,
        lr=train_lr,
        train_batch_size=1,
        grad_accum_steps=1,
    )
    print(f"adapter_dir: {adapter_dir}")

    print("loading_base_plus_lora_for_inference...")
    tokenizer = AutoTokenizer.from_pretrained(train_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        train_model_path,
        device_map={"": "cuda:0"},
        dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    messages = [
        {
            "role": "system",
            "content": (
                "あなたは技術文書要約アシスタントです。"
                "回答は日本語で、intel の oneAPI/SYCL 入門資料の内容だけに基づいてください。"
            ),
        },
        {
            "role": "user",
            "content": (
                "このチャンクの要点を簡潔に要約しつつ、"
                "oneAPI/SYCL 入門として重要な点を説明してください。"
                "特に、入手・導入方法、参照すべき入門ドキュメント、"
                "SYCL サンプルコードの狙いを含めてください。"
                "回答では関連箇所を思い出して照合した流れがわかるように書いてください。"
            ),
        },
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tok = tokenizer([prompt], return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = tok["input_ids"].to(device)
    attention_mask = tok["attention_mask"].to(device)

    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=infer_max_new_tokens,
            do_sample=False,
        )
    new_ids = out[:, input_ids.shape[1] :]
    text = tokenizer.decode(new_ids[0], skip_special_tokens=True).strip()
    print("inference_output:")
    print(text)

    assert len(chunks) > 0
    assert "test.md" in sources
    assert qa_pairs_path.exists() and qa_pairs_path.stat().st_size > 0
    assert Path(adapter_dir).exists()
    assert bool(text)

    print("PASS: sft_primer chunk + lora train + inference test")


if __name__ == "__main__":
    asyncio.run(_run())
