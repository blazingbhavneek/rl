from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from sft_primer.train import process_input_folder_to_chunks, train_model_on_qa_pairs


def _run() -> None:
    md_path = Path("/media/blazingbhavneek/Common/Code/rl/sft_primer/input/intel/test.md")
    input_root = str(md_path.parents[1])
    output_root = str(md_path.parents[2] / "output")
    input_folder = md_path.parent.name

    print("=== SFT Primer Test ===")
    print(f"md_path: {md_path}")
    print(f"input_root: {input_root}")
    print(f"output_root: {output_root}")
    print(f"input_folder: {input_folder}")

    if not md_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {md_path}")

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

    qa_pairs = []
    questions = [
        "このチャンクの要点を簡潔に要約してください。",
        "このチャンクで説明される oneAPI/SYCL の目的は何ですか。",
        "このチャンクに含まれる手順や利用方法を説明してください。",
        "このチャンクに出てくる重要用語を挙げて説明してください。",
        "このチャンクの内容から初心者向けの注意点を述べてください。",
    ]
    for row in chunks:
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        source = str(row.get("source", "unknown"))
        chunk_index = int(row.get("chunk_index", -1))
        for q in questions:
            qa_pairs.append(
                {
                    "question": q,
                    "reasoning": (
                        f"source={source}, chunk_index={chunk_index} の本文を根拠に、"
                        "質問に対応する文を抽出して簡潔に再構成する。"
                    ),
                    "answer": text,
                }
            )
    if not qa_pairs:
        raise ValueError("No QA pairs could be created from chunks.")
    print(f"chunks_for_qa: {len(chunks)}")
    print(f"qa_pairs_count: {len(qa_pairs)}")
    qa_pairs_path = Path(output_root) / input_folder / "qa_pair.json"
    qa_pairs_path.parent.mkdir(parents=True, exist_ok=True)
    with qa_pairs_path.open("w", encoding="utf-8") as f:
        for row in qa_pairs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"qa_pairs_file: {qa_pairs_path}")

    train_model_path = os.environ.get(
        "MODEL_PATH",
        "/media/blazingbhavneek/Common/Code/sglangServer/Infer/Qwen/Qwen3-1.7B",
    )
    print(f"train_model_path: {train_model_path}")
    print("starting_lora_training: epochs=1")
    adapter_dir = train_model_on_qa_pairs(
        qa_pairs=qa_pairs,
        train_model_path=train_model_path,
        output_dir=output_root,
        epochs=1,
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
        {"role": "system", "content": "あなたは簡潔な日本語アシスタントです。"},
        {"role": "user", "content": "この資料が説明している oneAPI/SYCL の要点を2行で述べてください。"},
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
    _run()
