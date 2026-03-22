#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

# Allow direct execution: python sft_primer/test_md_lora.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sft_primer.config import build_config
from sft_primer.train import train as train_sft


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tiny LoRA SFT primer on a markdown file.")
    p.add_argument(
        "--model-path",
        type=str,
        default="/media/blazingbhavneek/Common/Code/sglangServer/Infer/Qwen/Qwen3.5-0.8B",
    )
    p.add_argument(
        "--source-md",
        type=str,
        default="/media/blazingbhavneek/Common/Code/datagen/parser/tests/output/test.md",
    )
    p.add_argument("--library", type=str, default="qwen35_testmd")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-samples", type=int, default=12)
    p.add_argument("--max-seq", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _copy_md_into_repo(src: Path, dst_docs: Path) -> Path:
    if not src.exists():
        raise FileNotFoundError(f"source markdown not found: {src}")
    dst_docs.mkdir(parents=True, exist_ok=True)
    dst = dst_docs / src.name
    shutil.copy2(src, dst)
    return dst


def _paragraphs(md_text: str) -> List[str]:
    out: List[str] = []
    for block in md_text.split("\n\n"):
        s = block.strip()
        if len(s) < 40:
            continue
        out.append(s)
    return out


def _make_rows(paras: List[str], tokenizer, max_samples: int, max_seq: int, seed: int) -> List[Dict]:
    rnd = random.Random(seed)
    rnd.shuffle(paras)
    rows: List[Dict] = []

    base_prompt = (
        "You are reading project documentation. Learn and answer precisely from it.\n"
        "Continue the assistant response with correct technical details from docs.\n\n"
        "Assistant:"
    )
    prompt_ids = tokenizer.encode(base_prompt, add_special_tokens=False)
    budget = max(64, int(max_seq) - len(prompt_ids) - 4)

    for para in paras:
        comp_ids = tokenizer.encode(" " + para, add_special_tokens=False)[:budget]
        if len(comp_ids) < 16:
            continue
        rows.append(
            {
                "prompt_ids": prompt_ids,
                "completion_ids": comp_ids,
            }
        )
        if len(rows) >= max_samples:
            break
    return rows


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _split_train_val(rows: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    if len(rows) < 3:
        raise RuntimeError("Need at least 3 rows for train/val split.")
    n_val = max(1, int(round(len(rows) * 0.2)))
    n_val = min(n_val, len(rows) - 1)
    return rows[n_val:], rows[:n_val]


def _smoke_generate(model_path: str, lora_dir: str) -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    model = PeftModel.from_pretrained(model, lora_dir)
    model.eval()

    prompt = "Summarize what this documentation is mainly about in 3 bullet points.\n\nAssistant:"
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.0,
            do_sample=False,
        )
    text = tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    print("\n[sft_primer] generation smoke output:\n")
    print(text.strip())


def main() -> None:
    args = parse_args()

    cfg = build_config(args.library)
    cfg.MODEL_PATH = args.model_path
    cfg.DOCS_FOLDER = "sft_primer/docs"
    cfg.ATTN_IMPLEMENTATION = "sdpa"
    cfg.EPOCHS = int(args.epochs)
    cfg.GRAD_ACCUM_STEPS = 1
    cfg.MICRO_BATCH_SIZE = 1
    cfg.EARLY_STOPPING_PATIENCE = 1
    cfg.MAX_SEQ_LEN = int(args.max_seq)
    cfg.MAX_COMPLETION_TOKENS = int(args.max_seq)
    cfg.SEED = int(args.seed)
    cfg.USE_AUGMENTATION = False
    cfg.OPTIMIZER = "adamw"
    cfg.LORA_LAYERS_FRAC = 1.0  # vanilla LoRA path for tiny primer
    cfg.ensure_dirs()

    src = Path(args.source_md)
    copied = _copy_md_into_repo(src, Path(cfg.DOCS_FOLDER))
    print(f"[sft_primer] copied source markdown to: {copied}", flush=True)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.MODEL_PATH)
    paras = _paragraphs(copied.read_text(encoding="utf-8", errors="ignore"))
    rows = _make_rows(paras, tok, max_samples=int(args.max_samples), max_seq=int(args.max_seq), seed=int(args.seed))
    if len(rows) < 3:
        raise RuntimeError(f"Not enough rows extracted from markdown. got={len(rows)}")
    train_rows, val_rows = _split_train_val(rows)

    train_path = Path(cfg.DATASET_DIR) / "train.jsonl"
    val_path = Path(cfg.DATASET_DIR) / "val.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(val_path, val_rows)
    print(
        f"[sft_primer] dataset ready train={len(train_rows)} val={len(val_rows)} "
        f"at {train_path.parent}",
        flush=True,
    )

    final_lora = train_sft(str(train_path), str(val_path), config=cfg)
    print(f"[sft_primer] final_lora={final_lora}", flush=True)

    _smoke_generate(cfg.MODEL_PATH, final_lora)


if __name__ == "__main__":
    main()
