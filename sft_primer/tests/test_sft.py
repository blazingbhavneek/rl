from __future__ import annotations

import asyncio
import json
from pathlib import Path

import torch
from peft import PeftModel
from tqdm.asyncio import tqdm as async_tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from inference.vllm_engine import VLLMEngine
from sft_primer.train import (
    detect_model_family,
    generate_qa_for_chunk_with_lookbehind,
    process_input_folder_to_chunks,
    train_model_on_qa_pairs,
)

# ---------------------------------------------------------------------------
# Config — edit these before running
# ---------------------------------------------------------------------------

INPUT_FOLDER_PATH      = Path("/media/blazingbhavneek/Common/Code/rl/sft_primer/input/intel")
MODEL_PATH             = "/media/blazingbhavneek/Common/Code/sglangServer/Infer/Qwen/Qwen3-1.7B"

QA_PORT                = 30000
CHUNK_CONCURRENCY      = 10
TRAIN_EPOCHS           = 15
TRAIN_LR               = 2e-4
TRAIN_BATCH_SIZE       = 4
GRAD_ACCUM_STEPS       = 8
TRAIN_ON_REASONING     = False
INFER_MAX_NEW_TOKENS   = 8192
GPU_MEMORY_UTILIZATION = 0.90
SAVE_EVERY_STEPS       = 50           # save LoRA every N optimizer steps; 0 = disabled

# ---------------------------------------------------------------------------
# Hardcoded QA sample — used to sanity-check the pipeline without chunking.
# Source: LLVM Language Reference — exception handling instructions.
# ---------------------------------------------------------------------------

LLVM_QA_SAMPLE = {
    "question": (
        "What is the purpose of the catchswitch instruction in LLVM IR, "
        "and what constraint does it impose on its position within a basic block?"
    ),
    "reasoning": (
        "The question asks about two things: the purpose of catchswitch and its "
        "positional constraint. "
        "From the LLVM reference: catchswitch is used by LLVM's exception handling "
        "system to describe the set of possible catch handlers that may be executed "
        "by the EH personality routine. "
        "Regarding its position: the reference states it is both a terminator and a "
        "'pad' instruction, meaning it must be both the first non-phi instruction "
        "and last instruction in the basic block — therefore it must be the only "
        "non-phi instruction in the block. "
        "Conclusion: its purpose is EH handler dispatch, and it must be the sole "
        "non-phi instruction in its basic block."
    ),
    "answer": (
        "The catchswitch instruction is used by LLVM's exception handling system to "
        "describe the set of possible catch handlers that may be executed by the EH "
        "personality routine. It transfers control to one of the listed handler "
        "successors or continues unwinding via the unwind label. "
        "Because it is both a terminator and a pad instruction, it must be the first "
        "non-phi instruction and the last instruction in its basic block — making it "
        "the only non-phi instruction in that block."
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_qa_pairs(path: Path) -> list[dict]:
    pairs = []
    with path.open("r", encoding="utf-8") as f:
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
                pairs.append(row)
    return pairs


def _infer(model, tokenizer, question: str, model_family: str) -> str:
    """Run greedy inference for a single question, return decoded answer text."""
    messages = [{"role": "user", "content": question}]
    enable_thinking = model_family in ("qwen3", "qwen3_5")
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    tok = tokenizer([prompt], return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = tok["input_ids"].to(device)
    attention_mask = tok["attention_mask"].to(device)

    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=INFER_MAX_NEW_TOKENS,
            do_sample=False,
        )
    return tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True).strip()


def _compare_base_vs_lora(adapter_dir: str) -> None:
    """Load base model and base+LoRA, run LLVM question on both, print comparison."""
    model_family = detect_model_family(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print("loading_base_model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map={"": "cuda:0"},
        torch_dtype=torch.bfloat16,
    )
    base_model.eval()

    print("--- base model answer ---")
    base_answer = _infer(base_model, tokenizer, LLVM_QA_SAMPLE["question"], model_family)
    print(base_answer)

    print("\nloading_lora_adapter...")
    lora_model = PeftModel.from_pretrained(base_model, adapter_dir)
    lora_model.eval()

    print("--- lora model answer ---")
    lora_answer = _infer(lora_model, tokenizer, LLVM_QA_SAMPLE["question"], model_family)
    print(lora_answer)

    print("\n--- question ---")
    print(LLVM_QA_SAMPLE["question"])
    print("\n--- expected answer ---")
    print(LLVM_QA_SAMPLE["answer"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def _run() -> None:
    input_root = str(INPUT_FOLDER_PATH.parent)
    output_root = str(INPUT_FOLDER_PATH.parent.parent / "output")
    input_folder = INPUT_FOLDER_PATH.name
    model_family = detect_model_family(MODEL_PATH)

    print("=== SFT Primer Test ===")
    print(f"input_folder:          {INPUT_FOLDER_PATH}")
    print(f"model_path:            {MODEL_PATH}")
    print(f"model_family:          {model_family}")
    print(f"train_on_reasoning:    {TRAIN_ON_REASONING}")
    print(f"train_epochs:          {TRAIN_EPOCHS}")
    print(f"train_lr:              {TRAIN_LR}")
    print(f"train_batch_size:      {TRAIN_BATCH_SIZE}")
    print(f"grad_accum_steps:      {GRAD_ACCUM_STEPS}")
    print(f"save_every_steps:      {SAVE_EVERY_STEPS}")
    print(f"qa_port:               {QA_PORT}")

    if not INPUT_FOLDER_PATH.exists() or not INPUT_FOLDER_PATH.is_dir():
        raise FileNotFoundError(f"Input folder not found: {INPUT_FOLDER_PATH}")

    # --- check if final LoRA already exists; if so, skip training and compare ---
    model_key = (
        MODEL_PATH.strip("/")
        .replace("\\", "_").replace("/", "_").replace(":", "_")
    )
    final_lora_dir = Path(output_root) / input_folder / "lora" / model_key / "final"
    if final_lora_dir.exists():
        print(f"\nfinal_lora_found: {final_lora_dir}")
        print("skipping_training — running base vs lora comparison instead\n")
        _compare_base_vs_lora(str(final_lora_dir))
        return

    # --- chunks ---
    out_path = Path(output_root) / input_folder / "chunks.json"
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"chunks_checkpoint_found: {out_path}")
    else:
        out_path = process_input_folder_to_chunks(
            input_folder=input_folder,
            input_root=input_root,
            output_root=output_root,
        )
    chunks = json.loads(Path(out_path).read_text(encoding="utf-8"))
    chunk_texts = [
        str(r.get("text", "")).strip()
        for r in chunks
        if str(r.get("text", "")).strip()
    ]
    print(f"chunks_count: {len(chunk_texts)}")
    if not chunk_texts:
        raise ValueError("No non-empty chunks found.")

    # --- QA pairs ---
    qa_pairs_path = Path(output_root) / input_folder / "qa_pair.json"
    qa_pairs = (
        _load_qa_pairs(qa_pairs_path)
        if qa_pairs_path.exists() and qa_pairs_path.stat().st_size > 0
        else []
    )

    if qa_pairs:
        print(f"qa_checkpoint_found: {qa_pairs_path} ({len(qa_pairs)} pairs)")
    else:
        engine = VLLMEngine(
            model_path=MODEL_PATH,
            engine_kwargs={
                "base_url": f"http://127.0.0.1:{QA_PORT}/v1",
                "model_name": MODEL_PATH,
                "api_key": "",
                "save_vllm_logs": True,
                "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
                "reasoning_parser": "qwen3",
                "enable_auto_tool_choice": True,
                "tool_call_parser": "hermes",
            },
        )
        try:
            print("starting_qa_engine...")
            await engine.start()
            await engine.init()
            print("qa_engine_ready")

            sem = asyncio.Semaphore(max(1, CHUNK_CONCURRENCY))

            async def _run_chunk(i: int) -> tuple[int, list[dict]]:
                async with sem:
                    try:
                        rows = await generate_qa_for_chunk_with_lookbehind(
                            chunks=chunk_texts,
                            chunk_index=i,
                            lookbehind_chunks=2,
                            num_questions=5,
                            model_path=MODEL_PATH,
                            port=QA_PORT,
                        )
                    except Exception:
                        return i, []
                return i, [
                    r for r in rows
                    if isinstance(r, dict)
                    and str(r.get("question", "")).strip()
                    and str(r.get("answer", "")).strip()
                    and str(r.get("reasoning", "")).strip()
                ]

            tasks = [asyncio.create_task(_run_chunk(i)) for i in range(len(chunk_texts))]
            qa_pairs_path.parent.mkdir(parents=True, exist_ok=True)
            with qa_pairs_path.open("w", encoding="utf-8") as f:
                with async_tqdm(total=len(tasks), desc="Generating QA", unit="chunk") as pbar:
                    for fut in asyncio.as_completed(tasks):
                        _, rows = await fut
                        for row in rows:
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        f.flush()
                        pbar.update(1)
        finally:
            print("shutting_down_qa_engine...")
            await engine.shutdown()

        qa_pairs = _load_qa_pairs(qa_pairs_path)

    if not qa_pairs:
        raise ValueError("No QA pairs could be created from chunks.")

    qa_pairs.insert(0, LLVM_QA_SAMPLE)
    print(f"qa_pairs_count: {len(qa_pairs)} (includes 1 hardcoded LLVM sample)")

    # --- training ---
    print(f"starting_lora_training: epochs={TRAIN_EPOCHS} train_on_reasoning={TRAIN_ON_REASONING}")
    adapter_dir = train_model_on_qa_pairs(
        qa_pairs=qa_pairs,
        train_model_path=MODEL_PATH,
        output_dir=str(Path(output_root) / input_folder),
        epochs=TRAIN_EPOCHS,
        lr=TRAIN_LR,
        train_batch_size=TRAIN_BATCH_SIZE,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        train_on_reasoning=TRAIN_ON_REASONING,
        save_every_steps=SAVE_EVERY_STEPS,
    )
    print(f"adapter_dir: {adapter_dir}")

    # --- compare base vs trained LoRA ---
    print("\nrunning_base_vs_lora_comparison...")
    _compare_base_vs_lora(adapter_dir)

    print("\nPASS: sft_primer chunk + lora train + inference test")


if __name__ == "__main__":
    asyncio.run(_run())
