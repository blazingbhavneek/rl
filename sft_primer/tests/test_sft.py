import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from sft_primer.train import detect_model_family, train_model_on_qa_pairs

# ---------------------------------------------------------------------------
# Config — edit these before running
# ---------------------------------------------------------------------------

INPUT_FOLDER_PATH = Path("/media/blazingbhavneek/Common/Code/rl/sft_primer/input/intel")
MODEL_PATH = (
    "/media/blazingbhavneek/Common/Code/sglangServer/Infer/google/gemma-4-E2B-it"
)

TRAIN_EPOCHS = 5
TRAIN_LR = 2e-4
TRAIN_BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
TRAIN_CHUNK_SIZE = 256
TRAIN_LOGPROB_CHUNK = 128
TRAIN_TOKEN_CHUNK_SIZE = 128
OFFLOAD_PREFIX_TO_CPU = False
MAX_SAMPLE_TOKENS = 2048
TRAIN_ON_REASONING = True
INFER_MAX_NEW_TOKENS = 8192
GPU_MEMORY_UTILIZATION = 0.90
SAVE_EVERY_STEPS = 50  # save LoRA every N optimizer steps; 0 = disabled
QA_PAIRS_PATH = Path(
    "/media/blazingbhavneek/Common/Code/rl/sft_primer/output/intel/qa_pair.jsonl"
)

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
            if isinstance(row, dict):
                if (
                    "messages" in row
                    and isinstance(row["messages"], list)
                    and len(row["messages"]) >= 2
                ):
                    pairs.append(row)
                elif (
                    str(row.get("question", "")).strip()
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
            messages,
            tokenize=False,
            add_generation_prompt=True,
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
    return tokenizer.decode(
        out[0, input_ids.shape[1] :], skip_special_tokens=True
    ).strip()


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
    base_answer = _infer(
        base_model, tokenizer, LLVM_QA_SAMPLE["question"], model_family
    )
    print(base_answer)

    print("\nloading_lora_adapter...")
    lora_model = PeftModel.from_pretrained(base_model, adapter_dir)
    lora_model.eval()

    print("--- lora model answer ---")
    lora_answer = _infer(
        lora_model, tokenizer, LLVM_QA_SAMPLE["question"], model_family
    )
    print(lora_answer)

    print("\n--- question ---")
    print(LLVM_QA_SAMPLE["question"])
    print("\n--- expected answer ---")
    print(LLVM_QA_SAMPLE["answer"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _run() -> None:
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
    print(f"train_chunk_size:      {TRAIN_CHUNK_SIZE}")
    print(f"logprob_chunk_size:    {TRAIN_LOGPROB_CHUNK}")
    print(f"train_token_chunk:     {TRAIN_TOKEN_CHUNK_SIZE}")
    print(f"offload_prefix_cpu:    {OFFLOAD_PREFIX_TO_CPU}")
    print(f"max_sample_tokens:     {MAX_SAMPLE_TOKENS}")
    print(f"save_every_steps:      {SAVE_EVERY_STEPS}")
    print(f"qa_pairs_path:         {QA_PAIRS_PATH}")

    if not INPUT_FOLDER_PATH.exists() or not INPUT_FOLDER_PATH.is_dir():
        raise FileNotFoundError(f"Input folder not found: {INPUT_FOLDER_PATH}")

    # --- check if final LoRA already exists; if so, skip training and compare ---
    model_key = (
        MODEL_PATH.strip("/").replace("\\", "_").replace("/", "_").replace(":", "_")
    )
    final_lora_dir = Path(output_root) / input_folder / "lora" / model_key / "final"
    if final_lora_dir.exists():
        print(f"\nfinal_lora_found: {final_lora_dir}")
        print("skipping_training — running base vs lora comparison instead\n")
        _compare_base_vs_lora(str(final_lora_dir))
        return

    qa_pairs = (
        _load_qa_pairs(QA_PAIRS_PATH)
        if QA_PAIRS_PATH.exists() and QA_PAIRS_PATH.stat().st_size > 0
        else []
    )
    if not qa_pairs:
        raise ValueError(f"No QA pairs found at {QA_PAIRS_PATH}")

    print(f"qa_pairs_found:        {QA_PAIRS_PATH} ({len(qa_pairs)} pairs)")
    qa_pairs.insert(0, LLVM_QA_SAMPLE)
    print(f"qa_pairs_count: {len(qa_pairs)} (includes 1 hardcoded LLVM sample)")

    # --- training ---
    print(
        f"starting_lora_training: epochs={TRAIN_EPOCHS} train_on_reasoning={TRAIN_ON_REASONING}"
    )
    adapter_dir = train_model_on_qa_pairs(
        qa_pairs=qa_pairs,
        train_model_path=MODEL_PATH,
        output_dir=str(Path(output_root) / input_folder),
        epochs=TRAIN_EPOCHS,
        lr=TRAIN_LR,
        train_batch_size=TRAIN_BATCH_SIZE,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        chunk_size=TRAIN_CHUNK_SIZE,
        logprob_chunk_size=TRAIN_LOGPROB_CHUNK,
        token_chunk_size=TRAIN_TOKEN_CHUNK_SIZE,
        offload_prefix_to_cpu=OFFLOAD_PREFIX_TO_CPU,
        max_sample_tokens=MAX_SAMPLE_TOKENS,
        train_on_reasoning=TRAIN_ON_REASONING,
        save_every_steps=SAVE_EVERY_STEPS,
    )
    print(f"adapter_dir: {adapter_dir}")

    # --- compare base vs trained LoRA ---
    print("\nrunning_base_vs_lora_comparison...")
    _compare_base_vs_lora(adapter_dir)

    print("\nPASS: sft_primer chunk + lora train + inference test")


if __name__ == "__main__":
    _run()
