import gc
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model.config import ModelConfig
from model.gemma4 import Gemma4Model


MODEL_PATH = "/media/blazingbhavneek/Common/Code/sglangServer/Infer/google/gemma-4-E2B-it"
ADAPTER_PATH = Path(
    "/media/blazingbhavneek/Common/Code/rl/sft_primer/output/intel/lora/"
    "media_blazingbhavneek_Common_Code_sglangServer_Infer_google_gemma-4-E2B-it/step_50"
)
PROMPT = "Explain what LLVM is in two short bullet points."
MAX_NEW_TOKENS = 128
LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _build_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except TypeError:
        # Backward compatibility with tokenizers that do not support newer kwargs.
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _load_hf_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map={"": device},
        dtype=dtype,
    ).eval()


def _generate(model, tokenizer, prompt_text: str, label: str):
    batch = tokenizer([prompt_text], return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    new_text = tokenizer.decode(
        output_ids[0, input_ids.shape[1]:],
        skip_special_tokens=False,
    )
    print(f"\n--- {label} ---")
    print(f"new_token_ids: {output_ids[0, input_ids.shape[1]:].tolist()}")
    print(f"full_text: {full_text!r}")
    print(f"new_text:  {new_text!r}")
    return output_ids


def _load_lora_model() -> Gemma4Model:
    config = ModelConfig(
        lora=LORA_TARGETS,
        lora_fraction=0.25,
        lora_rank=128,
        lora_alpha=256,
        chunk_size=64,
        logprob_chunk_size=64,
        token_chunk_size=256,
        offload_prefix_to_cpu=False,
        cuda_device_index=0,
        use_grad_checkpoint=False,
    )
    model = Gemma4Model(model_path=MODEL_PATH, config=config)
    model.model.eval()
    model.load_lora_adapter("trained", str(ADAPTER_PATH), is_trainable=False)
    model.set_active_lora_adapter("trained")
    return model


def _run() -> None:
    print("=== Gemma LoRA Generation Test ===")
    print(f"model_path:   {MODEL_PATH}")
    print(f"adapter_path: {ADAPTER_PATH}")
    print(f"prompt:       {PROMPT}")

    if not ADAPTER_PATH.exists():
        raise FileNotFoundError(f"LoRA adapter not found: {ADAPTER_PATH}")

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_text = _build_prompt(tokenizer, PROMPT)
    print("\n--- prompt_text ---")
    print(repr(prompt_text))

    print("\nloading_hf_model...")
    hf_model = _load_hf_model()
    hf_first = _generate(hf_model, tokenizer, prompt_text, "hf_run_1")
    hf_second = _generate(hf_model, tokenizer, prompt_text, "hf_run_2")
    if not torch.equal(hf_first, hf_second):
        raise AssertionError("HF greedy generation is not deterministic")

    del hf_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nloading_lora_model...")
    lora_model = _load_lora_model()
    lora_ids = _generate(lora_model.model, tokenizer, prompt_text, "lora_run")

    print("\n--- comparison ---")
    print(f"hf_repeatable: {torch.equal(hf_first, hf_second)}")
    print(f"hf_vs_lora_same: {torch.equal(hf_first, lora_ids)}")
    if torch.equal(hf_first, lora_ids):
        print("LoRA did not change the greedy output for this prompt.")
    else:
        print("LoRA changed the greedy output for this prompt.")

    print("\nPASS: gemma lora generation test")


if __name__ == "__main__":
    _run()
