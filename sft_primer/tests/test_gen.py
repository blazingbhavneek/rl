import gc

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = (
    "/media/blazingbhavneek/Common/Code/sglangServer/Infer/google/gemma-4-E2B-it"
)
ADAPTER_PATH = "/media/blazingbhavneek/Common/Code/rl/sft_primer/output/intel/lora/media_blazingbhavneek_Common_Code_sglangServer_Infer_google_gemma-4-E2B-it/epoch_1/"
PROMPT = "Explain what LLVM is in two short bullet points."
MAX_NEW_TOKENS = 128


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
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def _load_hf_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map={"": device},
        dtype=dtype,
    ).eval()


def _load_hf_lora_model():
    base_model = _load_hf_model()
    return PeftModel.from_pretrained(base_model, ADAPTER_PATH).eval()


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
        output_ids[0, input_ids.shape[1] :],
        skip_special_tokens=False,
    )
    print(f"\n--- {label} ---")
    print(f"new_text:  {new_text!r}")
    return output_ids


def _run() -> None:
    print("=== Gemma HF+LoRA Generation Test ===")
    print(f"model_path:   {MODEL_PATH}")
    print(f"adapter_path: {ADAPTER_PATH}")
    print(f"prompt:       {PROMPT}")

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

    print("\nloading_hf_lora_model...")
    lora_model = _load_hf_lora_model()
    lora_first = _generate(lora_model, tokenizer, prompt_text, "hf_lora_run_1")
    lora_second = _generate(lora_model, tokenizer, prompt_text, "hf_lora_run_2")
    if not torch.equal(lora_first, lora_second):
        raise AssertionError("HF+LoRA greedy generation is not deterministic")

    print("\n--- comparison ---")
    print(f"hf_repeatable: {torch.equal(hf_first, hf_second)}")
    print(f"hf_lora_repeatable: {torch.equal(lora_first, lora_second)}")
    print(f"hf_vs_hf_lora_same: {torch.equal(hf_first, lora_first)}")

    print("\nPASS: gemma hf+lora generation test")


if __name__ == "__main__":
    _run()
