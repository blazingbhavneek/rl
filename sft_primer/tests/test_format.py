import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from sft_primer.tests.test_sft import MODEL_PATH
except ImportError:
    from test_sft import MODEL_PATH

SYSTEM_PROMPT = "<|think|>\nYou are a careful assistant. Think before answering."
USER_PROMPT = "Explain the exact chat format you are using right now."


def main() -> None:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).eval()
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    batch = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.inference_mode():
        out = model.generate(**batch, max_new_tokens=1000, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    raw = tokenizer.decode(out[0][batch["input_ids"].shape[1]:], skip_special_tokens=False)
    full = tokenizer.decode(out[0], skip_special_tokens=False)
    print("MESSAGES:\n", json.dumps(messages, indent=2, ensure_ascii=False))
    print("\nPROMPT:\n", prompt)
    print("\nRAW RESPONSE:\n", raw)
    print("\nFULL TEXT:\n", full)


if __name__ == "__main__":
    main()
