import os
import gc
import warnings

import torch

from model.config import ModelConfig
from model.gemma4 import Gemma4Model

warnings.filterwarnings(
    "ignore",
    message=r".*Dynamo detected a call to a `functools\.lru_cache`-wrapped function.*",
    category=UserWarning,
)

DEFAULT_GEMMA4_MODEL_PATH = "/media/blazingbhavneek/Common/Code/sglangServer/Infer/google/gemma-4-E2B-it"
DEFAULT_GEMMA4_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _build_test_config(lora_targets: list[str], *, chunk_size: int, use_grad_checkpoint: bool) -> ModelConfig:
    return ModelConfig(
        lora=lora_targets,
        lora_fraction=0.25,
        lora_rank=128,
        lora_alpha=256,
        chunk_size=chunk_size,
        cuda_device_index=0,
        use_grad_checkpoint=use_grad_checkpoint,
    )


def run_greedy_generation_parity(
    model_path: str, model_cls: type, lora_targets: list[str], max_new_tokens: int = 10
) -> None:
    print("[gen-parity] building config")
    config = _build_test_config(
        lora_targets,
        chunk_size=64,
        use_grad_checkpoint=False,
    )

    print("[gen-parity] loading model")
    model = model_cls(model_path=model_path, config=config)
    model.model.eval()

    print("[gen-parity] loading tokenizer")
    tokenizer = model.tokenizer

    messages = [
        [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Give one short greeting."},
        ]
    ]
    prompt = tokenizer.apply_chat_template(
        messages[0], tokenize=False, add_generation_prompt=True
    )
    tok = tokenizer([prompt], return_tensors="pt", padding=True)
    device = next(model.model.parameters()).device
    input_ids = tok["input_ids"].to(device)
    attention_mask = tok["attention_mask"].to(device)

    hf_ids = input_ids.clone()
    hf_mask = attention_mask.clone()
    custom_ids = input_ids.clone()
    custom_mask = attention_mask.clone()

    print(f"[gen-parity] running greedy decode for {max_new_tokens} tokens")
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            hf_logits = model.model(input_ids=hf_ids, attention_mask=hf_mask).logits
            hf_next = hf_logits[:, -1, :].argmax(dim=-1, keepdim=True)
            hf_ids = torch.cat([hf_ids, hf_next], dim=1)
            hf_mask = torch.cat([hf_mask, torch.ones_like(hf_next)], dim=1)

            prefix_hidden, pos_ids, shared_kv_states, per_layer_inputs = model._forward_prefix(
                custom_ids,
                custom_mask,
            )
            suffix_hidden = model._forward_suffix(
                prefix_hidden,
                pos_ids,
                custom_mask,
                shared_kv_states,
                per_layer_inputs,
            )
            custom_logits = model._lm_head_logits_chunked(suffix_hidden)
            custom_next = custom_logits[:, -1, :].argmax(dim=-1, keepdim=True)
            custom_ids = torch.cat([custom_ids, custom_next], dim=1)
            custom_mask = torch.cat([custom_mask, torch.ones_like(custom_next)], dim=1)

    hf_new = hf_ids[:, input_ids.shape[1] :]
    custom_new = custom_ids[:, input_ids.shape[1] :]
    assert torch.equal(
        hf_new, custom_new
    ), "greedy generation parity failed: generated token ids differ"

    hf_text = tokenizer.decode(hf_new[0], skip_special_tokens=True)
    custom_text = tokenizer.decode(custom_new[0], skip_special_tokens=True)
    print(f"[gen-parity] hf_new_tokens={hf_new[0].tolist()}")
    print(f"[gen-parity] custom_new_tokens={custom_new[0].tolist()}")
    print(f"[gen-parity] hf_text={hf_text!r}")
    print(f"[gen-parity] custom_text={custom_text!r}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run_greedy_generation_parity(
        DEFAULT_GEMMA4_MODEL_PATH,
        Gemma4Model,
        DEFAULT_GEMMA4_LORA_TARGETS,
    )
    print("PASS: gemma4 greedy generation parity")
