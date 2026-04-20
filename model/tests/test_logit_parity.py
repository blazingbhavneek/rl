import gc
import time
import warnings

import torch

from model.config import ModelConfig
from model.gemma4 import Gemma4Model

warnings.filterwarnings(
    "ignore",
    message=r".*Dynamo detected a call to a `functools\.lru_cache`-wrapped function.*",
    category=UserWarning,
)

DEFAULT_GEMMA4_MODEL_PATH = (
    "/media/blazingbhavneek/Common/Code/sglangServer/Infer/google/gemma-4-E2B-it"
)
DEFAULT_GEMMA4_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _build_test_config(
    lora_targets: list[str], *, chunk_size: int, use_grad_checkpoint: bool
) -> ModelConfig:
    return ModelConfig(
        lora=lora_targets,
        lora_fraction=0.25,
        lora_rank=128,
        lora_alpha=256,
        chunk_size=chunk_size,
        cuda_device_index=0,
        use_grad_checkpoint=use_grad_checkpoint,
    )


def run_logit_parity(model_path: str, model_cls: type, lora_targets: list[str]) -> None:
    t0 = time.perf_counter()

    print("[parity] building config")
    config = _build_test_config(
        lora_targets,
        chunk_size=5,
        use_grad_checkpoint=False,
    )

    print("[parity] loading custom model")
    t_load0 = time.perf_counter()
    custom_model = model_cls(model_path=model_path, config=config)
    custom_model.model.eval()
    print(f"[parity] custom model loaded in {time.perf_counter() - t_load0:.2f}s")

    print("[parity] loading tokenizer")
    tokenizer = custom_model.tokenizer

    print("[parity] building test message batch")
    messages_batch = [
        [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Say hello in one word."},
        ],
        [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Write a small essay on AI"},
        ],
    ]

    print("[parity] applying chat template")
    prompts = [
        tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=True,
        )
        for convo in messages_batch
    ]
    print("[parity] tokenizing + padding")
    tokenized = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
    )
    model_device = next(custom_model.model.parameters()).device
    input_ids = tokenized["input_ids"].to(model_device)
    attention_mask = tokenized["attention_mask"].to(model_device)
    print(
        f"[parity] input_ids shape={tuple(input_ids.shape)} "
        f"attention_mask shape={tuple(attention_mask.shape)}"
    )

    split_layer = int(getattr(custom_model, "_prefix_split_layer", 0))
    layer_idx = max(0, split_layer - 1)
    print(f"[parity] split_layer={split_layer} hook_layer={layer_idx}")

    hook_cache: dict[str, torch.Tensor] = {}

    def _capture_boundary_output(_module, _inputs, output):
        hook_cache["hidden"] = (
            output[0] if isinstance(output, tuple) else output
        ).detach()

    hook = custom_model._layers[layer_idx].register_forward_hook(
        _capture_boundary_output
    )

    print("[parity] running HF-style forward on loaded model")
    t_hf = time.perf_counter()
    with torch.inference_mode():
        # HF-style reference path on the same loaded model instance.
        hf_logits = custom_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits
    hook.remove()
    print(f"[parity] hf-style forward done in {time.perf_counter() - t_hf:.2f}s")

    with torch.inference_mode():
        prefix_bundle = custom_model._build_prefix_bundle(input_ids, attention_mask)
        custom_prefix_hidden = prefix_bundle.hidden_prefix

    if "hidden" in hook_cache:
        hf_boundary_hidden = hook_cache["hidden"]
        boundary_max_abs_diff = (
            (hf_boundary_hidden.float() - custom_prefix_hidden.float())
            .abs()
            .max()
            .item()
        )
        print(
            f"[parity] boundary hidden shape={tuple(custom_prefix_hidden.shape)} "
            f"boundary_max_abs_diff={boundary_max_abs_diff:.6f}"
        )
    else:
        print("[parity] warning: boundary hook did not capture hidden output")

    print("[parity] running custom forward (full lm_head)")
    t_custom = time.perf_counter()
    with torch.inference_mode():
        custom_logits_full = custom_model.forward(messages_batch)
    print(f"[parity] custom full forward done in {time.perf_counter() - t_custom:.2f}s")

    assert hf_logits.shape == custom_logits_full.shape
    max_abs_diff = (hf_logits.float() - custom_logits_full.float()).abs().max().item()
    print(
        f"[parity] full logits shape={tuple(custom_logits_full.shape)} "
        f"full_max_abs_diff={max_abs_diff:.6f}"
    )

    print("[parity] running custom forward (chunked lm_head)")
    custom_model.chunk_size = 64
    t_chunk = time.perf_counter()
    with torch.inference_mode():
        custom_logits_chunked = custom_model.forward(messages_batch)
    print(
        f"[parity] custom chunked forward done in {time.perf_counter() - t_chunk:.2f}s"
    )

    chunked_vs_hf = (
        (hf_logits.float() - custom_logits_chunked.float()).abs().max().item()
    )
    chunked_vs_full = (
        (custom_logits_full.float() - custom_logits_chunked.float()).abs().max().item()
    )
    print(
        f"[parity] chunked logits shape={tuple(custom_logits_chunked.shape)} "
        f"chunked_vs_hf_max_abs_diff={chunked_vs_hf:.6f} "
        f"chunked_vs_full_max_abs_diff={chunked_vs_full:.6f}"
    )
    print(f"[parity] total elapsed={time.perf_counter() - t0:.2f}s")
    assert max_abs_diff < 1e-2, f"logit parity failed: max_abs_diff={max_abs_diff}"
    assert (
        chunked_vs_hf < 1e-2
    ), f"chunked hf parity failed: max_abs_diff={chunked_vs_hf}"
    assert (
        chunked_vs_full < 1e-2
    ), f"chunked full parity failed: max_abs_diff={chunked_vs_full}"


if __name__ == "__main__":
    run_logit_parity(
        DEFAULT_GEMMA4_MODEL_PATH,
        Gemma4Model,
        DEFAULT_GEMMA4_LORA_TARGETS,
    )
    print("PASS: gemma4 parity")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
