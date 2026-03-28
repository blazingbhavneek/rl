import time
import warnings
import gc
import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)

from model.config import ModelConfig
from model.gptoss import GptOssModel
from model.qwen3 import Qwen3Model
from model.qwen3_5 import Qwen3_5Model

warnings.filterwarnings(
    "ignore",
    message=r".*Dynamo detected a call to a `functools\.lru_cache`-wrapped function.*",
    category=UserWarning,
)


def _collect_peft_grads(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    grads: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            grads[name] = torch.zeros_like(param, dtype=torch.float32)
        else:
            grads[name] = param.grad.detach().float().clone()
    return grads


def _zero_all_grads(model: torch.nn.Module) -> None:
    for p in model.parameters():
        p.grad = None


def _force_qwen_torch_linear_attention_kernels(model: torch.nn.Module) -> None:
    base = model.base_model.model if hasattr(model, "base_model") else model
    inner = base.model if hasattr(base, "model") else base
    if hasattr(inner, "language_model"):
        inner = inner.language_model
    if not hasattr(inner, "layers"):
        return
    for layer in inner.layers:
        if hasattr(layer, "linear_attn"):
            layer.linear_attn.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
            layer.linear_attn.recurrent_gated_delta_rule = (
                torch_recurrent_gated_delta_rule
            )


def run_grad_parity(
    model_path: str, model_cls: type, lora_targets: list[str]
) -> None:
    t0 = time.perf_counter()

    print("[grad-parity] building config")
    seed = 1337
    config = ModelConfig(
        lora=lora_targets,
        lora_fraction=0.25,
        lora_rank=128,
        lora_alpha=256,
        chunk_size=10,
        cuda_device_index=0,
        use_grad_checkpoint=True,
    )
    # LoRA-only parity thresholds (BF16, chunked path):
    # We compare adapter grads only, so tiny near-zero entries can inflate relative error.
    # Keep abs + rel_l2 as primary stability checks; masked relative is secondary.
    abs_tolerance = 5e-4
    rel_l2_tolerance = 1e-2
    rel_tolerance = 0.1
    epsilon = 1e-10
    rel_ref_floor = 1e-3

    print("[grad-parity] loading baseline model")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    t_load0 = time.perf_counter()
    baseline_model = model_cls(model_path=model_path, config=config)
    baseline_model.model.eval()
    if model_cls is Qwen3_5Model:
        _force_qwen_torch_linear_attention_kernels(baseline_model.model)
    print(f"[grad-parity] baseline model loaded in {time.perf_counter() - t_load0:.2f}s")

    print("[grad-parity] loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    messages_batch = [
        [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Say hello in one word."},
        ],
    ]
    completion_texts = ["hello"]

    print("[grad-parity] building prompt/completion tensors")
    prompts = [
        tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=True,
        )
        for convo in messages_batch
    ]
    prompt_tok = tokenizer(prompts, return_tensors="pt", padding=True)
    completion_tok = tokenizer(
        completion_texts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )

    model_device = next(baseline_model.model.parameters()).device
    prompt_ids = prompt_tok["input_ids"].to(model_device)
    prompt_mask = prompt_tok["attention_mask"].to(model_device)
    completion_ids = completion_tok["input_ids"].to(model_device)
    completion_mask = completion_tok["attention_mask"].to(
        model_device, dtype=torch.float32
    )

    # Allow single prompt with completion batch if needed.
    if prompt_ids.shape[0] == 1 and completion_ids.shape[0] > 1:
        prompt_ids = prompt_ids.expand(completion_ids.shape[0], prompt_ids.shape[1])
        prompt_mask = prompt_mask.expand(completion_ids.shape[0], prompt_mask.shape[1])

    full_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    full_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    completion_len = completion_ids.shape[1]

    print("[grad-parity] HF baseline backward")
    _zero_all_grads(baseline_model.model)
    out = baseline_model.model(input_ids=full_ids, attention_mask=full_mask).logits
    logits_comp = out[:, -completion_len:, :]
    token_logprobs = (
        torch.log_softmax(logits_comp.float(), dim=-1)
        .gather(
            dim=-1,
            index=completion_ids.unsqueeze(-1),
        )
        .squeeze(-1)
    )
    hf_loss = -(
        (token_logprobs * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
    )
    hf_loss.backward()
    hf_lora_grads = _collect_peft_grads(baseline_model.model)

    # Reuse the same loaded model for custom backward.
    # This avoids a second 20B model load spike on smaller GPUs.
    # Validity: we do not call optimizer.step(), so weights are unchanged.
    custom_model = baseline_model
    _zero_all_grads(custom_model.model)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Custom path, but with logits-based backward to match HF loss construction exactly.
    with torch.inference_mode():
        hidden_prefix, pos_ids = custom_model._forward_prefix(full_ids, full_mask)
    hidden_for_suffix = hidden_prefix.clone().detach().requires_grad_(True)
    hidden_suffix = custom_model._forward_suffix(
        hidden_for_suffix,
        pos_ids.clone().detach(),
        full_mask,
    )
    custom_logits = custom_model._lm_head_logits_chunked(hidden_suffix)
    custom_logits_comp = custom_logits[:, -completion_len:, :]
    custom_token_logprobs = (
        torch.log_softmax(custom_logits_comp.float(), dim=-1)
        .gather(
            dim=-1,
            index=completion_ids.unsqueeze(-1),
        )
        .squeeze(-1)
    )
    custom_loss_t = -(
        (custom_token_logprobs * completion_mask).sum()
        / completion_mask.sum().clamp(min=1.0)
    )
    custom_loss_t.backward()
    custom_stats = {
        "loss": float(custom_loss_t.item()),
    }
    custom_lora_grads = _collect_peft_grads(custom_model.model)

    hf_keys = sorted(hf_lora_grads.keys())
    custom_keys = sorted(custom_lora_grads.keys())
    assert hf_keys == custom_keys, "LoRA grad parameter sets differ"

    max_abs_diff = 0.0
    mean_abs_diff = 0.0
    mean_rel_diff = 0.0
    mean_rel_diff_masked = 0.0
    sq_diff = 0.0
    sq_ref = 0.0
    n = 0
    eps = epsilon
    for name in hf_keys:
        ref = hf_lora_grads[name]
        cur = custom_lora_grads[name]
        abs_diff = (ref - cur).abs()
        rel_diff = abs_diff / (ref.abs() + eps)
        mask = ref.abs() > rel_ref_floor
        if mask.any():
            rel_diff_masked = (abs_diff[mask] / (ref.abs()[mask] + eps)).mean().item()
        else:
            rel_diff_masked = 0.0
        max_abs_diff = max(max_abs_diff, float(abs_diff.max().item()))
        mean_abs_diff += float(abs_diff.mean().item())
        mean_rel_diff += float(rel_diff.mean().item())
        mean_rel_diff_masked += float(rel_diff_masked)
        sq_diff += float(((ref - cur) ** 2).sum().item())
        sq_ref += float((ref**2).sum().item())
        n += 1

    mean_abs_diff = mean_abs_diff / max(1, n)
    mean_rel_diff = mean_rel_diff / max(1, n)
    mean_rel_diff_masked = mean_rel_diff_masked / max(1, n)
    rel_l2 = (sq_diff ** 0.5) / max(1e-12, sq_ref ** 0.5)

    print(
        f"[grad-parity] peft_params={n} "
        f"hf_loss={hf_loss.item():.6f} custom_loss={custom_stats['loss']:.6f} "
        f"max_abs_diff={max_abs_diff:.6f} mean_abs_diff={mean_abs_diff:.6f} "
        f"mean_rel_diff={mean_rel_diff:.6f} "
        f"mean_rel_diff_masked={mean_rel_diff_masked:.6f} "
        f"rel_l2={rel_l2:.6e}"
    )
    print(f"[grad-parity] total elapsed={time.perf_counter() - t0:.2f}s")

    # PEFT parity: use stable metrics across all trainable adapter params.
    assert mean_abs_diff < abs_tolerance, f"PEFT grad parity failed: mean_abs_diff={mean_abs_diff}"
    assert rel_l2 < rel_l2_tolerance, f"PEFT grad parity failed: rel_l2={rel_l2}"
    assert mean_rel_diff_masked < rel_tolerance, f"PEFT grad parity failed: mean_rel_diff_masked={mean_rel_diff_masked}"
    del baseline_model
    del custom_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # run_grad_parity(
    #     "/media/blazingbhavneek/Common/Code/sglangServer/Infer/openai/gpt-oss-20b",
    #     GptOssModel,
    #     ["q_proj", "k_proj", "v_proj", "o_proj"],
    # )
    # print("PASS: gpt-oss grad parity")
    # gc.collect()
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()
    run_grad_parity(
        "/media/blazingbhavneek/Common/Code/sglangServer/Infer/Qwen/Qwen3-1.7B",
        Qwen3Model,
        ["gate_proj", "up_proj", "down_proj"],
    )
    print("PASS: qwen3 grad parity")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("PASS: run_grad_parity")
