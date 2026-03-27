import time
import warnings

import torch
from transformers import AutoTokenizer

from model.config import ModelConfig
from model.gptoss import GptOssModel

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


def _peft_param_type(name: str) -> str:
    if ".lora_A." in name:
        return "lora_A"
    if ".lora_B." in name:
        return "lora_B"
    if "lora_embedding_A" in name:
        return "lora_embedding_A"
    if "lora_embedding_B" in name:
        return "lora_embedding_B"
    if "modules_to_save" in name:
        return "modules_to_save"
    return "other_peft"


def test_gpt_oss_grad_parity_hf_vs_custom_backward() -> None:
    t0 = time.perf_counter()
    model_path = (
        "/media/blazingbhavneek/Common/Code/sglangServer/Infer/openai/gpt-oss-20b"
    )

    print("[grad-parity] building config")
    seed = 1337
    config = ModelConfig(
        lora=["q_proj", "k_proj", "v_proj", "o_proj"],
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
    baseline_model = GptOssModel(model_path=model_path, config=config)
    baseline_model.model.eval()
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

    sampled_abs_diffs = []
    global_abs_sum = 0.0
    global_abs_numel = 0
    global_abs_max = 0.0
    type_stats: dict[str, dict[str, float]] = {}
    for name in hf_keys:
        abs_diff = (hf_lora_grads[name] - custom_lora_grads[name]).abs()
        flat = abs_diff.reshape(-1)
        global_abs_sum += float(flat.sum().item())
        global_abs_numel += int(flat.numel())
        global_abs_max = max(global_abs_max, float(flat.max().item()))

        # Keep percentile estimation cheap on very large parameter sets.
        if flat.numel() > 8192:
            step = max(1, flat.numel() // 8192)
            sampled_abs_diffs.append(flat[::step])
        else:
            sampled_abs_diffs.append(flat)

        ptype = _peft_param_type(name)
        bucket = type_stats.setdefault(
            ptype,
            {
                "sum_abs": 0.0,
                "numel": 0.0,
                "max_abs": 0.0,
                "sum_rel_masked": 0.0,
                "count_params": 0.0,
            },
        )
        mask = hf_lora_grads[name].abs() > rel_ref_floor
        if mask.any():
            rel_masked = (
                abs_diff[mask] / (hf_lora_grads[name].abs()[mask] + eps)
            ).mean().item()
        else:
            rel_masked = 0.0
        bucket["sum_abs"] += float(flat.sum().item())
        bucket["numel"] += float(flat.numel())
        bucket["max_abs"] = max(bucket["max_abs"], float(flat.max().item()))
        bucket["sum_rel_masked"] += float(rel_masked)
        bucket["count_params"] += 1.0

    flat_abs_sample = (
        torch.cat(sampled_abs_diffs, dim=0)
        if sampled_abs_diffs
        else torch.zeros(1)
    )
    p50 = torch.quantile(flat_abs_sample, 0.50).item()
    p90 = torch.quantile(flat_abs_sample, 0.90).item()
    p99 = torch.quantile(flat_abs_sample, 0.99).item()
    p999 = torch.quantile(flat_abs_sample, 0.999).item()
    global_abs_mean = global_abs_sum / max(1, global_abs_numel)

    print(
        f"[grad-parity] peft_params={n} "
        f"hf_loss={hf_loss.item():.6f} custom_loss={custom_stats['loss']:.6f} "
        f"max_abs_diff={max_abs_diff:.6f} mean_abs_diff={mean_abs_diff:.6f} "
        f"mean_rel_diff={mean_rel_diff:.6f} "
        f"mean_rel_diff_masked={mean_rel_diff_masked:.6f} "
        f"rel_l2={rel_l2:.6e}"
    )
    print(
        f"[grad-parity] peft abs-diff distribution: "
        f"numel={global_abs_numel} mean={global_abs_mean:.6f} "
        f"p50~={p50:.6f} p90~={p90:.6f} p99~={p99:.6f} p999~={p999:.6f} "
        f"max={global_abs_max:.6f} sample_numel={flat_abs_sample.numel()}"
    )
    print("[grad-parity] PEFT type stats (all layers):")
    for ptype in sorted(type_stats.keys()):
        b = type_stats[ptype]
        type_mean_abs = b["sum_abs"] / max(1.0, b["numel"])
        type_mean_rel_masked = b["sum_rel_masked"] / max(1.0, b["count_params"])
        print(
            f"[grad-parity]   {ptype}: "
            f"params={int(b['count_params'])} numel={int(b['numel'])} "
            f"mean_abs={type_mean_abs:.6f} max_abs={b['max_abs']:.6f} "
            f"mean_rel_masked={type_mean_rel_masked:.6f}"
        )
    print(f"[grad-parity] total elapsed={time.perf_counter() - t0:.2f}s")

    # PEFT parity: use stable metrics across all trainable adapter params.
    assert mean_abs_diff < abs_tolerance, f"PEFT grad parity failed: mean_abs_diff={mean_abs_diff}"
    assert rel_l2 < rel_l2_tolerance, f"PEFT grad parity failed: rel_l2={rel_l2}"
    assert mean_rel_diff_masked < rel_tolerance, f"PEFT grad parity failed: mean_rel_diff_masked={mean_rel_diff_masked}"


if __name__ == "__main__":
    test_gpt_oss_grad_parity_hf_vs_custom_backward()
    print("PASS: test_gpt_oss_grad_parity_hf_vs_custom_backward")
