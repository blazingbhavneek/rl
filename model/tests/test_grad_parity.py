import time
import warnings
import gc
import torch

from model.config import ModelConfig
from model.gemma4 import Gemma4Model

try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        torch_chunk_gated_delta_rule,
        torch_recurrent_gated_delta_rule,
    )
except Exception:
    torch_chunk_gated_delta_rule = None
    torch_recurrent_gated_delta_rule = None

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


def _tensor_diff_stats(ref: torch.Tensor, cur: torch.Tensor) -> tuple[float, float, float]:
    ref_f = ref.detach().float()
    cur_f = cur.detach().float()
    diff = ref_f - cur_f
    max_abs_diff = float(diff.abs().max().item())
    mean_abs_diff = float(diff.abs().mean().item())
    rel_l2 = float(diff.norm().item() / max(1e-12, ref_f.norm().item()))
    return max_abs_diff, mean_abs_diff, rel_l2


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
    if torch_chunk_gated_delta_rule is None or torch_recurrent_gated_delta_rule is None:
        return
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


def _capture_suffix_grad_outputs(
    layers: list[torch.nn.Module],
    prefix_split_layer: int,
    *,
    capture: str = "output",
) -> tuple[dict[int, torch.Tensor], list[torch.utils.hooks.RemovableHandle]]:
    if capture not in {"input", "output"}:
        raise ValueError(f"unsupported capture mode: {capture}")

    captured: dict[int, torch.Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    for abs_idx, layer in enumerate(layers[prefix_split_layer:], start=prefix_split_layer):
        def _hook(_module, grad_input, grad_output, *, _abs_idx=abs_idx):
            grad_tuple = grad_output if capture == "output" else grad_input
            if not grad_tuple:
                return
            grad = grad_tuple[0]
            if grad is None:
                return
            captured[_abs_idx] = grad.detach().float().clone()

        handles.append(layer.register_full_backward_hook(_hook))

    return captured, handles


def run_grad_parity(
    model_path: str, model_cls: type, lora_targets: list[str]
) -> None:
    t0 = time.perf_counter()

    print("[grad-parity] building config")
    seed = 1337
    config = _build_test_config(
        lora_targets,
        chunk_size=10,
        use_grad_checkpoint=False,
    )
    # LoRA-only parity thresholds (BF16, chunked path):
    # We compare adapter grads only, so tiny near-zero entries can inflate relative error.
    # Keep abs + rel_l2 as primary stability checks; masked relative is secondary.
    #
    # rel_l2=1e-2 is intentionally tight but realistic for BF16. With the prefix/suffix split,
    # small roundoff at the boundary can accumulate through the suffix stack and show up most
    # strongly in early-layer grad_out diagnostics, then decay toward the loss. That monotonic
    # decay pattern is expected BF16 noise, not necessarily a correctness bug.
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
    if model_cls.__name__ == "Qwen3_5Model":
        _force_qwen_torch_linear_attention_kernels(baseline_model.model)
    print(f"[grad-parity] baseline model loaded in {time.perf_counter() - t_load0:.2f}s")

    print("[grad-parity] loading tokenizer")
    tokenizer = baseline_model.tokenizer

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
    prefix_split_layer = int(getattr(baseline_model, "_prefix_split_layer", 0))

    print("[prefix-parity] running boundary diagnostic")
    if prefix_split_layer <= 0:
        print("[prefix-parity] skipped: prefix_split_layer=0")
    else:
        hf_prefix_cache: dict[str, torch.Tensor] = {}

        def _capture_prefix_boundary(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hf_prefix_cache["hidden"] = hidden.detach().float().clone()

        prefix_handle = baseline_model._layers[prefix_split_layer - 1].register_forward_hook(
            _capture_prefix_boundary
        )
        _zero_all_grads(baseline_model.model)
        _ = baseline_model.model(input_ids=full_ids, attention_mask=full_mask).logits
        prefix_handle.remove()

        with torch.inference_mode():
            custom_prefix_hidden, _, _, _, _, _ = baseline_model._forward_prefix(full_ids, full_mask)
        prefix_max_abs_diff, prefix_mean_abs_diff, prefix_rel_l2 = _tensor_diff_stats(
            hf_prefix_cache["hidden"],
            custom_prefix_hidden,
        )
        print(
            f"[prefix-parity] max_abs_diff={prefix_max_abs_diff:.6f} "
            f"mean_abs_diff={prefix_mean_abs_diff:.6f} "
            f"rel_l2={prefix_rel_l2:.6e}"
        )

    print("[grad-parity] HF baseline backward")
    _zero_all_grads(baseline_model.model)
    hf_suffix_grad_outputs, hf_grad_handles = _capture_suffix_grad_outputs(
        baseline_model._layers,
        prefix_split_layer,
    )
    hf_suffix_grad_inputs, hf_grad_input_handles = _capture_suffix_grad_outputs(
        baseline_model._layers,
        prefix_split_layer,
        capture="input",
    )
    out = baseline_model.model(input_ids=full_ids, attention_mask=full_mask).logits
    # Align with the actual training objective: logits at position t predict token t+1.
    logits_comp = out[:, -(completion_len + 1):-1, :]
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
    for handle in hf_grad_handles:
        handle.remove()
    for handle in hf_grad_input_handles:
        handle.remove()
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
        hidden_prefix, pos_ids, shared_kv_states, per_layer_inputs, _, _ = custom_model._forward_prefix(
            full_ids,
            full_mask,
        )
    hidden_for_suffix = hidden_prefix.clone().detach().requires_grad_(True)
    pos_for_suffix = pos_ids.clone().detach()
    shared_kv_for_suffix = custom_model._detach_shared_kv_states(shared_kv_states)
    per_layer_inputs_for_suffix = (
        per_layer_inputs.clone().detach() if per_layer_inputs is not None else None
    )

    custom_suffix_grad_outputs, custom_grad_handles = _capture_suffix_grad_outputs(
        custom_model._layers,
        prefix_split_layer,
    )
    hidden_suffix = custom_model._forward_suffix(
        hidden_for_suffix,
        pos_for_suffix,
        full_mask,
        shared_kv_for_suffix,
        per_layer_inputs_for_suffix,
    )
    custom_logits = custom_model._lm_head_logits_chunked(hidden_suffix)
    custom_logits_comp = custom_logits[:, -(completion_len + 1):-1, :]
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
    for handle in custom_grad_handles:
        handle.remove()
    custom_stats = {
        "loss": float(custom_loss_t.item()),
    }
    custom_lora_grads = _collect_peft_grads(custom_model.model)

    if prefix_split_layer < len(custom_model._layers):
        # Best-effort boundary check:
        # hidden_for_suffix.grad is d(loss)/d(suffix_input). The comparable HF quantity is
        # grad_input[0] for the first suffix layer. In practice this may be missing when the
        # frozen prefix does not participate in the HF autograd traversal, so treat this as a
        # diagnostic only and rely on LoRA grad parity for pass/fail.
        hf_boundary_grad = hf_suffix_grad_inputs.get(prefix_split_layer)
        custom_boundary_grad = hidden_for_suffix.grad
        if hf_boundary_grad is None or custom_boundary_grad is None:
            print(
                f"[boundary-grad] missing_capture "
                f"hf={hf_boundary_grad is not None} custom={custom_boundary_grad is not None}"
            )
        else:
            boundary_max_abs_diff, boundary_mean_abs_diff, boundary_rel_l2 = _tensor_diff_stats(
                hf_boundary_grad,
                custom_boundary_grad,
            )
            print(
                f"[boundary-grad] suffix_layer=0 abs={prefix_split_layer} "
                f"metric=grad_input "
                f"hf_max={float(hf_boundary_grad.abs().max().item()):.6f} "
                f"custom_max={float(custom_boundary_grad.float().abs().max().item()):.6f} "
                f"max_abs_diff={boundary_max_abs_diff:.6f} "
                f"mean_abs_diff={boundary_mean_abs_diff:.6f} "
                f"rel_l2={boundary_rel_l2:.6e}"
            )

    first_divergent_layer = None
    for abs_idx in range(prefix_split_layer, len(custom_model._layers)):
        hf_grad = hf_suffix_grad_outputs.get(abs_idx)
        custom_grad = custom_suffix_grad_outputs.get(abs_idx)
        if hf_grad is None or custom_grad is None:
            print(
                f"[layer-parity] suffix layer {abs_idx - prefix_split_layer} "
                f"(abs={abs_idx}) missing_grad_capture "
                f"hf={hf_grad is not None} custom={custom_grad is not None}"
            )
            if first_divergent_layer is None:
                first_divergent_layer = abs_idx
            continue
        layer_max_abs_diff, _, layer_rel_l2 = _tensor_diff_stats(hf_grad, custom_grad)
        print(
            f"[layer-parity] suffix layer {abs_idx - prefix_split_layer} "
            f"(abs={abs_idx}) grad_out max_abs_diff={layer_max_abs_diff:.6f} "
            f"rel_l2={layer_rel_l2:.6e}"
        )
        if first_divergent_layer is None and (layer_max_abs_diff > 1e-5 or layer_rel_l2 > 1e-5):
            first_divergent_layer = abs_idx

    if first_divergent_layer is None:
        print("[layer-parity] first_divergent_layer=none")
    else:
        print(
            f"[layer-parity] first_divergent_layer suffix={first_divergent_layer - prefix_split_layer} "
            f"abs={first_divergent_layer}"
        )

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
    run_grad_parity(
        DEFAULT_GEMMA4_MODEL_PATH,
        Gemma4Model,
        DEFAULT_GEMMA4_LORA_TARGETS,
    )
    print("PASS: gemma4 grad parity")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("PASS: run_grad_parity")
