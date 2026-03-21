from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Dict

import torch
from torch import Tensor

from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import LoraConfig, TaskType, get_peft_model

from backprop import BackpropConfig, LoRABackprop, StreamingBackprop


def _build_lora(model: torch.nn.Module, rank: int, alpha: int, dropout: float) -> torch.nn.Module:
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )
    peft_model = get_peft_model(model, cfg)
    peft_model.train()
    return peft_model


def _collect_lora_grads(model: torch.nn.Module) -> Dict[str, Tensor]:
    out: Dict[str, Tensor] = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if "lora" in name.lower() or "adapter" in name.lower():
            out[name] = p.grad.detach().float().cpu().clone()
    return out


def _run_backward(
    bp,
    model: torch.nn.Module,
    prompt_ids: Tensor,
    completion_ids_1d: Tensor,
) -> Dict[str, Tensor]:
    model.zero_grad(set_to_none=True)

    completion_ids = completion_ids_1d.unsqueeze(0)
    completion_mask = torch.ones_like(completion_ids, dtype=torch.float32)

    def loss_fn(log_probs: Tensor, _idx: int) -> Tensor:
        return -log_probs.mean()

    _ = bp.backward_on_batch(
        model=model,
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        loss_fn=loss_fn,
        loss_scale=1.0,
        lora_path=None,
    )
    return _collect_lora_grads(model)


def _compare_grads(a: Dict[str, Tensor], b: Dict[str, Tensor]) -> Dict[str, float]:
    common = sorted(set(a) & set(b))
    if not common:
        raise RuntimeError("No overlapping LoRA gradients found between runs")

    max_abs = 0.0
    mean_abs_sum = 0.0
    n = 0
    dot = 0.0
    na = 0.0
    nb = 0.0

    for k in common:
        ga = a[k].reshape(-1)
        gb = b[k].reshape(-1)
        diff = (ga - gb).abs()
        max_abs = max(max_abs, diff.max().item())
        mean_abs_sum += diff.mean().item()
        dot += torch.dot(ga, gb).item()
        na += torch.dot(ga, ga).item()
        nb += torch.dot(gb, gb).item()
        n += 1

    cos = dot / ((na ** 0.5) * (nb ** 0.5) + 1e-12)
    return {
        "num_common_lora_tensors": float(n),
        "cosine_similarity": float(cos),
        "max_abs_diff": float(max_abs),
        "mean_of_tensor_mean_abs_diff": float(mean_abs_sum / max(1, n)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Compare LoRA grads from LoRABackprop vs StreamingBackprop")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--prompt", type=str, default="Solve this coding task:")
    p.add_argument("--completion", type=str, default="Use dynamic programming with memoization.")
    p.add_argument("--top-frac", type=float, default=0.25)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    args = p.parse_args()

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)

    model = _build_lora(model, rank=args.rank, alpha=args.alpha, dropout=args.dropout)
    model.to(device)

    prompt_ids = torch.tensor(
        [tokenizer.encode(args.prompt, add_special_tokens=False)],
        device=device,
        dtype=torch.long,
    )
    completion_ids = torch.tensor(
        tokenizer.encode(args.completion, add_special_tokens=False),
        device=device,
        dtype=torch.long,
    )

    if completion_ids.numel() == 0:
        raise ValueError("Completion tokenization is empty")

    cfg_stream = BackpropConfig(top_frac=args.top_frac, logit_chunk=64, offload_prefix_cpu=True)
    cfg_lora = BackpropConfig(top_frac=1.0, logit_chunk=64, offload_prefix_cpu=False)

    bp_stream = StreamingBackprop(model=model, config=cfg_stream)
    bp_lora = LoRABackprop(model=model, config=cfg_lora)

    grads_stream = _run_backward(bp_stream, model, prompt_ids, completion_ids)
    grads_lora = _run_backward(bp_lora, model, prompt_ids, completion_ids)

    report = {
        "model_path": args.model_path,
        "config_stream": asdict(cfg_stream),
        "config_lora": asdict(cfg_lora),
        "comparison": _compare_grads(grads_lora, grads_stream),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
