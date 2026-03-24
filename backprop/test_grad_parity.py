from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import LoraConfig, TaskType, get_peft_model

from backprop import BackpropConfig, ChunkSizeProfiler, LoRABackprop, StreamingBackprop

log = logging.getLogger("test_grad_parity")


def _stage(message: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[test_grad_parity {ts}] {message}", flush=True)


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


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _collect_lora_grads(model: torch.nn.Module) -> Dict[str, Tensor]:
    out: Dict[str, Tensor] = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if "lora" in name.lower() or "adapter" in name.lower():
            out[name] = p.grad.detach().float().cpu().clone()
    return out


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


def _build_completion_batch(
    tokenizer,
    base_text: str,
    batch_size: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor, List[str]]:
    texts: List[str] = []
    token_rows: List[List[int]] = []

    for i in range(batch_size):
        txt = f"{base_text} Variant {i}."
        ids = tokenizer.encode(txt, add_special_tokens=False)
        if not ids:
            ids = tokenizer.encode(base_text, add_special_tokens=False)
        texts.append(txt)
        token_rows.append(ids)

    max_len = max(len(x) for x in token_rows)
    completion_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    completion_mask = torch.zeros((batch_size, max_len), dtype=torch.float32, device=device)

    for i, row in enumerate(token_rows):
        ln = len(row)
        completion_ids[i, :ln] = torch.tensor(row, dtype=torch.long, device=device)
        completion_mask[i, :ln] = 1.0

    return completion_ids, completion_mask, texts


def _check_batched_logprob_parity(
    bp: StreamingBackprop,
    model: torch.nn.Module,
    prompt_ids: Tensor,
    completion_ids: Tensor,
    completion_mask: Tensor,
) -> Dict[str, float]:
    # Batched call (B, T_c)
    batch_log_probs = bp.compute_logprobs(model, prompt_ids, completion_ids)

    # Reference: stack of single-sample calls.
    ref_rows: List[Tensor] = []
    for i in range(completion_ids.shape[0]):
        ln = int(completion_mask[i].sum().item())
        sample_lp = bp.compute_logprobs(model, prompt_ids, completion_ids[i, :ln])
        padded = torch.zeros_like(completion_ids[i], dtype=sample_lp.dtype)
        padded[:ln] = sample_lp
        ref_rows.append(padded)

    ref = torch.stack(ref_rows, dim=0)
    diff = (batch_log_probs.detach().float() - ref.detach().float()).abs()

    return {
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
    }


def _run_backward(
    bp,
    model: torch.nn.Module,
    prompt_ids: Tensor,
    completion_ids: Tensor,
    completion_mask: Tensor,
    use_batch_loss: bool,
) -> Tuple[Dict[str, Tensor], Dict[str, float]]:
    model.zero_grad(set_to_none=True)

    if use_batch_loss:
        def loss_fn_batch(batch_log_probs: Tensor, batch_mask: Tensor, hidden_comp=None) -> Tensor:
            del hidden_comp
            denom = batch_mask.sum().clamp(min=1.0)
            return -((batch_log_probs * batch_mask).sum() / denom)

        stats = bp.backward_on_batch(
            model=model,
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            loss_fn=loss_fn_batch,
            loss_scale=1.0,
            lora_path=None,
        )
    else:
        def loss_fn(log_probs: Tensor, gen_idx: int, hidden_comp=None) -> Tensor:
            del hidden_comp
            mask = completion_mask[gen_idx].to(log_probs.device, non_blocking=True)
            return -((log_probs * mask).sum() / mask.sum().clamp(min=1.0))

        stats = bp.backward_on_batch(
            model=model,
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            loss_fn=loss_fn,
            loss_scale=1.0,
            lora_path=None,
        )

    grads = _collect_lora_grads(model)
    return grads, {k: float(v) for k, v in stats.items()}


def _measure_backward_runtime(
    bp,
    model: torch.nn.Module,
    prompt_ids: Tensor,
    completion_ids: Tensor,
    completion_mask: Tensor,
    use_batch_loss: bool,
    device: torch.device,
    iters: int,
) -> Dict[str, float]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    elapsed_ms: List[float] = []
    for _ in range(max(1, iters)):
        _sync(device)
        t0 = time.perf_counter()
        _run_backward(
            bp=bp,
            model=model,
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            use_batch_loss=use_batch_loss,
        )
        _sync(device)
        elapsed_ms.append((time.perf_counter() - t0) * 1000.0)

    peak_vram_bytes = 0.0
    peak_vram_ratio = 0.0
    if device.type == "cuda":
        peak_vram_bytes = float(torch.cuda.max_memory_reserved(device))
        total_vram = float(torch.cuda.get_device_properties(device).total_memory)
        peak_vram_ratio = peak_vram_bytes / max(1.0, total_vram)

    elapsed_sorted = sorted(elapsed_ms)
    median_ms = elapsed_sorted[len(elapsed_sorted) // 2]

    return {
        "iters": float(len(elapsed_ms)),
        "mean_ms": float(sum(elapsed_ms) / max(1, len(elapsed_ms))),
        "median_ms": float(median_ms),
        "min_ms": float(min(elapsed_ms)),
        "max_ms": float(max(elapsed_ms)),
        "peak_vram_bytes": float(peak_vram_bytes),
        "peak_vram_ratio": float(peak_vram_ratio),
    }


def _run_chunk_profiler_smoke(
    bp: StreamingBackprop,
    model: torch.nn.Module,
    model_path: str,
    device: torch.device,
    dtype: torch.dtype,
    top_frac: float,
    profile_buckets: List[int],
    profile_batches: List[int],
    profile_max_chunk_cap: int,
    profile_vram_ratio: float,
) -> Dict[str, Any]:
    _stage("chunk_profiler: setup")
    base, _ = bp.adapter.unwrap(model)
    lm_head = bp.adapter.get_lm_head(base)
    model_cfg = getattr(base, "config", getattr(model, "config", None))
    hidden_size = int(getattr(model_cfg, "hidden_size"))
    vocab_size = int(getattr(model_cfg, "vocab_size"))

    profiler = ChunkSizeProfiler(
        lm_head=lm_head,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        device=device,
        model_path=model_path,
        sglang_mem_frac=0.0,
        top_frac=float(top_frac),
        cache_dir="./outputs/chunk_profile_test",
        dtype=dtype,
        batch_candidates=profile_batches,
        max_chunk_cap=int(profile_max_chunk_cap),
        vram_safety_ratio=float(profile_vram_ratio),
    )
    profiler.SEQ_BUCKETS = list(profile_buckets)
    _stage(
        "chunk_profiler: start "
        f"(buckets={list(profile_buckets)}, batches={list(profile_batches)}, "
        f"max_chunk_cap={profile_max_chunk_cap}, vram_max_ratio={profile_vram_ratio})"
    )
    table = profiler.load_or_profile()
    _stage("chunk_profiler: done")

    grid_summary: Dict[str, Any] = {}
    for bucket in profiler.SEQ_BUCKETS:
        bucket_map = profiler.profile_grid.get(int(bucket), {})
        grid_summary[str(bucket)] = {}
        for batch_size, entry in sorted(bucket_map.items(), key=lambda x: x[0], reverse=True):
            chosen = dict(entry.get("chosen_metrics", {}))
            grid_summary[str(bucket)][str(batch_size)] = {
                "best_chunk": int(entry.get("best_chunk", profiler.MIN_CHUNK)),
                "safe": bool(entry.get("safe", False)),
                "peak_vram_ratio": chosen.get("peak_vram_ratio"),
                "max_gpu_util": chosen.get("max_gpu_util"),
                "elapsed_ms": chosen.get("elapsed_ms"),
            }

    return {
        "legacy_table_batch1": {str(k): int(v) for k, v in table.items()},
        "grid_summary": grid_summary,
        "batch_candidates": list(profiler.batch_candidates),
        "buckets": list(profiler.SEQ_BUCKETS),
        "max_chunk_cap": int(profiler.max_chunk_cap),
        "vram_safety_ratio": float(profiler.vram_safety_ratio),
    }


def main() -> None:
    # Keep test defaults in one place so the CLI stays simple.
    DEFAULT_PROMPT = "Solve this coding task:"
    DEFAULT_COMPLETION = "Use dynamic programming with memoization."
    DEFAULT_TOP_FRAC = 0.25
    DEFAULT_RANK = 8
    DEFAULT_ALPHA = 16
    DEFAULT_DROPOUT = 0.0
    DEFAULT_PROFILE_BUCKETS = [2000, 4000, 8000, 16000, 32000, 64000, 96000, 128000, 130000]
    DEFAULT_PROFILE_BATCHES = [32, 16, 8, 4, 2, 1]
    DEFAULT_PROFILE_MAX_CHUNK_CAP = 32000
    DEFAULT_PROFILE_VRAM_MAX_RATIO = 0.95

    p = argparse.ArgumentParser(
        description="Validate batched StreamingBackprop parity/speed and batch-aware ChunkSizeProfiler",
    )
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--bench-iters", type=int, default=3)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--skip-chunk-profiler", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _stage("start")
    _stage(
        f"args: model_path={args.model_path} batch_size={args.batch_size} "
        f"bench_iters={args.bench_iters} dtype={args.dtype} "
        f"skip_chunk_profiler={bool(args.skip_chunk_profiler)}"
    )

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _stage(f"device: {device}")

    _stage("tokenizer: loading")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    _stage("model: loading base model")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)

    _stage("model: attaching LoRA")
    model = _build_lora(
        model,
        rank=DEFAULT_RANK,
        alpha=DEFAULT_ALPHA,
        dropout=DEFAULT_DROPOUT,
    )
    model.to(device)

    prompt_ids = torch.tensor(
        [tokenizer.encode(DEFAULT_PROMPT, add_special_tokens=False)],
        device=device,
        dtype=torch.long,
    )

    completion_ids, completion_mask, completion_texts = _build_completion_batch(
        tokenizer=tokenizer,
        base_text=DEFAULT_COMPLETION,
        batch_size=max(1, int(args.batch_size)),
        device=device,
    )

    _stage("backprop: configure")
    cfg_stream = BackpropConfig(top_frac=DEFAULT_TOP_FRAC, logit_chunk=64, use_torch_compile=True)
    cfg_lora = BackpropConfig(top_frac=1.0, logit_chunk=64, use_torch_compile=True)

    _stage("backprop: initialize StreamingBackprop + LoRABackprop (torch.compile setup)")
    bp_stream = StreamingBackprop(model=model, config=cfg_stream)
    bp_lora = LoRABackprop(model=model, config=cfg_lora)
    _stage("backprop: initialized")

    # 1) Logprob parity: batched compute vs stack of single-sample computes.
    _stage("check: batched logprob parity")
    logprob_parity = _check_batched_logprob_parity(
        bp=bp_stream,
        model=model,
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
    )

    # 2) Gradient parity: legacy per-sample loss_fn path vs new batch loss path.
    _stage("check: gradient parity (legacy per-sample loss)")
    initial_lora_state = bp_stream.get_current_lora_state()

    grads_legacy, stats_legacy = _run_backward(
        bp=bp_stream,
        model=model,
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        use_batch_loss=False,
    )

    bp_stream.restore_lora_state(initial_lora_state)
    model.zero_grad(set_to_none=True)

    _stage("check: gradient parity (batch loss)")
    grads_batch, stats_batch = _run_backward(
        bp=bp_stream,
        model=model,
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        use_batch_loss=True,
    )

    grad_parity_legacy_vs_batch = _compare_grads(grads_legacy, grads_batch)

    # 3) Optional reference parity: streaming vs LoRA on first sample only.
    _stage("check: gradient parity (streaming vs lora single sample)")
    first_completion = completion_ids[0, : int(completion_mask[0].sum().item())]
    grads_stream_single, _ = _run_backward(
        bp=bp_stream,
        model=model,
        prompt_ids=prompt_ids,
        completion_ids=first_completion.unsqueeze(0),
        completion_mask=torch.ones((1, first_completion.numel()), dtype=torch.float32, device=device),
        use_batch_loss=False,
    )
    grads_lora_single, _ = _run_backward(
        bp=bp_lora,
        model=model,
        prompt_ids=prompt_ids,
        completion_ids=first_completion.unsqueeze(0),
        completion_mask=torch.ones((1, first_completion.numel()), dtype=torch.float32, device=device),
        use_batch_loss=False,
    )
    grad_parity_stream_vs_lora_single = _compare_grads(grads_stream_single, grads_lora_single)

    # 4) Runtime and memory comparison.
    _stage("bench: legacy loss_fn runtime")
    bp_stream.restore_lora_state(initial_lora_state)
    model.zero_grad(set_to_none=True)
    perf_legacy = _measure_backward_runtime(
        bp=bp_stream,
        model=model,
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        use_batch_loss=False,
        device=device,
        iters=args.bench_iters,
    )

    bp_stream.restore_lora_state(initial_lora_state)
    model.zero_grad(set_to_none=True)
    _stage("bench: batch loss_fn runtime")
    perf_batch = _measure_backward_runtime(
        bp=bp_stream,
        model=model,
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        use_batch_loss=True,
        device=device,
        iters=args.bench_iters,
    )

    chunk_profiler_report: Optional[Dict[str, Any]] = None
    if not args.skip_chunk_profiler:
        _stage("check: chunk profiler grid")
        chunk_profiler_report = _run_chunk_profiler_smoke(
            bp=bp_stream,
            model=model,
            model_path=args.model_path,
            device=device,
            dtype=dtype,
            top_frac=float(DEFAULT_TOP_FRAC),
            profile_buckets=list(DEFAULT_PROFILE_BUCKETS),
            profile_batches=list(DEFAULT_PROFILE_BATCHES),
            profile_max_chunk_cap=int(DEFAULT_PROFILE_MAX_CHUNK_CAP),
            profile_vram_ratio=float(DEFAULT_PROFILE_VRAM_MAX_RATIO),
        )

    _stage("report: writing JSON")
    report = {
        "model_path": args.model_path,
        "device": str(device),
        "dtype": args.dtype,
        "batch_size": int(completion_ids.shape[0]),
        "completion_text_samples": completion_texts,
        "config_stream": asdict(cfg_stream),
        "config_lora": asdict(cfg_lora),
        "hardcoded_defaults": {
            "prompt": DEFAULT_PROMPT,
            "completion": DEFAULT_COMPLETION,
            "lora_rank": DEFAULT_RANK,
            "lora_alpha": DEFAULT_ALPHA,
            "lora_dropout": DEFAULT_DROPOUT,
            "top_frac": DEFAULT_TOP_FRAC,
            "profile_buckets": DEFAULT_PROFILE_BUCKETS,
            "profile_batches": DEFAULT_PROFILE_BATCHES,
            "profile_max_chunk_cap": DEFAULT_PROFILE_MAX_CHUNK_CAP,
            "profile_vram_max_ratio": DEFAULT_PROFILE_VRAM_MAX_RATIO,
        },
        "logprob_parity_batched_vs_stacked_single": logprob_parity,
        "grad_parity_legacy_vs_batch_loss": grad_parity_legacy_vs_batch,
        "grad_parity_stream_vs_lora_single": grad_parity_stream_vs_lora_single,
        "stats_legacy": stats_legacy,
        "stats_batch": stats_batch,
        "perf_legacy_loss_fn": perf_legacy,
        "perf_batch_loss_fn": perf_batch,
        "chunk_profiler": chunk_profiler_report,
    }
    print(json.dumps(report, indent=2))
    _stage("done")


if __name__ == "__main__":
    main()
