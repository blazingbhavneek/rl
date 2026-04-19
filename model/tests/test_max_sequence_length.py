"""
Max sequence length search for Gemma4Model.

Uses model.backward() — the actual training path — and searches for the
largest sequence length that fits in GPU memory for the requested batch sizes.

Compares two gradient-checkpointing strategies on the same model instance:
  streaming  — _StreamCheckpointFunction (token_chunk_size > 0)
  standard   — torch.utils.checkpoint.checkpoint (token_chunk_size = 0)
"""
import csv
import gc
import time
import warnings
from pathlib import Path
from typing import NamedTuple

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
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

DEFAULT_TOKEN_CHUNK_SIZE = 128
BENCH_BATCH_SIZES = [1, 4]
CHECKPOINT_SEARCH_CONFIGS = [
    ("streaming", DEFAULT_TOKEN_CHUNK_SIZE),
    ("standard", 0),
]
DEFAULT_START_SEQ_LEN = 512
DEFAULT_SEARCH_STEP   = 256


SEARCH_BATCH_SIZES = BENCH_BATCH_SIZES[:]


USE_COMPILE = True  # set True to benchmark torch.compile (first step ~60s warmup, then faster)


def _build_config() -> ModelConfig:
    return ModelConfig(
        lora=DEFAULT_GEMMA4_LORA_TARGETS,
        lora_fraction=0.25,
        lora_rank=128,
        lora_alpha=256,
        chunk_size=256,
        logprob_chunk_size=128,
        token_chunk_size=DEFAULT_TOKEN_CHUNK_SIZE,
        use_grad_checkpoint=True,
        use_compile=USE_COMPILE,
        attn_implementation="sdpa",
        cuda_device_index=0,
    )


def _zero_all_grads(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.grad = None


def _make_sample(
    tokenizer,
    target_total_tokens: int,
) -> tuple[list[dict], str]:
    """
    Build a chat-formatted (messages, completion_text) pair whose tokenized
    length approximates target_total_tokens (~30% completion, ~70% prompt).
    """
    system = (
        "<|think|>\nYou are a careful technical assistant. "
        "Think step by step before answering."
    )
    question = (
        "Explain the following algorithms in detail, covering time complexity, "
        "space complexity, and real-world use cases with concrete examples: "
        "merge sort, quick sort, heap sort, radix sort. "
        "Compare their practical performance under different input distributions "
        "and discuss when each is preferable in production systems."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_len = len(tokenizer(prompt_str, add_special_tokens=False)["input_ids"])
    target_completion_tokens = max(32, target_total_tokens - prompt_len)

    snippet = (
        "The algorithm works by dividing the input into smaller subproblems, "
        "solving each recursively, and combining results efficiently. "
        "This divide-and-conquer approach yields O(n log n) average-case complexity. "
        "Memory usage depends on whether the implementation is in-place or uses auxiliary space. "
        "Cache locality significantly affects real-world throughput on large inputs. "
    )
    snippet_ids = tokenizer(snippet, add_special_tokens=False)["input_ids"]
    reps = (target_completion_tokens // max(1, len(snippet_ids))) + 2
    completion_ids = (snippet_ids * reps)[:target_completion_tokens]
    completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    completion_text += "<turn|>"  # Gemma4 EOS
    return messages, completion_text


def _loss_fn_batch(batch_log_probs, batch_mask, hidden_batch=None):
    del hidden_batch
    return -((batch_log_probs * batch_mask).sum() / batch_mask.sum().clamp(min=1.0))


_loss_fn_batch.__name__ = "loss_fn_batch"
_loss_fn_batch._streaming_reduction = "masked_mean_logprob"


def _cleanup(model: Gemma4Model) -> None:
    _zero_all_grads(model.model)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


class MaxFitResult(NamedTuple):
    label: str
    batch_size: int
    max_seq_len: int
    max_batch_tokens: int
    peak_cuda_mb: float
    time_s: float


def _make_algo_loss_fn(algo: str):
    """
    Return a loss_fn compatible with model.backward().

    sft  — streaming masked-mean logprob (memory-optimal, no full logprob tensor).
    grpo — advantage-weighted loss; must materialise full [G, T] logprob tensor
           before backward, so OOM limit is lower than SFT at the same G.
    """
    if algo == "sft":
        def _sft(batch_log_probs, batch_mask, hidden_batch=None):
            del hidden_batch
            return -((batch_log_probs * batch_mask).sum() / batch_mask.sum().clamp(min=1.0))
        _sft._streaming_reduction = "masked_mean_logprob"
        return _sft

    if algo == "grpo":
        def _grpo(batch_log_probs, batch_mask, hidden_batch=None):
            del hidden_batch
            G = batch_log_probs.shape[0]
            # Synthetic centred advantages — half positive, half negative.
            adv = torch.linspace(-1.0, 1.0, G, device=batch_log_probs.device)
            adv = adv - adv.mean()
            weighted = (batch_log_probs * batch_mask) * adv.unsqueeze(1)
            return -(weighted.sum() / batch_mask.sum().clamp(min=1.0))
        _grpo.__name__ = "loss_fn_batch"
        return _grpo

    raise ValueError(f"unknown algo: {algo!r}")


def _find_max_seq(
    label: str,
    model: Gemma4Model,
    batch_size: int,
    loss_fn,
    *,
    start_seq_len: int = DEFAULT_START_SEQ_LEN,
    search_step: int = DEFAULT_SEARCH_STEP,
    max_supported: int | None = None,
) -> MaxFitResult:
    """Binary-search for the largest sequence length that fits in GPU memory for a fixed batch size."""
    if max_supported is None:
        max_supported = int(getattr(model._inner_model.config, "max_position_embeddings", 131072))

    def _try(seq_len: int) -> tuple[bool, float, float]:
        messages, completion_text = _make_sample(model.tokenizer, seq_len)
        messages_batch = [messages] * batch_size
        completion_texts = [completion_text] * batch_size
        try:
            _zero_all_grads(model.model)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.backward(
                messages=messages_batch,
                completion_texts=completion_texts,
                loss_fn=loss_fn,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            peak_mb = float(torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else 0.0
            print(f"[{label}] seq={seq_len} PASS peak={peak_mb:.0f}MB time={elapsed:.2f}s")
            return True, peak_mb, elapsed
        except RuntimeError as exc:
            if _is_oom(exc):
                print(f"[{label}] seq={seq_len} OOM")
                return False, 0.0, 0.0
            raise
        finally:
            _cleanup(model)

    seq_len = min(start_seq_len, max_supported)
    last_pass, last_peak, last_time = 0, 0.0, 0.0
    first_fail = None

    # Exponential probe up
    while seq_len <= max_supported:
        ok, peak, elapsed = _try(seq_len)
        if ok:
            last_pass, last_peak, last_time = seq_len, peak, elapsed
            if seq_len == max_supported:
                break
            next_seq = min(seq_len * 2, max_supported)
            if next_seq == seq_len:
                break
            seq_len = next_seq
        else:
            first_fail = seq_len
            break

    # Binary refine
    if first_fail is not None:
        lo, hi = last_pass + search_step, first_fail - search_step
        while lo <= hi:
            mid = ((lo + hi) // (2 * search_step)) * search_step
            mid = max(lo, min(hi, mid))
            if mid <= last_pass:
                mid = last_pass + search_step
            if mid >= first_fail:
                break
            ok, peak, elapsed = _try(mid)
            if ok:
                last_pass, last_peak, last_time = mid, peak, elapsed
                lo = mid + search_step
            else:
                first_fail = mid
                hi = mid - search_step

    return MaxFitResult(
        label=label,
        batch_size=batch_size,
        max_seq_len=last_pass,
        max_batch_tokens=batch_size * last_pass,
        peak_cuda_mb=last_peak,
        time_s=last_time,
    )


def run_max_sequence_length_test(model_path: str) -> None:
    print("[bench] loading model")
    config = _build_config()
    model = Gemma4Model(model_path=model_path, config=config)
    model.model.eval()
    print(
        f"[bench] prefix_split={model._prefix_split_layer}/{len(model._layers)} layers "
        f"chunk_size={config.chunk_size} token_chunk={config.token_chunk_size}"
    )

    max_supported = int(getattr(model._inner_model.config, "max_position_embeddings", 131072))

    print("\n[bench] === max sequence search by checkpoint algo ===")
    checkpoint_max_results: list[MaxFitResult] = []
    for algo_label, token_chunk_size in CHECKPOINT_SEARCH_CONFIGS:
        model._use_grad_checkpoint = True
        model._token_chunk_size = token_chunk_size
        for batch_size in SEARCH_BATCH_SIZES:
            r = _find_max_seq(
                f"{algo_label}-b{batch_size}",
                model,
                batch_size,
                _loss_fn_batch,
                start_seq_len=DEFAULT_START_SEQ_LEN,
                search_step=DEFAULT_SEARCH_STEP,
                max_supported=max_supported,
            )
            checkpoint_max_results.append(r)
            print(
                f"[max-seq]   {algo_label:9s} batch={batch_size:2d} "
                f"max_seq={r.max_seq_len:6d} max_batch_tokens={r.max_batch_tokens:7d} "
                f"peak={r.peak_cuda_mb:.0f}MB time={r.time_s:.2f}s"
            )

    print("\n[bench] === max sequence summary by checkpoint algo ===")
    for algo_label, _ in CHECKPOINT_SEARCH_CONFIGS:
        algo_results = [r for r in checkpoint_max_results if r.label.startswith(f"{algo_label}-")]
        for r in sorted(algo_results, key=lambda item: item.batch_size):
            print(
                f"  {algo_label:9s} batch={r.batch_size:2d} "
                f"max_seq={r.max_seq_len:6d} max_batch_tokens={r.max_batch_tokens:7d} "
                f"time={r.time_s:.2f}s"
            )

    csv_path = Path(__file__).with_name("test_max_sequence_length_max_seq.csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "batch_size", "max_seq_len", "max_batch_tokens", "peak_cuda_mb", "time_s"])
        for r in checkpoint_max_results:
            writer.writerow([
                r.label,
                r.batch_size,
                r.max_seq_len,
                r.max_batch_tokens,
                f"{r.peak_cuda_mb:.1f}",
                f"{r.time_s:.2f}",
            ])
    print(f"\n[bench] csv={csv_path}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run_max_sequence_length_test(DEFAULT_GEMMA4_MODEL_PATH)
    print("PASS: gemma4 max-seq search")
