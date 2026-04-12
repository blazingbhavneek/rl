import csv
import gc
import time
import warnings
from pathlib import Path

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

DEFAULT_COMPLETION_LEN = 4
DEFAULT_START_SEQ_LEN = 256
DEFAULT_SEARCH_STEP = 256
DEFAULT_TOKEN_CHUNK_SIZE = 32
PLOT_BASENAME = "test_max_sequence_length_memory"


def _build_config() -> ModelConfig:
    return ModelConfig(
        lora=DEFAULT_GEMMA4_LORA_TARGETS,
        lora_fraction=0.25,
        lora_rank=128,
        lora_alpha=256,
        chunk_size=DEFAULT_TOKEN_CHUNK_SIZE,
        token_chunk_size=DEFAULT_TOKEN_CHUNK_SIZE,
        offload_prefix_to_cpu=True,
        cuda_device_index=0,
        use_grad_checkpoint=True,
        attn_implementation="sdpa",
    )


def _zero_all_grads(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.grad = None


def _build_test_tensors(
    seq_len: int,
    completion_len: int,
    *,
    vocab_size: int,
    device: torch.device,
    tokenizer,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    base_token_id = tokenizer.eos_token_id
    if base_token_id is None:
        base_token_id = tokenizer.pad_token_id
    if base_token_id is None:
        base_token_id = 1

    vocab_size = max(2, int(vocab_size))
    base_token_id = int(base_token_id) % vocab_size
    pattern = torch.arange(seq_len, device=device, dtype=torch.long) % 17
    full_ids = ((pattern + base_token_id) % vocab_size).unsqueeze(0)
    full_mask = torch.ones_like(full_ids)
    completion_ids = full_ids[:, -completion_len:].clone()
    completion_mask = torch.ones((1, completion_len), device=device, dtype=torch.float32)
    return full_ids, full_mask, completion_ids, completion_mask


def _run_streaming_backward(
    model: Gemma4Model,
    full_ids: torch.Tensor,
    full_mask: torch.Tensor,
    completion_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    completion_len: int,
) -> float:
    with torch.inference_mode():
        prefix_bundle = model._build_prefix_bundle(full_ids, full_mask)

    with torch.enable_grad():
        hidden_suffix = model._run_suffix_from_prefix_bundle(prefix_bundle, full_mask)
        hidden_comp = hidden_suffix[:, -(completion_len + 1):-1, :]
        token_logprobs = model._token_logprobs_chunked(hidden_comp, completion_ids)
        loss = -((token_logprobs * completion_mask).sum() / completion_mask.sum().clamp(min=1.0))
        loss.backward()
    return float(loss.item())


def _run_hf_backward(
    model: Gemma4Model,
    full_ids: torch.Tensor,
    full_mask: torch.Tensor,
    completion_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    completion_len: int,
) -> float:
    with torch.enable_grad():
        outputs = model._inner_model(
            input_ids=full_ids,
            attention_mask=full_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            per_layer_inputs=None,
            use_cache=False,
        )
        hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
        hidden_comp = hidden_states[:, -(completion_len + 1):-1, :]
        token_logprobs = model._token_logprobs_chunked(hidden_comp, completion_ids)
        loss = -((token_logprobs * completion_mask).sum() / completion_mask.sum().clamp(min=1.0))
        loss.backward()
    return float(loss.item())


def _cleanup(model: Gemma4Model) -> None:
    _zero_all_grads(model.model)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


def _measure_peak_cuda_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / 1024**2)


def _run_candidate(
    label: str,
    runner,
    model: Gemma4Model,
    seq_len: int,
    measurements: dict[int, float],
) -> tuple[bool, float | None]:
    device = next(model.model.parameters()).device
    vocab_size = int(getattr(model._inner_model.config, "vocab_size", 0))
    full_ids, full_mask, completion_ids, completion_mask = _build_test_tensors(
        seq_len,
        DEFAULT_COMPLETION_LEN,
        vocab_size=vocab_size,
        device=device,
        tokenizer=model.tokenizer,
    )

    try:
        _zero_all_grads(model.model)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        loss = runner(
            model,
            full_ids,
            full_mask,
            completion_ids,
            completion_mask,
            DEFAULT_COMPLETION_LEN,
        )
        peak_mb = _measure_peak_cuda_mb()
        measurements[seq_len] = max(measurements.get(seq_len, 0.0), peak_mb)
        print(f"[{label}] seq_len={seq_len} loss={loss:.6f} peak_cuda_mb={peak_mb:.1f} PASS")
        return True, loss
    except RuntimeError as exc:
        if not _is_oom(exc):
            raise
        print(f"[{label}] seq_len={seq_len} OOM")
        return False, None
    finally:
        _cleanup(model)


def _find_limit(
    label: str,
    runner,
    model: Gemma4Model,
    *,
    start_seq_len: int,
    max_supported: int,
    search_step: int,
) -> tuple[int, dict[int, float]]:
    measurements: dict[int, float] = {}

    seq_len = min(start_seq_len, max_supported)
    last_pass = 0
    first_fail = None

    while seq_len <= max_supported:
        passed, _ = _run_candidate(label, runner, model, seq_len, measurements)
        if passed:
            last_pass = seq_len
            if seq_len == max_supported:
                return last_pass, measurements
            next_seq = seq_len * 2
            if next_seq > max_supported:
                next_seq = max_supported
            if next_seq == seq_len:
                return last_pass, measurements
            seq_len = next_seq
        else:
            first_fail = seq_len
            break

    if first_fail is None:
        return last_pass, measurements

    lo = last_pass + search_step
    hi = first_fail - search_step
    while lo <= hi:
        mid = ((lo + hi) // (2 * search_step)) * search_step
        if mid < lo:
            mid = lo
        if mid > hi:
            mid = hi
        if mid <= last_pass:
            mid = last_pass + search_step
        if mid >= first_fail:
            break

        passed, _ = _run_candidate(label, runner, model, mid, measurements)
        if passed:
            last_pass = mid
            lo = mid + search_step
        else:
            first_fail = mid
            hi = mid - search_step

    return last_pass, measurements


def _write_measurements_csv(
    path: Path,
    hf_measurements: dict[int, float],
    stream_measurements: dict[int, float],
) -> None:
    seq_lens = sorted(set(hf_measurements) | set(stream_measurements))
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seq_len", "hf_peak_cuda_mb", "stream_peak_cuda_mb"])
        for seq_len in seq_lens:
            writer.writerow([
                seq_len,
                hf_measurements.get(seq_len, ""),
                stream_measurements.get(seq_len, ""),
            ])


def _write_svg_plot(
    path: Path,
    hf_measurements: dict[int, float],
    stream_measurements: dict[int, float],
) -> None:
    seq_lens = sorted(set(hf_measurements) | set(stream_measurements))
    if not seq_lens:
        return

    width = 960
    height = 540
    margin_left = 80
    margin_right = 24
    margin_top = 40
    margin_bottom = 60
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    max_x = max(seq_lens)
    max_y = max([1.0, *hf_measurements.values(), *stream_measurements.values()])

    def x_px(x: float) -> float:
        if max_x == min(seq_lens):
            return margin_left + plot_w / 2
        return margin_left + (x - min(seq_lens)) * plot_w / (max_x - min(seq_lens))

    def y_px(y: float) -> float:
        return margin_top + plot_h - (y * plot_h / max_y)

    def points(measurements: dict[int, float]) -> str:
        return " ".join(f"{x_px(x):.1f},{y_px(y):.1f}" for x, y in sorted(measurements.items()))

    x_ticks = seq_lens
    y_ticks = [max_y * frac / 4 for frac in range(5)]

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<style>text{font-family:monospace;font-size:12px} .grid{stroke:#ddd;stroke-width:1} .axis{stroke:#333;stroke-width:2} .hf{fill:none;stroke:#d1495b;stroke-width:3} .stream{fill:none;stroke:#00798c;stroke-width:3}</style>',
        f'<text x="{width/2:.1f}" y="24" text-anchor="middle">Gemma4 Peak CUDA Memory vs Sequence Length</text>',
        f'<line class="axis" x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}"/>',
        f'<line class="axis" x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}"/>',
    ]

    for tick in x_ticks:
        x = x_px(tick)
        svg.append(f'<line class="grid" x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{margin_top + plot_h}"/>')
        svg.append(f'<text x="{x:.1f}" y="{margin_top + plot_h + 20}" text-anchor="middle">{tick}</text>')

    for tick in y_ticks:
        y = y_px(tick)
        svg.append(f'<line class="grid" x1="{margin_left}" y1="{y:.1f}" x2="{margin_left + plot_w}" y2="{y:.1f}"/>')
        svg.append(f'<text x="{margin_left - 10}" y="{y + 4:.1f}" text-anchor="end">{tick:.0f}</text>')

    if hf_measurements:
        svg.append(f'<polyline class="hf" points="{points(hf_measurements)}"/>')
    if stream_measurements:
        svg.append(f'<polyline class="stream" points="{points(stream_measurements)}"/>')

    legend_x = margin_left + 12
    legend_y = margin_top + 12
    svg.extend([
        f'<line class="hf" x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 32}" y2="{legend_y}"/>',
        f'<text x="{legend_x + 40}" y="{legend_y + 4}">HF Backprop</text>',
        f'<line class="stream" x1="{legend_x}" y1="{legend_y + 20}" x2="{legend_x + 32}" y2="{legend_y + 20}"/>',
        f'<text x="{legend_x + 40}" y="{legend_y + 24}">Prefix/Suffix Backprop</text>',
        f'<text x="{width/2:.1f}" y="{height - 12}" text-anchor="middle">Sequence Length</text>',
        f'<text x="18" y="{height/2:.1f}" text-anchor="middle" transform="rotate(-90 18,{height/2:.1f})">Peak CUDA Memory (MB)</text>',
        "</svg>",
    ])

    path.write_text("\n".join(svg))


def _write_plot(
    base_path: Path,
    hf_measurements: dict[int, float],
    stream_measurements: dict[int, float],
) -> Path:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        svg_path = base_path.with_suffix(".svg")
        _write_svg_plot(svg_path, hf_measurements, stream_measurements)
        return svg_path

    fig, ax = plt.subplots(figsize=(10, 6))
    if hf_measurements:
        xs = sorted(hf_measurements)
        ax.plot(xs, [hf_measurements[x] for x in xs], marker="o", label="HF Backprop")
    if stream_measurements:
        xs = sorted(stream_measurements)
        ax.plot(xs, [stream_measurements[x] for x in xs], marker="o", label="Prefix/Suffix Backprop")
    ax.set_title("Gemma4 Peak CUDA Memory vs Sequence Length")
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Peak CUDA Memory (MB)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    png_path = base_path.with_suffix(".png")
    fig.savefig(png_path)
    plt.close(fig)
    return png_path


def run_max_sequence_length_test(model_path: str) -> None:
    print("[max-seq] loading model")
    t0 = time.perf_counter()
    model = Gemma4Model(model_path=model_path, config=_build_config())
    model.model.eval()
    print(f"[max-seq] model loaded in {time.perf_counter() - t0:.2f}s")

    max_supported = int(getattr(model._inner_model.config, "max_position_embeddings", 0))
    if max_supported <= 0:
        raise AssertionError("max_position_embeddings not found in model config")

    offload_enabled = bool(getattr(model, "_offload_prefix_to_cpu", False))
    print(
        f"[max-seq] configured_max={max_supported} "
        f"completion_len={DEFAULT_COMPLETION_LEN} "
        f"token_chunk_size={DEFAULT_TOKEN_CHUNK_SIZE} "
        f"offload_prefix_to_cpu={offload_enabled}"
    )

    hf_max_seq_len, hf_measurements = _find_limit(
        "hf-max-seq",
        _run_hf_backward,
        model,
        start_seq_len=min(DEFAULT_START_SEQ_LEN, max_supported),
        max_supported=max_supported,
        search_step=DEFAULT_SEARCH_STEP,
    )
    stream_max_seq_len, stream_measurements = _find_limit(
        "stream-max-seq",
        _run_streaming_backward,
        model,
        start_seq_len=min(DEFAULT_START_SEQ_LEN, max_supported),
        max_supported=max_supported,
        search_step=DEFAULT_SEARCH_STEP,
    )

    ratio = None
    if hf_max_seq_len > 0:
        ratio = stream_max_seq_len / hf_max_seq_len

    print(f"[hf-max-seq] maximum backprop sequence length: {hf_max_seq_len}")
    print(f"[stream-max-seq] maximum backprop sequence length: {stream_max_seq_len}")
    if ratio is not None:
        print(f"[compare] hf_max={hf_max_seq_len} stream_max={stream_max_seq_len} improvement={ratio:.2f}x")

    plot_base = Path(__file__).with_name(PLOT_BASENAME)
    csv_path = plot_base.with_suffix(".csv")
    _write_measurements_csv(csv_path, hf_measurements, stream_measurements)
    plot_path = _write_plot(plot_base, hf_measurements, stream_measurements)
    print(f"[max-seq] measurements_csv={csv_path}")
    print(f"[max-seq] memory_plot={plot_path}")

    if hf_max_seq_len <= 0 and stream_max_seq_len <= 0:
        raise AssertionError("No sequence length passed before OOM")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run_max_sequence_length_test(DEFAULT_GEMMA4_MODEL_PATH)
    print("PASS: gemma4 max backprop sequence length")
