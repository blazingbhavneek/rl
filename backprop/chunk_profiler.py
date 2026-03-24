from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import nn

log = logging.getLogger(__name__)


class _GpuUtilSampler:
    """Best-effort GPU utilization sampler.

    Returns a max utilization ratio in [0, 1] when available, otherwise None.
    """

    def __init__(self, device_index: int, poll_interval_s: float = 0.02) -> None:
        self.device_index = int(device_index)
        self.poll_interval_s = float(max(0.005, poll_interval_s))
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._max_util_ratio: Optional[float] = None

        self._nvml = None
        self._nvml_handle = None
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        except Exception:
            self._nvml = None
            self._nvml_handle = None

    def _read_util_ratio(self) -> Optional[float]:
        if self._nvml is not None and self._nvml_handle is not None:
            try:
                util = self._nvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                return max(0.0, min(1.0, float(util.gpu) / 100.0))
            except Exception:
                return None

        if hasattr(torch.cuda, "utilization"):
            try:
                util_pct = float(torch.cuda.utilization(self.device_index))
                return max(0.0, min(1.0, util_pct / 100.0))
            except Exception:
                return None
        return None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            util = self._read_util_ratio()
            if util is not None:
                if self._max_util_ratio is None:
                    self._max_util_ratio = util
                else:
                    self._max_util_ratio = max(self._max_util_ratio, util)
            time.sleep(self.poll_interval_s)

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> Optional[float]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
        return self._max_util_ratio


class ChunkSizeProfiler:
    SCHEMA_VERSION = 3
    # Sequence-length buckets as upper bounds:
    # 0-2k, 2k-4k, 4k-8k, 8k-16k, 16k-32k, 32k-64k, 64k-96k, 96k-128k, 128k-max.
    SEQ_BUCKETS = [2_000, 4_000, 8_000, 16_000, 32_000, 64_000, 96_000, 128_000, 130_000]
    MIN_CHUNK = 64

    def __init__(
        self,
        lm_head: nn.Module,
        hidden_size: int,
        vocab_size: int,
        device: torch.device,
        model_path: str,
        sglang_mem_frac: float,
        top_frac: float,
        cache_dir: str,
        dtype: torch.dtype = torch.bfloat16,
        batch_candidates: Optional[List[int]] = None,
        max_chunk_cap: int = 32000,
        vram_safety_ratio: float = 0.95,
    ) -> None:
        self.lm_head = lm_head
        self.hidden_size = int(hidden_size)
        self.vocab_size = int(vocab_size)
        self.device = device
        self.model_path = model_path
        self.sglang_mem_frac = float(sglang_mem_frac)
        self.top_frac = float(top_frac)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dtype = dtype

        candidates = list(batch_candidates) if batch_candidates is not None else [32, 16, 8, 4, 2, 1]
        self.batch_candidates = sorted({int(x) for x in candidates if int(x) > 0}, reverse=True)
        if not self.batch_candidates:
            self.batch_candidates = [1]

        self.max_chunk_cap = int(max(self.MIN_CHUNK, max_chunk_cap))
        self.vram_safety_ratio = float(max(0.1, min(0.99, vram_safety_ratio)))
        # Effective safety considers both this module's headroom and memory already reserved by sglang.
        self.effective_vram_safety_ratio = float(
            max(0.1, min(self.vram_safety_ratio, 1.0 - max(0.0, min(0.99, self.sglang_mem_frac))))
        )

        self.cache_key = self._make_cache_key(
            model_path=self.model_path,
            sglang_mem_frac=self.sglang_mem_frac,
            top_frac=self.top_frac,
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            dtype=str(self.dtype).replace("torch.", ""),
            batch_candidates=self.batch_candidates,
            max_chunk_cap=self.max_chunk_cap,
            vram_safety_ratio=self.vram_safety_ratio,
        )
        self.cache_path = self.cache_dir / f"chunk_profile_{self.cache_key}.json"

        # Compatibility table used by existing callers (batch_size=1 view).
        self.profile_table: Dict[int, int] = {}
        # Full profiling grid: bucket -> batch -> entry
        self.profile_grid: Dict[int, Dict[int, Dict[str, Any]]] = {}
        self.last_cache_hit: bool = False

    @staticmethod
    def _make_cache_key(
        model_path: str,
        sglang_mem_frac: float,
        top_frac: float,
        hidden_size: int,
        vocab_size: int,
        dtype: str,
        batch_candidates: List[int],
        max_chunk_cap: int,
        vram_safety_ratio: float,
    ) -> str:
        payload = json.dumps(
            {
                "schema": ChunkSizeProfiler.SCHEMA_VERSION,
                "model_path": model_path,
                "sglang_mem_frac": round(sglang_mem_frac, 3),
                "top_frac": round(top_frac, 3),
                "hidden_size": hidden_size,
                "vocab_size": vocab_size,
                "dtype": dtype,
                "batch_candidates": list(batch_candidates),
                "max_chunk_cap": int(max_chunk_cap),
                "vram_safety_ratio": round(vram_safety_ratio, 4),
            },
            sort_keys=True,
        )
        return hashlib.md5(payload.encode()).hexdigest()[:12]

    def _build_chunk_candidates(self) -> List[int]:
        cap = max(self.MIN_CHUNK, min(self.vocab_size, self.max_chunk_cap))
        highest_pow2 = 1
        while highest_pow2 * 2 <= cap:
            highest_pow2 *= 2

        out: List[int] = []
        current = highest_pow2
        while current >= self.MIN_CHUNK:
            out.append(int(current))
            current //= 2

        if self.MIN_CHUNK not in out:
            out.append(self.MIN_CHUNK)
        return out

    @staticmethod
    def _pick_best_safe_candidate(candidate_metrics: Dict[int, Dict[str, Any]]) -> tuple[int, Dict[str, Any]]:
        safe_candidates: List[tuple[int, Dict[str, Any]]] = []
        for chunk_size, metrics in candidate_metrics.items():
            if bool(metrics.get("safe", False)):
                safe_candidates.append((int(chunk_size), metrics))

        if safe_candidates:
            # Prefer the point that uses the most VRAM while still safe.
            # Tie-breakers: higher GPU util, then larger chunk.
            safe_candidates.sort(
                key=lambda x: (
                    float(x[1].get("peak_vram_ratio") or -1.0),
                    float(x[1].get("max_gpu_util") or -1.0),
                    int(x[0]),
                ),
                reverse=True,
            )
            return int(safe_candidates[0][0]), dict(safe_candidates[0][1])

        # No safe candidate found. Fall back to smallest chunk metrics if present.
        if candidate_metrics:
            min_chunk = min(candidate_metrics.keys())
            return int(min_chunk), dict(candidate_metrics[min_chunk])

        return ChunkSizeProfiler.MIN_CHUNK, {
            "oom": True,
            "safe": False,
            "peak_vram_bytes": None,
            "peak_vram_ratio": None,
            "max_gpu_util": None,
            "elapsed_ms": None,
        }

    @staticmethod
    def _clone_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "oom": bool(metrics.get("oom", False)),
            "safe": bool(metrics.get("safe", False)),
            "peak_vram_bytes": metrics.get("peak_vram_bytes"),
            "peak_vram_ratio": metrics.get("peak_vram_ratio"),
            "max_gpu_util": metrics.get("max_gpu_util"),
            "elapsed_ms": metrics.get("elapsed_ms"),
        }

    def _legacy_table_from_grid(self) -> Dict[int, int]:
        table: Dict[int, int] = {}
        for bucket in self.SEQ_BUCKETS:
            bucket_grid = self.profile_grid.get(int(bucket), {})
            best = self.MIN_CHUNK
            # Prefer batch=1 entry for compatibility.
            entry = bucket_grid.get(1)
            if isinstance(entry, dict):
                best = int(entry.get("best_chunk", self.MIN_CHUNK))
            table[int(bucket)] = int(best)
        return table

    def _parse_cached_grid(self, raw_grid: Dict[str, Any]) -> Dict[int, Dict[int, Dict[str, Any]]]:
        parsed: Dict[int, Dict[int, Dict[str, Any]]] = {}
        for bucket_key, batch_map in raw_grid.items():
            bucket = int(bucket_key)
            parsed[bucket] = {}
            for batch_key, entry in batch_map.items():
                batch = int(batch_key)
                parsed[bucket][batch] = dict(entry)
                parsed[bucket][batch]["best_chunk"] = int(parsed[bucket][batch].get("best_chunk", self.MIN_CHUNK))
        return parsed

    def load_or_profile(self) -> Dict[int, int]:
        log.info(
            "ChunkSizeProfiler: cache_key=%s cache_path=%s",
            self.cache_key,
            self.cache_path,
        )
        if self.cache_path.exists():
            self.last_cache_hit = True
            with self.cache_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)

            if isinstance(raw, dict) and int(raw.get("schema_version", 0)) == self.SCHEMA_VERSION:
                grid_raw = raw.get("profile_grid", {})
                self.profile_grid = self._parse_cached_grid(grid_raw)
                self.profile_table = self._legacy_table_from_grid()
                log.info("ChunkSizeProfiler: cache hit, loaded grid for %d buckets", len(self.profile_grid))
                return dict(self.profile_table)

            # Backward compatibility: old cache format {bucket: chunk}
            if isinstance(raw, dict):
                old_table = {int(k): int(v) for k, v in raw.items()}
                self.profile_table = dict(old_table)
                self.profile_grid = {
                    int(bucket): {
                        1: {
                            "best_chunk": int(chunk),
                            "safe": True,
                            "chosen_metrics": {
                                "oom": False,
                                "safe": True,
                                "peak_vram_bytes": None,
                                "peak_vram_ratio": None,
                                "max_gpu_util": None,
                                "elapsed_ms": None,
                            },
                            "candidate_metrics": {},
                        }
                    }
                    for bucket, chunk in old_table.items()
                }
                log.info("ChunkSizeProfiler: cache hit (legacy format), loaded %d buckets", len(self.profile_table))
                return dict(self.profile_table)

        self.last_cache_hit = False
        log.info("ChunkSizeProfiler: cache miss, profiling chunks")
        table = self.profile()

        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "cache_key": self.cache_key,
            "profile_grid": {
                str(bucket): {str(batch): entry for batch, entry in batch_map.items()}
                for bucket, batch_map in self.profile_grid.items()
            },
            "profile_table": {str(k): int(v) for k, v in self.profile_table.items()},
            "config": {
                "batch_candidates": list(self.batch_candidates),
                "max_chunk_cap": int(self.max_chunk_cap),
                "vram_safety_ratio": float(self.vram_safety_ratio),
                "effective_vram_safety_ratio": float(self.effective_vram_safety_ratio),
                "sglang_mem_frac": float(self.sglang_mem_frac),
            },
        }
        with self.cache_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f)
        log.info("ChunkSizeProfiler: profile complete and cached")
        return dict(table)

    def _chunked_lm_head_logprobs(
        self,
        hidden: torch.Tensor,
        token_ids: torch.Tensor,
        chunk: int,
    ) -> torch.Tensor:
        # hidden: (B, T, H), token_ids: (B, T)
        batch_size, seq_len, hidden_size = hidden.shape
        flat_tokens = batch_size * seq_len
        hidden_flat = hidden.reshape(flat_tokens, hidden_size)
        token_flat = token_ids.reshape(flat_tokens)

        lse = torch.empty(flat_tokens, device=self.device, dtype=torch.float32)
        weight = self.lm_head.weight.detach()
        weight_fp32 = weight.float()
        bias = getattr(self.lm_head, "bias", None)
        bias_fp32 = bias.detach().float() if bias is not None else None

        with torch.no_grad():
            for t0 in range(0, flat_tokens, chunk):
                t1 = min(t0 + chunk, flat_tokens)
                h = hidden_flat[t0:t1].detach().float()
                logits = h @ weight_fp32.T
                if bias_fp32 is not None:
                    logits = logits + bias_fp32
                lse[t0:t1] = torch.logsumexp(logits, dim=-1)

        w_sel = self.lm_head.weight[token_flat]
        tok_logits = (hidden_flat * w_sel).sum(dim=-1)
        if bias is not None:
            tok_logits = tok_logits + bias[token_flat]

        return (tok_logits.float() - lse).reshape(batch_size, seq_len)

    @staticmethod
    def _is_oom(exc: RuntimeError) -> bool:
        text = str(exc).lower()
        return "out of memory" in text or "cuda error: out of memory" in text

    def _profile_candidate(self, seq_len: int, batch_size: int, chunk_size: int) -> Dict[str, Any]:
        hidden = None
        token_ids = None
        sampler: Optional[_GpuUtilSampler] = None
        try:
            device_index = self.device.index if self.device.index is not None else torch.cuda.current_device()
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)

            sampler = _GpuUtilSampler(device_index=device_index)
            sampler.start()
            t0 = time.perf_counter()

            hidden = torch.zeros(
                (batch_size, seq_len, self.hidden_size),
                dtype=self.dtype,
                device=self.device,
                requires_grad=True,
            )
            token_ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=self.device)
            out = self._chunked_lm_head_logprobs(hidden, token_ids, chunk_size)
            loss = out.mean()
            loss.backward()

            torch.cuda.synchronize(self.device)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            max_gpu_util = sampler.stop() if sampler is not None else None

            peak_vram_bytes = int(torch.cuda.max_memory_reserved(self.device))
            total_vram_bytes = int(torch.cuda.get_device_properties(self.device).total_memory)
            peak_vram_ratio = float(peak_vram_bytes / max(1, total_vram_bytes))
            safe = bool(peak_vram_ratio <= self.effective_vram_safety_ratio)
            return {
                "oom": False,
                "safe": safe,
                "peak_vram_bytes": peak_vram_bytes,
                "peak_vram_ratio": peak_vram_ratio,
                "max_gpu_util": max_gpu_util,
                "elapsed_ms": float(elapsed_ms),
            }
        except RuntimeError as exc:
            if sampler is not None:
                sampler.stop()
            if not self._is_oom(exc):
                raise
            return {
                "oom": True,
                "safe": False,
                "peak_vram_bytes": None,
                "peak_vram_ratio": None,
                "max_gpu_util": None,
                "elapsed_ms": None,
            }
        finally:
            del hidden, token_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def profile(self) -> Dict[int, int]:
        if self.device.type != "cuda":
            self.profile_table = {bucket: self.MIN_CHUNK for bucket in self.SEQ_BUCKETS}
            self.profile_grid = {
                int(bucket): {
                    int(batch): {
                        "best_chunk": int(self.MIN_CHUNK),
                        "safe": True,
                        "chosen_metrics": {
                            "oom": False,
                            "safe": True,
                            "peak_vram_bytes": None,
                            "peak_vram_ratio": None,
                            "max_gpu_util": None,
                            "elapsed_ms": None,
                        },
                        "candidate_metrics": {},
                    }
                    for batch in self.batch_candidates
                }
                for bucket in self.SEQ_BUCKETS
            }
            log.warning(
                "ChunkSizeProfiler: non-cuda device detected, falling back to MIN_CHUNK=%s",
                self.MIN_CHUNK,
            )
            return dict(self.profile_table)

        chunk_candidates = self._build_chunk_candidates()
        grid: Dict[int, Dict[int, Dict[str, Any]]] = {}
        buckets_desc = sorted((int(x) for x in self.SEQ_BUCKETS), reverse=True)
        inferred_safe_batch_threshold: Optional[int] = None

        for bucket in buckets_desc:
            bucket_grid: Dict[int, Dict[str, Any]] = {}
            log.info("ChunkSizeProfiler: profiling bucket=%s", bucket)

            for batch_size in self.batch_candidates:
                # Monotonic inference:
                # If a batch fits at a larger bucket, all smaller batches also fit for this and smaller buckets.
                if inferred_safe_batch_threshold is not None and int(batch_size) <= int(inferred_safe_batch_threshold):
                    inferred_chunk = self.MIN_CHUNK
                    inferred_metrics: Dict[str, Any] = {
                        "oom": False,
                        "safe": True,
                        "peak_vram_bytes": None,
                        "peak_vram_ratio": None,
                        "max_gpu_util": None,
                        "elapsed_ms": None,
                        "inferred": True,
                        "inferred_from_batch_threshold": int(inferred_safe_batch_threshold),
                    }

                    # Use closest already-profiled larger batch in this bucket when available.
                    profiled_larger = [b for b in bucket_grid.keys() if int(b) > int(batch_size)]
                    if profiled_larger:
                        src_b = min(profiled_larger)
                        src_entry = bucket_grid[src_b]
                        inferred_chunk = int(src_entry.get("best_chunk", self.MIN_CHUNK))
                        inferred_metrics = self._clone_metrics(dict(src_entry.get("chosen_metrics", {})))
                        inferred_metrics["safe"] = True
                        inferred_metrics["oom"] = False
                        inferred_metrics["inferred"] = True
                        inferred_metrics["inferred_from"] = f"bucket={bucket},batch={src_b}"

                    bucket_grid[int(batch_size)] = {
                        "best_chunk": int(inferred_chunk),
                        "safe": True,
                        "chosen_metrics": inferred_metrics,
                        "candidate_metrics": {},
                    }
                    log.info(
                        "ChunkSizeProfiler: bucket=%s batch=%s -> inferred safe, chunk=%s "
                        "(threshold batch=%s)",
                        bucket,
                        batch_size,
                        inferred_chunk,
                        inferred_safe_batch_threshold,
                    )
                    continue

                candidate_metrics: Dict[int, Dict[str, Any]] = {}

                for chunk_size in chunk_candidates:
                    metrics = self._profile_candidate(
                        seq_len=int(bucket),
                        batch_size=int(batch_size),
                        chunk_size=int(chunk_size),
                    )
                    candidate_metrics[int(chunk_size)] = metrics

                best_chunk, chosen_metrics = self._pick_best_safe_candidate(candidate_metrics)

                bucket_grid[int(batch_size)] = {
                    "best_chunk": int(best_chunk),
                    "safe": bool(chosen_metrics.get("safe", False)),
                    "chosen_metrics": chosen_metrics,
                    "candidate_metrics": {str(k): v for k, v in candidate_metrics.items()},
                }
                log.info(
                    "ChunkSizeProfiler: bucket=%s batch=%s -> chunk=%s safe=%s vram_ratio=%s util=%s",
                    bucket,
                    batch_size,
                    best_chunk,
                    bucket_grid[int(batch_size)]["safe"],
                    chosen_metrics.get("peak_vram_ratio"),
                    chosen_metrics.get("max_gpu_util"),
                )

                # Once we observe a safe batch at this (largest so far) bucket, all smaller batches are safe
                # for this and all following smaller buckets, so we can skip profiling those cells.
                if bucket_grid[int(batch_size)]["safe"]:
                    if inferred_safe_batch_threshold is None:
                        inferred_safe_batch_threshold = int(batch_size)
                    else:
                        inferred_safe_batch_threshold = max(inferred_safe_batch_threshold, int(batch_size))

            grid[int(bucket)] = bucket_grid

        # Keep external shape stable (same bucket order as SEQ_BUCKETS).
        self.profile_grid = {int(b): grid.get(int(b), {}) for b in self.SEQ_BUCKETS}
        self.profile_table = self._legacy_table_from_grid()
        return dict(self.profile_table)

    def _best_chunk_for_bucket(self, bucket: int, batch_size: int) -> int:
        bucket_grid = self.profile_grid.get(int(bucket), {})
        if not bucket_grid:
            return self.MIN_CHUNK

        requested = int(max(1, batch_size))
        # Fallback policy: exact batch first, then smaller batches only.
        if requested in bucket_grid:
            return int(bucket_grid[requested].get("best_chunk", self.MIN_CHUNK))

        for b in sorted(bucket_grid.keys(), reverse=True):
            if b <= requested:
                return int(bucket_grid[b].get("best_chunk", self.MIN_CHUNK))

        return self.MIN_CHUNK

    def get_chunk_size(self, t_c: int, batch_size: int = 1) -> int:
        if not self.profile_table and not self.profile_grid:
            self.load_or_profile()

        for bucket in self.SEQ_BUCKETS:
            if int(t_c) <= bucket:
                return int(self._best_chunk_for_bucket(bucket, int(batch_size)))
        return self.MIN_CHUNK

    def invalidate(self) -> None:
        if self.cache_path.exists():
            self.cache_path.unlink()
        self.profile_table = {}
        self.profile_grid = {}
