from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict

import torch
from torch import nn

log = logging.getLogger(__name__)


class ChunkSizeProfiler:
    SEQ_BUCKETS = [4_000, 8_000, 16_000, 32_000, 48_000, 64_000, 96_000, 130_000]
    CHUNK_CANDIDATES = [4096, 2048, 1024, 512, 256, 128, 64]
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

        self.cache_key = self._make_cache_key(
            model_path=self.model_path,
            sglang_mem_frac=self.sglang_mem_frac,
            top_frac=self.top_frac,
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            dtype=str(self.dtype).replace("torch.", ""),
        )
        self.cache_path = self.cache_dir / f"chunk_profile_{self.cache_key}.json"
        self.profile_table: Dict[int, int] = {}
        self.last_cache_hit: bool = False

    @staticmethod
    def _make_cache_key(
        model_path: str,
        sglang_mem_frac: float,
        top_frac: float,
        hidden_size: int,
        vocab_size: int,
        dtype: str,
    ) -> str:
        payload = json.dumps(
            {
                "model_path": model_path,
                "sglang_mem_frac": round(sglang_mem_frac, 3),
                "top_frac": round(top_frac, 3),
                "hidden_size": hidden_size,
                "vocab_size": vocab_size,
                "dtype": dtype,
            },
            sort_keys=True,
        )
        return hashlib.md5(payload.encode()).hexdigest()[:12]

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
            self.profile_table = {int(k): int(v) for k, v in raw.items()}
            log.info("ChunkSizeProfiler: cache hit, loaded %d buckets", len(self.profile_table))
            return dict(self.profile_table)

        self.last_cache_hit = False
        log.info("ChunkSizeProfiler: cache miss, profiling chunks")
        table = self.profile()
        with self.cache_path.open("w", encoding="utf-8") as f:
            json.dump({str(k): int(v) for k, v in table.items()}, f)
        log.info("ChunkSizeProfiler: profile complete and cached")
        return dict(table)

    def _chunked_lm_head_logprobs(
        self,
        hidden: torch.Tensor,
        token_ids: torch.Tensor,
        chunk: int,
    ) -> torch.Tensor:
        t_c = hidden.shape[0]
        lse = torch.empty(t_c, device=self.device, dtype=torch.float32)
        weight = self.lm_head.weight.detach()

        with torch.no_grad():
            for t0 in range(0, t_c, chunk):
                t1 = min(t0 + chunk, t_c)
                h = hidden[t0:t1].detach().float()
                logits = h @ weight.float().T
                if getattr(self.lm_head, "bias", None) is not None:
                    logits = logits + self.lm_head.bias.detach().float()
                lse[t0:t1] = torch.logsumexp(logits, dim=-1)

        w_sel = self.lm_head.weight[token_ids]
        tok_logits = (hidden * w_sel).sum(dim=-1)
        if getattr(self.lm_head, "bias", None) is not None:
            tok_logits = tok_logits + self.lm_head.bias[token_ids]
        return tok_logits.float() - lse

    @staticmethod
    def _is_oom(exc: RuntimeError) -> bool:
        text = str(exc).lower()
        return "out of memory" in text or "cuda error: out of memory" in text

    def profile(self) -> Dict[int, int]:
        if self.device.type != "cuda":
            self.profile_table = {bucket: self.MIN_CHUNK for bucket in self.SEQ_BUCKETS}
            log.warning(
                "ChunkSizeProfiler: non-cuda device detected, falling back to MIN_CHUNK=%s",
                self.MIN_CHUNK,
            )
            return dict(self.profile_table)

        table: Dict[int, int] = {}
        for bucket in self.SEQ_BUCKETS:
            chosen = self.MIN_CHUNK
            log.info("ChunkSizeProfiler: profiling bucket=%s", bucket)
            for chunk in self.CHUNK_CANDIDATES:
                hidden = None
                token_ids = None
                try:
                    hidden = torch.zeros(
                        (bucket, self.hidden_size),
                        dtype=self.dtype,
                        device=self.device,
                        requires_grad=True,
                    )
                    token_ids = torch.zeros((bucket,), dtype=torch.long, device=self.device)
                    out = self._chunked_lm_head_logprobs(hidden, token_ids, chunk)
                    loss = out.mean()
                    loss.backward()
                    chosen = chunk
                    break
                except RuntimeError as exc:
                    if not self._is_oom(exc):
                        raise
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    chosen = self.MIN_CHUNK
                finally:
                    del hidden, token_ids
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            table[int(bucket)] = int(chosen)
            log.info("ChunkSizeProfiler: bucket=%s -> chunk=%s", bucket, chosen)

        self.profile_table = table
        return dict(table)

    def get_chunk_size(self, t_c: int) -> int:
        if not self.profile_table:
            self.load_or_profile()
        for bucket in self.SEQ_BUCKETS:
            if t_c <= bucket:
                return int(self.profile_table.get(bucket, self.MIN_CHUNK))
        return self.MIN_CHUNK

    def invalidate(self) -> None:
        if self.cache_path.exists():
            self.cache_path.unlink()
        self.profile_table = {}
