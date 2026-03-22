from __future__ import annotations

import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from backprop import BackpropConfig, ChunkSizeProfiler, StreamingBackprop


@dataclass
class Sample:
    prompt_ids: List[int]
    completion_ids: List[int]


class SFTTrainer:
    def __init__(self, config) -> None:
        self.cfg = config
        self.tokenizer = None
        self.model = None
        self.backprop = None
        self.profiler = None
        self.optimizer = None
        self.scheduler = None
        self.checkpoint_dir = Path(self.cfg.CHECKPOINT_DIR)
        self.resume_lora_dir = self.checkpoint_dir / "latest_lora"
        self.resume_best_lora_dir = self.checkpoint_dir / "best_lora"
        self.progress_path = self.checkpoint_dir / "progress.json"
        self.trainer_state_path = self.checkpoint_dir / "trainer_state.pt"
        self.total_samples_processed = 0
        self.total_optimizer_steps = 0

    def train(self, dataset_path: str, val_path: str) -> str:
        train_rows = self._load_samples(dataset_path)
        val_rows = self._load_samples(val_path)
        if not train_rows:
            raise RuntimeError(f"No training rows in {dataset_path}")
        if not val_rows:
            raise RuntimeError(f"No validation rows in {val_path}")

        self._init_model()
        self._init_chunk_profiler()
        self._init_optimizer_scheduler(num_samples=len(train_rows), epochs=int(self.cfg.EPOCHS))

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        start_epoch, best_val_loss, best_epoch, patience_counter = self._maybe_resume()
        if start_epoch > int(self.cfg.EPOCHS):
            print(
                f"[sft_primer] checkpoint indicates training already complete "
                f"(last_epoch={start_epoch - 1}, configured_epochs={self.cfg.EPOCHS})",
                flush=True,
            )
            print(f"[sft_primer] RL pipeline checkpoint: {self.cfg.FINAL_LORA_DIR}", flush=True)
            return self.cfg.FINAL_LORA_DIR

        lora_dir = Path(self.cfg.LORA_DIR)
        final_dir = Path(self.cfg.FINAL_LORA_DIR)
        lora_dir.mkdir(parents=True, exist_ok=True)
        samples_before_run = int(self.total_samples_processed)

        for epoch in range(start_epoch, int(self.cfg.EPOCHS) + 1):
            train_loss, epoch_samples_processed = self._train_epoch(train_rows, epoch)
            val_loss = self._eval_epoch(val_rows)
            delta = val_loss - best_val_loss if math.isfinite(best_val_loss) else 0.0
            print(
                f"[sft_primer] epoch {epoch} | train_loss={train_loss:.4f} "
                f"| val_loss={val_loss:.4f} | delta={delta:+.4f} "
                f"| epoch_samples={epoch_samples_processed} "
                f"| total_samples={self.total_samples_processed}",
                flush=True,
            )

            epoch_dir = lora_dir / f"epoch_{epoch}"
            self.backprop.save_lora(str(epoch_dir))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                if final_dir.exists():
                    shutil.rmtree(final_dir)
                shutil.copytree(epoch_dir, final_dir)
                if self.resume_best_lora_dir.exists():
                    shutil.rmtree(self.resume_best_lora_dir)
                shutil.copytree(epoch_dir, self.resume_best_lora_dir)
            else:
                patience_counter += 1
            self._save_checkpoint(
                epoch,
                best_val_loss,
                best_epoch,
                patience_counter,
                epoch_samples_processed=epoch_samples_processed,
            )
            if patience_counter >= int(self.cfg.EARLY_STOPPING_PATIENCE):
                print("[sft_primer] Early stopping - val loss not improving", flush=True)
                break

        print(f"[sft_primer] best_epoch={best_epoch} best_val_loss={best_val_loss:.4f}", flush=True)
        print(
            f"[sft_primer] samples_processed_this_run={self.total_samples_processed - samples_before_run} "
            f"(cumulative={self.total_samples_processed})",
            flush=True,
        )
        print(f"[sft_primer] RL pipeline checkpoint: {self.cfg.FINAL_LORA_DIR}", flush=True)
        return self.cfg.FINAL_LORA_DIR

    def _maybe_resume(self) -> Tuple[int, float, int, int]:
        if not self.progress_path.exists():
            return 1, float("inf"), -1, 0

        progress = json.loads(self.progress_path.read_text(encoding="utf-8"))
        last_epoch = int(progress.get("last_completed_epoch", 0))
        best_val_loss = float(progress.get("best_val_loss", float("inf")))
        best_epoch = int(progress.get("best_epoch", -1))
        patience_counter = int(progress.get("patience_counter", 0))
        self.total_samples_processed = int(progress.get("total_samples_processed", 0))
        self.total_optimizer_steps = int(progress.get("total_optimizer_steps", 0))

        if self.resume_lora_dir.exists():
            self.backprop.load_lora(str(self.resume_lora_dir))
            print(f"[sft_primer] resumed LoRA from {self.resume_lora_dir}", flush=True)

        if self.trainer_state_path.exists():
            state = torch.load(self.trainer_state_path, map_location="cpu")
            opt_state = state.get("optimizer")
            sch_state = state.get("scheduler")
            if opt_state is not None:
                self.optimizer.load_state_dict(opt_state)
            if sch_state is not None:
                self.scheduler.load_state_dict(sch_state)
            print("[sft_primer] resumed optimizer/scheduler state", flush=True)

        if self.resume_best_lora_dir.exists() and not Path(self.cfg.FINAL_LORA_DIR).exists():
            shutil.copytree(self.resume_best_lora_dir, Path(self.cfg.FINAL_LORA_DIR))

        start_epoch = last_epoch + 1
        print(
            f"[sft_primer] resuming from checkpoint: start_epoch={start_epoch} "
            f"best_epoch={best_epoch} best_val_loss={best_val_loss:.4f} patience={patience_counter} "
            f"total_samples_processed={self.total_samples_processed} "
            f"total_optimizer_steps={self.total_optimizer_steps}",
            flush=True,
        )
        return start_epoch, best_val_loss, best_epoch, patience_counter

    def _save_checkpoint(
        self,
        completed_epoch: int,
        best_val_loss: float,
        best_epoch: int,
        patience_counter: int,
        epoch_samples_processed: int,
    ) -> None:
        if self.resume_lora_dir.exists():
            shutil.rmtree(self.resume_lora_dir)
        self.backprop.save_lora(str(self.resume_lora_dir))

        state = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        torch.save(state, self.trainer_state_path)

        progress = {
            "last_completed_epoch": int(completed_epoch),
            "best_val_loss": float(best_val_loss),
            "best_epoch": int(best_epoch),
            "patience_counter": int(patience_counter),
            "last_epoch_samples_processed": int(epoch_samples_processed),
            "total_samples_processed": int(self.total_samples_processed),
            "total_optimizer_steps": int(self.total_optimizer_steps),
        }
        self.progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")

    def _dtype(self) -> torch.dtype:
        m = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        return m.get(self.cfg.DTYPE, torch.bfloat16)

    def _device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _init_model(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except Exception as exc:
            raise RuntimeError("sft_primer.train requires PEFT") from exc

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.MODEL_PATH)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.MODEL_PATH,
            dtype=self._dtype(),
            low_cpu_mem_usage=True,
            attn_implementation=self.cfg.ATTN_IMPLEMENTATION,
        )

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=int(self.cfg.LORA_RANK),
            lora_alpha=int(self.cfg.LORA_ALPHA),
            lora_dropout=float(self.cfg.LORA_DROPOUT),
            target_modules=list(self.cfg.LORA_TARGET),
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_cfg)
        self.model.to(self._device())
        self.model.train()

        bp_cfg = BackpropConfig(
            top_frac=float(self.cfg.LORA_LAYERS_FRAC),
            use_grad_checkpoint=True,
            offload_prefix_cpu=True,
        )
        self.backprop = StreamingBackprop(self.model, config=bp_cfg)

    def _init_chunk_profiler(self) -> None:
        base, _ = self.backprop.adapter.unwrap(self.model)
        lm_head = self.backprop.adapter.get_lm_head(base)
        model_cfg = getattr(base, "config", getattr(self.model, "config", None))
        hidden_size = int(getattr(model_cfg, "hidden_size"))
        vocab_size = int(getattr(model_cfg, "vocab_size"))

        profiler = ChunkSizeProfiler(
            lm_head=lm_head,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            device=self._device(),
            model_path=self.cfg.MODEL_PATH,
            sglang_mem_frac=float(self.cfg.SGLANG_MEM_FRAC),
            top_frac=float(self.cfg.LORA_LAYERS_FRAC),
            cache_dir=self.cfg.CHUNK_PROFILE_DIR,
            dtype=self._dtype(),
        )
        if bool(self.cfg.FORCE_REPROFILE) or bool(getattr(self.cfg, "FORCE_REPROFLE", False)):
            profiler.invalidate()
        table = profiler.load_or_profile()
        self.backprop.chunk_profiler = profiler
        self.backprop.config.logit_chunk = profiler.get_chunk_size(int(self.cfg.MAX_COMPLETION_TOKENS))
        self.profiler = profiler
        pretty = ", ".join([f"{k}:{v}" for k, v in sorted(table.items())])
        print(f"[sft_primer] chunk_profile: {pretty}", flush=True)

    def _init_optimizer_scheduler(self, num_samples: int, epochs: int) -> None:
        params = [p for p in self.model.parameters() if p.requires_grad]
        if str(self.cfg.OPTIMIZER).lower() == "adamw_8bit":
            try:
                import bitsandbytes as bnb

                self.optimizer = bnb.optim.AdamW8bit(params, lr=self.cfg.LR, weight_decay=self.cfg.WEIGHT_DECAY)
            except Exception:
                self.optimizer = AdamW(params, lr=self.cfg.LR, weight_decay=self.cfg.WEIGHT_DECAY)
        else:
            self.optimizer = AdamW(params, lr=self.cfg.LR, weight_decay=self.cfg.WEIGHT_DECAY)

        total_updates = max(1, math.ceil((num_samples * epochs) / max(1, int(self.cfg.GRAD_ACCUM_STEPS))))
        warmup = int(total_updates * float(self.cfg.WARMUP_RATIO))

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return float(step) / float(max(1, warmup))
            progress = (step - warmup) / float(max(1, total_updates - warmup))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def _train_epoch(self, train_rows: List[Sample], epoch: int) -> Tuple[float, int]:
        rng = random.Random(int(self.cfg.SEED) + epoch)
        rows = list(train_rows)
        rng.shuffle(rows)

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        grad_accum = max(1, int(self.cfg.GRAD_ACCUM_STEPS))
        micro_batch_size = max(1, int(self.cfg.MICRO_BATCH_SIZE))
        epoch_samples_processed = 0
        accum_micro_steps = 0

        for mb_start in range(0, len(rows), micro_batch_size):
            mb = rows[mb_start : mb_start + micro_batch_size]
            mb_size = max(1, len(mb))
            for row in mb:
                prompt_ids = torch.tensor([row.prompt_ids], dtype=torch.long, device=self._device())
                completion_1d = torch.tensor(row.completion_ids, dtype=torch.long, device=self._device())
                completion_ids = completion_1d.unsqueeze(0)
                completion_mask = torch.ones_like(completion_ids, dtype=torch.float32)

                loss_fn = self._make_sft_loss_fn(completion_mask)
                metrics = self.backprop.backward_on_batch(
                    self.model,
                    prompt_ids,
                    completion_ids,
                    completion_mask,
                    loss_fn,
                    loss_scale=1.0 / (grad_accum * mb_size),
                    lora_path=None,
                )
                total_loss += float(metrics["loss"])
                epoch_samples_processed += 1
                self.total_samples_processed += 1

            accum_micro_steps += 1
            is_step = (accum_micro_steps % grad_accum == 0) or (mb_start + mb_size >= len(rows))
            if is_step:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg.MAX_GRAD_NORM))
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.total_optimizer_steps += 1

        return total_loss / max(1, len(rows)), epoch_samples_processed

    def _eval_epoch(self, val_rows: List[Sample]) -> float:
        self.model.eval()
        total_nll = 0.0
        total_tokens = 0

        with torch.inference_mode():
            for row in val_rows:
                prompt_ids = row.prompt_ids
                completion_ids = row.completion_ids
                full_ids = prompt_ids + completion_ids
                labels = [-100] * len(prompt_ids) + completion_ids

                input_ids = torch.tensor([full_ids], dtype=torch.long, device=self._device())
                label_ids = torch.tensor([labels], dtype=torch.long, device=self._device())

                out = self.model(input_ids=input_ids, labels=label_ids)
                token_count = max(1, len(completion_ids))
                total_nll += float(out.loss.item()) * token_count
                total_tokens += token_count

        self.model.train()
        return total_nll / max(1, total_tokens)

    @staticmethod
    def _make_sft_loss_fn(completion_mask: torch.Tensor) -> Callable:
        def loss_fn(log_probs, gen_idx: int, hidden_comp=None):
            del hidden_comp
            mask = completion_mask[gen_idx].to(log_probs.device)
            denom = mask.sum().clamp(min=1)
            return -((log_probs * mask).sum() / denom)

        return loss_fn

    @staticmethod
    def _load_samples(path: str) -> List[Sample]:
        rows: List[Sample] = []
        p = Path(path)
        if not p.exists():
            return rows
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                rows.append(Sample(prompt_ids=obj["prompt_ids"], completion_ids=obj["completion_ids"]))
        return rows


def train(dataset_path: str, val_path: str, config) -> str:
    trainer = SFTTrainer(config)
    return trainer.train(dataset_path, val_path)
