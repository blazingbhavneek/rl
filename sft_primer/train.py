from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, List, Literal

import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer


from model.config import ModelConfig
from model.gemma4 import Gemma4Model
from model.gptoss import GptOssModel
from model.qwen3 import Qwen3Model
from model.qwen3_5 import Qwen3_5Model

ModelFamily = Literal["qwen3", "qwen3_5", "gpt-oss", "gemma4"]

def detect_model_family(model_path: str) -> ModelFamily:
    lower = model_path.lower()
    if "gemma" in lower:
        return "gemma4"
    if "gpt-oss" in lower:
        return "gpt-oss"
    if "qwen3.5" in lower or "qwen3_5" in lower:
        return "qwen3_5"
    return "qwen3"


def build_train_samples(
    qa_pairs: List[Dict[str, str]],
    model_family: str,
    train_on_reasoning: bool = True,
) -> List[Dict]:
    """
    Convert raw QA rows into {messages, completion_text} dicts ready for
    trainer.backward().

    The prompt is kept as a message-dict list so backward() can call
    apply_chat_template(..., add_generation_prompt=True) internally.

    The completion is formatted manually using the model's literal token strings.
    For Gemma 4 we also inject the required system prompt if train_on_reasoning is enabled.
    """
    samples = []
    gemma_system = "<|think|>\nYou are a careful assistant. Think before answering."

    for row in qa_pairs:
        if "messages" in row and isinstance(row["messages"], list) and len(row["messages"]) >= 2:
            prompt_messages = row["messages"][:-1]
            last = row["messages"][-1]
            answer = str(last.get("content", "")).strip()
            reasoning = str(last.get("reasoning", "")).strip()
        else:
            question = str(row.get("question", "")).strip()
            if not question:
                continue
            prompt_messages = []
            if model_family == "gemma4" and train_on_reasoning:
                prompt_messages.append({"role": "system", "content": gemma_system})
                
            prompt_messages.append({"role": "user", "content": question})
            answer = str(row.get("answer", "")).strip()
            reasoning = str(row.get("reasoning", "")).strip()

        if not answer:
            continue

        if model_family == "gemma4":
            if train_on_reasoning and reasoning:
                completion_text = f"<|channel>thought\n{reasoning}<channel|>{answer}<turn|>"
            else:
                completion_text = f"{answer}<turn|>"
        else:  # qwen3, qwen3_5, gpt-oss
            if train_on_reasoning and reasoning:
                completion_text = f"<think>\n{reasoning}\n</think>\n{answer}<|im_end|>"
            else:
                completion_text = f"{answer}<|im_end|>"

        samples.append({"messages": prompt_messages, "completion_text": completion_text})

    return samples


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model_on_qa_pairs(
    qa_pairs: List[Dict[str, str]],
    train_model_path: str,
    output_dir: str,
    epochs: int = 1,
    lr: float = 2e-4,
    weight_decay: float = 0.01,
    grad_accum_steps: int = 1,
    max_grad_norm: float = 1.0,
    lora_rank: int = 128,
    lora_alpha: int = 256,
    lora_target: List[str] | None = None,
    lora_layers_frac: float = 0.25,
    seed: int = 42,
    train_batch_size: int = 8,
    train_on_reasoning: bool = True,
    chunk_size: int = 256,
    logprob_chunk_size: int | None = None,
    token_chunk_size: int | None = None,
    offload_prefix_to_cpu: bool = False,
    max_sample_tokens: int | None = None,
    save_every_steps: int = 0,          # 0 = disabled; N = save every N optimizer steps
) -> str:
    if not qa_pairs:
        raise ValueError("qa_pairs is empty.")

    model_family = detect_model_family(train_model_path)

    # --- model class + default LoRA targets ---
    if model_family == "gemma4":
        model_cls = Gemma4Model
        lora_target = lora_target or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    elif model_family == "gpt-oss":
        model_cls = GptOssModel
        lora_target = lora_target or ["q_proj", "k_proj", "v_proj", "o_proj"]
    elif model_family == "qwen3_5":
        model_cls = Qwen3_5Model
        lora_target = lora_target or ["gate_proj", "up_proj", "down_proj"]
    else:
        model_cls = Qwen3Model
        lora_target = lora_target or ["gate_proj", "up_proj", "down_proj"]

    # --- checkpoint resume ---
    model_key = (
        train_model_path.strip("/")
        .replace("\\", "_").replace("/", "_").replace(":", "_")
    )
    output_root = Path(output_dir)
    lora_model_dir = output_root / "lora" / model_key
    lora_model_dir.mkdir(parents=True, exist_ok=True)

    completed_epoch = 0
    for p in lora_model_dir.glob("epoch_*"):
        if p.is_dir() and p.name.split("_", 1)[1].isdigit():
            completed_epoch = max(completed_epoch, int(p.name.split("_", 1)[1]))
    if completed_epoch >= int(epochs):
        final_dir = lora_model_dir / "final"
        if final_dir.exists():
            return str(final_dir)

    # --- build samples ---
    rng = random.Random(int(seed))
    rows = list(qa_pairs)
    rng.shuffle(rows)

    train_samples = build_train_samples(rows, model_family, train_on_reasoning)
    if not train_samples:
        raise ValueError("No valid train samples after QA conversion.")

    max_sample_tokens = int(max_sample_tokens) if max_sample_tokens is not None else None
    if max_sample_tokens is not None:
        tokenizer = AutoTokenizer.from_pretrained(train_model_path)
        filtered_samples = []
        skipped_samples = 0
        max_seen_tokens = 0

        for sample in train_samples:
            prompt = tokenizer.apply_chat_template(
                sample["messages"],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_len = int(
                tokenizer(prompt, add_special_tokens=False, return_attention_mask=False)["input_ids"].__len__()
            )
            completion_len = int(
                tokenizer(
                    sample["completion_text"],
                    add_special_tokens=False,
                    return_attention_mask=False,
                )["input_ids"].__len__()
            )
            total_len = prompt_len + completion_len
            max_seen_tokens = max(max_seen_tokens, total_len)
            if total_len > max_sample_tokens:
                skipped_samples += 1
                continue
            filtered_samples.append(sample)

        print(
            f"[sft_primer] sample_length_filter "
            f"max_sample_tokens={max_sample_tokens} "
            f"kept={len(filtered_samples)} "
            f"skipped={skipped_samples} "
            f"max_seen={max_seen_tokens}",
            flush=True,
        )
        train_samples = filtered_samples
        if not train_samples:
            raise ValueError(
                f"All train samples were filtered out by max_sample_tokens={max_sample_tokens}."
            )

    # --- model ---
    train_cfg = ModelConfig(
        lora=list(lora_target),
        lora_fraction=float(lora_layers_frac),
        lora_rank=int(lora_rank),
        lora_alpha=int(lora_alpha),
        chunk_size=int(chunk_size),
        logprob_chunk_size=(int(logprob_chunk_size) if logprob_chunk_size is not None else None),
        token_chunk_size=(int(token_chunk_size) if token_chunk_size is not None else None),
        offload_prefix_to_cpu=bool(offload_prefix_to_cpu),
        cuda_device_index=0,
        use_grad_checkpoint=True,
        attn_implementation="sdpa",
    )
    trainer = model_cls(model_path=train_model_path, config=train_cfg)

    if completed_epoch > 0:
        from peft import PeftModel
        resume_dir = lora_model_dir / f"epoch_{completed_epoch}"
        if resume_dir.exists():
            trainer.model = PeftModel.from_pretrained(
                trainer.model, str(resume_dir), is_trainable=True,
            )

    trainer.model.train()

    # --- optimizer + scheduler ---
    optimizer = AdamW(
        [p for p in trainer.model.parameters() if p.requires_grad],
        lr=float(lr),
        weight_decay=float(weight_decay),
    )

    # We batch samples directly. The model backward path constructs each
    # sample's full prompt+completion sequence before padding, so mixed prompt
    # lengths do not create padding in the middle of the sequence.
    train_batch_size = max(1, int(train_batch_size))
    grad_accum_steps = max(1, int(grad_accum_steps))
    effective_accum = grad_accum_steps * train_batch_size   # total samples per update

    n_samples = len(train_samples)
    num_micro_batches = max(1, math.ceil(n_samples / train_batch_size))
    total_updates = max(1, math.ceil((num_micro_batches * int(epochs)) / grad_accum_steps))

    # Restore step_count so the LR scheduler continues its curve on resume.
    steps_per_epoch = max(1, math.ceil(num_micro_batches / grad_accum_steps))
    step_count = completed_epoch * steps_per_epoch

    def lr_lambda(step: int) -> float:
        warmup = max(20, int(0.1 * total_updates))
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_updates - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    # Advance the scheduler to where we left off after resume.
    for _ in range(step_count):
        scheduler.step()

    def loss_fn_batch(batch_log_probs, batch_mask, hidden_batch=None):
        del hidden_batch
        return -((batch_log_probs * batch_mask).sum() / batch_mask.sum().clamp(min=1.0))
    loss_fn_batch._streaming_reduction = "masked_mean_logprob"

    # --- training loop ---
    for epoch in range(completed_epoch + 1, int(epochs) + 1):
        rng.shuffle(train_samples)
        optimizer.zero_grad(set_to_none=True)
        running_valid_tokens = 0.0
        running_token_loss = 0.0
        seen_micro = 0

        pbar = tqdm(
            range(0, n_samples, train_batch_size),
            total=num_micro_batches,
            desc=f"Epoch {epoch}/{epochs}",
            unit="batch",
            dynamic_ncols=True,
        )
        for batch_start in pbar:
            batch = train_samples[batch_start: batch_start + train_batch_size]
            batch_size = len(batch)
            if not batch:
                continue

            # Compute how many samples belong to the current accumulation window
            # so that the last (possibly short) window is scaled correctly.
            update_idx = batch_start // effective_accum
            window_start = update_idx * effective_accum
            window_end = min(window_start + effective_accum, n_samples)
            window_size = window_end - window_start

            stats = trainer.backward(
                messages=[sample["messages"] for sample in batch],
                completion_texts=[sample["completion_text"] for sample in batch],
                loss_fn=loss_fn_batch,
                loss_scale=float(batch_size) / float(window_size),
            )
            seen_micro += batch_size
            running_valid_tokens += float(stats.get("valid_tokens", 0.0))
            running_token_loss += float(stats.get("loss", 0.0)) * float(stats.get("valid_tokens", 1.0))

            is_last_batch = (batch_start + train_batch_size >= n_samples)
            if seen_micro % effective_accum == 0 or is_last_batch:
                torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), float(max_grad_norm))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step_count += 1

                # --- step-level checkpoint ---
                if save_every_steps > 0 and step_count % save_every_steps == 0:
                    step_dir = lora_model_dir / f"step_{step_count}"
                    step_dir.mkdir(parents=True, exist_ok=True)
                    trainer.model.save_pretrained(str(step_dir))
                    print(f"[sft_primer] step_checkpoint saved: {step_dir}", flush=True)

            current_lr = scheduler.get_last_lr()[0]
            # Report token-weighted mean loss in the progress bar.
            mean_loss = running_token_loss / max(1.0, running_valid_tokens)
            pbar.set_postfix(
                loss=f"{float(stats.get('loss', 0.0)):.4f}",
                tok_mean=f"{mean_loss:.4f}",
                lr=f"{current_lr:.2e}",
                step=step_count,
            )

        pbar.close()
        epoch_mean_loss = running_token_loss / max(1.0, running_valid_tokens)
        print(
            f"[sft_primer] epoch={epoch} "
            f"tok_mean_loss={epoch_mean_loss:.4f} "
            f"optimizer_steps={step_count} "
            f"micro_steps={seen_micro} "
            f"valid_tokens={int(running_valid_tokens)} "
            f"samples={n_samples}",
            flush=True,
        )

        epoch_dir = lora_model_dir / f"epoch_{epoch}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(str(epoch_dir))

    final_dir = lora_model_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(final_dir))
    return str(final_dir)
