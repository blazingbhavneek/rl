import torch
from typing import Callable, Optional


class BaseModel:

    def __init__(self, model_path: str, config: object) -> None:

        # Shared runtime knobs used by model-specific subclasses.
        self.model_path = model_path
        
        self.config = config
        
        self.cuda_device_index = getattr(config, "cuda_device_index", 0)
        
        self.chunk_size = getattr(config, "chunk_size")
        
        self.logprob_chunk_size = getattr(config, "logprob_chunk_size", None)
        
        self.use_streaming = self.chunk_size is not None
        
        self.setup()

    # Subclasses provide concrete setup/forward implementations.
    def forward(self, messages: list[list[dict]]):
        """Run chat-message batch through prefix -> suffix -> lm_head."""
        
        # 1) Convert chat messages into plain prompt strings.
        prompts = [
            self.tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=True,
            )
            for convo in messages
        ]

        # 2) Tokenize and pad batch.
        tokenized = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        )
        
        model_device = next(self.model.parameters()).device
        
        input_ids = tokenized["input_ids"]
        input_ids = input_ids.to(model_device, non_blocking=True)
        # input_ids: [B, T_total]

        attention_mask = tokenized["attention_mask"]
        attention_mask = attention_mask.to(model_device, non_blocking=True)
        # attention_mask: [B, T_total]

        # 3) Prefix is frozen: run under inference mode.
        with torch.inference_mode():
            hidden_prefix, pos_ids = self._forward_prefix(input_ids, attention_mask)
        # hidden_prefix: [B, T_total, H]
        # pos_ids: [B, T_total]

        # 4) Boundary handoff for trainable suffix path.
        hidden_for_suffix = hidden_prefix.clone().detach().requires_grad_(True)
        pos_for_suffix = pos_ids.clone().detach()
        # hidden_for_suffix: [B, T_total, H] (leaf for suffix autograd)
        # pos_for_suffix: [B, T_total]
        
        del hidden_prefix
        del pos_ids

        # 5) Trainable suffix + output projection.
        hidden_suffix = self._forward_suffix(
            hidden_for_suffix,
            pos_for_suffix,
            attention_mask,
        )
        # hidden_suffix: [B, T_total, H] (post-norm suffix output)

        return self._lm_head_logits_chunked(hidden_suffix)

    def _lm_head_logits_chunked(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to logits, optionally chunked over sequence."""
        # hidden_states: [B, T, H]
        lm_head = self._lm_head

        # Fast/full path: exact single matmul.
        if self.chunk_size is None:
            logits = lm_head(hidden_states)
            # logits: [B, T, V]
            return logits

        # Memory path: project token slices, then stitch back on sequence axis.
        # This matches StreamBP-style chunking over sequence length.
        seq_len = hidden_states.shape[1]
        chunk = max(1, int(self.chunk_size))
        token_chunks = []

        for start in range(0, seq_len, chunk):
            end = min(start + chunk, seq_len)
            h_slice = hidden_states[:, start:end, :]
            # h_slice: [B, t_chunk, H]
            logits_part = lm_head(h_slice)
            # logits_part: [B, t_chunk, V]
            token_chunks.append(logits_part)

        return torch.cat(token_chunks, dim=1)
        # output logits: [B, T, V]

    def _token_logprobs_chunked(
        self,
        hidden_comp: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-token logprobs for target token ids.

        Naming legend:
        - B: batch size
        - T_c: completion length
        - H: hidden size
        - V: vocabulary size

        Data flow / Shapes:
        - hidden_comp: [B, T_c, H]
        - token_ids:   [B, T_c]
        - lm_head.weight: [V, H]
        - return:      [B, T_c]
        """
        lm_head = self._lm_head

        _, completion_len, _ = hidden_comp.shape
        chunk_source = self.logprob_chunk_size if self.logprob_chunk_size is not None else self.chunk_size
        chunk = max(1, int(chunk_source)) if chunk_source is not None else completion_len

        # Exact differentiable path: per token-slice compute full-vocab logits,
        # then log_softmax + gather. This matches HF math while controlling peak memory.
        token_logprob_chunks = []
        for start in range(0, completion_len, chunk):
            end = min(start + chunk, completion_len)
            hidden_slice = hidden_comp[:, start:end, :]
            token_slice = token_ids[:, start:end].to(hidden_comp.device, non_blocking=True)
            # hidden_slice: [B, t_chunk, H], token_slice: [B, t_chunk]

            # Use lm_head module call (same op path as HF) for better grad parity.
            logits = lm_head(hidden_slice)
            # logits: [B, t_chunk, V]

            token_logprobs = torch.log_softmax(logits.float(), dim=-1).gather(
                dim=-1,
                index=token_slice.unsqueeze(-1),
            ).squeeze(-1)
            # token_logprobs: [B, t_chunk]
            token_logprob_chunks.append(token_logprobs)

        return torch.cat(token_logprob_chunks, dim=1)

    def _backward_masked_mean_logprob_chunked(
        self,
        hidden_comp: torch.Tensor,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor,
        *,
        loss_scale: float,
    ) -> tuple[float, float]:
        lm_head = self._lm_head
        _, completion_len, _ = hidden_comp.shape
        chunk_source = self.logprob_chunk_size if self.logprob_chunk_size is not None else self.chunk_size
        chunk = max(1, int(chunk_source)) if chunk_source is not None else completion_len
        valid_tokens = token_mask.sum().clamp(min=1.0)
        total_logp_sum = 0.0

        for start in range(0, completion_len, chunk):
            end = min(start + chunk, completion_len)
            hidden_slice = hidden_comp[:, start:end, :]
            token_slice = token_ids[:, start:end].to(hidden_comp.device, non_blocking=True)
            mask_slice = token_mask[:, start:end]
            logits = lm_head(hidden_slice)
            token_logprobs = torch.log_softmax(logits.float(), dim=-1).gather(
                dim=-1,
                index=token_slice.unsqueeze(-1),
            ).squeeze(-1)
            chunk_loss = -((token_logprobs * mask_slice).sum() / valid_tokens)
            (chunk_loss * float(loss_scale)).backward(retain_graph=end < completion_len)
            with torch.no_grad():
                total_logp_sum += float((token_logprobs * mask_slice).sum().item())

        valid_tokens_value = float(valid_tokens.item())
        mean_logp = total_logp_sum / valid_tokens_value
        total_loss = -total_logp_sum / valid_tokens_value
        return total_loss, mean_logp

    def backward(
        self,
        messages: list[list[dict]],
        completion_texts: list[str],
        loss_fn: Callable,
        loss_scale: float = 1.0,
    ) -> dict[str, float]:
        """
        Compute and accumulate gradients for one rollout batch.

        This method does NOT call optimizer.step() or zero_grad().

        Naming legend:
        - G: number of completions in the rollout batch
        - T_prompt: prompt token length after padding
        - T_c: completion token length after padding
        - T_total: T_prompt + T_c
        """
        if not completion_texts:
            raise ValueError("completion_texts cannot be empty")

        prompts = [
            self.tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=True,
            )
            for convo in messages
        ]
        model_device = next(self.model.parameters()).device
        num_generations = len(completion_texts)
        if len(prompts) == 1 and num_generations > 1:
            prompts = prompts * num_generations
        if len(prompts) != num_generations:
            raise ValueError(
                f"prompt batch ({len(prompts)}) must be 1 or match completion batch ({num_generations})"
            )

        prompt_token_lists = [
            self.tokenizer(prompt, add_special_tokens=False, return_attention_mask=False)["input_ids"]
            for prompt in prompts
        ]
        completion_token_lists = [
            self.tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
            for text in completion_texts
        ]
        full_token_lists = [
            prompt_ids + completion_ids
            for prompt_ids, completion_ids in zip(prompt_token_lists, completion_token_lists)
        ]
        max_total_len = max(len(ids) for ids in full_token_lists)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("tokenizer.pad_token_id must be set")

        full_ids_cpu = []
        full_mask_cpu = []
        target_ids_cpu = []
        completion_target_mask_cpu = []
        for prompt_ids, completion_ids, full_ids_list in zip(
            prompt_token_lists,
            completion_token_lists,
            full_token_lists,
        ):
            pad_len = max_total_len - len(full_ids_list)
            full_ids_cpu.append(full_ids_list + ([pad_token_id] * pad_len))
            full_mask_cpu.append(([1] * len(full_ids_list)) + ([0] * pad_len))
            target_ids_cpu.append(full_ids_list[1:] + ([pad_token_id] * (pad_len + 1)))
            completion_target_mask_cpu.append(
                ([0] * max(0, len(prompt_ids) - 1))
                + ([1] * len(completion_ids))
                + ([0] * pad_len)
            )

        full_ids = torch.tensor(full_ids_cpu, device=model_device, dtype=torch.long)
        full_mask = torch.tensor(full_mask_cpu, device=model_device, dtype=torch.long)
        target_ids = torch.tensor(target_ids_cpu, device=model_device, dtype=torch.long)
        completion_mask = torch.tensor(
            completion_target_mask_cpu,
            device=model_device,
            dtype=torch.float32,
        )
        # full_ids/full_mask: [G, T_total]
        # target_ids/completion_mask: [G, T_total - 1]

        # Frozen prefix once.
        with torch.inference_mode():
            hidden_prefix, pos_ids = self._forward_prefix(full_ids, full_mask)
        # hidden_prefix: [G, T_total, H], pos_ids: [G, T_total]

        # Trainable suffix once.
        hidden_for_suffix = hidden_prefix.clone().detach().requires_grad_(True)
        hidden_suffix = self._forward_suffix(hidden_for_suffix, pos_ids.detach(), full_mask)
        hidden_comp = hidden_suffix[:, :-1, :]
        # hidden_suffix: [G, T_total, H]
        # hidden_comp: [G, T_total - 1, H]

        # Batched token logprobs.
        batch_logprobs = self._token_logprobs_chunked(hidden_comp, target_ids)
        # batch_logprobs: [G, T_total - 1]

        # Prefer batch loss; fallback to per-sample legacy style.
        batch_loss_callable: Optional[Callable] = None
        explicit_batch = getattr(loss_fn, "loss_fn_batch", None)
        if callable(explicit_batch):
            batch_loss_callable = explicit_batch
        elif getattr(loss_fn, "__name__", "") == "loss_fn_batch":
            batch_loss_callable = loss_fn

        total_loss = 0.0
        mean_logp = None
        if batch_loss_callable is not None:
            stream_mode = getattr(batch_loss_callable, "_streaming_reduction", None)
            if stream_mode == "masked_mean_logprob":
                total_loss, mean_logp = self._backward_masked_mean_logprob_chunked(
                    hidden_comp,
                    target_ids,
                    completion_mask,
                    loss_scale=float(loss_scale),
                )
            else:
                batch_loss = batch_loss_callable(batch_logprobs, completion_mask, hidden_comp)
                (batch_loss * float(loss_scale)).backward()
                total_loss = float(batch_loss.item())
        else:
            for idx in range(num_generations):
                sample_loss = loss_fn(batch_logprobs[idx], idx, hidden_comp[idx])
                retain_graph = idx < (num_generations - 1)
                (sample_loss * float(loss_scale)).backward(retain_graph=retain_graph)
                total_loss += float(sample_loss.item())
            total_loss = total_loss / max(1, num_generations)

        with torch.no_grad():
            valid_tokens = completion_mask.sum().clamp(min=1.0)
            if mean_logp is None:
                mean_logp = float((batch_logprobs * completion_mask).sum().item() / valid_tokens.item())

        return {
            "loss": float(total_loss),
            "mean_logp": mean_logp,
            "batch_size": float(num_generations),
            "valid_tokens": float(completion_mask.sum().item()),
        }

    def load_lora_adapter(
        self,
        adapter_name: str,
        adapter_path: str,
        *,
        is_trainable: bool = False,
    ) -> None:
        if not adapter_name:
            raise ValueError("adapter_name cannot be empty")
        if not adapter_path:
            raise ValueError("adapter_path cannot be empty")
        if not hasattr(self.model, "load_adapter"):
            raise RuntimeError("model does not support LoRA adapter loading")
        self.model.load_adapter(
            adapter_path,
            adapter_name=adapter_name,
            is_trainable=bool(is_trainable),
        )

    def set_active_lora_adapter(self, adapter_name: str) -> None:
        if not adapter_name:
            raise ValueError("adapter_name cannot be empty")
        if not hasattr(self.model, "set_adapter"):
            raise RuntimeError("model does not support LoRA adapter switching")
        self.model.set_adapter(adapter_name)

    def save_lora_adapter(self, adapter_name: str, save_path: str) -> None:
        if not hasattr(self.model, "peft_config"):
            raise RuntimeError("Model is not a PEFT model")
        actual_adapters = list(self.model.peft_config.keys())
        if not actual_adapters:
            raise RuntimeError("No LoRA adapters found in model")
        # PEFT names the adapter "default" when created via get_peft_model.
        # Use exact name if it exists, otherwise fall back to first available.
        save_name = adapter_name if adapter_name in actual_adapters else actual_adapters[0]
        self.model.save_pretrained(save_path, selected_adapters=[save_name])
