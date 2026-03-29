import torch
from typing import Callable, Optional


class BaseModel:

    def __init__(self, model_path: str, config: object) -> None:

        # Shared runtime knobs used by model-specific subclasses.
        self.model_path = model_path
        
        self.config = config
        
        self.cuda_device_index = getattr(config, "cuda_device_index", 0)
        
        self.chunk_size = getattr(config, "chunk_size")
        
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
        chunk = max(1, int(self.chunk_size)) if self.chunk_size is not None else completion_len

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
        tokenized = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        )
        model_device = next(self.model.parameters()).device
        prompt_ids = tokenized["input_ids"].to(model_device, non_blocking=True)
        prompt_mask = tokenized["attention_mask"].to(model_device, non_blocking=True)
        # prompt_ids: [B_prompt, T_prompt]
        # prompt_mask: [B_prompt, T_prompt]

        completion_tok = self.tokenizer(
            completion_texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        completion_ids = completion_tok["input_ids"].to(model_device, non_blocking=True)
        completion_mask = completion_tok["attention_mask"].to(
            model_device, dtype=torch.float32, non_blocking=True
        )
        # completion_ids: [G, T_c]
        # completion_mask: [G, T_c]

        # Allow single prompt with batched completions.
        num_generations = completion_ids.shape[0]
        if prompt_ids.shape[0] == 1 and num_generations > 1:
            prompt_ids = prompt_ids.expand(num_generations, prompt_ids.shape[1])
            prompt_mask = prompt_mask.expand(num_generations, prompt_mask.shape[1])
        if prompt_ids.shape[0] != num_generations:
            raise ValueError(
                f"prompt batch ({prompt_ids.shape[0]}) must be 1 or match completion batch ({num_generations})"
            )

        full_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        full_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        completion_len = completion_ids.shape[1]
        # full_ids/full_mask: [G, T_total], where T_total = T_prompt + T_c

        # Frozen prefix once.
        with torch.inference_mode():
            hidden_prefix, pos_ids = self._forward_prefix(full_ids, full_mask)
        # hidden_prefix: [G, T_total, H], pos_ids: [G, T_total]

        # Trainable suffix once.
        hidden_for_suffix = hidden_prefix.clone().detach().requires_grad_(True)
        hidden_suffix = self._forward_suffix(hidden_for_suffix, pos_ids.detach(), full_mask)
        hidden_comp = hidden_suffix[:, -completion_len:, :]
        # hidden_suffix: [G, T_total, H]
        # hidden_comp: [G, T_c, H] (completion slice only)

        # Batched token logprobs.
        batch_logprobs = self._token_logprobs_chunked(hidden_comp, completion_ids)
        # batch_logprobs: [G, T_c]

        # Prefer batch loss; fallback to per-sample legacy style.
        batch_loss_callable: Optional[Callable] = None
        explicit_batch = getattr(loss_fn, "loss_fn_batch", None)
        if callable(explicit_batch):
            batch_loss_callable = explicit_batch
        elif getattr(loss_fn, "__name__", "") == "loss_fn_batch":
            batch_loss_callable = loss_fn

        total_loss = 0.0
        if batch_loss_callable is not None:
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
