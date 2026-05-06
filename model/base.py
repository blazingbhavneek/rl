# BaseModel: the shared training loop for all model implementations.
#
# High-level overview of what this class does:
#
#   FORWARD (inference):
#     input tokens → embed → prefix layers (frozen) → suffix layers → lm_head → logits
#
#   BACKWARD (training):
#     The same pass, but split into two phases:
#       Phase 1 (no grad): run prefix layers once, cache result in a PrefixBundle.
#       Phase 2 (grad on): run suffix layers G times (once per completion),
#                          reusing the cached prefix — this is where LoRA lives.
#     The loss is computed inside Phase 2 in memory-efficient token chunks,
#     then .backward() accumulates gradients into the LoRA parameters.
#
#   WHY split prefix/suffix?
#     In RL training we generate G completions per prompt (e.g. G=8).
#     The prompt is the same for all G, so the prefix layers would compute
#     the exact same thing G times. By caching the prefix output, we run
#     those frozen layers only once per prompt — saving ~50% of compute.

from typing import Callable, Optional

import torch

from .utils.prefix import PrefixBundle


class BaseModel:

    def __init__(
        self,
        model_path: str,
        config: object,
        *,
        lora_path: str | None = None,
        lora_adapter_name: str = "default",
        lora_is_trainable: bool = True,
    ) -> None:

        # Shared runtime knobs used by model-specific subclasses.
        self.model_path = model_path

        self.config = config

        self.lora_path = lora_path

        self.lora_adapter_name = lora_adapter_name

        self.lora_is_trainable = bool(lora_is_trainable)

        self.cuda_device_index = getattr(config, "cuda_device_index", 0)

        self.chunk_size = getattr(config, "chunk_size")
        # chunk_size controls how many tokens are projected through lm_head at once.
        # The lm_head projects [hidden_dim → vocab_size]. For Gemma vocab ~256k,
        # doing the full sequence in one shot can OOM. Chunking caps peak memory.
        # None = no chunking (faster but more memory).

        self.logprob_chunk_size = getattr(config, "logprob_chunk_size", None)
        # Separate chunk size for log-probability computation specifically.
        # Falls back to chunk_size if not set. Useful if logprob passes need
        # a different memory budget than inference passes.

        self.use_streaming = self.chunk_size is not None

        self.setup()  # Subclass (e.g. Gemma4Model) loads the actual model here

    def forward(self, messages: list[list[dict]]):
        """Run chat-message batch through prefix -> suffix -> lm_head."""

        # Convert each conversation (list of {"role":..., "content":...} dicts)
        # into a plain string using the model's chat template.
        # e.g. "<start_of_turn>user\nSolve x^2=4<end_of_turn><start_of_turn>model\n"
        prompts = [
            self.tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=True,
            )
            for convo in messages
        ]

        # Tokenize all prompts together, padding shorter ones to match the longest.
        tokenized = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        )

        model_device = next(self.model.parameters()).device

        input_ids = tokenized["input_ids"].to(model_device, non_blocking=True)
        attention_mask = tokenized["attention_mask"].to(model_device, non_blocking=True)
        # attention_mask is 1 for real tokens, 0 for padding tokens.
        # Padding tokens are at the left (because tokenizer pads left by default).

        with torch.inference_mode():
            # inference_mode() is like no_grad() but also disables version tracking,
            # making it slightly faster and more memory-efficient.
            prefix_bundle = self._build_prefix_bundle(input_ids, attention_mask)
            hidden_suffix = self._run_suffix_from_prefix_bundle(
                prefix_bundle, attention_mask
            )
            return self._lm_head_logits_chunked(hidden_suffix)

    def _postprocess_logits(self, logits: torch.Tensor) -> torch.Tensor:
        # Base implementation: no post-processing.
        # Gemma4Model overrides this to apply logit softcapping.
        return logits

    def _build_prefix_bundle(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> PrefixBundle:
        # Base implementation: simple prefix with no shared KV states.
        # Gemma4Model overrides this with the full chunked + KV-sharing implementation.
        hidden_prefix, position_ids = self._forward_prefix(input_ids, attention_mask)
        return PrefixBundle(
            hidden_prefix=hidden_prefix,
            position_ids=position_ids,
            shared_kv_states={},
        )

    def _run_suffix_from_prefix_bundle(
        self,
        bundle: PrefixBundle,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Base implementation: detach the prefix output so gradients don't flow
        # back into the frozen prefix layers, then run the suffix.
        hidden_for_suffix = (
            bundle.hidden_prefix.clone()
            .detach()
            .requires_grad_(torch.is_grad_enabled())
            # .detach() cuts the computation graph at the prefix/suffix boundary.
            # .requires_grad_(True) re-enables grad tracking from this point onward,
            # so the suffix layers and LoRA weights get gradients.
        )
        position_ids = bundle.position_ids.clone().detach()
        return self._forward_suffix(hidden_for_suffix, position_ids, attention_mask)

    def _lm_head_logits_chunked(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to logits, optionally chunked over sequence."""
        lm_head = self._lm_head

        if self.chunk_size is None:
            # Fast path: single matrix multiply for the entire sequence at once.
            return self._postprocess_logits(lm_head(hidden_states))

        # Memory-efficient path: project chunk_size tokens at a time, then concatenate.
        # lm_head maps [hidden_dim=3072 → vocab=256k]. For a 4096-token sequence,
        # the full projection would need ~3GB just for the output logits. Chunking
        # caps this to chunk_size * vocab_size * 2 bytes (bfloat16) at any moment.
        seq_len = hidden_states.shape[1]
        chunk = max(1, int(self.chunk_size))
        token_chunks = []
        for start in range(0, seq_len, chunk):
            end = min(start + chunk, seq_len)
            logits_part = self._postprocess_logits(lm_head(hidden_states[:, start:end, :]))
            token_chunks.append(logits_part)
        return torch.cat(token_chunks, dim=1)

    def _token_logprobs_chunked(
        self,
        hidden_comp: torch.Tensor,  # [B, T_c, H] — hidden states for completion tokens only
        token_ids: torch.Tensor,    # [B, T_c]   — the actual token IDs at each position
    ) -> torch.Tensor:
        """
        Compute per-token log-probabilities for the target token IDs.

        For each position t, we want: log P(token_ids[t] | context up to t).
        This is: log_softmax(lm_head(hidden_comp[t]))[token_ids[t]].

        Shapes: hidden_comp [B, T_c, H] → logits [B, T_c, V] → logprobs [B, T_c]
        """
        lm_head = self._lm_head
        _, completion_len, _ = hidden_comp.shape

        chunk_source = self.logprob_chunk_size if self.logprob_chunk_size is not None else self.chunk_size
        chunk = max(1, int(chunk_source)) if chunk_source is not None else completion_len

        token_logprob_chunks = []
        for start in range(0, completion_len, chunk):
            end = min(start + chunk, completion_len)
            hidden_slice = hidden_comp[:, start:end, :]
            token_slice = token_ids[:, start:end].to(hidden_comp.device, non_blocking=True)

            logits = self._postprocess_logits(lm_head(hidden_slice))  # [B, t_chunk, V]
            token_logprobs = (
                torch.log_softmax(logits.float(), dim=-1)
                # .float() upcasts to fp32 for numerical precision — softmax over 256k vocab
                # can have very small values that lose precision in bfloat16.
                .gather(dim=-1, index=token_slice.unsqueeze(-1))
                # .gather picks out the log-prob of the specific token at each position.
                # unsqueeze(-1) adds a dim so gather's index shape matches.
                .squeeze(-1)  # Remove that extra dim: [B, t_chunk, 1] → [B, t_chunk]
            )
            token_logprob_chunks.append(token_logprobs)

        return torch.cat(token_logprob_chunks, dim=1)  # [B, T_c]

    def _backward_masked_mean_logprob_chunked(
        self,
        hidden_comp: torch.Tensor,   # [B, T_c, H]
        token_ids: torch.Tensor,     # [B, T_c]
        token_mask: torch.Tensor,    # [B, T_c] — 1 for real tokens, 0 for padding/prompt
        *,
        loss_scale: float,
    ) -> tuple[float, float]:
        # Memory-efficient backward for the simple "minimize negative log-prob" loss.
        # The trick: we never materialise all per-token logprobs at once.
        # Instead we:
        #   1. Detach hidden_comp and re-attach with requires_grad so we can
        #      accumulate a gradient w.r.t. hidden states chunk by chunk.
        #   2. For each chunk: compute logits → logprobs → chunk loss → .backward()
        #      This immediately frees the logit tensor (256k vocab) before the next chunk.
        #   3. After all chunks, propagate the accumulated gradient from hidden_comp_detach
        #      back through the rest of the graph (suffix layers → LoRA weights).
        lm_head = self._lm_head
        _, completion_len, _ = hidden_comp.shape
        chunk_source = (
            self.logprob_chunk_size
            if self.logprob_chunk_size is not None
            else self.chunk_size
        )
        chunk = (
            max(1, int(chunk_source)) if chunk_source is not None else completion_len
        )
        valid_tokens = token_mask.sum().clamp(min=1.0)
        total_logp_sum = 0.0

        # Detach from the main graph so chunk-wise .backward() calls don't interfere.
        # We'll manually propagate the gradient at the end.
        hidden_comp_detach = hidden_comp.detach().requires_grad_(True)

        for start in range(0, completion_len, chunk):
            end = min(start + chunk, completion_len)
            hidden_slice = hidden_comp_detach[:, start:end, :]
            token_slice = token_ids[:, start:end].to(
                hidden_comp_detach.device, non_blocking=True
            )
            mask_slice = token_mask[:, start:end]
            logits = self._postprocess_logits(lm_head(hidden_slice))
            token_logprobs = (
                torch.log_softmax(logits.float(), dim=-1)
                .gather(
                    dim=-1,
                    index=token_slice.unsqueeze(-1),
                )
                .squeeze(-1)
            )
            chunk_loss = -((token_logprobs * mask_slice).sum() / valid_tokens)
            # Negative because we *minimise* loss = maximise log-probability.
            (chunk_loss * float(loss_scale)).backward()
            # This .backward() accumulates grad into hidden_comp_detach.grad.
            # The logit tensor is freed immediately after, saving memory.
            with torch.no_grad():
                total_logp_sum += float((token_logprobs * mask_slice).sum().item())

        # Now propagate the accumulated gradient from the detached copy back
        # through the real hidden_comp tensor and into the suffix/LoRA layers.
        if hidden_comp_detach.grad is not None:
            hidden_comp.backward(gradient=hidden_comp_detach.grad)

        valid_tokens_value = float(valid_tokens.item())
        return -total_logp_sum / valid_tokens_value, total_logp_sum / valid_tokens_value

    def backward(
        self,
        messages: list[list[dict]],
        completion_texts: list[str],
        loss_fn: Callable,
        loss_scale: float = 1.0,
    ) -> dict[str, float]:
        """
        Compute and accumulate gradients for one rollout batch.
        Does NOT call optimizer.step() or zero_grad() — the caller does that.

        G = number of completions, T_prompt = prompt tokens, T_c = completion tokens.
        """
        if not completion_texts:
            raise ValueError("completion_texts cannot be empty")

        prompts = [
            self.tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=True
            )
            for convo in messages
        ]
        model_device = next(self.model.parameters()).device
        num_generations = len(completion_texts)

        # Allow passing a single prompt that is broadcast to all G completions.
        if len(prompts) == 1 and num_generations > 1:
            prompts = prompts * num_generations
        if len(prompts) != num_generations:
            raise ValueError(
                f"prompt batch ({len(prompts)}) must be 1 or match completion batch ({num_generations})"
            )

        # Tokenize prompts and completions separately so we know which tokens
        # belong to the prompt (frozen) vs the completion (trainable).
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

        # Build padded tensors manually so we have exact control over which
        # positions are prompt vs completion vs padding.
        full_ids_cpu = []
        full_mask_cpu = []
        target_ids_cpu = []
        completion_target_mask_cpu = []
        for prompt_ids, completion_ids, full_ids_list in zip(
            prompt_token_lists, completion_token_lists, full_token_lists
        ):
            pad_len = max_total_len - len(full_ids_list)
            full_ids_cpu.append(full_ids_list + ([pad_token_id] * pad_len))
            full_mask_cpu.append(([1] * len(full_ids_list)) + ([0] * pad_len))

            # Language modelling target: predict the next token at each position.
            # So target_ids[t] = full_ids[t+1], shifted left by one.
            target_ids_cpu.append(full_ids_list[1:] + ([pad_token_id] * (pad_len + 1)))

            # We only compute loss on completion tokens, not on prompt tokens or padding.
            # len(prompt_ids)-1 because target_ids is shifted — the first completion target
            # is at position len(prompt_ids)-1 in the target array.
            completion_target_mask_cpu.append(
                ([0] * max(0, len(prompt_ids) - 1))
                + ([1] * len(completion_ids))
                + ([0] * pad_len)
            )

        full_ids = torch.tensor(full_ids_cpu, device=model_device, dtype=torch.long)
        full_mask = torch.tensor(full_mask_cpu, device=model_device, dtype=torch.long)
        target_ids = torch.tensor(target_ids_cpu, device=model_device, dtype=torch.long)
        completion_mask = torch.tensor(
            completion_target_mask_cpu, device=model_device, dtype=torch.float32
        )

        # === PHASE 1: Frozen prefix (no gradients) ===
        # Run the prompt through the frozen prefix layers once, cache the output.
        with torch.inference_mode():
            prefix_bundle = self._build_prefix_bundle(full_ids, full_mask)

        # === PHASE 2: Trainable suffix (gradients enabled) ===
        with torch.enable_grad():
            hidden_suffix = self._run_suffix_from_prefix_bundle(prefix_bundle, full_mask)
            # hidden_suffix: [G, T_total, H]

            # We want log-probs at position t to predict token t+1, so slice off the last
            # hidden state (it has no target) and the first target (it's the prompt's first token).
            hidden_comp = hidden_suffix[:, :-1, :]  # [G, T_total-1, H]

            # Three possible loss paths, in order of preference:
            batch_loss_callable: Optional[Callable] = None
            explicit_batch = getattr(loss_fn, "loss_fn_batch", None)
            if callable(explicit_batch):
                batch_loss_callable = explicit_batch
            elif getattr(loss_fn, "__name__", "") == "loss_fn_batch":
                batch_loss_callable = loss_fn

            total_loss = 0.0
            mean_logp = None
            batch_logprobs = None

            if batch_loss_callable is not None:

                # TODO: Use this mode in GRPO too?
                stream_mode = getattr(batch_loss_callable, "_streaming_reduction", None)
                if stream_mode == "masked_mean_logprob":
                    # Most memory-efficient path: computes and backprops chunk-by-chunk,
                    # never materialising the full [G, T_total, V] logit tensor.
                    total_loss, mean_logp = self._backward_masked_mean_logprob_chunked(
                        hidden_comp, target_ids, completion_mask, loss_scale=float(loss_scale)
                    )
                else:
                    # Batch loss path: materialise all logprobs, let the algo compute loss.
                    batch_logprobs = self._token_logprobs_chunked(hidden_comp, target_ids)
                    batch_loss = batch_loss_callable(batch_logprobs, completion_mask, hidden_comp)
                    (batch_loss * float(loss_scale)).backward()
                    total_loss = float(batch_loss.item())
            else:
                # Legacy per-sample path: call loss_fn once per completion.
                # retain_graph=True needed for all but the last sample because all G
                # samples share the same computation graph (hidden_suffix).
                batch_logprobs = self._token_logprobs_chunked(hidden_comp, target_ids)
                for idx in range(num_generations):
                    sample_loss = loss_fn(batch_logprobs[idx], idx, hidden_comp[idx])
                    retain_graph = idx < (num_generations - 1)
                    (sample_loss * float(loss_scale)).backward(
                        retain_graph=retain_graph
                    )
                    total_loss += float(sample_loss.item())
                total_loss = total_loss / max(1, num_generations)

        # Compute mean log-prob for logging (not for loss).
        with torch.no_grad():
            valid_tokens = completion_mask.sum().clamp(min=1.0)
            if mean_logp is None:
                if batch_logprobs is None:
                    batch_logprobs = self._token_logprobs_chunked(
                        hidden_comp, target_ids
                    )
                mean_logp = float(
                    (batch_logprobs * completion_mask).sum().item()
                    / valid_tokens.item()
                )

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
        save_name = (
            adapter_name if adapter_name in actual_adapters else actual_adapters[0]
        )
        self.model.save_pretrained(save_path, selected_adapters=[save_name])
