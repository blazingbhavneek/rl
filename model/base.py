import torch


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
        
        attention_mask = tokenized["attention_mask"]
        attention_mask = attention_mask.to(model_device, non_blocking=True)

        # 3) Prefix is frozen: run under inference mode.
        with torch.inference_mode():
            hidden_prefix, pos_ids = self._forward_prefix(input_ids, attention_mask)

        # 4) Boundary handoff for trainable suffix path.
        hidden_for_suffix = hidden_prefix.clone().detach().requires_grad_(True)
        pos_for_suffix = pos_ids.clone().detach()
        
        del hidden_prefix
        del pos_ids

        # 5) Trainable suffix + output projection.
        hidden_suffix = self._forward_suffix(
            hidden_for_suffix,
            pos_for_suffix,
            attention_mask,
        )
        
        return self._lm_head_logits_chunked(hidden_suffix)

    def _lm_head_logits_chunked(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to logits, optionally chunked over sequence."""
        lm_head = self._lm_head
        weight = lm_head.weight
        bias = getattr(lm_head, "bias", None)

        # Fast path: one full projection.
        if self.chunk_size is None:
            logits = hidden_states @ weight.t()
            
            if bias is not None:
                logits = logits + bias
            
            return logits

        # Memory path: project small token slices, then stitch back on seq axis.
        seq_len = hidden_states.shape[1]
        chunk = max(1, int(self.chunk_size))
        token_chunks = []
        
        for start in range(0, seq_len, chunk):
            end = min(start + chunk, seq_len)
            h_slice = hidden_states[:, start:end, :]
            logits_part = h_slice @ weight.t()
            
            if bias is not None:
                logits_part = logits_part + bias
            
            token_chunks.append(logits_part)
        
        return torch.cat(token_chunks, dim=1)

    def backward(self) -> None:
        pass
