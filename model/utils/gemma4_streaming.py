# gemma4_streaming.py — chunked/streaming execution of Gemma 4 transformer layers.
#
# Why does this file exist?
#   During training with gradient checkpointing, we want to process the sequence
#   in small token chunks rather than all at once. This trades extra compute
#   (we re-run some operations) for much lower peak GPU memory.
#
#   The challenge: transformer attention is *global* — every query token attends
#   to every key/value token. So when we process a chunk of query tokens, we still
#   need the full K/V history up to that point.
#
#   This file solves that by:
#     1. Pre-computing K/V for the entire sequence ("prepare_for_replay").
#     2. Processing query chunks one at a time, slicing K/V as needed.
#     3. Implementing a custom autograd Function (_StreamCheckpointFunction) that
#        runs forward in no-grad chunks (saving memory) and reruns each chunk
#        during backward (gradient checkpointing).

from typing import Callable

import torch
import torch._functorch.config as _functorch_config
import torch.nn as nn

# Donated buffers are incompatible with retain_graph=True used in
# _StreamCheckpointFunction.backward. Disable globally before any compilation.
_functorch_config.donated_buffer = False

from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)

from .attn import apply_rotary_pos_emb as _apply_rotary_pos_emb
from .attn import repeat_kv as _repeat_kv


def _slice_shared_kv_states(
    shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]],
    seq_end: int | None,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    # When processing a chunk ending at position `seq_end`, KV-sharing layers
    # should only see K/V up to that position (causal constraint).
    if seq_end is None:
        return shared_kv_states
    return {
        key: (
            key_states[:, :, :seq_end, :],
            value_states[:, :, :seq_end, :],
        )
        for key, (key_states, value_states) in shared_kv_states.items()
    }


def _normalize_chunk_size(chunk_size: int | None) -> int:
    # Convert None or 0 to 0 (meaning "disabled"), positive int stays as-is.
    chunk = int(chunk_size or 0)
    return chunk if chunk > 0 else 0


def _get_chunk_kv_range(
    *,
    start: int,     # first query token index in this chunk
    end: int,       # one past last query token index
    sliding_window: int | None,  # window size for sliding-window attention layers
) -> tuple[int, int]:
    # For a chunk of query tokens [start, end), determine which K/V tokens are needed.
    # Full attention: need all tokens from position 0 up to `end`.
    # Sliding-window attention: only need the last `sliding_window` tokens before `end`.
    if sliding_window is None:
        return 0, end
    return max(0, start - sliding_window + 1), end


# =============================================================================
# _StreamCheckpointFunction — custom autograd for chunked gradient checkpointing
# =============================================================================

class _StreamCheckpointFunction(torch.autograd.Function):
    # Standard gradient checkpointing (torch.utils.checkpoint) reruns the entire
    # layer during backward. Our version reruns it *chunk by chunk*, which means
    # we never need to hold the full-sequence activation in memory during backward.
    #
    # Forward: run each chunk with no_grad, store only the input hidden states.
    # Backward: rerun each chunk with grad enabled, call backward on each chunk.

    @staticmethod
    def forward(
        ctx,
        run_function: Callable,  # A _LayerReplay object wrapping one transformer layer
        chunk_size: int,
        offload_to_cpu: bool,    # If True, move saved hidden states to CPU to free GPU RAM
        hidden_states: torch.Tensor,  # [batch, seq_len, hidden_dim]
    ) -> torch.Tensor:
        ctx.run_function = run_function
        ctx.chunk_size = max(1, int(chunk_size))

        # Save the input hidden states for the backward pass (needed to recompute).
        saved_hidden = hidden_states.detach()
        if offload_to_cpu and saved_hidden.device.type != "cpu":
            saved_hidden = saved_hidden.to("cpu", non_blocking=True)
            # non_blocking=True lets the CPU transfer happen in the background
            # while the GPU continues working on the next chunk.
        ctx.save_for_backward(saved_hidden)

        seq_len = hidden_states.shape[1]
        with torch.no_grad():
            if hasattr(ctx.run_function, "prepare_for_replay"):
                # Pre-compute K/V for the full sequence so individual chunks can slice it.
                ctx.run_function.prepare_for_replay(hidden_states, requires_grad=False)
            outputs = torch.empty_like(hidden_states)
            for start in range(0, seq_len, ctx.chunk_size):
                end = min(start + ctx.chunk_size, seq_len)
                outputs[:, start:end, :] = run_function(
                    hidden_states, chunk_range=(start, end)
                )
            if hasattr(ctx.run_function, "clear_replay_cache"):
                ctx.run_function.clear_replay_cache()
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Retrieve the saved input and move it back to GPU if it was offloaded.
        (hidden_states_saved,) = ctx.saved_tensors
        hidden_states_detached = (
            hidden_states_saved.to(device=grad_output.device, non_blocking=True)
            .detach()
            .requires_grad_(True)
        )
        seq_len = grad_output.shape[1]

        shared_graph = False
        if hasattr(ctx.run_function, "prepare_for_replay"):
            # Re-run K/V precompute with grad enabled. The K/V computation shares
            # a graph across all chunks, so we set retain_graph=True for all but the last.
            with torch.enable_grad():
                ctx.run_function.prepare_for_replay(
                    hidden_states_detached, requires_grad=True
                )
            shared_graph = True

        try:
            for start in range(0, seq_len, ctx.chunk_size):
                end = min(start + ctx.chunk_size, seq_len)
                with torch.enable_grad():
                    # Recompute the forward pass for this chunk (gradient checkpointing).
                    output_chunk = ctx.run_function(
                        hidden_states_detached, chunk_range=(start, end)
                    )
                    torch.autograd.backward(
                        output_chunk,
                        grad_tensors=grad_output[:, start:end, :].detach(),
                        # retain_graph=True if there are more chunks — the K/V graph
                        # is shared across all chunks and must not be freed early.
                        retain_graph=shared_graph and end < seq_len,
                    )
        finally:
            if hasattr(ctx.run_function, "clear_replay_cache"):
                ctx.run_function.clear_replay_cache()

        # Return gradients for each argument of forward().
        # None for run_function, chunk_size, offload_to_cpu (not tensors).
        return None, None, None, hidden_states_detached.grad


_COMPILE_MODE = "default"


def _stream_attention_forward(
    module,
    query: torch.Tensor,   # [batch, heads, chunk_len, head_dim]
    key: torch.Tensor,     # [batch, kv_heads, kv_len, head_dim]
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Standard scaled dot-product attention, implemented manually (not Flash Attention)
    # so it works with arbitrary chunk sizes and custom masks.
    key_states = _repeat_kv(key, module.num_key_value_groups)
    value_states = _repeat_kv(value, module.num_key_value_groups)

    # Scaled dot-product: (Q @ K^T) / sqrt(head_dim)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * module.scaling

    # Gemma 4 uses logit softcapping: tanh(w/cap)*cap keeps attention weights bounded.
    # This prevents any single token from dominating attention with extreme logits.
    softcap = getattr(module.config, "attn_logit_softcapping", None)
    if softcap is not None:
        attn_weights = torch.tanh(attn_weights / softcap) * softcap

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
        # The mask is an additive bias: 0 for allowed positions, -inf for masked.
        # Adding -inf before softmax makes those positions contribute ~0 to attention.

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    if module.training and module.attention_dropout:
        attn_weights = nn.functional.dropout(attn_weights, p=module.attention_dropout, training=True)

    attn_output = torch.matmul(attn_weights, value_states)
    return attn_output.transpose(1, 2), attn_weights  # transpose back to [batch, seq, heads, dim]


# =============================================================================
# _LayerReplay — wraps a single transformer layer for chunked execution
# =============================================================================

class _LayerReplay:
    # This is the callable that _StreamCheckpointFunction receives.
    # It bundles one transformer layer with all the context it needs
    # (masks, position embeddings, shared KV, per-layer inputs) so that
    # calling layer_replay(hidden, chunk_range=(start, end)) just works.

    def __init__(
        self,
        layer,
        abs_idx: int,              # Absolute layer index in the full transformer stack
        current_per_layer_input_source: torch.Tensor | None,
        self_outer,                # Reference to the parent Gemma4Model (to call helper methods)
        mask_mapping: dict,
        position_embeddings: dict,
        position_ids: torch.Tensor,
        shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]],
    ):
        self.layer = layer
        self.abs_idx = abs_idx
        self.current_per_layer_input_source = current_per_layer_input_source
        self.self_outer = self_outer
        self.mask_mapping = mask_mapping
        self.position_embeddings = position_embeddings
        self.position_ids = position_ids
        self.shared_kv_states = shared_kv_states
        self.shared_kv_local: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None
        self._materialized_shared_kv: (
            dict[int, tuple[torch.Tensor, torch.Tensor]] | None
        ) = None

    def _get_materialized_shared_kv(
        self, device: torch.device,
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        # Lazily move shared KV states to the right device and cache the result.
        # "Materialise" = move from CPU (if offloaded) to GPU, once per layer replay.
        if self._materialized_shared_kv is None:
            self._materialized_shared_kv = self.self_outer._materialize_shared_kv_states(
                self.shared_kv_states, device, None,
            )
        return self._materialized_shared_kv

    def prepare_for_replay(self, h: torch.Tensor, requires_grad: bool) -> None:
        # Pre-compute K/V for the full sequence so chunks can slice it cheaply.
        # Called once before chunked processing begins (both in forward and backward).
        self.shared_kv_local = self._get_materialized_shared_kv(h.device)
        if isinstance(self.layer, _StreamGemmaDecoderLayer):
            self.layer.prepare_for_replay(
                h,
                self.position_embeddings[self.self_outer._layer_types[self.abs_idx]],
                self.shared_kv_local,
            )

    def clear_replay_cache(self) -> None:
        # Free K/V cache after all chunks are done to avoid holding it in memory.
        if isinstance(self.layer, _StreamGemmaDecoderLayer):
            self.layer.clear_replay_cache()
        self.shared_kv_local = None
        self._materialized_shared_kv = None

    def __call__(
        self,
        h: torch.Tensor,
        chunk_range: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        seq_end = None if chunk_range is None else chunk_range[1]

        # Slice per-layer input to match the current chunk, if applicable.
        per_layer_input_local = None
        if self.current_per_layer_input_source is not None:
            if chunk_range is None:
                per_layer_input_local = self.self_outer._move_tensor(
                    self.current_per_layer_input_source, h.device
                )
            else:
                per_layer_input_local = self.self_outer._move_tensor(
                    self.current_per_layer_input_source[:, chunk_range[0]:chunk_range[1], :],
                    h.device,
                )

        # Get shared KV, sliced to only include tokens up to the current chunk end.
        full_shared_kv = self.shared_kv_local
        if full_shared_kv is None:
            full_shared_kv = self._get_materialized_shared_kv(h.device)
        if seq_end is None or seq_end == h.shape[1]:
            shared_kv_local = full_shared_kv
        else:
            shared_kv_local = _slice_shared_kv_states(full_shared_kv, seq_end)

        output = _run_layer(
            self.layer, h, self.mask_mapping, self.position_embeddings,
            self.position_ids, shared_kv_local,
            self.self_outer._layer_types[self.abs_idx],
            per_layer_input=per_layer_input_local,
            chunk_range=chunk_range,
        )

        # After processing the last chunk, propagate any newly computed shared KV
        # states back to the master dict so subsequent layers can use them.
        if seq_end is None or seq_end == h.shape[1]:
            for key, value in full_shared_kv.items():
                self.shared_kv_states[key] = value
        return output


# =============================================================================
# _StreamGemmaAttention — attention module that supports chunked query processing
# =============================================================================

class _StreamGemmaAttention(nn.Module):
    # Wraps the original HuggingFace Gemma attention module.
    # When called without chunk_range, it delegates directly to the base attention.
    # When called with chunk_range, it handles:
    #   - Computing Q for only the chunk's tokens
    #   - Retrieving K/V for the appropriate range (from cache or recomputing)
    #   - Building the correct causal mask for this chunk

    def __init__(self, base_attn: nn.Module):
        super().__init__()
        self.base_attn = base_attn
        self._replay_key_states: torch.Tensor | None = None
        self._replay_value_states: torch.Tensor | None = None

    def __getattr__(self, name: str):
        # Transparent attribute forwarding: anything not found on this wrapper
        # (like q_proj, k_proj, config, layer_idx, etc.) is looked up on base_attn.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_attn, name)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None,
        shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]],
        past_key_values=None,
        chunk_range: tuple[int, int] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        full_hidden_seq_len = kwargs.pop("full_hidden_seq_len", hidden_states.shape[1])

        if chunk_range is None:
            # No chunking: delegate to the unmodified HuggingFace attention.
            return self.base_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                shared_kv_states=shared_kv_states,
                past_key_values=past_key_values,
                **kwargs,
            )

        batch_size = hidden_states.shape[0]
        start, end = chunk_range
        chunk_len = end - start
        cos, sin = position_embeddings

        # For sliding-window attention layers, we don't need the full KV history —
        # only the last `sliding_window` tokens are relevant for this chunk.
        sliding_window = None
        if getattr(self, "layer_type", None) == "sliding_attention":
            sliding_window = getattr(self.config, "sliding_window", None)
        kv_start, kv_end = _get_chunk_kv_range(start=start, end=end, sliding_window=sliding_window)

        # need_full_store: whether this layer needs to save its K/V for a later layer to reuse.
        # Only happens at the last chunk (end == full sequence length), so we compute
        # the full K/V once and store it, rather than partially storing per chunk.
        need_full_store = (
            self.store_full_length_kv
            and not self.is_kv_shared_layer
            and end == full_hidden_seq_len
        )

        # Compute Q for just the chunk tokens.
        query_input = (
            hidden_states if hidden_states.shape[1] == chunk_len
            else hidden_states[:, start:end, :]
        )
        query_states = self.q_proj(query_input).view(batch_size, chunk_len, -1, self.head_dim)
        query_states = self.q_norm(query_states)
        # Apply RoPE rotation at the chunk's actual positions (start to end).
        query_states = _apply_rotary_pos_emb(
            query_states, cos[:, start:end, :], sin[:, start:end, :], unsqueeze_dim=2,
        ).transpose(1, 2)  # → [batch, heads, chunk_len, head_dim]

        full_key_states = None
        full_value_states = None

        if self._replay_key_states is not None and self._replay_value_states is not None:
            # Fast path: K/V was precomputed for the full sequence in prepare_replay_kv.
            # Just slice to the range this chunk needs.
            key_states = self._replay_key_states[:, :, kv_start:kv_end, :]
            value_states = self._replay_value_states[:, :, kv_start:kv_end, :]
            if need_full_store:
                full_key_states = self._replay_key_states[:, :, :end, :]
                full_value_states = self._replay_value_states[:, :, :end, :]

        elif self.is_kv_shared_layer:
            # This layer reuses K/V from another layer (KV-sharing, a Gemma 4 feature).
            # Just fetch from the shared dict and slice.
            key_states, value_states = shared_kv_states[self.kv_shared_layer_index]
            key_states = key_states[:, :, kv_start:kv_end, :].to(query_states.device, non_blocking=True)
            value_states = value_states[:, :, kv_start:kv_end, :].to(query_states.device, non_blocking=True)

        else:
            # No cache: compute K/V from the hidden states for the KV range.
            kv_hidden = hidden_states[:, kv_start:kv_end, :]
            kv_len = kv_end - kv_start
            key_states = self.k_proj(kv_hidden).view(batch_size, kv_len, -1, self.head_dim)
            value_states = (
                self.v_proj(kv_hidden).view(batch_size, kv_len, -1, self.head_dim)
                if self.v_proj is not None else key_states
            )
            key_states = self.k_norm(key_states)
            key_states = _apply_rotary_pos_emb(
                key_states, cos[:, kv_start:kv_end, :], sin[:, kv_start:kv_end, :], unsqueeze_dim=2,
            ).transpose(1, 2)
            value_states = self.v_norm(value_states).transpose(1, 2)

            if need_full_store:
                # We need full K/V for storage. If the KV range already starts at 0,
                # we already have it. Otherwise recompute from the full range.
                if kv_start == 0:
                    full_key_states = key_states
                    full_value_states = value_states
                else:
                    full_kv_hidden = hidden_states[:, :end, :]
                    full_key_states = self.k_proj(full_kv_hidden).view(batch_size, end, -1, self.head_dim)
                    full_value_states = (
                        self.v_proj(full_kv_hidden).view(batch_size, end, -1, self.head_dim)
                        if self.v_proj is not None else full_key_states
                    )
                    full_key_states = self.k_norm(full_key_states)
                    full_key_states = _apply_rotary_pos_emb(
                        full_key_states, cos[:, :end, :], sin[:, :end, :], unsqueeze_dim=2,
                    ).transpose(1, 2)
                    full_value_states = self.v_norm(full_value_states).transpose(1, 2)

        if need_full_store:
            if full_key_states is None or full_value_states is None:
                raise RuntimeError(
                    "full shared kv store requested but full kv states were not prepared"
                )
            shared_kv_states[self.layer_idx] = (full_key_states, full_value_states)

        # Build the attention mask for just this chunk's query vs KV range.
        chunk_mask = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # 2D mask means we're in chunked mode — build a precise slice mask.
                chunk_mask = _make_chunk_attention_mask(
                    attention_mask, start=start, end=end,
                    kv_start=kv_start, kv_end=kv_end,
                    dtype=query_states.dtype, device=query_states.device,
                    sliding_window=sliding_window,
                )
            else:
                # 4D mask was pre-built for the full sequence — just slice it.
                chunk_mask = attention_mask[:, :, start:end, kv_start:kv_end]

        attn_output, attn_weights = _stream_attention_forward(
            self, query_states, key_states, value_states, chunk_mask,
        )
        attn_output = attn_output.reshape(batch_size, chunk_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    def prepare_replay_kv(
        self,
        hidden_states: torch.Tensor,       # Full-sequence hidden states
        position_embeddings: torch.Tensor,
        shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        # Pre-compute K/V for the entire sequence and cache it.
        # This is called once before chunked processing so individual chunks
        # don't each have to recompute K/V from scratch.
        cos, sin = position_embeddings
        if self.is_kv_shared_layer:
            # KV-sharing: just point to what the shared layer already computed.
            key_states, value_states = shared_kv_states[self.kv_shared_layer_index]
            self._replay_key_states = key_states
            self._replay_value_states = value_states
            return

        batch_size, seq_len = hidden_states.shape[:2]
        key_states = self.k_proj(hidden_states).view(batch_size, seq_len, -1, self.head_dim)
        value_states = (
            self.v_proj(hidden_states).view(batch_size, seq_len, -1, self.head_dim)
            if self.v_proj is not None else key_states
        )
        key_states = self.k_norm(key_states)
        key_states = _apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2).transpose(1, 2)
        value_states = self.v_norm(value_states).transpose(1, 2)
        self._replay_key_states = key_states
        self._replay_value_states = value_states
        if self.store_full_length_kv:
            # Store in the shared dict so downstream KV-sharing layers can find it.
            shared_kv_states[self.layer_idx] = (key_states, value_states)

    def clear_replay_cache(self) -> None:
        self._replay_key_states = None
        self._replay_value_states = None


# =============================================================================
# _StreamGemmaDecoderLayer — wraps one full transformer layer for chunked forward
# =============================================================================

class _StreamGemmaDecoderLayer(nn.Module):
    # Wraps a single Gemma decoder layer (attention + MLP + norms) to support chunked execution.
    # When chunk_range is None, it passes through to the original layer unchanged.
    # When chunk_range=(start, end), it manually computes just those query positions
    # while still attending to the full KV context.

    def __init__(self, base_layer: nn.Module, use_compile: bool = False):
        super().__init__()
        self.base_layer = base_layer
        self._use_compile = use_compile
        # Wrap the attention module too, so it also understands chunk_range.
        if not isinstance(self.base_layer.self_attn, _StreamGemmaAttention):
            self.base_layer.self_attn = _StreamGemmaAttention(self.base_layer.self_attn)
        self._replay_shared_kv_states: (
            dict[int, tuple[torch.Tensor, torch.Tensor]] | None
        ) = None

    def __getattr__(self, name: str):
        # Transparent forwarding: let callers access base_layer attributes directly.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_layer, name)

    def forward(
        self,
        hidden_states: torch.Tensor,
        per_layer_input: torch.Tensor = None,
        shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
        position_embeddings: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        chunk_range: tuple[int, int] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if chunk_range is None:
            # Full-sequence path: no chunking, delegate to the original layer.
            return self.base_layer(
                hidden_states=hidden_states,
                per_layer_input=per_layer_input,
                shared_kv_states=shared_kv_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

        # Chunked path: process only tokens [start, end) but with full KV context.
        start, end = chunk_range
        chunk_len = end - start
        residual = hidden_states[:, start:end, :]  # The skip connection for this chunk
        full_hidden_seq_len = hidden_states.shape[1]

        use_replay_cache = self._replay_shared_kv_states is not None
        if use_replay_cache:
            # K/V was precomputed — apply LayerNorm only to the chunk tokens.
            hidden_states_norm = self.base_layer.input_layernorm(hidden_states[:, start:end, :])
        else:
            # No cache — pass hidden states up to `end` so attention can compute K/V inline.
            hidden_states_norm = self.base_layer.input_layernorm(hidden_states[:, :end, :])

        attn_out, _ = self.base_layer.self_attn(
            hidden_states=hidden_states_norm,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            shared_kv_states=self._replay_shared_kv_states or shared_kv_states,
            position_ids=position_ids,
            past_key_values=past_key_values,
            chunk_range=chunk_range,
            full_hidden_seq_len=full_hidden_seq_len,
            **kwargs,
        )
        # Post-attention residual: LayerNorm → add skip connection
        hidden_states_chunk = self.base_layer.post_attention_layernorm(attn_out)
        hidden_states_chunk = residual + hidden_states_chunk

        # MLP block
        residual = hidden_states_chunk
        hidden_states_chunk = self.base_layer.pre_feedforward_layernorm(hidden_states_chunk)
        hidden_states_chunk = self.base_layer.mlp(hidden_states_chunk)

        if getattr(self.base_layer, "enable_moe_block", False):
            # Mixture-of-Experts (MoE) block: Gemma 4 alternates between dense MLP
            # layers and MoE layers. In MoE, a router decides which expert(s) to use
            # for each token, allowing the model to specialise different experts for
            # different types of content without increasing inference cost.
            hidden_states_1 = self.base_layer.post_feedforward_layernorm_1(hidden_states_chunk)
            hidden_states_flat = residual.reshape(-1, residual.shape[-1])
            _, top_k_weights, top_k_index = self.base_layer.router(hidden_states_flat)
            # Router selects the top-k experts for each token. top_k_index says which
            # experts, top_k_weights says how much to weight each expert's output.
            hidden_states_2 = self.base_layer.pre_feedforward_layernorm_2(hidden_states_flat)
            hidden_states_2 = self.base_layer.experts(hidden_states_2, top_k_index, top_k_weights)
            hidden_states_2 = hidden_states_2.reshape(residual.shape)
            hidden_states_2 = self.base_layer.post_feedforward_layernorm_2(hidden_states_2)
            hidden_states_chunk = hidden_states_1 + hidden_states_2

        hidden_states_chunk = self.base_layer.post_feedforward_layernorm(hidden_states_chunk)
        hidden_states_chunk = residual + hidden_states_chunk

        if getattr(self.base_layer, "hidden_size_per_layer_input", 0):
            # Per-layer input: Gemma 4 can inject a small extra vector into each layer.
            # This is gated and projected before adding to the residual stream.
            residual = hidden_states_chunk
            if per_layer_input is None:
                raise RuntimeError("per_layer_input is required for chunked Gemma4 replay")
            if per_layer_input.shape[1] != chunk_len:
                per_layer_input = per_layer_input[:, start:end, :]
            hidden_states_chunk = self.base_layer.per_layer_input_gate(hidden_states_chunk)
            hidden_states_chunk = self.base_layer.act_fn(hidden_states_chunk)
            hidden_states_chunk = hidden_states_chunk * per_layer_input
            hidden_states_chunk = self.base_layer.per_layer_projection(hidden_states_chunk)
            hidden_states_chunk = self.base_layer.post_per_layer_input_norm(hidden_states_chunk)
            hidden_states_chunk = residual + hidden_states_chunk

        # layer_scalar is a learned per-layer scale factor applied to the output.
        return hidden_states_chunk * self.base_layer.layer_scalar

    def prepare_for_replay(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        # Store the shared KV dict reference and trigger K/V precomputation in attention.
        self._replay_shared_kv_states = shared_kv_states
        hidden_states_norm = self.base_layer.input_layernorm(hidden_states)
        self.base_layer.self_attn.prepare_replay_kv(
            hidden_states_norm, position_embeddings, shared_kv_states
        )

    def clear_replay_cache(self) -> None:
        self._replay_shared_kv_states = None
        self.base_layer.self_attn.clear_replay_cache()


def _get_layer_types(inner_model) -> list[str]:
    # Read layer types from the model config.
    # Gemma 4 alternates between "full_attention" (attends to all prior tokens)
    # and "sliding_attention" (attends only to the last sliding_window tokens).
    # Sliding attention is cheaper but has limited context per layer.
    cfg = inner_model.config
    if hasattr(cfg, "layer_types"):
        return list(cfg.layer_types)
    return ["full_attention"] * len(inner_model.layers)


def _make_chunk_attention_mask(
    raw_attention_mask: torch.Tensor | None,
    *,
    start: int,
    end: int,
    kv_start: int,
    kv_end: int,
    dtype: torch.dtype,
    device: torch.device,
    sliding_window: int | None = None,
) -> torch.Tensor | None:
    # Build a [1, 1, chunk_len, kv_len] attention bias mask for one chunk.
    # Values: 0 where attention is allowed, -inf where it should be masked out.
    #
    # Causal constraint: token at position q can only attend to tokens at positions k <= q.
    q_pos = torch.arange(start, end, device=device).unsqueeze(1)     # [chunk_len, 1]
    k_pos = torch.arange(kv_start, kv_end, device=device).unsqueeze(0)  # [1, kv_len]
    causal = k_pos <= q_pos  # True where attention is allowed

    if sliding_window is not None:
        # Additional constraint for sliding-window: token q can't attend to tokens
        # more than `sliding_window` steps in the past.
        causal = causal & ((q_pos - k_pos) < sliding_window)

    mask = torch.zeros((end - start, kv_end - kv_start), device=device, dtype=dtype)
    mask = mask.masked_fill(~causal, torch.finfo(dtype).min)  # -inf for masked positions
    mask = mask.unsqueeze(0).unsqueeze(0)  # → [1, 1, chunk_len, kv_len]

    if raw_attention_mask is not None:
        # Also mask out padding tokens (where raw_attention_mask == 0).
        pad = raw_attention_mask[:, kv_start:kv_end].to(device=device, non_blocking=True)
        pad_mask = (pad == 0).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, kv_len]
        mask = mask.masked_fill(pad_mask, torch.finfo(dtype).min)
    return mask


def _build_mask_mapping(
    inner_model, hidden_states, attention_mask, position_ids, inputs_embeds_kwarg: str
):
    # Ask HuggingFace to build the full causal attention masks for all layer types.
    # These are the standard 4D masks [batch, 1, seq_len, seq_len] that HF uses.
    # We build them once and cache them so they don't need to be recomputed per layer.
    cfg = inner_model.config
    mask_kwargs = {
        "config": cfg,
        inputs_embeds_kwarg: hidden_states,
        "attention_mask": attention_mask,
        "past_key_values": None,
        "position_ids": position_ids,
    }
    if inputs_embeds_kwarg == "inputs_embeds":
        mask_kwargs["cache_position"] = None
    full_mask = create_causal_mask(**mask_kwargs)
    sliding_mask = create_sliding_window_causal_mask(**mask_kwargs)
    return {
        "full_attention": full_mask,
        "sliding_attention": sliding_mask,
    }


def _to_normal_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
    # Detach a tensor from any computation graph and clone it into a plain tensor.
    # "inference_mode=False" context is needed because inference_mode tensors
    # can't be cloned into grad-capable tensors directly.
    if tensor is None:
        return None
    with torch.inference_mode(False):
        return tensor.clone().detach()


def _build_runtime(
    inner_model,
    layer_types: list[str],
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    build_full_masks: bool,
    causal_mask_inputs_embeds_kwarg: str,
) -> tuple[dict, dict]:
    # Build two things needed by every layer in the stack:
    #   1. mask_mapping: attention masks keyed by layer type
    #   2. position_embeddings: RoPE (cos, sin) keyed by layer type
    #
    # build_full_masks=True → 4D masks (pre-built for the whole sequence, fast per-layer).
    # build_full_masks=False → 2D masks (chunked mode; masks are built per-chunk later).
    if build_full_masks:
        mask_mapping = _build_mask_mapping(
            inner_model, hidden_states.detach(), attention_mask,
            position_ids, causal_mask_inputs_embeds_kwarg,
        )
    else:
        # In chunked mode, pass the raw 2D attention mask through.
        # _make_chunk_attention_mask will turn it into a proper 4D mask per chunk.
        mask_mapping = {}
        for layer_type in set(layer_types):
            safe_mask = attention_mask
            if safe_mask is not None and not torch.is_floating_point(safe_mask):
                safe_mask = safe_mask.to(dtype=torch.float32)
            mask_mapping[layer_type] = safe_mask

    # Build RoPE embeddings (cos, sin) once for the full sequence.
    # Different layer types can have different rotary embedding configurations.
    position_embeddings = {}
    for layer_type in set(layer_types):
        position_embeddings[layer_type] = inner_model.rotary_emb(
            hidden_states.detach(), position_ids, layer_type=layer_type,
        )
    return mask_mapping, position_embeddings


def _run_layer(
    layer,
    hidden_states: torch.Tensor,
    mask_mapping: dict,
    position_embeddings: dict,
    position_ids: torch.Tensor,
    shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]],
    layer_type: str,
    per_layer_input: torch.Tensor | None = None,
    chunk_range: tuple[int, int] | None = None,
) -> torch.Tensor:
    # Unified entry point for running one transformer layer (chunked or full).
    # Looks up the correct mask and position embeddings for this layer's type,
    # then calls the layer with all arguments in the expected format.
    attention_mask = mask_mapping.get(layer_type, mask_mapping["full_attention"])
    if attention_mask is not None and torch.is_tensor(attention_mask):
        # Cast mask to match hidden_states dtype (bfloat16) for the attention matmul.
        attention_mask = attention_mask.to(dtype=hidden_states.dtype)
    output = layer(
        hidden_states=hidden_states,
        per_layer_input=per_layer_input,
        shared_kv_states=shared_kv_states,
        position_embeddings=position_embeddings.get(layer_type),
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=None,
        chunk_range=chunk_range,
    )
    # Some HF layer implementations return a tuple (hidden_states, extras).
    # We only want the hidden states tensor.
    return output if not isinstance(output, tuple) else output[0]
