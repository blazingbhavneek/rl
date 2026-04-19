import gc
import warnings

import torch

from model.config import ModelConfig
from model.gemma4 import (
    Gemma4Model,
    _apply_rotary_pos_emb,
    _make_chunk_attention_mask,
    _run_layer,
    _stream_attention_forward,
)
from model.tests.test_max_sequence_length import (
    DEFAULT_GEMMA4_LORA_TARGETS,
    DEFAULT_GEMMA4_MODEL_PATH,
    _make_sample,
)

warnings.filterwarnings(
    "ignore",
    message=r".*Dynamo detected a call to a `functools\.lru_cache`-wrapped function.*",
    category=UserWarning,
)

DEFAULT_TEST_SEQ_LEN = 1536
DEFAULT_CHUNK_LEN = 128
TEST_CHUNK_STARTS = [64, 640]


def _build_config() -> ModelConfig:
    return ModelConfig(
        lora=DEFAULT_GEMMA4_LORA_TARGETS,
        lora_fraction=0.25,
        lora_rank=128,
        lora_alpha=256,
        prefix_token_chunk_size=768,
        suffix_token_chunk_size=128,
        use_grad_checkpoint=False,
        use_compile=False,
        attn_implementation="sdpa",
        cuda_device_index=0,
    )


def _build_full_inputs(model: Gemma4Model, target_total_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    messages, completion_text = _make_sample(model.tokenizer, target_total_tokens)
    prompt = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = model.tokenizer(prompt, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    completion_ids = model.tokenizer(
        completion_text,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    full_ids = torch.tensor([prompt_ids + completion_ids], device=model._model_device, dtype=torch.long)
    full_mask = torch.ones_like(full_ids)
    return full_ids, full_mask


def _find_target_layers(model: Gemma4Model) -> tuple[int, int]:
    direct_idx = -1
    shared_idx = -1
    for idx, layer in enumerate(model._layers):
        attn = layer.base_layer.self_attn
        if model._layer_types[idx] != "sliding_attention":
            continue
        if direct_idx < 0 and getattr(attn, "store_full_length_kv", False):
            direct_idx = idx
        if shared_idx < 0 and getattr(attn, "is_kv_shared_layer", False):
            shared_idx = idx
        if direct_idx >= 0 and shared_idx >= 0:
            return direct_idx, shared_idx
    raise AssertionError("could not find both direct and shared sliding-attention layers")


def _prepare_layer_inputs(
    model: Gemma4Model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_layer_idx: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[int, tuple[torch.Tensor, torch.Tensor]], dict, dict]:
    batch_size, seq_len = input_ids.shape
    hidden_states = model._inner_model.embed_tokens(input_ids)
    position_ids = (
        torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
        .unsqueeze(0)
        .expand(batch_size, seq_len)
    )
    mask_mapping, position_embeddings = model._build_runtime(
        hidden_states,
        attention_mask,
        position_ids,
        build_full_masks=True,
    )

    shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    raw_per_layer_inputs = model._get_per_layer_inputs(input_ids)
    per_layer_inputs = model._project_per_layer_inputs(hidden_states, raw_per_layer_inputs)

    for idx in range(target_layer_idx):
        per_layer_input = per_layer_inputs[:, :, idx, :] if per_layer_inputs is not None else None
        hidden_states = _run_layer(
            model._layers[idx],
            hidden_states,
            mask_mapping,
            position_embeddings,
            position_ids,
            shared_kv_states,
            model._layer_types[idx],
            per_layer_input=per_layer_input,
        )

    return hidden_states, position_ids, shared_kv_states, mask_mapping, position_embeddings


def _reference_full_kv_attention(
    attn,
    hidden_states_norm: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
    shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]],
    *,
    start: int,
    end: int,
) -> torch.Tensor:
    batch_size = hidden_states_norm.shape[0]
    chunk_len = end - start
    cos, sin = position_embeddings
    query_states = attn.q_proj(hidden_states_norm[:, start:end, :]).view(batch_size, chunk_len, -1, attn.head_dim)
    query_states = attn.q_norm(query_states)
    query_states = _apply_rotary_pos_emb(
        query_states,
        cos[:, start:end, :],
        sin[:, start:end, :],
        unsqueeze_dim=2,
    ).transpose(1, 2)

    if attn.is_kv_shared_layer:
        key_states, value_states = shared_kv_states[attn.kv_shared_layer_index]
        key_states = key_states[:, :, :end, :]
        value_states = value_states[:, :, :end, :]
    else:
        kv_hidden = hidden_states_norm[:, :end, :]
        key_states = attn.k_proj(kv_hidden).view(batch_size, end, -1, attn.head_dim)
        value_states = (
            attn.v_proj(kv_hidden).view(batch_size, end, -1, attn.head_dim)
            if attn.v_proj is not None
            else key_states
        )
        key_states = attn.k_norm(key_states)
        key_states = _apply_rotary_pos_emb(
            key_states,
            cos[:, :end, :],
            sin[:, :end, :],
            unsqueeze_dim=2,
        ).transpose(1, 2)
        value_states = attn.v_norm(value_states).transpose(1, 2)

    chunk_mask = _make_chunk_attention_mask(
        attention_mask,
        start=start,
        end=end,
        kv_start=0,
        kv_end=end,
        dtype=query_states.dtype,
        device=query_states.device,
        sliding_window=getattr(attn, "sliding_window", None),
    )
    attn_output, _ = _stream_attention_forward(
        attn,
        query_states,
        key_states,
        value_states,
        chunk_mask,
    )
    attn_output = attn_output.reshape(batch_size, chunk_len, -1)
    return attn.o_proj(attn_output)


def run_sliding_window_chunking_test(model_path: str) -> None:
    model = Gemma4Model(model_path=model_path, config=_build_config())
    model.model.eval()
    input_ids, attention_mask = _build_full_inputs(model, DEFAULT_TEST_SEQ_LEN)
    direct_idx, shared_idx = _find_target_layers(model)

    cases = [
        ("direct", direct_idx),
        ("shared", shared_idx),
    ]

    with torch.inference_mode():
        for label, layer_idx in cases:
            hidden_states, _, shared_kv_states, _, position_embeddings = _prepare_layer_inputs(
                model,
                input_ids,
                attention_mask,
                layer_idx,
            )
            layer = model._layers[layer_idx]
            attn = layer.base_layer.self_attn
            hidden_states_norm = layer.base_layer.input_layernorm(hidden_states)
            for start in TEST_CHUNK_STARTS:
                end = min(start + DEFAULT_CHUNK_LEN, hidden_states.shape[1])
                reference = _reference_full_kv_attention(
                    attn,
                    hidden_states_norm,
                    position_embeddings[model._layer_types[layer_idx]],
                    attention_mask,
                    shared_kv_states,
                    start=start,
                    end=end,
                )
                current, _ = attn(
                    hidden_states=hidden_states_norm,
                    position_embeddings=position_embeddings[model._layer_types[layer_idx]],
                    attention_mask=attention_mask,
                    shared_kv_states=shared_kv_states,
                    chunk_range=(start, end),
                    full_hidden_seq_len=hidden_states.shape[1],
                )
                max_abs_diff = float((reference.float() - current.float()).abs().max().item())
                print(
                    f"[sliding-chunk] case={label} layer={layer_idx} "
                    f"start={start} end={end} max_abs_diff={max_abs_diff:.6f}"
                )
                assert max_abs_diff < 5e-2, (
                    f"sliding chunk mismatch for case={label} layer={layer_idx} "
                    f"start={start} end={end}: max_abs_diff={max_abs_diff}"
                )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run_sliding_window_chunking_test(DEFAULT_GEMMA4_MODEL_PATH)
    print("PASS: gemma4 sliding chunking")
