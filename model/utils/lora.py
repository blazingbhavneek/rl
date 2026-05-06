# LoRA adapter save/load normalization utilities.
#
# LoRA (Low-Rank Adaptation) works by adding a small pair of matrices (A and B,
# rank r << hidden_dim) to each target linear layer. Instead of updating the full
# weight matrix W, the update is: W_new = W + (B @ A) * (alpha/r).
# This means we only need to save/load A and B, which is much smaller than W.
#
# This file handles one specific problem: our streaming wrappers rename layers
# (e.g. layers.5.self_attn becomes layers.5.base_layer.self_attn.base_attn).
# When PEFT saves adapter weights, the keys in the file reflect those wrapper names.
# But if you load that file on a model without the wrappers, the keys won't match.
# The normalization functions here strip the wrapper-specific path segments so
# the saved adapter files are portable and standard-PEFT-compatible.

import json
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file

_ADAPTER_CONFIG_NAME = "adapter_config.json"
_ADAPTER_SAFE_NAME = "adapter_model.safetensors"  # Modern PEFT save format
_ADAPTER_BIN_NAME = "adapter_model.bin"            # Legacy PyTorch format

# Matches ".layers.5." inside a tensor key name, capturing the index (5).
_LAYER_INDEX_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")

# Matches ".layers.5.base_layer." — the extra segment added by _StreamGemmaDecoderLayer.
# We replace it with just ".layers.5." to undo the wrapper rename.
_WRAPPED_LAYER_RE = re.compile(r"(\.layers\.\d+)\.base_layer\.")


def _coerce_str_list(value) -> list[str]:
    # Safely convert None / a single string / a list into a list of strings.
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _coerce_int_list(value) -> list[int]:
    # Same idea but for integers, and deduplicates + sorts.
    if value is None:
        return []
    if isinstance(value, int):
        return [int(value)]
    return sorted({int(item) for item in value})


def _lora_target_leaf(target: str) -> str:
    # Extract the final module name from a dotted path.
    # e.g. "model.layers.5.self_attn.q_proj" → "q_proj"
    # We only need the leaf name because PEFT matches by leaf when layers_to_transform is set.
    return str(target).rsplit(".", 1)[-1]


def _layer_indices_from_names(names: list[str]) -> list[int]:
    # Given a list of tensor key names, extract the layer indices they belong to.
    # e.g. ["base_model.model.model.layers.26.self_attn.q_proj.lora_A.weight"] → [26]
    # Used to infer which layers a saved adapter covers, without reading the config.
    indices = set()
    for name in names:
        match = _LAYER_INDEX_RE.search(str(name))
        if match is not None:
            indices.add(int(match.group(1)))
    return sorted(indices)


def _normalize_lora_state_key(key: str) -> str:
    # Strip wrapper-specific segments from a tensor key so it matches standard PEFT naming.
    # Step 1: ".layers.5.base_layer.self_attn..." → ".layers.5.self_attn..."
    key = _WRAPPED_LAYER_RE.sub(r"\1.", key)
    # Step 2: ".self_attn.base_attn." → ".self_attn."  (from _StreamGemmaAttention wrapper)
    return key.replace(".self_attn.base_attn.", ".self_attn.")


def _adapter_state_path(adapter_dir: Path) -> Path | None:
    # Return the path to the adapter weights file, preferring .safetensors over .bin.
    safe_path = adapter_dir / _ADAPTER_SAFE_NAME
    if safe_path.exists():
        return safe_path
    bin_path = adapter_dir / _ADAPTER_BIN_NAME
    if bin_path.exists():
        return bin_path
    return None


def _adapter_state_keys(state_path: Path | None) -> list[str]:
    # Return the list of tensor names stored in the adapter weights file.
    # For .safetensors we can read just the keys without loading all the tensors into memory.
    if state_path is None:
        return []
    if state_path.name == _ADAPTER_SAFE_NAME:
        with safe_open(str(state_path), framework="pt", device="cpu") as handle:
            return list(handle.keys())
    state = torch.load(state_path, map_location="cpu")
    if not isinstance(state, dict):
        return []
    return list(state.keys())


def _adapter_safetensors_metadata(state_path: Path) -> dict[str, str] | None:
    # Read the optional string metadata stored in the safetensors header.
    # PEFT uses this for format_version, etc. We preserve it during normalization.
    with safe_open(str(state_path), framework="pt", device="cpu") as handle:
        return handle.metadata()


def _load_adapter_config_dict(adapter_dir: Path) -> dict:
    config_path = adapter_dir / _ADAPTER_CONFIG_NAME
    if not config_path.exists():
        raise FileNotFoundError(f"LoRA adapter config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_adapter_config_dict(adapter_dir: Path, config_data: dict) -> None:
    with (adapter_dir / _ADAPTER_CONFIG_NAME).open("w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2, sort_keys=True)
        f.write("\n")


def _normalized_lora_config_dict(config_data: dict, state_keys: list[str]) -> dict:
    # Produce a canonical version of adapter_config.json:
    #   - target_modules: only the leaf names (no full paths), sorted
    #   - layers_to_transform: the actual layer indices, inferred from state keys if missing
    #   - layers_pattern: always "layers" so PEFT knows how to find them
    normalized = dict(config_data)

    targets = _coerce_str_list(normalized.get("target_modules"))
    if targets:
        normalized["target_modules"] = sorted(
            {_lora_target_leaf(target) for target in targets}
        )

    target_layers = _layer_indices_from_names(targets)
    configured_layers = _coerce_int_list(normalized.get("layers_to_transform"))
    state_layers = _layer_indices_from_names(state_keys)
    # Priority: explicit target paths > config field > inferred from saved tensor names
    layer_indices = target_layers or configured_layers or state_layers
    if layer_indices:
        normalized["layers_to_transform"] = layer_indices
        normalized["layers_pattern"] = "layers"

    return normalized


def _normalize_adapter_state_file(src: Path, dst: Path) -> None:
    # Rewrite every tensor key in the adapter weights file using _normalize_lora_state_key,
    # then save to dst. This strips wrapper path segments so the file is portable.
    if src.name == _ADAPTER_SAFE_NAME:
        tensors = load_safetensors_file(str(src), device="cpu")
        metadata = _adapter_safetensors_metadata(src)
        normalized_tensors = {}
        for key, tensor in tensors.items():
            normalized_key = _normalize_lora_state_key(key)
            if normalized_key in normalized_tensors:
                raise RuntimeError(
                    f"duplicate LoRA tensor key after normalization: {normalized_key}"
                )
            normalized_tensors[normalized_key] = tensor
        save_safetensors_file(normalized_tensors, str(dst), metadata=metadata)
        return

    state = torch.load(src, map_location="cpu")
    if not isinstance(state, dict):
        raise RuntimeError(f"unsupported LoRA adapter state format: {src}")
    normalized_state = {}
    for key, tensor in state.items():
        normalized_key = _normalize_lora_state_key(key)
        if normalized_key in normalized_state:
            raise RuntimeError(
                f"duplicate LoRA tensor key after normalization: {normalized_key}"
            )
        normalized_state[normalized_key] = tensor
    torch.save(normalized_state, dst)


def _normalize_lora_adapter_dir(
    adapter_dir: Path, output_dir: Path | None = None
) -> Path:
    # Main entry point: normalize both the config and the weights in an adapter directory.
    # If output_dir is None, modifies in-place (writes to a temp file then replaces).
    # If output_dir is provided, writes the clean version there (used during load).
    adapter_dir = adapter_dir.expanduser().resolve()
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"LoRA adapter path not found: {adapter_dir}")

    state_path = _adapter_state_path(adapter_dir)
    state_keys = _adapter_state_keys(state_path)
    config_data = _load_adapter_config_dict(adapter_dir)
    normalized_config = _normalized_lora_config_dict(config_data, state_keys)
    dirty_config = normalized_config != config_data
    dirty_state = any(_normalize_lora_state_key(key) != key for key in state_keys)

    if output_dir is None and not dirty_config and not dirty_state:
        return adapter_dir  # Already clean, nothing to do

    target_dir = adapter_dir if output_dir is None else output_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    _write_adapter_config_dict(target_dir, normalized_config)

    if state_path is not None:
        target_state_path = target_dir / state_path.name
        if dirty_state or target_state_path != state_path:
            if dirty_state:
                # Write to a temp file first, then atomically rename — avoids
                # a corrupt state if the process dies mid-write.
                write_path = target_state_path
                if target_state_path == state_path:
                    write_path = target_state_path.with_name(
                        f".{target_state_path.name}.tmp"
                    )
                _normalize_adapter_state_file(state_path, write_path)
                if write_path != target_state_path:
                    write_path.replace(target_state_path)
            else:
                shutil.copy2(state_path, target_state_path)

    return target_dir


def _assert_clean_lora_adapter_dir(adapter_dir: Path) -> None:
    # Sanity check after saving: verify the adapter directory meets all requirements
    # that make it loadable by standard PEFT on any model variant (with or without wrappers).
    config_data = _load_adapter_config_dict(adapter_dir)

    targets = config_data.get("target_modules")
    if not isinstance(targets, list) or not targets:
        raise RuntimeError(
            "Gemma4 LoRA adapter must save leaf target_modules as a non-empty list"
        )

    # Make sure no full dotted paths snuck in — PEFT expects leaf names only.
    full_targets = [target for target in targets if "." in str(target)]
    if full_targets:
        raise RuntimeError(
            f"Gemma4 LoRA adapter saved full target paths: {full_targets[:3]}"
        )

    if not config_data.get("layers_to_transform"):
        raise RuntimeError("Gemma4 LoRA adapter saved without layers_to_transform")

    if config_data.get("layers_pattern") != "layers":
        raise RuntimeError("Gemma4 LoRA adapter saved without layers_pattern='layers'")

    # Make sure no wrapper-specific key names ended up in the saved weights.
    bad_keys = [
        key
        for key in _adapter_state_keys(_adapter_state_path(adapter_dir))
        if "base_layer" in key or "base_attn" in key
    ]
    if bad_keys:
        raise RuntimeError(
            f"Gemma4 LoRA adapter saved wrapper-specific tensor keys: {bad_keys[:3]}"
        )
