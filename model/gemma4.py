import inspect
import math
import shutil
import tempfile
from collections import deque
from pathlib import Path

import torch
import torch._functorch.config as _functorch_config
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import create_causal_mask

from .base import BaseModel
from .utils.attn import apply_rotary_pos_emb as _apply_rotary_pos_emb
from .utils.attn import repeat_kv as _repeat_kv
from .utils.gemma4_streaming import (
    _COMPILE_MODE,
    _build_runtime,
    _get_layer_types,
    _LayerReplay,
    _normalize_chunk_size,
    _run_layer,
    _StreamCheckpointFunction,
    _StreamGemmaAttention,
    _StreamGemmaDecoderLayer,
    _to_normal_tensor,
)
from .utils.lora import (
    _adapter_state_keys,
    _adapter_state_path,
    _assert_clean_lora_adapter_dir,
    _coerce_int_list,
    _coerce_str_list,
    _layer_indices_from_names,
    _load_adapter_config_dict,
    _lora_target_leaf,
    _normalize_lora_adapter_dir,
    _normalize_lora_state_key,
    _normalized_lora_config_dict,
)
from .utils.prefix import PrefixBundle

_functorch_config.donated_buffer = False

_GEMMA4_DEFAULT_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


class Gemma4Model(BaseModel):
    def setup(self) -> None:
        device = f"cuda:{self.cuda_device_index}"
        attn_impl = getattr(self.config, "attn_implementation", "eager")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map={"": device},
            dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        )
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False

        self.lora = getattr(self.config, "lora", list(_GEMMA4_DEFAULT_LORA_TARGETS))
        self.lora_fraction = float(getattr(self.config, "lora_fraction", 0.5))
        self._use_grad_checkpoint = bool(
            getattr(self.config, "use_grad_checkpoint", False)
        )
        self._use_compile = bool(getattr(self.config, "use_compile", False))
        self._offload_prefix_to_cpu = bool(
            getattr(self.config, "offload_prefix_to_cpu", False)
        )
        self._lora_temp_dirs: list[tempfile.TemporaryDirectory] = []
        self._token_chunk_size = _normalize_chunk_size(
            getattr(self.config, "token_chunk_size", None)
        )
        self._prefix_token_chunk_size = _normalize_chunk_size(
            getattr(self.config, "prefix_token_chunk_size", None)
        )
        if self._prefix_token_chunk_size == 0:
            self._prefix_token_chunk_size = self._token_chunk_size
        self._suffix_token_chunk_size = _normalize_chunk_size(
            getattr(self.config, "suffix_token_chunk_size", None)
        )
        if self._suffix_token_chunk_size == 0:
            self._suffix_token_chunk_size = self._token_chunk_size

        self._inner_model, self._lm_head = self._resolve_text_stack(self.model)
        self._layers = self._inner_model.layers
        self._layer_types = _get_layer_types(self._inner_model)
        self._prefix_split_layer = self._compute_prefix_end(
            len(self._layers), self.lora_fraction
        )
        self._causal_mask_inputs_embeds_kwarg = (
            "inputs_embeds"
            if "inputs_embeds" in inspect.signature(create_causal_mask).parameters
            else "input_embeds"
        )

        if self.lora_path:
            lora_path = self._prepare_lora_path_for_load(self.lora_path)
            self._infer_lora_metadata_from_adapter(lora_path)
            self.model = PeftModel.from_pretrained(
                self.model,
                str(lora_path),
                adapter_name=self.lora_adapter_name,
                is_trainable=self.lora_is_trainable,
            )
        else:
            lora_targets = self._requested_lora_targets()
            suffix_layer_indices = list(
                range(self._prefix_split_layer, len(self._layers))
            )
            target_modules = self._available_lora_leaf_targets(
                lora_targets,
                suffix_layer_indices,
            )
            lora_rank = int(getattr(self.config, "lora_rank", 128))
            lora_alpha = int(getattr(self.config, "lora_alpha", lora_rank * 2))
            self.lora = target_modules
            self.lora_rank = lora_rank
            self.lora_alpha = lora_alpha
            lora_cfg = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=target_modules,
                layers_to_transform=suffix_layer_indices,
                layers_pattern="layers",
            )
            self.model = get_peft_model(
                self.model,
                lora_cfg,
                adapter_name=self.lora_adapter_name,
            )

        self._inner_model, self._lm_head = self._resolve_text_stack(self.model)
        self._layers = self._inner_model.layers
        self._layer_types = _get_layer_types(self._inner_model)
        self._wrap_suffix_layers_for_streaming()
        self._model_device = next(self.model.parameters()).device

        for param in self.model.parameters():
            param.requires_grad_(False)
        for name, param in self.model.named_parameters():
            if "lora_" in name and self.lora_is_trainable:
                param.requires_grad_(True)

    _SUFFIX_NORM_ATTRS = (
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
    )

    def _requested_lora_targets(self) -> list[str]:
        if isinstance(self.lora, list):
            return [str(target) for target in self.lora]
        return list(_GEMMA4_DEFAULT_LORA_TARGETS)

    def _prepare_lora_path_for_load(self, adapter_path: str) -> Path:
        adapter_dir = Path(adapter_path).expanduser().resolve()
        if not adapter_dir.is_dir():
            raise FileNotFoundError(f"LoRA adapter path not found: {adapter_dir}")

        state_path = _adapter_state_path(adapter_dir)
        state_keys = _adapter_state_keys(state_path)
        config_data = _load_adapter_config_dict(adapter_dir)
        normalized_config = _normalized_lora_config_dict(config_data, state_keys)
        dirty_config = normalized_config != config_data
        dirty_state = any(_normalize_lora_state_key(key) != key for key in state_keys)
        if not dirty_config and not dirty_state:
            return adapter_dir

        tmp = tempfile.TemporaryDirectory(prefix="gemma4_lora_")
        self._lora_temp_dirs.append(tmp)
        return _normalize_lora_adapter_dir(adapter_dir, Path(tmp.name))

    def _infer_lora_metadata_from_adapter(self, adapter_path: Path) -> None:
        peft_config = PeftConfig.from_pretrained(str(adapter_path))
        config_data = _load_adapter_config_dict(adapter_path)

        targets = _coerce_str_list(config_data.get("target_modules"))
        if not targets:
            targets = _coerce_str_list(getattr(peft_config, "target_modules", None))
        self.lora = sorted({_lora_target_leaf(target) for target in targets})

        self.lora_rank = int(getattr(peft_config, "r", config_data.get("r", 128)))
        self.lora_alpha = int(
            getattr(
                peft_config,
                "lora_alpha",
                config_data.get("lora_alpha", self.lora_rank * 2),
            )
        )

        layer_indices = _coerce_int_list(config_data.get("layers_to_transform"))
        if not layer_indices:
            layer_indices = _layer_indices_from_names(
                _adapter_state_keys(_adapter_state_path(adapter_path))
            )

        if layer_indices:
            self._prefix_split_layer = min(layer_indices)
        else:
            self._prefix_split_layer = 0
        if len(self._layers) > 0:
            self.lora_fraction = (len(self._layers) - self._prefix_split_layer) / len(
                self._layers
            )

    def _available_lora_leaf_targets(
        self,
        requested_targets: list[str],
        layer_indices: list[int],
    ) -> list[str]:
        requested = {str(target) for target in requested_targets}
        if not requested:
            raise ValueError("Gemma4 LoRA target list cannot be empty")
        layer_index_set = set(layer_indices)
        found = set()

        for name, module in self.model.named_modules():
            parts = name.split(".")
            layer_idx = None
            for i in range(len(parts) - 1):
                if parts[i] == "layers" and parts[i + 1].isdigit():
                    layer_idx = int(parts[i + 1])
                    break
            if layer_idx is None or layer_idx not in layer_index_set:
                continue
            leaf = parts[-1]
            if leaf in requested and hasattr(module, "weight"):
                found.add(leaf)

        if not found:
            raise ValueError(
                "None of the requested Gemma4 LoRA targets exist in the selected suffix layers: "
                f"{sorted(requested)}"
            )
        return sorted(found)

    def load_lora_adapter(
        self,
        adapter_name: str,
        adapter_path: str,
        *,
        is_trainable: bool = False,
    ) -> None:
        normalized_path = self._prepare_lora_path_for_load(adapter_path)
        unwrapped = self._unwrap_streaming_layers_for_peft()
        try:
            super().load_lora_adapter(
                adapter_name,
                str(normalized_path),
                is_trainable=is_trainable,
            )
        finally:
            if unwrapped:
                self._wrap_suffix_layers_for_streaming()

    def save_lora_adapter(self, adapter_name: str, save_path: str) -> None:
        save_dir = Path(save_path).expanduser().resolve()
        if not hasattr(self.model, "peft_config"):
            raise RuntimeError("Model is not a PEFT model")
        actual_adapters = list(self.model.peft_config.keys())
        if not actual_adapters:
            raise RuntimeError("No LoRA adapters found in model")

        save_name = (
            adapter_name if adapter_name in actual_adapters else actual_adapters[0]
        )
        unwrapped = self._unwrap_streaming_layers_for_peft()
        try:
            self.model.save_pretrained(str(save_dir), selected_adapters=[save_name])
        finally:
            if unwrapped:
                self._wrap_suffix_layers_for_streaming()

        adapter_dir = save_dir / save_name if save_name != "default" else save_dir
        if adapter_dir != save_dir and adapter_dir.exists():
            for child in adapter_dir.iterdir():
                dst = save_dir / child.name
                if dst.exists():
                    if dst.is_dir():
                        shutil.rmtree(dst)
                    else:
                        dst.unlink()
                shutil.move(str(child), str(dst))
            adapter_dir.rmdir()

        _normalize_lora_adapter_dir(save_dir)
        _assert_clean_lora_adapter_dir(save_dir)

    def _unwrap_streaming_layers_for_peft(self) -> bool:
        changed = False
        for idx, layer in enumerate(self._inner_model.layers):
            if isinstance(layer, _StreamGemmaDecoderLayer):
                base_layer = layer.base_layer
                if isinstance(base_layer.self_attn, _StreamGemmaAttention):
                    base_layer.self_attn = base_layer.self_attn.base_attn
                self._inner_model.layers[idx] = base_layer
                changed = True
            elif hasattr(layer, "self_attn") and isinstance(
                layer.self_attn, _StreamGemmaAttention
            ):
                layer.self_attn = layer.self_attn.base_attn
                changed = True
        if changed:
            self._layers = self._inner_model.layers
        return changed

    def _wrap_suffix_layers_for_streaming(self) -> None:
        use_compile = self._use_compile
        for idx in range(len(self._layers)):
            layer = self._layers[idx]
            if isinstance(layer, _StreamGemmaDecoderLayer):
                layer._use_compile = use_compile
                continue
            new_layer = _StreamGemmaDecoderLayer(layer, use_compile=use_compile)
            if use_compile and idx >= self._prefix_split_layer:
                base = new_layer.base_layer
                # Compile MLP only. The suffix path alternates between inference-mode
                # prefix work and grad-enabled replay, and compiled RMSNorm modules
                # hit Dynamo's recompile limit under that mixed-mode workload.
                if hasattr(base, "mlp") and base.mlp is not None:
                    base.mlp = torch.compile(
                        base.mlp, dynamic=True, fullgraph=True, mode=_COMPILE_MODE
                    )
            self._inner_model.layers[idx] = new_layer
        self._layers = self._inner_model.layers

    def _resolve_text_stack(self, model):
        seen = set()
        queue = deque([model])
        ordered_candidates = []

        while queue:
            candidate = queue.popleft()
            if candidate is None or id(candidate) in seen:
                continue
            seen.add(id(candidate))
            ordered_candidates.append(candidate)
            for attr in ("base_model", "model"):
                if hasattr(candidate, attr):
                    queue.append(getattr(candidate, attr))

        inner_model = None
        for candidate in ordered_candidates:
            if hasattr(candidate, "language_model"):
                inner_model = candidate.language_model
                break
            if hasattr(candidate, "layers") and hasattr(candidate, "embed_tokens"):
                inner_model = candidate
                break
        if inner_model is None:
            raise RuntimeError(
                "Unable to resolve Gemma 4 text stack from loaded model. "
                f"candidate_types={[type(c).__name__ for c in ordered_candidates]}"
            )

        lm_head = None
        for candidate in ordered_candidates:
            if hasattr(candidate, "lm_head"):
                lm_head = candidate.lm_head
                break
        if lm_head is None:
            raise RuntimeError(
                "Unable to resolve lm_head from loaded Gemma 4 model. "
                f"candidate_types={[type(c).__name__ for c in ordered_candidates]}"
            )

        return inner_model, lm_head

    def _compute_prefix_end(self, n_layers: int, frac: float) -> int:
        if frac <= 0.0:
            return n_layers
        if frac >= 1.0:
            return 0
        return int(math.floor(n_layers * (1.0 - frac)))

    def _debug_lora_state(self) -> dict:
        total = sum(p.numel() for p in self.model.parameters())
        lora_params = {n: p for n, p in self.model.named_parameters() if "lora_" in n}
        trainable_lora = sum(p.numel() for p in lora_params.values() if p.requires_grad)
        sample = next(iter(lora_params), None)
        return {
            "total_params": total,
            "lora_params": sum(p.numel() for p in lora_params.values()),
            "trainable_lora": trainable_lora,
            "sample_lora_name": sample,
            "peft_config_keys": list(getattr(self.model, "peft_config", {}).keys()),
            "prefix_split_layer": self._prefix_split_layer,
        }

    def _postprocess_logits(self, logits: torch.Tensor) -> torch.Tensor:
        softcap = getattr(self._inner_model.config, "final_logit_softcapping", None)
        if softcap is not None:
            logits = torch.tanh(logits / softcap) * softcap
        return logits

    def _get_per_layer_inputs(self, input_ids: torch.Tensor) -> torch.Tensor | None:
        if not getattr(self._inner_model.config, "hidden_size_per_layer_input", None):
            return None
        inputs_embeds = self._inner_model.embed_tokens(input_ids)
        return self._inner_model.get_per_layer_inputs(input_ids, inputs_embeds)

    def _project_per_layer_inputs(
        self,
        hidden_states: torch.Tensor,
        per_layer_inputs: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if per_layer_inputs is None:
            return None
        return self._inner_model.project_per_layer_inputs(
            hidden_states, per_layer_inputs
        )

    def _mask_mapping_has_full_masks(self, mask_mapping: dict | None) -> bool:
        if mask_mapping is None:
            return True
        saw_tensor = False
        for mask in mask_mapping.values():
            if mask is None:
                continue
            if torch.is_tensor(mask):
                saw_tensor = True
                if mask.dim() >= 4:
                    return True
                if mask.dim() == 2:
                    return False
        return not saw_tensor

    def _build_runtime(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        *,
        build_full_masks: bool,
    ) -> tuple[dict, dict]:
        return _build_runtime(
            self._inner_model,
            self._layer_types,
            hidden_states.detach(),
            attention_mask,
            position_ids,
            build_full_masks=build_full_masks,
            causal_mask_inputs_embeds_kwarg=self._causal_mask_inputs_embeds_kwarg,
        )

    def _detach_shared_kv_states(
        self,
        shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]],
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        return {
            key: (_to_normal_tensor(key_states), _to_normal_tensor(value_states))
            for key, (key_states, value_states) in shared_kv_states.items()
        }

    def _move_tensor(
        self, tensor: torch.Tensor | None, device: torch.device
    ) -> torch.Tensor | None:
        if tensor is None:
            return None
        if tensor.device == device:
            return tensor
        return tensor.to(device=device, non_blocking=True)

    def _materialize_shared_kv_states(
        self,
        shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]],
        device: torch.device,
        seq_end: int | None = None,
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        materialized = {}
        for key, (key_states, value_states) in shared_kv_states.items():
            if seq_end is not None:
                key_states = key_states[:, :, :seq_end, :]
                value_states = value_states[:, :, :seq_end, :]
            materialized[key] = (
                self._move_tensor(key_states, device),
                self._move_tensor(value_states, device),
            )
        return materialized

    def _offload_prefix_bundle_to_cpu(self, bundle: PrefixBundle) -> PrefixBundle:
        return PrefixBundle(
            hidden_prefix=bundle.hidden_prefix.to("cpu", non_blocking=True),
            position_ids=bundle.position_ids.to("cpu", non_blocking=True),
            shared_kv_states={
                key: (
                    key_states.to("cpu", non_blocking=True),
                    value_states.to("cpu", non_blocking=True),
                )
                for key, (key_states, value_states) in bundle.shared_kv_states.items()
            },
            per_layer_inputs=(
                bundle.per_layer_inputs.to("cpu", non_blocking=True)
                if bundle.per_layer_inputs is not None
                else None
            ),
        )

    def _build_prefix_bundle(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> PrefixBundle:
        (
            hidden_prefix,
            position_ids,
            shared_kv_states,
            per_layer_inputs,
            mask_mapping,
            position_embeddings,
        ) = self._forward_prefix(
            input_ids,
            attention_mask,
        )
        bundle = PrefixBundle(
            hidden_prefix=_to_normal_tensor(hidden_prefix),
            position_ids=_to_normal_tensor(position_ids),
            shared_kv_states=self._detach_shared_kv_states(shared_kv_states),
            per_layer_inputs=_to_normal_tensor(per_layer_inputs),
            mask_mapping=mask_mapping,
            position_embeddings=position_embeddings,
        )
        if self._offload_prefix_to_cpu:
            bundle = self._offload_prefix_bundle_to_cpu(bundle)
        return bundle

    def _prepare_suffix_inputs_from_bundle(
        self,
        bundle: PrefixBundle,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        dict[int, tuple[torch.Tensor, torch.Tensor]],
        torch.Tensor | None,
    ]:
        model_device = self._model_device
        hidden_prefix = self._move_tensor(bundle.hidden_prefix, model_device)
        if hidden_prefix is None:
            raise RuntimeError("prefix bundle did not contain hidden_prefix")
        position_ids = self._move_tensor(bundle.position_ids, model_device)
        if position_ids is None:
            raise RuntimeError("prefix bundle did not contain position_ids")
        return (
            hidden_prefix.detach().requires_grad_(torch.is_grad_enabled()),
            position_ids,
            bundle.shared_kv_states,
            bundle.per_layer_inputs,
        )

    def _run_suffix_from_prefix_bundle(
        self,
        bundle: PrefixBundle,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        bundle = bundle.clone_for_autograd()
        hidden_for_suffix, position_ids, shared_kv_states, per_layer_inputs = (
            self._prepare_suffix_inputs_from_bundle(bundle)
        )
        return self._forward_suffix(
            hidden_for_suffix,
            position_ids,
            attention_mask,
            shared_kv_states,
            per_layer_inputs,
            prebuilt_mask_mapping=bundle.mask_mapping,
            prebuilt_position_embeddings=bundle.position_embeddings,
        )

    def _forward_prefix(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        dict[int, tuple[torch.Tensor, torch.Tensor]],
        torch.Tensor | None,
        dict,
        dict,
    ]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        hidden_states = self._inner_model.embed_tokens(input_ids)
        position_ids = (
            torch.arange(seq_len, device=device, dtype=torch.long)
            .unsqueeze(0)
            .expand(batch_size, seq_len)
        )
        use_chunked_prefix = self._prefix_token_chunk_size > 0

        mask_mapping, position_embeddings = self._build_runtime(
            hidden_states,
            attention_mask,
            position_ids,
            build_full_masks=not use_chunked_prefix,
        )

        shared_kv_states = {}
        raw_per_layer_inputs = self._get_per_layer_inputs(input_ids)
        per_layer_inputs = self._project_per_layer_inputs(
            hidden_states, raw_per_layer_inputs
        )

        for idx, layer in enumerate(self._layers[: self._prefix_split_layer]):
            per_layer_input = (
                per_layer_inputs[:, :, idx, :] if per_layer_inputs is not None else None
            )
            if use_chunked_prefix and isinstance(layer, _StreamGemmaDecoderLayer):
                layer.prepare_for_replay(
                    hidden_states,
                    position_embeddings[self._layer_types[idx]],
                    shared_kv_states,
                )
                output_chunks = torch.empty_like(hidden_states)
                for start in range(0, seq_len, self._prefix_token_chunk_size):
                    end = min(start + self._prefix_token_chunk_size, seq_len)
                    per_layer_chunk = None
                    if per_layer_input is not None:
                        per_layer_chunk = per_layer_input[:, start:end, :]
                    output_chunks[:, start:end, :] = _run_layer(
                        layer,
                        hidden_states,
                        mask_mapping,
                        position_embeddings,
                        position_ids,
                        shared_kv_states,
                        self._layer_types[idx],
                        per_layer_input=per_layer_chunk,
                        chunk_range=(start, end),
                    )
                layer.clear_replay_cache()
                hidden_states = output_chunks
            else:
                hidden_states = _run_layer(
                    layer,
                    hidden_states,
                    mask_mapping,
                    position_embeddings,
                    position_ids,
                    shared_kv_states,
                    self._layer_types[idx],
                    per_layer_input=per_layer_input,
                )

        return (
            hidden_states,
            position_ids,
            shared_kv_states,
            per_layer_inputs,
            mask_mapping,
            position_embeddings,
        )

    def _forward_suffix(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]],
        per_layer_inputs: torch.Tensor | None,
        *,
        prebuilt_mask_mapping: dict | None = None,
        prebuilt_position_embeddings: dict | None = None,
    ) -> torch.Tensor:
        use_streaming_suffix = (
            self._use_grad_checkpoint
            and torch.is_grad_enabled()
            and self._suffix_token_chunk_size > 0
        )
        build_full_masks = not use_streaming_suffix
        can_reuse_prebuilt = (
            prebuilt_mask_mapping is not None
            and prebuilt_position_embeddings is not None
            and self._mask_mapping_has_full_masks(prebuilt_mask_mapping)
            == build_full_masks
        )
        if can_reuse_prebuilt:
            mask_mapping = prebuilt_mask_mapping
            position_embeddings = prebuilt_position_embeddings
        else:
            mask_mapping, position_embeddings = self._build_runtime(
                hidden_states,
                attention_mask,
                position_ids,
                build_full_masks=build_full_masks,
            )

        for offset, layer in enumerate(self._layers[self._prefix_split_layer :]):
            abs_idx = self._prefix_split_layer + offset
            current_per_layer_input_source = None
            if per_layer_inputs is not None:
                current_per_layer_input_source = per_layer_inputs[:, :, abs_idx, :]

            layer_replay = _LayerReplay(
                layer,
                abs_idx,
                current_per_layer_input_source,
                self,
                mask_mapping,
                position_embeddings,
                position_ids,
                shared_kv_states,
            )

            if use_streaming_suffix:
                hidden_states = _StreamCheckpointFunction.apply(
                    layer_replay,
                    self._suffix_token_chunk_size,
                    self._offload_prefix_to_cpu,
                    hidden_states,
                )
            elif self._use_grad_checkpoint and torch.is_grad_enabled():
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer_replay, hidden_states, use_reentrant=False
                )
            else:
                hidden_states = layer_replay(hidden_states)

        return self._inner_model.norm(hidden_states)
