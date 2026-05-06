# Gemma4Model: concrete implementation of BaseModel for Google's Gemma 4 architecture.
#
# Gemma 4 has several features that need special handling:
#   - KV-sharing: some attention layers reuse K/V computed by an earlier layer.
#   - Sliding-window attention: alternates between full and local context layers.
#   - Per-layer inputs: an optional extra embedding injected into each layer.
#   - MoE (Mixture of Experts): some layers route each token to one of many experts.
#   - Logit softcapping: tanh-based final logit squashing for training stability.
#
# On top of this, we add:
#   - LoRA on the suffix layers only (prefix layers are frozen and unmodified).
#   - Optional chunked prefix / streaming suffix for reduced peak memory.
#   - torch.compile on the MLP of suffix layers for speed.

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

# These are the linear projections inside each transformer layer that we apply LoRA to.
# q/k/v/o = the attention projections; gate/up/down = the MLP (SwiGLU feed-forward) projections.
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
        # setup() is called by BaseModel.__init__ after storing config fields.
        # It must populate self.model, self.tokenizer, self._inner_model,
        # self._lm_head, self._layers, and self._layer_types.

        device = f"cuda:{self.cuda_device_index}"
        attn_impl = getattr(self.config, "attn_implementation", "eager")
        # "eager" = standard Python loop attention. Alternatives: "flash_attention_2", "sdpa".
        # We default to eager because our chunked streaming code manually implements attention.

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            # Gemma's tokenizer doesn't define a pad token by default.
            # Using EOS as pad is standard practice — pad tokens are masked out anyway.

        # TODO: Make the training process work on multiple GPUs
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map={"": device},  # Put all parameters on this single GPU
            dtype=torch.bfloat16,     # bfloat16 = 2 bytes/param, same range as float32 but less precision
            attn_implementation=attn_impl,
        )
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False
            # Disable KV cache (used for autoregressive generation).
            # During training we always process full sequences, so we don't need it.

        self.lora = getattr(self.config, "lora", list(_GEMMA4_DEFAULT_LORA_TARGETS))
        # Which linear layers to apply LoRA to. Can be overridden in the config.

        self.lora_fraction = float(getattr(self.config, "lora_fraction", 0.5))
        # What fraction of layers (from the end) get LoRA. 0.5 = last half of layers.
        # The first half runs frozen (prefix), the second half gets LoRA (suffix).
        # Larger fraction = more parameters to train but more memory.

        self._use_grad_checkpoint = bool(getattr(self.config, "use_grad_checkpoint", False))
        # If True, use gradient checkpointing in the suffix layers.
        # Trades ~33% extra compute for significantly lower peak memory
        # because intermediate activations don't need to be stored.

        self._use_compile = bool(getattr(self.config, "use_compile", False))
        # If True, JIT-compile suffix MLP modules with torch.compile.
        # Can give 20-40% speedup after a one-time warmup compilation cost.

        self._offload_prefix_to_cpu = bool(getattr(self.config, "offload_prefix_to_cpu", False))
        # If True, move the PrefixBundle to CPU RAM after computing it.
        # Frees GPU memory between the prefix and suffix passes.
        # Adds a small latency penalty to move it back before the suffix pass.

        self._lora_temp_dirs: list[tempfile.TemporaryDirectory] = []
        # Temporary directories created when normalizing LoRA adapters on load.
        # Kept alive here so they aren't garbage-collected while the model is running.

        # token_chunk_size: default chunk size for both prefix and suffix passes.
        self._token_chunk_size = _normalize_chunk_size(
            getattr(self.config, "token_chunk_size", None)
        )
        # Prefix and suffix can have different chunk sizes (prefix is no-grad so can be larger).
        # If not set separately, both fall back to token_chunk_size.
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

        # Resolve the actual transformer stack from the HF model.
        # HF wraps models differently depending on architecture (Gemma 4 has a VLM wrapper),
        # so we can't hardcode the path — we have to search for it.
        # TODO: check this helper function and simplify it
        self._inner_model, self._lm_head = self._resolve_text_stack(self.model)
        self._layers = self._inner_model.layers
        self._layer_types = _get_layer_types(self._inner_model)

        # Compute where the prefix ends and the suffix (with LoRA) begins.
        # e.g. 46 total layers, lora_fraction=0.5 → prefix_split_layer=23
        self._prefix_split_layer = self._compute_prefix_end(
            len(self._layers), self.lora_fraction
        )

        # HF changed the argument name of create_causal_mask between versions.
        # Detect which name it uses so we call it correctly.
        self._causal_mask_inputs_embeds_kwarg = (
            "inputs_embeds"
            if "inputs_embeds" in inspect.signature(create_causal_mask).parameters
            else "input_embeds"
        )

        if self.lora_path:
            # Load a previously saved LoRA adapter from disk.
            lora_path = self._prepare_lora_path_for_load(self.lora_path)
            # Read layer indices and target names from the adapter, so we know
            # where the prefix/suffix split was when the adapter was trained.
            self._infer_lora_metadata_from_adapter(lora_path)
            self.model = PeftModel.from_pretrained(
                self.model,
                str(lora_path),
                adapter_name=self.lora_adapter_name,
                is_trainable=self.lora_is_trainable,
            )
        else:
            # Create fresh LoRA adapters on the suffix layers.
            lora_targets = self._requested_lora_targets()
            suffix_layer_indices = list(range(self._prefix_split_layer, len(self._layers)))

            # Verify the requested targets actually exist in the suffix layers.
            # (Some model variants might not have all projections.)
            target_modules = self._available_lora_leaf_targets(
                lora_targets, suffix_layer_indices,
            )

            lora_rank = int(getattr(self.config, "lora_rank", 128))
            # LoRA rank r: the inner dimension of the A and B matrices.
            # Higher rank = more expressive but more parameters.
            # r=128 is large; common choices are 8-64 for smaller models.

            lora_alpha = int(getattr(self.config, "lora_alpha", lora_rank * 2))
            # Scaling factor: effective update = (B @ A) * (alpha / r).
            # alpha = 2*r is a common heuristic that keeps the effective learning rate
            # roughly independent of the rank choice.

            self.lora = target_modules
            self.lora_rank = lora_rank
            self.lora_alpha = lora_alpha
            lora_cfg = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=0.0,   # No dropout during RL training (we want deterministic gradients)
                bias="none",        # Don't add LoRA to bias terms
                task_type="CAUSAL_LM",
                target_modules=target_modules,
                layers_to_transform=suffix_layer_indices,  # Only apply to suffix layers
                layers_pattern="layers",  # Tell PEFT how to find layer indices in module names
            )
            self.model = get_peft_model(
                self.model,
                lora_cfg,
                adapter_name=self.lora_adapter_name,
            )

        # After wrapping with PEFT, the model structure changes (PEFT adds wrappers).
        # Re-resolve the inner model and layers so our references are up to date.
        self._inner_model, self._lm_head = self._resolve_text_stack(self.model)
        self._layers = self._inner_model.layers
        self._layer_types = _get_layer_types(self._inner_model)

        # Wrap all layers with _StreamGemmaDecoderLayer for chunked execution support.
        self._wrap_suffix_layers_for_streaming()

        self._model_device = next(self.model.parameters()).device

        # Freeze all parameters, then selectively unfreeze the LoRA ones.
        # This ensures that gradients only flow through the tiny LoRA A/B matrices,
        # not through the large frozen base model weights.
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
        # Return the list of target module names from config, or fall back to defaults.
        if isinstance(self.lora, list):
            return [str(target) for target in self.lora]
        return list(_GEMMA4_DEFAULT_LORA_TARGETS)

    def _prepare_lora_path_for_load(self, adapter_path: str) -> Path:
        # Check whether the adapter directory needs normalization before loading.
        # If the keys contain wrapper-specific paths (base_layer, base_attn), we
        # create a cleaned copy in a temp dir and return that path instead.
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
            return adapter_dir  # Already clean, load directly

        # Needs normalization: write a cleaned copy into a temp dir.
        tmp = tempfile.TemporaryDirectory(prefix="gemma4_lora_")
        self._lora_temp_dirs.append(tmp)  # Keep alive until model is destroyed
        return _normalize_lora_adapter_dir(adapter_dir, Path(tmp.name))

    def _infer_lora_metadata_from_adapter(self, adapter_path: Path) -> None:
        # When loading a saved adapter, read its config to set:
        #   - self.lora: which projections were trained
        #   - self.lora_rank / lora_alpha: the LoRA hyperparameters
        #   - self._prefix_split_layer: where the prefix/suffix boundary was
        # This ensures the current model setup matches what the adapter expects.
        peft_config = PeftConfig.from_pretrained(str(adapter_path))
        config_data = _load_adapter_config_dict(adapter_path)

        targets = _coerce_str_list(config_data.get("target_modules"))
        if not targets:
            targets = _coerce_str_list(getattr(peft_config, "target_modules", None))
        self.lora = sorted({_lora_target_leaf(target) for target in targets})

        self.lora_rank = int(getattr(peft_config, "r", config_data.get("r", 128)))
        self.lora_alpha = int(
            getattr(peft_config, "lora_alpha", config_data.get("lora_alpha", self.lora_rank * 2))
        )

        # Find the first layer index that has LoRA weights — that's the prefix/suffix split.
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
            self.lora_fraction = (len(self._layers) - self._prefix_split_layer) / len(self._layers)

    def _available_lora_leaf_targets(
        self,
        requested_targets: list[str],
        layer_indices: list[int],
    ) -> list[str]:
        # Walk all modules in the suffix layers and find which of the requested
        # target names actually exist. This handles model variants where some
        # projections (e.g. v_proj) might be absent or fused with others.
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
        # PEFT's load_adapter can't handle our streaming wrappers (the extra
        # base_layer/base_attn levels confuse its module name matching).
        # We temporarily unwrap, load, then re-wrap.
        normalized_path = self._prepare_lora_path_for_load(adapter_path)
        unwrapped = self._unwrap_streaming_layers_for_peft()
        try:
            super().load_lora_adapter(adapter_name, str(normalized_path), is_trainable=is_trainable)
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
        # Unwrap streaming layers so PEFT saves clean (non-wrapper) key names.
        unwrapped = self._unwrap_streaming_layers_for_peft()
        try:
            self.model.save_pretrained(str(save_dir), selected_adapters=[save_name])
        finally:
            if unwrapped:
                self._wrap_suffix_layers_for_streaming()

        # PEFT may save into a subdirectory named after the adapter.
        # Flatten that into save_dir so the path is predictable.
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

        # Normalize key names and run sanity checks on the saved directory.
        _normalize_lora_adapter_dir(save_dir)
        _assert_clean_lora_adapter_dir(save_dir)

    def _unwrap_streaming_layers_for_peft(self) -> bool:
        # Remove our streaming wrappers from the model temporarily.
        # PEFT's save/load code uses module.named_modules() and matches by name,
        # so it needs to see the original HF layer structure, not our wrappers.
        changed = False
        for idx, layer in enumerate(self._inner_model.layers):
            if isinstance(layer, _StreamGemmaDecoderLayer):
                base_layer = layer.base_layer
                if isinstance(base_layer.self_attn, _StreamGemmaAttention):
                    base_layer.self_attn = base_layer.self_attn.base_attn
                self._inner_model.layers[idx] = base_layer
                changed = True
            elif hasattr(layer, "self_attn") and isinstance(layer.self_attn, _StreamGemmaAttention):
                layer.self_attn = layer.self_attn.base_attn
                changed = True
        if changed:
            self._layers = self._inner_model.layers
        return changed

    def _wrap_suffix_layers_for_streaming(self) -> None:
        # Wrap every layer with _StreamGemmaDecoderLayer.
        # Only suffix layers (>= prefix_split_layer) get torch.compile on their MLP.
        use_compile = self._use_compile
        for idx in range(len(self._layers)):
            layer = self._layers[idx]
            if isinstance(layer, _StreamGemmaDecoderLayer):
                layer._use_compile = use_compile
                continue  # Already wrapped
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
        # Gemma 4 can be loaded as a pure language model or as a vision-language model.
        # In the VLM case the structure is: model → model.language_model → ...
        # In the LM case:                   model → model.model → layers, embed_tokens
        # PEFT adds another wrapper:        PeftModel → base_model → model → ...
        # We do a BFS to find the innermost object that has both .layers and .embed_tokens.
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
                inner_model = candidate.language_model  # VLM path
                break
            if hasattr(candidate, "layers") and hasattr(candidate, "embed_tokens"):
                inner_model = candidate  # LM path
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
        # Convert lora_fraction → first layer index that gets LoRA.
        # frac=0.5, n_layers=46 → floor(46 * 0.5) = 23 (layers 0-22 are prefix, 23-45 are suffix)
        # frac=0.0 means no LoRA at all (all layers are prefix).
        # frac=1.0 means LoRA on all layers (layer 0 is the first suffix layer).
        if frac <= 0.0:
            return n_layers
        if frac >= 1.0:
            return 0
        return int(math.floor(n_layers * (1.0 - frac)))

    def _debug_lora_state(self) -> dict:
        # Diagnostic helper: print LoRA parameter counts and which layers are trained.
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
        # Gemma 4 applies a final logit softcap: tanh(logits/cap) * cap.
        # This squashes extreme logits into [-cap, +cap], which makes the loss
        # landscape smoother and prevents probability from concentrating on one token.
        softcap = getattr(self._inner_model.config, "final_logit_softcapping", None)
        if softcap is not None:
            logits = torch.tanh(logits / softcap) * softcap
        return logits

    def _get_per_layer_inputs(self, input_ids: torch.Tensor) -> torch.Tensor | None:
        # Gemma 4 has an optional per-layer input: a small embedding vector computed
        # from the input tokens and injected into each transformer layer.
        # If this feature is not configured, return None and skip it everywhere.
        if not getattr(self._inner_model.config, "hidden_size_per_layer_input", None):
            return None
        inputs_embeds = self._inner_model.embed_tokens(input_ids)
        return self._inner_model.get_per_layer_inputs(input_ids, inputs_embeds)

    def _project_per_layer_inputs(
        self,
        hidden_states: torch.Tensor,
        per_layer_inputs: torch.Tensor | None,
    ) -> torch.Tensor | None:
        # Project per-layer inputs into the hidden dimension using the model's
        # learned projection. Returns None if per-layer inputs are not used.
        if per_layer_inputs is None:
            return None
        return self._inner_model.project_per_layer_inputs(hidden_states, per_layer_inputs)

    def _mask_mapping_has_full_masks(self, mask_mapping: dict | None) -> bool:
        # Detect whether a mask_mapping contains 4D (full) or 2D (chunked) masks.
        # Used to decide if we can reuse a prebuilt mask_mapping from the prefix pass
        # in the suffix pass, or if we need to rebuild it.
        # 4D mask (dim >= 4) = full pre-built mask, compatible with non-streaming suffix.
        # 2D mask (dim == 2) = raw attention mask, compatible with streaming suffix.
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
        # Thin wrapper around the module-level _build_runtime that fills in model-specific args.
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
        # Clone and detach the K/V tensors computed during the prefix pass.
        # We detach them so they're treated as constants during the suffix backward pass
        # (gradients should not flow back into the frozen prefix layers through the KV).
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
        # Move shared K/V states to the correct device (e.g. from CPU if offloaded)
        # and optionally truncate to seq_end tokens (for chunked suffix processing).
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
        # Move every tensor in the bundle to CPU RAM.
        # This frees GPU memory between the prefix and suffix passes, at the cost
        # of a PCIe transfer when the suffix pass fetches them back.
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
        # Run the frozen prefix layers and package the result into a PrefixBundle.
        # The bundle is then reused for all G completions without re-running prefix layers.
        (
            hidden_prefix,
            position_ids,
            shared_kv_states,
            per_layer_inputs,
            mask_mapping,
            position_embeddings,
        ) = self._forward_prefix(input_ids, attention_mask)

        bundle = PrefixBundle(
            hidden_prefix=_to_normal_tensor(hidden_prefix),  # detach from prefix compute graph
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
        # Unpack a PrefixBundle for consumption by _forward_suffix.
        # Moves hidden_prefix and position_ids to GPU if they were offloaded to CPU.
        # Sets requires_grad=True on hidden_prefix so gradients flow through suffix layers.
        model_device = self._model_device
        hidden_prefix = self._move_tensor(bundle.hidden_prefix, model_device)
        if hidden_prefix is None:
            raise RuntimeError("prefix bundle did not contain hidden_prefix")
        position_ids = self._move_tensor(bundle.position_ids, model_device)
        if position_ids is None:
            raise RuntimeError("prefix bundle did not contain position_ids")
        return (
            hidden_prefix.detach().requires_grad_(torch.is_grad_enabled()),
            # requires_grad_(True) only when called from within a grad-enabled context
            # (i.e. the backward pass). During inference (no grad), stays False.
            position_ids,
            bundle.shared_kv_states,
            bundle.per_layer_inputs,
        )

    def _run_suffix_from_prefix_bundle(
        self,
        bundle: PrefixBundle,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Entry point for running the suffix given a cached prefix bundle.
        # Clones the bundle first so autograd gets fresh tensor leaves to track.
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
        # Run the frozen prefix layers (layers 0 to prefix_split_layer-1) over the input.
        # Returns the hidden states after those layers, plus everything the suffix needs.
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Convert token IDs to dense vectors (embedding lookup).
        hidden_states = self._inner_model.embed_tokens(input_ids)

        # [0, 1, 2, ..., seq_len-1] repeated for each batch item.
        position_ids = (
            torch.arange(seq_len, device=device, dtype=torch.long)
            .unsqueeze(0)
            .expand(batch_size, seq_len)
        )

        use_chunked_prefix = self._prefix_token_chunk_size > 0
        # build_full_masks=True when we're NOT chunking — we can build the full causal
        # mask upfront and reuse it. When chunking, we build per-chunk masks on the fly.
        mask_mapping, position_embeddings = self._build_runtime(
            hidden_states,
            attention_mask,
            position_ids,
            build_full_masks=not use_chunked_prefix,
        )

        shared_kv_states = {}
        raw_per_layer_inputs = self._get_per_layer_inputs(input_ids)
        per_layer_inputs = self._project_per_layer_inputs(hidden_states, raw_per_layer_inputs)

        for idx, layer in enumerate(self._layers[: self._prefix_split_layer]):
            per_layer_input = (
                per_layer_inputs[:, :, idx, :] if per_layer_inputs is not None else None
            )
            if use_chunked_prefix and isinstance(layer, _StreamGemmaDecoderLayer):
                # Chunked prefix: pre-compute K/V for the full sequence, then
                # process query tokens chunk by chunk to reduce peak memory.
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
                # Non-chunked prefix: run the full sequence through in one shot.
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
        # Run the trainable suffix layers (prefix_split_layer to end).
        # These layers have LoRA and receive gradients during training.

        # Streaming suffix requires: grad checkpointing ON + chunk size set.
        use_streaming_suffix = (
            self._use_grad_checkpoint
            and torch.is_grad_enabled()
            and self._suffix_token_chunk_size > 0
        )
        # Streaming needs 2D masks (per-chunk mask is built on the fly).
        # Non-streaming needs 4D masks (full pre-built mask, faster).
        build_full_masks = not use_streaming_suffix

        # Check if we can reuse masks/embeddings already computed during the prefix pass.
        can_reuse_prebuilt = (
            prebuilt_mask_mapping is not None
            and prebuilt_position_embeddings is not None
            and self._mask_mapping_has_full_masks(prebuilt_mask_mapping) == build_full_masks
        )
        if can_reuse_prebuilt:
            mask_mapping = prebuilt_mask_mapping
            position_embeddings = prebuilt_position_embeddings
        else:
            # Rebuild — either no prebuilt mask or the mask type doesn't match the mode.
            mask_mapping, position_embeddings = self._build_runtime(
                hidden_states, attention_mask, position_ids, build_full_masks=build_full_masks,
            )

        for offset, layer in enumerate(self._layers[self._prefix_split_layer :]):
            abs_idx = self._prefix_split_layer + offset

            current_per_layer_input_source = None
            if per_layer_inputs is not None:
                current_per_layer_input_source = per_layer_inputs[:, :, abs_idx, :]

            # _LayerReplay bundles the layer with all its context so it can be called
            # as a simple function: layer_replay(hidden_states, chunk_range=...).
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
                # Chunked gradient checkpointing: forward in no-grad chunks,
                # backward reruns each chunk. Lowest memory.
                hidden_states = _StreamCheckpointFunction.apply(
                    layer_replay,
                    self._suffix_token_chunk_size,
                    self._offload_prefix_to_cpu,
                    hidden_states,
                )
            elif self._use_grad_checkpoint and torch.is_grad_enabled():
                # Standard gradient checkpointing: saves memory by recomputing
                # the entire layer during backward instead of storing activations.
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer_replay, hidden_states, use_reentrant=False
                )
            else:
                # No checkpointing: fastest but highest memory.
                hidden_states = layer_replay(hidden_states)

        # Final layer norm over the last suffix hidden states.
        return self._inner_model.norm(hidden_states)
