# region Imports

import math
from typing import Dict, Tuple

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import create_causal_mask

from .base import BaseModel

# endregion


class Qwen3_5Model(BaseModel):

    def setup(self) -> None:

        # region Base Load

        device = f"cuda:{self.cuda_device_index}"
        attn_impl = getattr(self.config, "attn_implementation", "eager")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map={"": device},
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        )
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False

        # endregion

        # region LoRA Config Input and Target Discovery

        self.lora = getattr(self.config, "lora")
        self.lora_fraction = getattr(self.config, "lora_fraction")

        base = self.model
        lora_targets = (
            self.lora
            if isinstance(self.lora, list)
            else [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        )
        target_modules, expert_param_targets = (
            self._get_lora_target_modules_and_expert_params(
                lora_targets,
                self.lora_fraction,
            )
        )

        # endregion

        # region LoRA Attach

        lora_rank = int(getattr(self.config, "lora_rank", 128))
        lora_alpha = int(getattr(self.config, "lora_alpha", lora_rank * 2))

        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
            target_parameters=expert_param_targets,
        )

        self.model = get_peft_model(base, lora_cfg)

        # endregion

        # region Prefix/Suffix Partition

        base_model = (
            self.model.base_model.model
            if hasattr(self.model, "base_model")
            else self.model
        )

        inner_candidate = (
            base_model.model if hasattr(base_model, "model") else base_model
        )
        if hasattr(inner_candidate, "language_model"):
            inner_candidate = inner_candidate.language_model
        if hasattr(inner_candidate, "model") and hasattr(inner_candidate.model, "layers"):
            inner_candidate = inner_candidate.model

        self._inner_model = inner_candidate
        self._lm_head = base_model.lm_head

        self._prefix_split_layer = self._get_lora_cutoff_index(self.lora_fraction)
        self._prefix_layers = tuple(
            self._inner_model.layers[: self._prefix_split_layer]
        )
        self._suffix_layers = tuple(
            self._inner_model.layers[self._prefix_split_layer :]
        )

        self._layer_types = list(getattr(self._inner_model.config, "layer_types", []))
        self._suffix_layer_types = {
            self._prefix_split_layer + i: (
                self._layer_types[self._prefix_split_layer + i]
                if (self._prefix_split_layer + i) < len(self._layer_types)
                else "full_attention"
            )
            for i in range(len(self._suffix_layers))
        }
        self._prefix_layer_types = {
            i: (self._layer_types[i] if i < len(self._layer_types) else "full_attention")
            for i in range(len(self._prefix_layers))
        }

        self._use_grad_checkpoint = bool(
            getattr(self.config, "use_grad_checkpoint", False)
        )
        self._compiled_prefix_depth = self._prefix_split_layer

        # endregion

        # region Prefix Runner Build

        def _prefix_body(
            hidden_states: torch.Tensor,
            causal_mask: torch.Tensor,
            linear_attn_mask: torch.Tensor | None,
            position_ids_layer: torch.Tensor,
            position_ids_rope: torch.Tensor,
            position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        ) -> torch.Tensor:
            layer_types = list(getattr(self._inner_model.config, "layer_types", []))
            for i in range(self._prefix_split_layer):
                layer = self._inner_model.layers[i]
                layer_type = (
                    layer_types[i] if i < len(layer_types) else "full_attention"
                )
                layer_mask = linear_attn_mask if layer_type == "linear_attention" else causal_mask
                out = layer(
                    hidden_states,
                    attention_mask=layer_mask,
                    position_ids=position_ids_layer,
                    past_key_values=None,
                    position_embeddings=position_embeddings,
                    use_cache=False,
                )
                hidden_states = out[0] if isinstance(out, tuple) else out
            return hidden_states

        self._prefix_body_eager = _prefix_body
        self._compiled_prefix_fn = _prefix_body
        if bool(getattr(self.config, "use_prefix_compile", False)):
            try:
                self._compiled_prefix_fn = torch.compile(_prefix_body, dynamic=False)
            except Exception:
                self._compiled_prefix_fn = _prefix_body

        # endregion

    def _get_lora_cutoff_index(self, frac: float) -> int:
        layer_indices = set()
        for name, _ in self.model.named_modules():
            parts = name.split(".")
            for i in range(len(parts) - 1):
                if parts[i] == "layers" and parts[i + 1].isdigit():
                    layer_indices.add(int(parts[i + 1]))

        if frac <= 0.0 or not layer_indices:
            return 0

        return int(math.floor(max(layer_indices) * (1.0 - frac)))

    def _get_lora_target_modules_and_expert_params(
        self, lora_targets: list[str], frac: float
    ) -> Tuple[list[str], list[str]]:
        cutoff = self._get_lora_cutoff_index(frac)

        module_leaf_targets = set(lora_targets)
        target_modules: list[str] = []
        for name, module in self.model.named_modules():
            parts = name.split(".")
            layer_idx = None
            for i in range(len(parts) - 1):
                if parts[i] == "layers" and parts[i + 1].isdigit():
                    layer_idx = int(parts[i + 1])
                    break

            if layer_idx is None or layer_idx < cutoff:
                continue
            if parts[-1] in module_leaf_targets and hasattr(module, "weight"):
                target_modules.append(name)

        expert_param_targets: list[str] = []
        for name, _ in self.model.named_parameters():
            parts = name.split(".")
            layer_idx = None
            for i in range(len(parts) - 1):
                if parts[i] == "layers" and parts[i + 1].isdigit():
                    layer_idx = int(parts[i + 1])
                    break

            if layer_idx is None or layer_idx < cutoff:
                continue

            leaf = parts[-1]
            if "experts" in name and leaf in module_leaf_targets:
                expert_param_targets.append(name)

        return sorted(set(target_modules)), sorted(set(expert_param_targets))

    def _identify_layer_types(self) -> Dict[int, str]:
        base = (
            self.model.base_model.model
            if hasattr(self.model, "base_model")
            else self.model
        )
        inner = base.model if hasattr(base, "model") else base
        if hasattr(inner, "language_model"):
            inner = inner.language_model
        if hasattr(inner, "model") and hasattr(inner.model, "layers"):
            inner = inner.model

        layer_types: Dict[int, str] = {}
        if hasattr(inner, "config") and hasattr(inner.config, "layer_types"):
            for i, layer_type in enumerate(inner.config.layer_types):
                layer_types[i] = layer_type
            return layer_types

        for i, layer in enumerate(inner.layers):
            if hasattr(layer, "layer_type"):
                layer_types[i] = str(layer.layer_type)
            elif hasattr(layer, "linear_attn"):
                layer_types[i] = "linear_attention"
            else:
                layer_types[i] = "full_attention"
        return layer_types

    def _forward_prefix(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        hidden_states = self._inner_model.embed_tokens(input_ids)

        batch_size, seq_len = input_ids.shape
        position_ids = (
            torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
            .unsqueeze(0)
            .expand(batch_size, seq_len)
        )
        position_ids_rope = position_ids.unsqueeze(0).expand(3, batch_size, seq_len)
        causal_mask = create_causal_mask(
            config=self._inner_model.config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=position_ids,
        )
        linear_attn_mask = attention_mask
        if attention_mask is not None and torch.all(attention_mask == 1):
            linear_attn_mask = None

        position_embeddings = self._inner_model.rotary_emb(hidden_states, position_ids_rope)

        try:
            hidden_states = self._compiled_prefix_fn(
                hidden_states,
                causal_mask,
                linear_attn_mask,
                position_ids,
                position_ids_rope,
                position_embeddings,
            )
        except Exception:
            self._compiled_prefix_fn = self._prefix_body_eager
            hidden_states = self._compiled_prefix_fn(
                hidden_states,
                causal_mask,
                linear_attn_mask,
                position_ids,
                position_ids_rope,
                position_embeddings,
            )

        return hidden_states, position_ids

    def _forward_suffix(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:

        batch_size, seq_len, _ = hidden_states.shape
        if position_ids.ndim != 2:
            raise ValueError("position_ids must be shape [B, T] for Qwen3.5")
        text_position_ids = position_ids
        rope_position_ids = position_ids.unsqueeze(0).expand(3, batch_size, seq_len)

        causal_mask = create_causal_mask(
            config=self._inner_model.config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=text_position_ids,
        )
        linear_attn_mask = attention_mask
        if attention_mask is not None and torch.all(attention_mask == 1):
            linear_attn_mask = None

        position_embeddings = self._inner_model.rotary_emb(
            hidden_states,
            rope_position_ids,
        )

        for i, layer in enumerate(self._suffix_layers):
            abs_idx = self._prefix_split_layer + i
            layer_type = self._suffix_layer_types.get(abs_idx, "full_attention")
            layer_mask = linear_attn_mask if layer_type == "linear_attention" else causal_mask

            if self._use_grad_checkpoint and torch.is_grad_enabled():
                if layer_mask is None:

                    def _layer_forward(
                        h: torch.Tensor,
                        cos: torch.Tensor,
                        sin: torch.Tensor,
                        _layer=layer,
                    ) -> torch.Tensor:
                        out = _layer(
                            h,
                            attention_mask=None,
                            position_ids=text_position_ids,
                            past_key_values=None,
                            position_embeddings=(cos, sin),
                            use_cache=False,
                        )
                        return out[0] if isinstance(out, tuple) else out

                    hidden_states = torch.utils.checkpoint.checkpoint(
                        _layer_forward,
                        hidden_states,
                        position_embeddings[0],
                        position_embeddings[1],
                        use_reentrant=False,
                    )
                else:

                    def _layer_forward(
                        h: torch.Tensor,
                        a: torch.Tensor,
                        cos: torch.Tensor,
                        sin: torch.Tensor,
                        _layer=layer,
                    ) -> torch.Tensor:
                        out = _layer(
                            h,
                            attention_mask=a,
                            position_ids=text_position_ids,
                            past_key_values=None,
                            position_embeddings=(cos, sin),
                            use_cache=False,
                        )
                        return out[0] if isinstance(out, tuple) else out

                    hidden_states = torch.utils.checkpoint.checkpoint(
                        _layer_forward,
                        hidden_states,
                        layer_mask,
                        position_embeddings[0],
                        position_embeddings[1],
                        use_reentrant=False,
                    )
            else:
                out = layer(
                    hidden_states,
                    attention_mask=layer_mask,
                    position_ids=text_position_ids,
                    past_key_values=None,
                    position_embeddings=position_embeddings,
                    use_cache=False,
                )
                hidden_states = out[0] if isinstance(out, tuple) else out

        hidden_states = self._inner_model.norm(hidden_states)
        return hidden_states
