# region Imports

import math
from typing import Dict, Tuple

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseModel

# endregion


class GptOssModel(BaseModel):

    def setup(self) -> None:

        # region Base Load

        # Load tokenizer/model once in setup so forward stays hot-path only.
        device = f"cuda:{self.cuda_device_index}"
        attn_impl = getattr(self.config, "attn_implementation", "eager")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map={"": device},
            use_cache=False,
            torch_dtype=torch.bfloat16,
            # Why: this remains configurable because backend choice is hardware-sensitive.
            # Some kernels are Hopper-only; some flex paths can fail to compile on smaller GPUs.
            attn_implementation=attn_impl,
        )

        # endregion

        # region LoRA Config Input and Target Discovery

        # Read LoRA target selection controls from config.
        
        self.lora = getattr(self.config, "lora")
        self.lora_fraction = getattr(self.config, "lora_fraction")
        
        # Split requested targets into PEFT module targets vs parameter targets.
        # Why: attention projections and MoE expert tensors are attached differently in PEFT.
        base = self.model
        lora_targets = (
            self.lora
            if isinstance(self.lora, list)
            else ["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        target_modules, expert_param_targets = (
            self._get_lora_target_modules_and_expert_params(
                lora_targets,
                self.lora_fraction,
            )
        )

        # endregion

        # region LoRA Attach

        # Build PEFT LoRA config and wrap the base model.
        
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

        # Resolve inner transformer and pre-split layers into prefix/suffix subsets.
        # Why: prefix is run as frozen context; suffix remains the trainable path.

        base_model = (
            self.model.base_model.model
            if hasattr(self.model, "base_model")
            else self.model
        )

        self._inner_model = (
            base_model.model if hasattr(base_model, "model") else base_model
        )
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

        # Optional memory-saving toggle for suffix backward path.
        self._use_grad_checkpoint = bool(
            getattr(self.config, "use_grad_checkpoint", False)
        )

        self._compiled_prefix_depth = self._prefix_split_layer
        
        # endregion

        # region Prefix Runner Build

        # Build callable that runs only prefix subset.
        # Why: keeping this isolated allows future compile/wrapping without touching suffix code.
        
        def _prefix_body(
            hidden_states: torch.Tensor,
            full_attention_mask: torch.Tensor,
            sliding_attention_mask: torch.Tensor,
            position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        ) -> torch.Tensor:
            for i, layer in enumerate(self._prefix_layers):
                layer_type = self._prefix_layer_types.get(i, "full_attention")
                layer_mask = (
                    sliding_attention_mask
                    if layer_type == "sliding_attention"
                    else full_attention_mask
                )
                out = layer(
                    hidden_states,
                    attention_mask=layer_mask,
                    position_embeddings=position_embeddings,
                )
                hidden_states = out[0] if isinstance(out, tuple) else out
            return hidden_states

        # Compile path is intentionally disabled for now.
        # Why: this graph hit reproducible torch.compile/inductor failures in local runs.
        # if hasattr(torch, "compile"):
        #     try:
        #         self._compiled_prefix_fn = torch.compile(_prefix_body, dynamic=True)
        #     except Exception:
        #         self._compiled_prefix_fn = _prefix_body
        # else:
        #     self._compiled_prefix_fn = _prefix_body
        self._compiled_prefix_fn = _prefix_body

        # endregion

    def _get_lora_cutoff_index(self, frac: float) -> int:
        # Collect numeric transformer layer indices from names like "...layers.<idx>...".
        layer_indices = set()
        for name, _ in self.model.named_modules():
            parts = name.split(".")
            for i in range(len(parts) - 1):
                if parts[i] == "layers" and parts[i + 1].isdigit():
                    layer_indices.add(int(parts[i + 1]))

        # If fraction is disabled or layers are not discoverable, start from index 0.
        if frac <= 0.0 or not layer_indices:
            return 0

        # Keep the top `frac` portion trainable by cutting off earlier layers.
        return int(math.floor(max(layer_indices) * (1.0 - frac)))

    def _get_lora_target_modules_and_expert_params(
        self, lora_targets: list[str], frac: float
    ) -> Tuple[list[str], list[str]]:
        # LoRA is applied only on suffix layers based on fraction.
        # Why: prefix is meant to stay frozen and cheap, suffix carries adaptation.
        cutoff = self._get_lora_cutoff_index(frac)

        # Split user targets into:
        # - attention module names that PEFT can target via `target_modules`
        # - MLP/expert parameter names to target via `target_parameters`
        attn_leaf_targets = {"q_proj", "k_proj", "v_proj", "o_proj"} & set(lora_targets)
        mlp_leaf_targets = set(lora_targets) - attn_leaf_targets

        attn_target_modules: list[str] = []
        for name, _ in self.model.named_modules():
            parts = name.split(".")

            # Find layer index from "...layers.<idx>..." path.
            layer_idx = None
            for i in range(len(parts) - 1):
                if parts[i] == "layers" and parts[i + 1].isdigit():
                    layer_idx = int(parts[i + 1])
                    break

            # Keep only modules in suffix layers and requested attention leaves.
            if layer_idx is None or layer_idx < cutoff:
                continue
            if parts[-1] in attn_leaf_targets:
                attn_target_modules.append(name)

        expert_param_targets: list[str] = []
        for name, _ in self.model.named_parameters():
            parts = name.split(".")

            # Find layer index from parameter path.
            layer_idx = None
            for i in range(len(parts) - 1):
                if parts[i] == "layers" and parts[i + 1].isdigit():
                    layer_idx = int(parts[i + 1])
                    break

            # Keep only suffix parameters.
            if layer_idx is None or layer_idx < cutoff:
                continue

            # For MLP/expert targets, only use actual weight tensors.
            leaf = parts[-1]
            if "experts" in name and leaf in mlp_leaf_targets:
                expert_param_targets.append(name)

        return attn_target_modules, expert_param_targets

    def _identify_layer_types(self) -> Dict[int, str]:
        # Resolve the inner transformer module used by GPT-OSS style models.
        base = (
            self.model.base_model.model
            if hasattr(self.model, "base_model")
            else self.model
        )
        inner = base.model

        layer_types: Dict[int, str] = {}

        # Prefer explicit layer typing from model config when available.
        if hasattr(inner, "config") and hasattr(inner.config, "layer_types"):
            for i, layer_type in enumerate(inner.config.layer_types):
                layer_types[i] = layer_type
            return layer_types

        # Fallback: infer from attention module attributes per layer.
        for i, layer in enumerate(inner.layers):
            attn = layer.self_attn
            if hasattr(attn, "sliding_window") and attn.sliding_window is not None:
                layer_types[i] = "sliding_attention"
            else:
                layer_types[i] = "full_attention"
        return layer_types

    def _forward_prefix(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run prefix layers and return raw prefix outputs.

        Memory notes for sequence length 132,000 (B=1, bf16 hidden states):
        - Hidden states: B * T * H * 2 bytes.
          Example at H=4096: 1 * 132000 * 4096 * 2 = 1,081,344,000 bytes (~1.01 GiB).
        - Position ids use int64: B * T * 8 bytes (~1.01 MiB), typically negligible vs hidden.
        """
        
        # Input is expected as token ids with shape [batch, seq_len].
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        # Build position ids explicitly.
        # Why: in this split/manual forward path we do not call the monolithic HF model.forward,
        # so we must provide the RoPE lookup positions ourselves.
        position_ids = (
            torch.arange(seq_len, dtype=torch.long)
            .unsqueeze(0)
            .expand(batch_size, seq_len)
            .to(input_ids)
        )

        # Token embedding lookup to get hidden states [batch, seq_len, hidden_size].
        hidden_states = self._inner_model.embed_tokens(input_ids)

        # Model config gives us head_dim needed by rotary embedding module.
        cfg = self._inner_model.config
        head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)

        # Dummy tensor is used only to satisfy rotary_emb API shape contract.
        dummy = hidden_states.new_zeros(1, seq_len, head_dim)

        # Precompute rotary cos/sin once for the full prefix pass.
        position_embeddings = self._inner_model.rotary_emb(dummy, position_ids)

        # Match HF GPT-OSS masking contract for per-layer attention type selection.
        # Why: full/sliding layer masks must match HF exactly for boundary/logit parity.
        from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

        mask_kwargs = {
            "config": self._inner_model.config,
            "inputs_embeds": hidden_states,
            "attention_mask": attention_mask,
            "past_key_values": None,
        }
        full_attention_mask = create_causal_mask(**mask_kwargs)
        sliding_attention_mask = create_sliding_window_causal_mask(**mask_kwargs)

        # Run the preselected prefix-layer subset through compiled prefix function.
        hidden_states = self._compiled_prefix_fn(
            hidden_states,
            full_attention_mask,
            sliding_attention_mask,
            position_embeddings,
        )

        # Return raw prefix outputs; boundary grad handoff is managed by BaseModel.forward.
        return hidden_states, position_ids

    def _forward_suffix(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run suffix layers and return final normalized hidden states.

        Terms used here:
        - hidden_states: layer input/output tensor with shape [B, T, H]. This is
          the main signal flowing between transformer blocks.
        - weights/parameters: trainable model tensors (for example projection and
          MLP weights). These are what optimization updates.
        - gradients: d(loss)/d(parameter) tensors computed in backward; they match
          parameter shapes and are used by the optimizer to update weights.
        - activations: outputs produced by forward ops (for example post-attention
          output, post-MLP output). Some of these are saved for backward.
        - intermediates: internal temporary activations inside a layer's op graph
          (for example q/k/v projections, softmax-related values, pre/post-norm states).
        - tensors used in backprop: saved activations/intermediates plus incoming
          gradient signals; autograd combines them to compute parameter gradients.
        - requires_grad: boolean tensor property that enables autograd tracking.
          Parameters typically have this enabled. Outputs inherit it if any op input
          requires grad.
        - compute graph / autograd graph: dynamic DAG built during forward.
          Nodes are differentiable ops; edges are tensors flowing between ops.
          Backward traverses this graph in reverse.
        - graph nodes / Function objects: autograd entries created per differentiable
          op (matmul, softmax, etc.). Each node defines backward behavior.
        - leaf tensors: graph-boundary tensors with no creator op. Model parameters
          are the canonical leaves, and gradients accumulate into `.grad` there.
        - non-leaf tensors: tensors produced by differentiable ops (`grad_fn` set).
          These are intermediates autograd differentiates through.
        - retained/saved tensors: subset of non-leaf tensors explicitly kept for
          backward (`save_for_backward`). These dominate forward->backward memory.
        - grad_fn: pointer on non-leaf tensors to the Function that created them.
          Backward follows these links in reverse to propagate gradients.

        Memory/computation notes (bf16 activations, bytes_per_elem=2):
        - Base hidden tensor memory is always B * T * H * 2 bytes.
          Example (B=1, T=132000, H=4096): ~1.01 GiB.

        Vanilla autograd (checkpointing OFF):
        - Forward stores many intermediates from deep inside each suffix layer.
        - Backward reuses them directly.
        - Effect: higher global VRAM peak, lower recompute cost.

        Gradient checkpointing (checkpointing ON):
        - Forward stores only checkpoint boundary tensors (end-to-end layer inputs/outputs),
          not deep internal intermediates for each checkpointed layer.
        - Backward reruns local layer forward(s) to recreate those intermediates on demand.
        - Effect: lower global VRAM peak, but extra compute and possible local transient
          peaks while each checkpointed layer is recomputed.

        Both modes are mathematically equivalent in output/gradients; tradeoff is
        memory footprint vs recomputation time.
        """
        # Recompute rotary embeddings once for the full suffix pass.
        cfg = self._inner_model.config
        seq_len = hidden_states.shape[1]
        head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        dummy = hidden_states.new_zeros(1, seq_len, head_dim)
        position_embeddings = self._inner_model.rotary_emb(dummy, position_ids)

        # Build the same mask mapping used by HF GPT-OSS forward.
        # Why: suffix parity breaks if full/sliding masks are not routed exactly per layer type.
        from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

        mask_kwargs = {
            "config": self._inner_model.config,
            "inputs_embeds": hidden_states,
            "attention_mask": attention_mask,
            "past_key_values": None,
        }
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
            "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
        }

        # Iterate through only the suffix subset selected during setup.
        for i, layer in enumerate(self._suffix_layers):
            abs_idx = self._prefix_split_layer + i
            layer_type = self._suffix_layer_types.get(abs_idx, "full_attention")
            layer_mask = causal_mask_mapping.get(layer_type, causal_mask_mapping["full_attention"])

            # Optional checkpointing: lower activation memory, higher recompute cost.
            if self._use_grad_checkpoint and torch.is_grad_enabled():
                # Capture static layer/mask in closure; only tensors are checkpoint inputs.
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
                        position_ids=position_ids,
                        past_key_values=None,
                        use_cache=False,
                        position_embeddings=(cos, sin),
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
                # Default path: regular suffix forward without checkpointing.
                out = layer(
                    hidden_states,
                    attention_mask=layer_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                    position_embeddings=position_embeddings,
                )
                hidden_states = out[0] if isinstance(out, tuple) else out

        # Match legacy path: apply final norm after suffix stack.
        hidden_states = self._inner_model.norm(hidden_states)
        return hidden_states
