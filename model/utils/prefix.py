# PrefixBundle: a container that holds everything computed during the frozen prefix pass.
#
# The big idea behind prefix/suffix splitting:
#   When training with RL, you generate G completions (e.g. 8) for one prompt.
#   The prompt (prefix) is the same for all G completions — so it's wasteful to
#   run it through the frozen layers G times. Instead, run it once, store the
#   result in a PrefixBundle, then run each completion through the trainable
#   suffix layers G times, reusing the cached prefix output.
#
#   This is the core memory trick that makes training large models affordable.

from dataclasses import dataclass

import torch


@dataclass
class PrefixBundle:
    hidden_prefix: torch.Tensor
    # The hidden states output by the last frozen prefix layer.
    # Shape: [batch, seq_len, hidden_dim].
    # This is the "summary" of the entire prompt that the suffix layers will continue from.

    position_ids: torch.Tensor
    # Token position indices [0, 1, 2, ..., seq_len-1] for the whole sequence.
    # Needed by the suffix layers to apply RoPE at the correct positions.

    shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]]
    # Gemma 4 has "KV-sharing" layers: some attention layers reuse the K/V tensors
    # computed by an earlier layer instead of computing their own.
    # This dict maps layer_index → (key_states, value_states) for those shared layers.
    # The suffix layers look up their KV here instead of recomputing from hidden states.

    per_layer_inputs: torch.Tensor | None = None
    # Gemma 4 has an optional per-layer input feature: a small extra vector injected
    # into each transformer layer. If the model config has hidden_size_per_layer_input,
    # this tensor holds those inputs for every layer. None if the feature is not used.

    mask_mapping: dict | None = None
    # Pre-built attention masks keyed by layer type ("full_attention", "sliding_attention").
    # Building these masks once in the prefix pass and reusing them in the suffix pass
    # avoids redundant computation. None if not yet built or using chunked prefix.

    position_embeddings: dict | None = None
    # Pre-built RoPE (cos, sin) tensors keyed by layer type.
    # Same idea as mask_mapping — compute once, reuse in suffix.

    def clone_for_autograd(self) -> "PrefixBundle":
        # Before running the suffix pass with gradients enabled, we clone the bundle.
        # Why: PyTorch's autograd tracks tensor operations. If the same tensor object
        # is used in both a no-grad prefix pass and a grad-enabled suffix pass,
        # the gradient graph can get confused. Cloning gives autograd fresh leaves to work with.
        return PrefixBundle(
            hidden_prefix=self.hidden_prefix,
            position_ids=self.position_ids,
            shared_kv_states=dict(self.shared_kv_states),  # shallow copy of the dict
            per_layer_inputs=self.per_layer_inputs,
            mask_mapping=(
                {
                    key: value.clone() if value is not None else None
                    for key, value in self.mask_mapping.items()
                }
                if self.mask_mapping is not None
                else None
            ),
            position_embeddings=(
                {
                    # cos and sin are cloned so autograd treats them as new leaves
                    key: (cos.clone(), sin.clone())
                    for key, (cos, sin) in self.position_embeddings.items()
                }
                if self.position_embeddings is not None
                else None
            ),
        )
