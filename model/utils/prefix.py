from dataclasses import dataclass

import torch


@dataclass
class PrefixBundle:
    hidden_prefix: torch.Tensor
    position_ids: torch.Tensor
    shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]]
    per_layer_inputs: torch.Tensor | None = None
    mask_mapping: dict | None = None
    position_embeddings: dict | None = None

    def clone_for_autograd(self) -> "PrefixBundle":
        return PrefixBundle(
            hidden_prefix=self.hidden_prefix,
            position_ids=self.position_ids,
            shared_kv_states=dict(self.shared_kv_states),
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
                    key: (cos.clone(), sin.clone())
                    for key, (cos, sin) in self.position_embeddings.items()
                }
                if self.position_embeddings is not None
                else None
            ),
        )
