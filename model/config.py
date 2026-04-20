from dataclasses import dataclass


@dataclass
class ModelConfig:

    lora: list[str] | str
    lora_fraction: float
    lora_rank: int = 128
    lora_alpha: int = 256

    chunk_size: int | None = None
    logprob_chunk_size: int | None = None
    token_chunk_size: int | None = None
    prefix_token_chunk_size: int | None = None
    suffix_token_chunk_size: int | None = None
    offload_prefix_to_cpu: bool = False

    attn_implementation: str = "eager"

    cuda_device_index: int = 0

    use_grad_checkpoint: bool = False
    use_compile: bool = False
