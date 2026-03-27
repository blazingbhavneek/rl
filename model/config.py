from dataclasses import dataclass


@dataclass
class ModelConfig:
    
    lora: list[str] | str
    lora_fraction: float
    lora_rank: int = 128
    lora_alpha: int = 256
    
    chunk_size: int | None = None
    
    attn_implementation: str = "eager"
    
    cuda_device_index: int = 0
    
    use_grad_checkpoint: bool = False
