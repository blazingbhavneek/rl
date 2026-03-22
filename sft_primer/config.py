from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class PrimerConfig:
    # Library identity
    LIBRARY_NAME: str = "mylib"

    # Input
    DOCS_FOLDER: str = "sft_primer/docs"

    # Model
    MODEL_PATH: str = "openai/gpt-oss-20b"
    ATTN_IMPLEMENTATION: str = "flash_attention_2"
    DTYPE: str = "bfloat16"

    # LoRA
    LORA_RANK: int = 128
    LORA_ALPHA: int = 256
    LORA_DROPOUT: float = 0.0
    LORA_TARGET: List[str] = None
    LORA_LAYERS_FRAC: float = 0.25

    # Runtime
    SGLANG_MEM_FRAC: float = 0.6
    MAX_SEQ_LEN: int = 8192
    MAX_COMPLETION_TOKENS: int = 2048
    TEMPERATURE_QUESTIONS: float = 0.8
    TEMPERATURE_ANSWERS: float = 0.1

    # Chunking
    CHUNK_SIZE: int = 2048
    CHUNK_OVERLAP: int = 512
    MIN_CHUNK_TOKENS: int = 256

    # QA generation
    SKIP_EXISTING_QA: bool = True
    MAX_QUESTIONS_PER_CHUNK: int = 20

    # Augmentation
    USE_AUGMENTATION: bool = True
    SKIP_EXISTING_AUG: bool = True
    EXAMPLES_PER_FUNCTION: int = 8

    # Data mix
    QA_WEIGHT: float = 1.0
    AUGMENTED_WEIGHT: float = 1.5

    # Training
    EPOCHS: int = 5
    LR: float = 2e-4
    WEIGHT_DECAY: float = 0.01
    WARMUP_RATIO: float = 0.05
    GRAD_ACCUM_STEPS: int = 8
    MICRO_BATCH_SIZE: int = 1
    MAX_GRAD_NORM: float = 1.0
    OPTIMIZER: str = "adamw_8bit"
    EVAL_SPLIT: float = 0.05
    EARLY_STOPPING_PATIENCE: int = 2

    # Chunk profiler
    FORCE_REPROFILE: bool = False
    FORCE_REPROFLE: bool = False  # backward-compatible typo alias

    # Repro
    SEED: int = 42

    def __post_init__(self) -> None:
        if self.LORA_TARGET is None:
            self.LORA_TARGET = ["q_proj", "k_proj", "v_proj", "o_proj"]
        if self.LORA_ALPHA <= 0:
            self.LORA_ALPHA = self.LORA_RANK * 2

    @property
    def BASE_OUTPUT_DIR(self) -> str:
        return f"sft_primer/outputs/{self.LIBRARY_NAME}"

    @property
    def GENERATED_DIR(self) -> str:
        return f"{self.BASE_OUTPUT_DIR}/generated"

    @property
    def DATASET_DIR(self) -> str:
        return f"{self.BASE_OUTPUT_DIR}/dataset"

    @property
    def CHUNK_PROFILE_DIR(self) -> str:
        return f"{self.BASE_OUTPUT_DIR}/chunk_profiles"

    @property
    def LORA_DIR(self) -> str:
        return f"{self.BASE_OUTPUT_DIR}/lora"

    @property
    def CHECKPOINT_DIR(self) -> str:
        return f"{self.BASE_OUTPUT_DIR}/checkpoint"

    @property
    def FINAL_LORA_DIR(self) -> str:
        return f"{self.LORA_DIR}/final"

    def ensure_dirs(self) -> None:
        for p in [
            self.GENERATED_DIR,
            self.DATASET_DIR,
            self.CHUNK_PROFILE_DIR,
            self.LORA_DIR,
            self.CHECKPOINT_DIR,
        ]:
            Path(p).mkdir(parents=True, exist_ok=True)


def build_config(library_name: Optional[str] = None) -> PrimerConfig:
    cfg = PrimerConfig()
    if library_name:
        cfg.LIBRARY_NAME = library_name
    cfg.ensure_dirs()
    return cfg
