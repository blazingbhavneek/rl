# region Training config
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainConfig:

    # --- model ---
    model_path: str
    model_type: str = "qwen3"
    lora_targets: list[str] = field(
        default_factory=lambda: ["gate_proj", "up_proj", "down_proj"]
    )
    lora_fraction: float = 0.5
    lora_rank: int = 64
    lora_alpha: int = 128
    chunk_size: int = 256
    cuda_device_index: int = 0
    use_grad_checkpoint: bool = True

    # --- inference engine ---
    engine_base_url: str = "http://127.0.0.1:8000/v1"
    engine_api_key: str = "EMPTY"
    engine_gpu_memory_utilization: float = 0.60
    engine_semaphore_limit: int = 64

    # --- external teacher ---
    teacher_base_url: str = None
    teacher_api_key: str = None
    teacher_model_name: str = None

    # --- rollouts ---
    n_rollouts: int = 8
    temperature: float = 0.7
    system_prompt: str = "Solve the problem in C. Return only one `c` block."
    max_tokens: int = 1024

    # --- teacher ---
    teacher_max_tokens: int = 8000
    teacher_temperature: float = 0.2
    enable_teacher: bool = False
    max_hint_attempts: int = 1
    hint_reward_discount: float = 0.7
    teacher_max_turns: int = 3

    # --- LoRA adapters ---
    run_dir: str = "fast_checkpoints3/grpo_run"
    student_adapter_name: str = "student"
    student_adapter_path: Optional[str] = None
    ref_adapter_name: Optional[str] = None
    ref_adapter_path: Optional[str] = None

    # --- GRPO algo ---
    kl_coeff: float = 0.0
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.28
    norm_advantages: bool = True

    # --- optimizer ---
    lr: float = 2e-5
    grad_clip: float = 1.0
    sft_lr: float = 5e-5  # FIX: restored from old pipeline (was 1e-5)

    # --- curriculum ---
    dataset_dir: str = "taskset/codeforces/data"
    train_steps: int = 10
    sample_size: int = 10
    curriculum_n_buckets: int = 9
    curriculum_initial_mean: float = 0.6
    curriculum_std: float = 1.0
    solve_threshold: float = 0.8
    consecutive_required: int = 2
    min_evaluated: int = 8
    shift_delta: float = 1.0
    shift_window_radius: int = 0
    rolling_window: int = 20
    require_full_bucket_coverage: bool = True

    # --- verifier ---
    verifier_timeout: float = 5.0
    verifier_workers: int = 16
    save_lora_every: int = 20
    gen_batch_size: int = 64

    # --- Embeddings for Teacher model usage
    docs_folder: str = "/home/seigyo/rl/sft_primer/input/moove"
    embedding_backend: str = os.environ.get("EMBEDDING_BACKEND", "huggingface")
    embedding_base_url: str = os.environ.get(
        "EMBEDDING_BASE_URL", "http://10.160.144.101:51028/v1"
    )
    embedding_api_key: str = os.environ.get("EMBEDDING_API_KEY", "EMPTY")
    embedding_model: str = os.environ.get("EMBEDDING_MODEL", "cl-nagoya/ruri-v3-30m")


# endregion Training config
