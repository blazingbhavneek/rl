from __future__ import annotations

from algo.base import AlgoConfig
from algo.sdpo import SDPOConfig
from backprop.base import BackpropConfig
from pipeline.base import RewardConfig

# Model path and output locations used by trainer startup/checkpointing.
MODEL_PATH = "/media/blazingbhavneek/Common/Code/sglangServer/Infer/Qwen/Qwen3.5-0.8B"
OUTPUT_DIR = "./outputs/qwen35_08b_sdpo"
# Precision used when loading the training model.
DTYPE = "bfloat16"

# PEFT LoRA adapter setup for the trainable model.
LORA_RANK = 128
LORA_ALPHA = LORA_RANK * 2
LORA_DROPOUT = 0.0
LORA_TARGET = ["q_proj", "k_proj", "v_proj", "o_proj"]
# Top fraction of transformer layers included in streaming backprop suffix.
LORA_LAYERS_FRAC = 0.25
# HF attention implementation used at model load time.
ATTN_IMPLEMENTATION = "sdpa"

# Inference engine wiring and default generation budgets.
USE_SERVER_ENGINE = False
SERVER_URL = "http://127.0.0.1:30000"
SERVER_MODEL = "default"
SGLANG_MEM_FRAC = 0.3
MAX_SEQ_LEN = 8192
MAX_COMPLETION_TOKENS = 4096
TEMPERATURE = 0.7
TOP_P = 0.95
ENGINE_POLL_INTERVAL_S = 10.0
# SGLang Qwen3.5 LoRA loader currently fails on some builds
# ('Qwen3_5Config' has no 'num_hidden_layers'). Keep False to train safely.
SYNC_ENGINE_LORA = False
# Safety margin to avoid boundary context-length rejections from the engine.
CONTEXT_SAFETY_MARGIN = 64

# Dataset location used by CurriculumLoader.
TASKSET_DIR = "tasksets/codeforces/data"

# Curriculum controls: problems sampled per step and bucket-shift policy.
CURRICULUM_X = 2
CURRICULUM_INITIAL_MEAN = 0.0
CURRICULUM_STD = 1.2
SOLVE_THRESHOLD = 0.8
CONSECUTIVE_REQUIRED = 2
MAX_STEPS = 2000
MIN_EVALUATED = 8
SHIFT_DELTA = 1.0
SHIFT_WINDOW_RADIUS = 0
ROLLING_WINDOW = 20
REQUIRE_FULL_BUCKET_COVERAGE = True

# Generation controls for RL rollout.
NUM_GENERATIONS = 8
STUDENT_SYSTEM_PROMPT = "Solve the programming problem and return only the final solution."
TEACHER_SYSTEM_PROMPT = (
    "You are a programming tutor. Diagnose the student's reasoning error and provide a targeted hint. "
    "Do not reveal the final answer."
)

# Optimizer/scheduler and gradient clipping.
LR = 5e-6
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
LR_SCHEDULER = "cosine"
OPTIMIZER = "adamw_8bit"
MAX_GRAD_NORM = 1.0

# SDPO-specific controls.
SDPO_CREDIT_ASSIGNMENT = "logit"
SDPO_TOP_K = 20
SDPO_TEACHER_REG = "ema"
SDPO_TEACHER_ALPHA = 0.01
SDPO_DIVERGENCE = "reverse_kl"
SDPO_LAMBDA_GRPO = 0.0

# Generic clipping/KL terms consumed by build_algo_config().
KL_COEFF = 0.0
CLIP_RATIO_LOW = 0.2
CLIP_RATIO_HIGH = 0.28

# Reward shaping consumed by build_reward_config().
REWARD_COMPILE = 1.0
REWARD_PER_TEST = 1.0
REWARD_LENGTH_PENALTY = 0.01
MIN_COMPLETION_TOKENS = 64
REWARD_ERROR_ENGAGE = 0.1

# Teacher phase controls (only active when USE_TEACHER_PIPELINE=True and SFT output enabled).
USE_TEACHER_PIPELINE = False
MAX_HINT_ATTEMPTS = 3
SFT_OUTPUT_DIR = None

# Optional teacher RAG config.
TEACHER_DOCS_FOLDER = None
TEACHER_EMBEDDING_URL = None

# Verifier process pool/timeout settings.
VERIFY_TIMEOUT_S = 5
VERIFY_WORKERS = 32

# Trainer logging/checkpoint cadence.
SAVE_STEPS = 50
LOG_STEPS = 1

# Chunk profiler cache path and force-refresh switch.
CHUNK_PROFILE_DIR = f"{OUTPUT_DIR}/chunk_profiles"
FORCE_REPROFILE = False


def build_backprop_config() -> BackpropConfig:
    return BackpropConfig(
        top_frac=LORA_LAYERS_FRAC,
        use_grad_checkpoint=True,
        offload_prefix_cpu=True,
    )


def build_sdpo_config() -> SDPOConfig:
    return SDPOConfig(
        credit_assignment=SDPO_CREDIT_ASSIGNMENT,
        top_k=SDPO_TOP_K,
        teacher_reg=SDPO_TEACHER_REG,
        teacher_alpha=SDPO_TEACHER_ALPHA,
        divergence=SDPO_DIVERGENCE,
        lambda_grpo=SDPO_LAMBDA_GRPO,
    )


def build_algo_config() -> AlgoConfig:
    return AlgoConfig(
        kl_coeff=KL_COEFF,
        clip_ratio_low=CLIP_RATIO_LOW,
        clip_ratio_high=CLIP_RATIO_HIGH,
        norm_advantages=True,
    )


def build_reward_config() -> RewardConfig:
    return RewardConfig(
        reward_compile=REWARD_COMPILE,
        reward_per_test=REWARD_PER_TEST,
        reward_length_penalty=REWARD_LENGTH_PENALTY,
        min_completion_tokens=MIN_COMPLETION_TOKENS,
        reward_error_engage=REWARD_ERROR_ENGAGE,
    )
