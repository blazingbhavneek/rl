"""
main.py — GRPO training entry point.

Set the ALL_CAPS constants below to configure a run, then execute:
    python main.py
"""

import asyncio
import random

import torch

from pipeline import GRPOPipeline, TrainConfig

# ── reproducibility ────────────────────────────────────────────────────────────
SEED = 42

# ── model ─────────────────────────────────────────────────────────────────────
# Supported model types: "qwen3" | "qwen3_5" | "gptoss"
#
# Each model type auto-selects:
#   - training model class (Qwen3Model / Qwen3_5Model / GptOssModel)
#   - vLLM reasoning_parser + tool_call_parser
#   - thinking/reasoning toggle for rollouts and teacher
#
# ┌──────────┬─────────────────────────────────────────────────────────────┐
# │ Type     │ Recommended LORA_TARGETS                                   │
# ├──────────┼─────────────────────────────────────────────────────────────┤
# │ qwen3    │ ["gate_proj", "up_proj", "down_proj"]                      │
# │ qwen3_5  │ ["gate_proj", "up_proj", "down_proj"]                      │
# │ gptoss   │ ["q_proj", "k_proj", "v_proj", "o_proj"]                   │
# └──────────┴─────────────────────────────────────────────────────────────┘

MODEL_TYPE = "qwen3"
MODEL_PATH = "Qwen/Qwen3-8B"

# which linear projections to attach LoRA to (suffix layers only, per lora_fraction)
LORA_TARGETS = ["gate_proj", "up_proj", "down_proj"]
LORA_FRACTION = 0.5  # fraction of layers from the top that are trainable
LORA_RANK = 64
LORA_ALPHA = 128  # typically 2× rank
CHUNK_SIZE = 10  # token chunk size for chunked logprob computation (memory vs speed)
CUDA_DEVICE = 0
GRAD_CHECKPOINT = True  # activation checkpointing in suffix layers

# ── inference engine (vLLM) ───────────────────────────────────────────────────
ENGINE_BASE_URL = "http://127.0.0.1:8000/v1"
ENGINE_API_KEY = "EMPTY"
# fraction of GPU VRAM reserved for vLLM; remainder goes to train_model
ENGINE_GPU_MEM = 0.50
ENGINE_SEMAPHORE = 32  # max concurrent generation requests

# ── rollouts ──────────────────────────────────────────────────────────────────
N_ROLLOUTS = 8  # completions sampled per problem per step
TEMPERATURE = 0.7

# system prompt shown to the student model during generation
SYSTEM_PROMPT = "Solve the problem in C. Return only one ```c``` code block."

# ── teacher correction ────────────────────────────────────────────────────────
TEACHER_TEMPERATURE = 0.2  # lower = more deterministic hints
MAX_HINT_ATTEMPTS = 2  # retries per failed rollout
HINT_REWARD_DISCOUNT = 0.7  # scale reward of teacher-assisted solutions (< 1 to distinguish from clean wins)
TEACHER_MAX_TURNS = 6  # max tool-call rounds; with tools=[] exits on 1st turn

# ── teacher: RAG agent (optional — uncomment to enable) ──────────────────────
# TEACHER_DOCS_FOLDER     = "/path/to/your/docs.md"
# TEACHER_RAG_PERSIST_DIR = "logs/chroma_teacher_rag"
# TEACHER_EMBEDDING_BACKEND  = "huggingface"
# TEACHER_EMBEDDING_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
# TEACHER_EMBEDDING_BASE_URL = ENGINE_BASE_URL
# TEACHER_EMBEDDING_API_KEY  = "EMPTY"

# ── LoRA adapters (set paths to enable; None = no adapter) ───────────────────
STUDENT_ADAPTER_NAME = "student"
STUDENT_ADAPTER_PATH = (
    "checkpoints/student_lora"  # path to saved student adapter, or None for base model
)
REF_ADAPTER_NAME = None  # reference adapter for KL penalty; None = disable KL
REF_ADAPTER_PATH = None

# ── GRPO algorithm ────────────────────────────────────────────────────────────
KL_COEFF = 0.0  # KL penalty coefficient (0 = off; set > 0 if using ref adapter)
CLIP_RATIO_LOW = 0.2  # PPO clip lower bound (1 - clip_low)
CLIP_RATIO_HIGH = 0.28  # PPO clip upper bound (1 + clip_high); asymmetric = tighter on positive advantage
NORM_ADVANTAGES = True  # group-normalize advantages before policy update

# ── optimizer ─────────────────────────────────────────────────────────────────
LR = 2e-5
GRAD_CLIP = 1.0  # max gradient norm before clipping

# ── curriculum ────────────────────────────────────────────────────────────────
DATASET_DIR = "taskset/codeforces/data"
TRAIN_STEPS = 10  # total optimizer steps
SAMPLE_SIZE = 2  # problems sampled per step
N_BUCKETS = 9  # difficulty buckets (e.g. Codeforces A–G mapped to 1–9)
INITIAL_MEAN = 0.6  # starting bucket mean (0=easiest, 1=hardest)
CURRICULUM_STD = 1.0  # spread of the Gaussian sampling distribution over buckets

# a problem "graduates" when its solve rate exceeds this for CONSECUTIVE_REQUIRED steps
SOLVE_THRESHOLD = 0.8
CONSECUTIVE_REQUIRED = 2

MIN_EVALUATED = 8  # minimum rollouts before a curriculum shift is considered
SHIFT_DELTA = 1.0  # how much the mean shifts right when graduation triggers
SHIFT_WINDOW_RADIUS = 0  # smooth shift over ± this many buckets (0 = hard shift)
ROLLING_WINDOW = 20  # steps of history used to compute rolling solve rate
REQUIRE_FULL_BUCKET_COVERAGE = True  # don't shift until every bucket has been sampled

# ── verifier ──────────────────────────────────────────────────────────────────
VERIFIER_TIMEOUT = 5.0  # seconds per test case execution
VERIFIER_WORKERS = 16  # parallel worker processes


async def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)

    config = TrainConfig(
        model_path=MODEL_PATH,
        model_type=MODEL_TYPE,
        lora_targets=LORA_TARGETS,
        lora_fraction=LORA_FRACTION,
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        chunk_size=CHUNK_SIZE,
        cuda_device_index=CUDA_DEVICE,
        use_grad_checkpoint=GRAD_CHECKPOINT,
        engine_base_url=ENGINE_BASE_URL,
        engine_api_key=ENGINE_API_KEY,
        engine_gpu_memory_utilization=ENGINE_GPU_MEM,
        engine_semaphore_limit=ENGINE_SEMAPHORE,
        n_rollouts=N_ROLLOUTS,
        temperature=TEMPERATURE,
        system_prompt=SYSTEM_PROMPT,
        teacher_temperature=TEACHER_TEMPERATURE,
        max_hint_attempts=MAX_HINT_ATTEMPTS,
        hint_reward_discount=HINT_REWARD_DISCOUNT,
        teacher_max_turns=TEACHER_MAX_TURNS,
        student_adapter_name=STUDENT_ADAPTER_NAME,
        student_adapter_path=STUDENT_ADAPTER_PATH,
        ref_adapter_name=REF_ADAPTER_NAME,
        ref_adapter_path=REF_ADAPTER_PATH,
        kl_coeff=KL_COEFF,
        clip_ratio_low=CLIP_RATIO_LOW,
        clip_ratio_high=CLIP_RATIO_HIGH,
        norm_advantages=NORM_ADVANTAGES,
        lr=LR,
        grad_clip=GRAD_CLIP,
        dataset_dir=DATASET_DIR,
        train_steps=TRAIN_STEPS,
        sample_size=SAMPLE_SIZE,
        curriculum_n_buckets=N_BUCKETS,
        curriculum_initial_mean=INITIAL_MEAN,
        curriculum_std=CURRICULUM_STD,
        solve_threshold=SOLVE_THRESHOLD,
        consecutive_required=CONSECUTIVE_REQUIRED,
        min_evaluated=MIN_EVALUATED,
        shift_delta=SHIFT_DELTA,
        shift_window_radius=SHIFT_WINDOW_RADIUS,
        rolling_window=ROLLING_WINDOW,
        require_full_bucket_coverage=REQUIRE_FULL_BUCKET_COVERAGE,
        verifier_timeout=VERIFIER_TIMEOUT,
        verifier_workers=VERIFIER_WORKERS,
    )

    # ── optional: custom teacher with tools ───────────────────────────────────
    # Leave teacher_client = None to use the default AgentClient(tools=[])
    # built inside pipeline.train(). Uncomment below to enable RAG instead.
    #
    teacher_client = None
    #
    # from client.agent import AgentClient
    # from client.tools import build_markdown_rag_tool
    # from pipeline import _get_profile
    # rag_tool = build_markdown_rag_tool(
    #     docs_folder=TEACHER_DOCS_FOLDER,
    #     persist_directory=TEACHER_RAG_PERSIST_DIR,
    #     embedding_backend=TEACHER_EMBEDDING_BACKEND,
    #     embedding_base_url=TEACHER_EMBEDDING_BASE_URL,
    #     embedding_api_key=TEACHER_EMBEDDING_API_KEY,
    #     embedding_model=TEACHER_EMBEDDING_MODEL,
    # )
    # teacher_client = AgentClient(
    #     base_url=ENGINE_BASE_URL,
    #     api_key=ENGINE_API_KEY,
    #     temperature=TEACHER_TEMPERATURE,
    #     max_output_tokens=512,
    #     system_prompt=(
    #         "You are a coding tutor with access to reference documentation. "
    #         "Always search the knowledge base first. "
    #         "Give a targeted hint only — do not provide the full solution."
    #     ),
    #     model=MODEL_PATH,
    #     tools=[rag_tool],
    #     max_turns=TEACHER_MAX_TURNS,
    #     extra_body=_get_profile(MODEL_TYPE).teacher_extra_body,
    # )
    # ─────────────────────────────────────────────────────────────────────────

    await GRPOPipeline(config).train(teacher_client=teacher_client)


if __name__ == "__main__":
    asyncio.run(main())
