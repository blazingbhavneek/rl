# RL Trainer Quickstart

## Run Training

Default (hides noisy sglang logs):

```bash
python trainer.py
```

Show sglang logs:

```bash
python trainer.py --show-sglang-logs
```

Switch offline backend in `config.py`:
- `OFFLINE_ENGINE_BACKEND = "sglang"` or `"vllm"`
- keep `USE_SERVER_ENGINE = False`

## Plot Saved Stats

Training writes:

- `OUTPUT_DIR/stats/run_meta.json`
- `OUTPUT_DIR/stats/steps.jsonl`

Generate PNG plots:

```bash
python scripts/plot_stats.py --run-dir outputs/qwen35_08b_sdpo --smooth 5
```

Or pass the file directly:

```bash
python scripts/plot_stats.py \
  --steps-file outputs/qwen35_08b_sdpo/stats/steps.jsonl \
  --out-dir outputs/qwen35_08b_sdpo/stats/plots \
  --smooth 5
```

Install dependency if needed:

```bash
pip install matplotlib
```

## Agent + Tools Demos (No Test Framework)

Tool-only demo (RAG + execute_code):

```bash
python client/tools/demo_tools.py
```

Agentic client demo (mocked multi-turn tool calling):

```bash
python client/demo_agent_client.py
```

## SFT Primer Quick Test (LoRA)

Copy a markdown doc, build a tiny dataset, run 1-epoch LoRA SFT, then do a generation smoke test:

```bash
/media/blazingbhavneek/Common/Code/miniconda3/envs/rl/bin/python sft_primer/test_md_lora.py \
  --model-path /media/blazingbhavneek/Common/Code/sglangServer/Infer/Qwen/Qwen3.5-0.8B \
  --source-md /media/blazingbhavneek/Common/Code/datagen/parser/tests/output/test.md \
  --library qwen35_testmd \
  --epochs 1 \
  --max-samples 12 \
  --max-seq 1024
```

Flags:
- `--max-samples`: maximum number of SFT examples built from the markdown for this primer run.
- `--max-seq`: token budget per example and sequence cap used by this primer config.

Outputs:
- dataset: `sft_primer/outputs/<library>/dataset/`
- checkpoints: `sft_primer/outputs/<library>/checkpoint/`
- final LoRA: `sft_primer/outputs/<library>/lora/final`

## Config (Short Explanations)

All config lives in [config.py](/media/blazingbhavneek/Common/Code/rl/config.py).

- `MODEL_PATH`, `OUTPUT_DIR`, `DTYPE`: model source, checkpoint/stats location, precision.
- `LORA_*`, `LORA_TARGET`, `LORA_LAYERS_FRAC`: LoRA adapter size/targets and streaming suffix fraction.
- `ATTN_IMPLEMENTATION`: attention backend used by HF model load.
- `USE_SERVER_ENGINE`, `SERVER_URL`, `SERVER_MODEL`: use remote server engine vs local offline engine.
- `TEACHER_SERVER_URL`, `TEACHER_SERVER_MODEL`: optional override for teacher AgentClient endpoint/model; if `None`, teacher uses `SERVER_URL`/`SERVER_MODEL`.
  If teacher endpoint is unreachable, trainer auto-falls back to `SimpleTurnClient` on the same rollout engine.
- `OFFLINE_ENGINE_BACKEND`: choose local backend (`sglang` or `vllm`).
- `SGLANG_MEM_FRAC`: SGLang GPU memory fraction.
- `VLLM_GPU_MEMORY_UTILIZATION`: vLLM GPU memory fraction.
- `INFERENCE_QUANTIZATION`: inference quantization mode (for example `awq`/`gptq`) if backend supports it.
- `MAX_SEQ_LEN`, `MAX_COMPLETION_TOKENS`, `CONTEXT_SAFETY_MARGIN`: context/completion limits and safety margin.
- `SYNC_ENGINE_LORA`: whether inference engine hot-swaps latest LoRA every step.
- `TASKSET_DIR`: Codeforces dataset path.
- `CURRICULUM_*`, `MIN_EVALUATED`, `SHIFT_*`, `ROLLING_WINDOW`, `REQUIRE_FULL_BUCKET_COVERAGE`: bucket sampling and curriculum shift behavior.
- `NUM_GENERATIONS`: rollouts per problem.
- `STUDENT_SYSTEM_PROMPT`, `TEACHER_SYSTEM_PROMPT`: prompts for student/teacher clients.
- `TEACHER_MAX_COMPLETION_TOKENS`, `TEACHER_REASONING_EFFORT`: teacher hint generation cap and OpenAI-compatible reasoning setting.
- `LR`, `WEIGHT_DECAY`, `WARMUP_RATIO`, `LR_SCHEDULER`, `OPTIMIZER`, `MAX_GRAD_NORM`: optimizer/scheduler/clipping.
- `SDPO_*`: SDPO loss and teacher regularization controls.
- `KL_COEFF`, `CLIP_RATIO_LOW`, `CLIP_RATIO_HIGH`: PPO-style KL/clipping controls used by algo config.
- `REWARD_*`, `MIN_COMPLETION_TOKENS`: reward shaping parameters in pipeline.
- `USE_TEACHER_PIPELINE`, `MAX_HINT_ATTEMPTS`, `SFT_OUTPUT_DIR`: enable/disable teacher-hint + SFT phase.
- `TEACHER_DOCS_FOLDER`, `TEACHER_EMBEDDING_URL`: optional teacher RAG.
- `VERIFY_TIMEOUT_S`, `VERIFY_WORKERS`: verifier timeout and worker pool.
- `SAVE_STEPS`, `LOG_STEPS`: checkpoint/log frequency.
- `CHUNK_PROFILE_DIR`, `FORCE_REPROFILE`: chunk profiler cache path and forced reprofiling toggle.
