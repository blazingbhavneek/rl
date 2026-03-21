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

## Config (Short Explanations)

All config lives in [config.py](/media/blazingbhavneek/Common/Code/rl/config.py).

- `MODEL_PATH`, `OUTPUT_DIR`, `DTYPE`: model source, checkpoint/stats location, precision.
- `LORA_*`, `LORA_TARGET`, `LORA_LAYERS_FRAC`: LoRA adapter size/targets and streaming suffix fraction.
- `ATTN_IMPLEMENTATION`: attention backend used by HF model load.
- `USE_SERVER_ENGINE`, `SERVER_URL`, `SERVER_MODEL`: use remote server engine vs local offline engine.
- `SGLANG_MEM_FRAC`, `MAX_SEQ_LEN`, `MAX_COMPLETION_TOKENS`, `CONTEXT_SAFETY_MARGIN`: inference memory/context/completion limits.
- `SYNC_ENGINE_LORA`: whether inference engine hot-swaps latest LoRA every step.
- `TASKSET_DIR`: Codeforces dataset path.
- `CURRICULUM_*`, `MIN_EVALUATED`, `SHIFT_*`, `ROLLING_WINDOW`, `REQUIRE_FULL_BUCKET_COVERAGE`: bucket sampling and curriculum shift behavior.
- `NUM_GENERATIONS`: rollouts per problem.
- `STUDENT_SYSTEM_PROMPT`, `TEACHER_SYSTEM_PROMPT`: prompts for student/teacher clients.
- `LR`, `WEIGHT_DECAY`, `WARMUP_RATIO`, `LR_SCHEDULER`, `OPTIMIZER`, `MAX_GRAD_NORM`: optimizer/scheduler/clipping.
- `SDPO_*`: SDPO loss and teacher regularization controls.
- `KL_COEFF`, `CLIP_RATIO_LOW`, `CLIP_RATIO_HIGH`: PPO-style KL/clipping controls used by algo config.
- `REWARD_*`, `MIN_COMPLETION_TOKENS`: reward shaping parameters in pipeline.
- `USE_TEACHER_PIPELINE`, `MAX_HINT_ATTEMPTS`, `SFT_OUTPUT_DIR`: enable/disable teacher-hint + SFT phase.
- `TEACHER_DOCS_FOLDER`, `TEACHER_EMBEDDING_URL`: optional teacher RAG.
- `VERIFY_TIMEOUT_S`, `VERIFY_WORKERS`: verifier timeout and worker pool.
- `SAVE_STEPS`, `LOG_STEPS`: checkpoint/log frequency.
- `CHUNK_PROFILE_DIR`, `FORCE_REPROFILE`: chunk profiler cache path and forced reprofiling toggle.
