from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

import config as cfg
from algo.sdpo import SDPOAlgo
from backprop import BackpropConfig, ChunkSizeProfiler, StreamingBackprop
from client import AgentClient, SimpleTurnClient
from inference import (
    BaseEngine,
    EngineManager,
    SGLangOfflineEngine,
    ServerEngine,
    WeightSwapMode,
)
from pipeline import SDPOTeacherPipeline
from tasksets import BucketDistribution, CurriculumLoader, Score
from tasksets.codeforces import CodeforcesVerifier
from tasksets.codeforces.tools import get_tools

log = logging.getLogger(__name__)


class ManagedEngineProxy(BaseEngine):
    def __init__(self, manager: EngineManager) -> None:
        self.manager = manager

    def generate(self, prompt, params):
        return self.manager.get_engine().generate(prompt, params)

    def generate_batch(self, prompts, params):
        return self.manager.get_engine().generate_batch(prompts, params)

    def swap_weights(self, checkpoint_path: str, mode: WeightSwapMode) -> None:
        self.manager.sync_weights(checkpoint_path, mode)

    def is_healthy(self) -> bool:
        return self.manager.get_engine().is_healthy()

    def shutdown(self) -> None:
        self.manager.shutdown()


class Trainer:
    def __init__(self, *, show_sglang_logs: bool = False) -> None:
        self.show_sglang_logs = bool(show_sglang_logs)
        self.output_dir = Path(cfg.OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stats_dir = self.output_dir / "stats"
        self.stats_dir.mkdir(parents=True, exist_ok=True)
        self.step_stats_path = self.stats_dir / "steps.jsonl"
        self.run_meta_path = self.stats_dir / "run_meta.json"
        self.progress_path = self.output_dir / "progress.json"
        self.latest_lora_path = self.output_dir / "latest_lora"
        self.trainer_state_path = self.output_dir / "trainer_state.pt"

        self.global_step = 0
        self.tokenizer = None
        self.model = None
        self.backprop = None
        self.profiler = None
        self.loader = None
        self.verifier = None
        self.algo = None
        self.optimizer = None
        self.scheduler = None
        self.engine_manager = None
        self.engine_proxy = None
        self.student_client = None
        self.teacher_client = None
        self.pipeline = None
        self._run_started_at = time.time()

    def setup(self) -> None:
        self._notice("setup:start")
        self._configure_sglang_logging()
        self._notice("setup:init_training_model")
        self._init_training_model()
        self._notice("setup:init_inference_manager")
        self._init_inference_manager()
        self._notice("setup:init_chunk_profiler")
        self._init_chunk_profiler()
        self._notice("setup:init_taskset")
        self._init_taskset()
        self._notice("setup:init_algo")
        self._init_algo()
        self._notice("setup:init_optimizer_scheduler")
        self._init_optimizer_scheduler()
        self._notice("setup:init_clients")
        self._init_clients()
        self._notice("setup:init_pipeline")
        self._init_pipeline()
        self._notice("setup:resume")
        self._resume_from_checkpoint()
        self._write_run_metadata()
        self._notice("setup:done")

    @staticmethod
    def _notice(message: str) -> None:
        print(f"[trainer] {message}", flush=True)

    def _configure_sglang_logging(self) -> None:
        if self.show_sglang_logs:
            log.info("sglang logs: enabled")
            return
        for name in [
            "sglang",
            "sglang.srt",
            "sglang.srt.entrypoints.engine",
            "sglang.srt.server_args",
            "sglang.srt.configs.model_config",
        ]:
            logging.getLogger(name).setLevel(logging.ERROR)
        log.info("sglang logs: hidden (use --show-sglang-logs to enable)")

    def _dtype(self) -> torch.dtype:
        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return mapping.get(cfg.DTYPE, torch.bfloat16)

    def _init_training_model(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except Exception as exc:
            raise RuntimeError("Trainer requires PEFT to attach LoRA adapters.") from exc

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_PATH)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.MODEL_PATH,
            dtype=self._dtype(),
            low_cpu_mem_usage=True,
            attn_implementation=cfg.ATTN_IMPLEMENTATION,
        )

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=cfg.LORA_RANK,
            lora_alpha=cfg.LORA_ALPHA,
            lora_dropout=cfg.LORA_DROPOUT,
            target_modules=list(cfg.LORA_TARGET),
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_cfg)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.train()

        bp_cfg: BackpropConfig = cfg.build_backprop_config()
        self.backprop = StreamingBackprop(self.model, config=bp_cfg)
        setattr(self.backprop, "enable_tqdm", True)
        log.info(
            "backprop=streaming top_frac=%.3f grad_ckpt=%s offload_prefix_cpu=%s",
            bp_cfg.top_frac,
            bp_cfg.use_grad_checkpoint,
            bp_cfg.offload_prefix_cpu,
        )

    def _make_engine(self) -> BaseEngine:
        if cfg.USE_SERVER_ENGINE or cfg.USE_TEACHER_PIPELINE:
            return ServerEngine(base_url=cfg.SERVER_URL, model=cfg.SERVER_MODEL)
        return SGLangOfflineEngine(
            cfg.MODEL_PATH,
            engine_kwargs={
                "mem_fraction_static": cfg.SGLANG_MEM_FRAC,
                "context_length": cfg.MAX_SEQ_LEN,
            },
        )

    def _init_inference_manager(self) -> None:
        self.engine_manager = EngineManager(
            engine=self._make_engine(),
            poll_interval=cfg.ENGINE_POLL_INTERVAL_S,
            engine_factory=self._make_engine,
        )
        self.engine_manager.start(checkpoint_path=None)
        self.engine_proxy = ManagedEngineProxy(self.engine_manager)
        log.info("inference engine initialized (server=%s)", cfg.USE_SERVER_ENGINE or cfg.USE_TEACHER_PIPELINE)

    def _init_chunk_profiler(self) -> None:
        log.info("running ChunkSizeProfiler before training loop")
        self._notice("ChunkSizeProfiler:start")
        base, _ = self.backprop.adapter.unwrap(self.model)
        lm_head = self.backprop.adapter.get_lm_head(base)
        model_cfg = getattr(base, "config", getattr(self.model, "config", None))
        hidden_size = int(getattr(model_cfg, "hidden_size"))
        vocab_size = int(getattr(model_cfg, "vocab_size"))
        device = next(self.model.parameters()).device

        self.profiler = ChunkSizeProfiler(
            lm_head=lm_head,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            device=device,
            model_path=cfg.MODEL_PATH,
            sglang_mem_frac=cfg.SGLANG_MEM_FRAC,
            top_frac=cfg.LORA_LAYERS_FRAC,
            cache_dir=cfg.CHUNK_PROFILE_DIR,
            dtype=self._dtype(),
        )
        if cfg.FORCE_REPROFILE:
            self.profiler.invalidate()
        table = self.profiler.load_or_profile()
        self._notice(
            f"ChunkSizeProfiler cache_hit={self.profiler.last_cache_hit} force_reprofile={bool(cfg.FORCE_REPROFILE)}"
        )
        self.backprop.chunk_profiler = self.profiler
        self.backprop.config.logit_chunk = self.profiler.get_chunk_size(cfg.MAX_COMPLETION_TOKENS)
        log.info(
            "ChunkSizeProfiler selected logit_chunk=%s for max_completion_tokens=%s",
            self.backprop.config.logit_chunk,
            cfg.MAX_COMPLETION_TOKENS,
        )
        pretty = ", ".join([f"{k}:{v}" for k, v in sorted(table.items())])
        log.info("ChunkSizeProfiler bucket->chunk: %s", pretty)
        self._notice(f"ChunkSizeProfiler bucket->chunk {pretty}")

    def _init_taskset(self) -> None:
        distribution = BucketDistribution(
            n_buckets=9,
            initial_mean=cfg.CURRICULUM_INITIAL_MEAN,
            std=cfg.CURRICULUM_STD,
        )
        self.loader = CurriculumLoader(
            dataset_dir=cfg.TASKSET_DIR,
            x=cfg.CURRICULUM_X,
            solve_threshold=cfg.SOLVE_THRESHOLD,
            consecutive_required=cfg.CONSECUTIVE_REQUIRED,
            max_steps=cfg.MAX_STEPS,
            distribution=distribution,
            min_evaluated=cfg.MIN_EVALUATED,
            shift_delta=cfg.SHIFT_DELTA,
            shift_window_radius=cfg.SHIFT_WINDOW_RADIUS,
            rolling_window=cfg.ROLLING_WINDOW,
            require_full_bucket_coverage=cfg.REQUIRE_FULL_BUCKET_COVERAGE,
        )
        self.verifier = CodeforcesVerifier(timeout=cfg.VERIFY_TIMEOUT_S, n_workers=cfg.VERIFY_WORKERS)
        self._notice("verifier:dependency_check:start")
        self.verifier.check_dependencies()
        self._notice("verifier:dependency_check:ok")

    def _init_algo(self) -> None:
        self.algo = SDPOAlgo(
            config=cfg.build_algo_config(),
            sdpo_config=cfg.build_sdpo_config(),
            backprop=self.backprop,
            model=self.model,
            tokenizer=self.tokenizer,
            current_lora_path_getter=lambda: getattr(self.backprop, "_current_lora_path", None),
        )

    def _init_optimizer_scheduler(self) -> None:
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = self._build_optimizer(params)
        self.scheduler = self._build_scheduler(self.optimizer)

    def _build_optimizer(self, params):
        if cfg.OPTIMIZER == "adamw_8bit":
            try:
                import bitsandbytes as bnb

                return bnb.optim.AdamW8bit(params, lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
            except Exception:
                log.warning("bitsandbytes unavailable; falling back to torch.optim.AdamW")
        return AdamW(params, lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

    def _build_scheduler(self, optimizer):
        if cfg.LR_SCHEDULER == "cosine":
            return CosineAnnealingLR(optimizer, T_max=max(1, cfg.MAX_STEPS))

        warmup_steps = max(1, int(cfg.MAX_STEPS * cfg.WARMUP_RATIO))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            return 1.0

        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    def _init_clients(self) -> None:
        self.student_client = SimpleTurnClient(
            engine=self.engine_proxy,
            tokenizer=self.tokenizer,
            system_prompt=cfg.STUDENT_SYSTEM_PROMPT,
            default_max_new_tokens=cfg.MAX_COMPLETION_TOKENS,
            default_temperature=cfg.TEMPERATURE,
            default_top_p=cfg.TOP_P,
            max_context_tokens=cfg.MAX_SEQ_LEN,
            context_safety_margin=cfg.CONTEXT_SAFETY_MARGIN,
        )

        if cfg.USE_TEACHER_PIPELINE:
            teacher_tools = get_tools(
                docs_folder=cfg.TEACHER_DOCS_FOLDER,
                embedding_model_url=cfg.TEACHER_EMBEDDING_URL,
            )
            self.teacher_client = AgentClient(
                engine=self.engine_proxy,
                tokenizer=self.tokenizer,
                system_prompt=cfg.TEACHER_SYSTEM_PROMPT,
                tools=teacher_tools,
                max_turns=10,
                server_url=cfg.SERVER_URL,
                model_name=cfg.SERVER_MODEL,
            )
        else:
            self.teacher_client = None

    def _init_pipeline(self) -> None:
        self.pipeline = SDPOTeacherPipeline(
            student_client=self.student_client,
            teacher_client=self.teacher_client,
            verifier=self.verifier,
            algo=self.algo,
            backprop=self.backprop,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            reward_config=cfg.build_reward_config(),
            sft_output_dir=cfg.SFT_OUTPUT_DIR,
            n_rollouts=cfg.NUM_GENERATIONS,
            max_hint_attempts=cfg.MAX_HINT_ATTEMPTS,
            extensions=[],
            show_tqdm=True,
            max_grad_norm=cfg.MAX_GRAD_NORM,
        )

    @staticmethod
    def _config_snapshot() -> Dict[str, object]:
        snap: Dict[str, object] = {}
        for k in dir(cfg):
            if not k.isupper():
                continue
            v = getattr(cfg, k)
            if isinstance(v, (str, int, float, bool)) or v is None:
                snap[k] = v
            elif isinstance(v, (list, tuple)):
                snap[k] = list(v)
        return snap

    def _write_run_metadata(self) -> None:
        payload = {
            "started_at_unix": self._run_started_at,
            "model_path": cfg.MODEL_PATH,
            "output_dir": str(self.output_dir),
            "config": self._config_snapshot(),
        }
        with self.run_meta_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        self._notice(f"stats:run_meta={self.run_meta_path}")

    def _append_step_stats(self, step_result, step: int, sampled_buckets: Dict[int, int]) -> None:
        row = {
            "ts": time.time(),
            "step": int(step),
            "sampled_buckets": {str(k): int(v) for k, v in sampled_buckets.items()},
            "n_problems": int(step_result.n_problems),
            "n_passed_rl": int(step_result.n_passed_rl),
            "n_passed_after_hint": int(step_result.n_passed_after_hint),
            "rl_stats": {k: float(v) for k, v in step_result.rl_stats.items()},
            "sft_stats": {k: float(v) for k, v in step_result.sft_stats.items()},
        }
        with self.step_stats_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _save_latest_lora(self) -> str:
        self.backprop.save_lora(str(self.latest_lora_path))
        return str(self.latest_lora_path)

    def _save_checkpoint(self, step: int, final: bool = False) -> None:
        step_path = self.output_dir / f"step_{step:05d}"
        self.backprop.save_lora(str(step_path))
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            },
            self.trainer_state_path,
        )
        progress = {
            "global_step": int(step),
            "latest_lora_path": str(step_path),
            "chunk_profile_hash": getattr(self.profiler, "cache_key", None),
            "loader_stats": self.loader.get_stats(),
            "final": bool(final),
        }
        with self.progress_path.open("w", encoding="utf-8") as f:
            json.dump(progress, f)

    def _resume_from_checkpoint(self) -> None:
        if not self.progress_path.exists():
            return

        with self.progress_path.open("r", encoding="utf-8") as f:
            progress = json.load(f)

        self.global_step = int(progress.get("global_step", 0))
        latest_lora_path = progress.get("latest_lora_path")
        if latest_lora_path:
            self.backprop.load_lora(latest_lora_path)
            if cfg.SYNC_ENGINE_LORA:
                try:
                    self.engine_manager.sync_weights(latest_lora_path, WeightSwapMode.LORA)
                except Exception:
                    log.exception(
                        "Engine LoRA sync failed during resume; continuing with current engine weights. "
                        "Training model LoRA is loaded correctly."
                    )
            else:
                log.warning("SYNC_ENGINE_LORA=False: skipping engine LoRA sync during resume")

        if self.trainer_state_path.exists():
            state = torch.load(self.trainer_state_path, map_location="cpu")
            if "optimizer" in state:
                self.optimizer.load_state_dict(state["optimizer"])
            if self.scheduler is not None and state.get("scheduler") is not None:
                self.scheduler.load_state_dict(state["scheduler"])

        current_hash = getattr(self.profiler, "cache_key", None)
        if progress.get("chunk_profile_hash") != current_hash:
            self.profiler.invalidate()
            self.profiler.load_or_profile()

    def _extract_scores(self, step_result) -> Tuple[List[str], List[Score]]:
        problem_ids: List[str] = []
        scores: List[Score] = []
        for pr in step_result.problem_rollouts:
            problem_ids.append(pr.problem.id)
            if pr.best_rollout is not None:
                scores.append(pr.best_rollout.score)
            elif pr.rollouts:
                scores.append(pr.rollouts[0].score)
            else:
                scores.append(Score(compiled=False, passed=0, total=0, error="no rollout"))
        return problem_ids, scores

    def _log(self, step_result, step: int, sampled_buckets: Dict[int, int]) -> None:
        if step % cfg.LOG_STEPS != 0:
            return
        rl_mean_reward = step_result.rl_stats.get("mean_reward", 0.0)
        sft_pairs = step_result.sft_stats.get("n_pairs_saved", 0.0)
        lr = 0.0
        if self.optimizer.param_groups:
            lr = float(self.optimizer.param_groups[0].get("lr", 0.0))
        grad_norm = float(step_result.rl_stats.get("grad_norm", 0.0))
        loss_val = float(step_result.rl_stats.get("bp_loss", step_result.rl_stats.get("loss", 0.0)))
        kl_val = float(
            step_result.rl_stats.get("bp_kl_loss", step_result.rl_stats.get("kl_loss", 0.0))
        )
        log.info(
            "step=%s sampled=%s n_problems=%s passed_rl=%s passed_after_hint=%s mean_reward=%.4f loss=%.4f kl=%.4f grad_norm=%.4f lr=%.2e sft_pairs=%s",
            step,
            sampled_buckets,
            step_result.n_problems,
            step_result.n_passed_rl,
            step_result.n_passed_after_hint,
            rl_mean_reward,
            loss_val,
            kl_val,
            grad_norm,
            lr,
            int(sft_pairs),
        )
        self._notice(
            "step="
            f"{step} sampled={sampled_buckets} mean_reward={rl_mean_reward:.4f} "
            f"loss={loss_val:.4f} kl={kl_val:.4f} "
            f"tlogp={float(step_result.rl_stats.get('mean_teacher_logprob', 0.0)):.4f} "
            f"succ={float(step_result.rl_stats.get('success_ratio', 0.0)):.2f} "
            f"grad_norm={grad_norm:.4f} lr={lr:.2e}"
        )

    def train(self) -> None:
        try:
            self.setup()
            while not self.loader.should_stop(self.global_step):
                problems = self.loader.sample(self.global_step)
                if not problems:
                    break
                sampled_buckets = dict(Counter([p.bucket for p in problems]))

                step_result = self.pipeline.run_step(problems, self.global_step)
                self._log(step_result, self.global_step, sampled_buckets)
                self._append_step_stats(step_result, self.global_step, sampled_buckets)

                problem_ids, scores = self._extract_scores(step_result)
                self.loader.update(problem_ids, scores, self.global_step)

                latest_lora_path = self._save_latest_lora()
                if cfg.SYNC_ENGINE_LORA:
                    try:
                        self.engine_manager.sync_weights(latest_lora_path, WeightSwapMode.LORA)
                    except Exception:
                        log.exception(
                            "Engine LoRA sync failed at step=%s; continuing without engine weight swap. "
                            "Rollouts will use engine's current weights until next successful sync.",
                            self.global_step,
                        )

                if self.global_step % cfg.SAVE_STEPS == 0:
                    self._save_checkpoint(self.global_step)

                self.global_step += 1

            log.info("Training complete")
            self._save_checkpoint(self.global_step, final=True)
            self.engine_manager.shutdown()
        except Exception:
            log.exception("Training failed with an exception")
            raise
        finally:
            pass


def train(*, show_sglang_logs: bool = False) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    Trainer(show_sglang_logs=show_sglang_logs).train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL pipeline")
    parser.add_argument(
        "--show-sglang-logs",
        action="store_true",
        help="Show verbose sglang logs (default: hidden).",
    )
    args = parser.parse_args()
    train(show_sglang_logs=bool(args.show_sglang_logs))
