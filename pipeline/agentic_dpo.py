from __future__ import annotations

import asyncio
import importlib
import json
import random
import re
import shutil
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch
from torch import Tensor
from tqdm.auto import tqdm

from inference import BaseEngine, VLLMEngine
from taskset import BucketDistribution, CurriculumLoader
from taskset.base import Problem, ProblemState, Score

from .grpo_teacher import _get_profile


@dataclass
class AgenticDPOConfig:
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

    # --- LoRA adapters ---
    run_dir: str = "checkpoints/agentic_dpo_run"
    student_adapter_name: str = "student"
    student_adapter_path: Optional[str] = None
    ref_adapter_name: Optional[str] = None
    ref_adapter_path: Optional[str] = None
    save_lora_every: int = 5

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

    # --- trace generation ---
    attempts_per_problem: int = 8
    temperature: float = 0.7
    system_prompt: str = "Solve the problem in C. Return only one ```c``` code block."
    max_tokens: int = 4000

    # --- dummy DPO ---
    dpo_lr: float = 5e-5
    dpo_beta: float = 0.1
    dummy_loss_mode: str = "best_of_n_sft"


class AgenticDPOPipeline:
    def __init__(self, config: AgenticDPOConfig) -> None:
        self.cfg = config
        self.profile = _get_profile(config.model_type)

        self._adapters_initialized = False
        self._lora_in_vllm = False
        self._active_engine_adapter = config.student_adapter_name
        self._task_stats: dict[str, dict[str, float]] = {}

        stats_path = Path(config.run_dir) / "task_stats.json"
        if stats_path.exists():
            try:
                with stats_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self._task_stats = data
                    tqdm.write(
                        f"[resume] loaded task_stats with {len(self._task_stats)} entries"
                    )
            except Exception:
                pass

    async def train(self) -> None:
        from model.config import ModelConfig

        cfg = self.cfg
        profile = self.profile

        model_cfg = ModelConfig(
            lora=cfg.lora_targets,
            lora_fraction=cfg.lora_fraction,
            lora_rank=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            chunk_size=cfg.chunk_size,
            cuda_device_index=cfg.cuda_device_index,
            use_grad_checkpoint=cfg.use_grad_checkpoint,
        )

        ModelCls = getattr(importlib.import_module(profile.module), profile.cls_name)
        train_model = self._build_train_model(ModelCls, model_cfg)

        trainable_params = [
            p for p in train_model.model.parameters() if p.requires_grad
        ]
        optimizer = torch.optim.AdamW(trainable_params, lr=cfg.dpo_lr)

        distribution = BucketDistribution(
            n_buckets=cfg.curriculum_n_buckets,
            initial_mean=cfg.curriculum_initial_mean,
            std=cfg.curriculum_std,
        )
        loader = CurriculumLoader(
            dataset_dir=str(Path(cfg.dataset_dir)),
            x=cfg.sample_size,
            solve_threshold=cfg.solve_threshold,
            consecutive_required=cfg.consecutive_required,
            max_steps=cfg.train_steps,
            distribution=distribution,
            min_evaluated=cfg.min_evaluated,
            shift_delta=cfg.shift_delta,
            shift_window_radius=cfg.shift_window_radius,
            rolling_window=cfg.rolling_window,
            require_full_bucket_coverage=cfg.require_full_bucket_coverage,
        )

        step = self._load_resume_state(optimizer=optimizer, loader=loader)
        optimizer.zero_grad(set_to_none=True)

        engine_kwargs: dict[str, object] = {
            "base_url": cfg.engine_base_url,
            "api_key": cfg.engine_api_key,
            "model_name": cfg.model_path,
            "save_vllm_logs": True,
            "enable_auto_tool_choice": True,
            "tool_call_parser": profile.tool_call_parser,
            "enable_lora": True,
            "enable_runtime_lora_updating": True,
            "max_loras": 1,
            "max_lora_rank": cfg.lora_rank,
            "gpu_memory_utilization": cfg.engine_gpu_memory_utilization,
            "initial_lora_name": cfg.student_adapter_name,
            "initial_lora_path": cfg.student_adapter_path,
        }
        if profile.reasoning_parser is not None:
            engine_kwargs["reasoning_parser"] = profile.reasoning_parser

        engine: BaseEngine = VLLMEngine(
            model_path=cfg.model_path, engine_kwargs=engine_kwargs
        )
        await self._engine_init(engine)

        verifier = self._build_verifier()
        verifier.check_dependencies()

        step_bar = tqdm(total=cfg.train_steps, initial=step, desc="agentic-dpo-steps")
        try:
            while not loader.should_stop(step):
                step_t0 = time.time()
                batch = loader.sample(step=step)
                if not batch:
                    break

                rows, best_scores, batch_stats = await self.run_batch(
                    batch=batch,
                    step=step,
                    engine=engine,
                    verifier=verifier.verify,
                    semaphore_limit=cfg.engine_semaphore_limit,
                )

                dummy_loss = self._dummy_dpo_step(
                    rows=rows,
                    train_model=train_model,
                    optimizer=optimizer,
                )

                problem_ids = [str(p.id) for p in batch]
                loader.update(problem_ids=problem_ids, scores=best_scores, step=step)

                self._write_step_artifacts(
                    step=step,
                    batch=batch,
                    rows=rows,
                    batch_stats=batch_stats,
                    dummy_loss=dummy_loss,
                )
                self._write_task_stats()

                if step > 0 and step % cfg.save_lora_every == 0:
                    save_dir = self._save_checkpoint(
                        step=step,
                        train_model=train_model,
                        optimizer=optimizer,
                        loader=loader,
                    )
                    await engine.swap_lora_adapter(
                        cfg.student_adapter_name,
                        str(save_dir),
                        load_inplace=False,
                    )
                    self._lora_in_vllm = True
                    self._active_engine_adapter = cfg.student_adapter_name

                combined_pass_rate = float(batch_stats.get("combined_pass_rate", 0.0))
                tqdm.write(
                    f"\nstep={step} "
                    f"mean_bucket={loader.distribution.mean:.2f} "
                    f"pass={combined_pass_rate:.3f} "
                    f"dummy_dpo_loss={dummy_loss:.6f} "
                    f"selected={int(batch_stats.get('n_selected_for_dummy_dpo', 0.0))} "
                    f"time={time.time() - step_t0:.1f}s"
                )
                step += 1
                step_bar.update(1)
        finally:
            step_bar.close()
            await self._engine_shutdown(engine)

    def _build_train_model(self, ModelCls, model_cfg):
        cfg = self.cfg
        adapter_path = None
        if cfg.student_adapter_path:
            candidate = Path(cfg.student_adapter_path).expanduser().resolve()
            if candidate.exists():
                adapter_path = str(candidate)
            else:
                tqdm.write(
                    f"[init] student adapter path not found, starting fresh: {candidate}"
                )

        if adapter_path and ModelCls.__name__ == "Gemma4Model":
            train_model = ModelCls(
                cfg.model_path,
                model_cfg,
                lora_path=adapter_path,
                lora_adapter_name=cfg.student_adapter_name,
                lora_is_trainable=True,
            )
            return train_model

        train_model = ModelCls(cfg.model_path, model_cfg)
        if adapter_path:
            train_model.load_lora_adapter(
                cfg.student_adapter_name,
                adapter_path,
                is_trainable=True,
            )
            try:
                train_model.set_active_lora_adapter(cfg.student_adapter_name)
            except Exception:
                pass
        return train_model

    def _build_verifier(self):
        try:
            from taskset.remote_docker import RemoteDockerVerifier  # type: ignore

            return RemoteDockerVerifier(timeout=self.cfg.verifier_timeout)
        except Exception:
            from taskset.codeforces import CodeforcesVerifier

            return CodeforcesVerifier(
                timeout=self.cfg.verifier_timeout,
                n_workers=self.cfg.verifier_workers,
            )

    def _load_resume_state(self, *, optimizer, loader: CurriculumLoader) -> int:
        cfg = self.cfg
        if not cfg.student_adapter_path:
            return 0

        adapter_path = Path(cfg.student_adapter_path).expanduser().resolve()
        meta_path = adapter_path / "meta.json"
        if not meta_path.exists():
            tqdm.write(
                f"[resume] adapter found without meta.json, starting fresh from step 0: {adapter_path.name}"
            )
            return 0

        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        step = int(meta["step"]) + 1

        optimizer_path = adapter_path / "optimizer.pt"
        if optimizer_path.exists():
            optimizer.load_state_dict(
                torch.load(
                    optimizer_path,
                    map_location="cpu",
                    weights_only=True,
                )
            )

        curriculum_path = adapter_path / "curriculum.pt"
        if curriculum_path.exists():
            self._loader_load_state_dict(
                loader,
                torch.load(curriculum_path, map_location="cpu"),
            )

        rng_path = adapter_path / "rng.pt"
        if rng_path.exists():
            rng = torch.load(rng_path, map_location="cpu")
            random.setstate(rng["python"])
            torch.random.set_rng_state(rng["torch"])
            if torch.cuda.is_available() and rng.get("cuda") is not None:
                torch.cuda.set_rng_state_all(rng["cuda"])

        tqdm.write(
            f"[resume] training checkpoint - resuming from step {step} (last completed: {step - 1})"
        )
        return step

    async def run_batch(
        self,
        batch: list[Problem],
        *,
        step: int,
        engine: BaseEngine,
        verifier: Callable[[Problem, str], Score],
        semaphore_limit: int = 32,
    ) -> tuple[list[dict], list[Score], dict]:
        cfg = self.cfg
        profile = self.profile

        await self._ensure_adapters(engine)

        sem = asyncio.Semaphore(max(1, semaphore_limit))
        use_lora = self._lora_in_vllm

        async def _generate(messages: list[dict]) -> Optional[str]:
            model_name = self._active_engine_adapter if use_lora else cfg.model_path
            payload = {
                "model": model_name,
                "messages": messages,
                "temperature": cfg.temperature,
                "max_tokens": cfg.max_tokens,
                "stream": False,
            }
            payload.update(profile.gen_extra_payload)

            async with sem:
                try:
                    resp = await engine._request_json("POST", "/chat/completions", payload)  # type: ignore[attr-defined]
                except Exception:
                    if not use_lora:
                        raise
                    fallback_payload = dict(payload)
                    fallback_payload["model"] = cfg.model_path
                    fallback_payload["lora_name"] = cfg.student_adapter_name
                    resp = await engine._request_json("POST", "/chat/completions", fallback_payload)  # type: ignore[attr-defined]

            choices = resp.get("choices") or []
            if not choices:
                return None
            raw_msg = choices[0].get("message") or {}
            reasoning = str(raw_msg.get("reasoning") or "").strip()
            content = str(raw_msg.get("content") or "").strip()
            if reasoning:
                return f"<think>\n{reasoning}\n</think>\n\n{content}".strip()
            return content or None

        async def _score(problem: Problem, text: str) -> tuple[Score, float, bool]:
            extracted = self._extract_c_code(text)
            sc = verifier(problem, extracted)
            if sc.passed == 0 and sc.total > 0:
                return sc, 0.0, False

            reward = (float(sc.passed) / float(sc.total)) if sc.total > 0 else 0.0
            details = sc.details or {}
            compile_logs = str(details.get("compile_logs", "") or "")
            stdout = str(details.get("stdout", "") or "")
            clean = self._is_clean_logs(compile_logs, stdout)
            reward += 0.1 if clean else -0.1
            passed = bool(sc.total > 0 and sc.passed == sc.total and clean)
            return sc, reward, passed

        gen_tasks: list[
            tuple[Problem, list[dict], asyncio.Future | asyncio.Task | object]
        ] = []
        for problem in batch:
            messages = []
            if cfg.system_prompt:
                messages.append({"role": "system", "content": cfg.system_prompt})
            messages.append({"role": "user", "content": str(problem.statement)})
            for _ in range(cfg.attempts_per_problem):
                gen_tasks.append((problem, messages, _generate(messages)))

        rows: list[dict] = []
        results = await asyncio.gather(*[t[2] for t in gen_tasks])
        for (problem, messages, _), txt in zip(gen_tasks, results):
            if txt is None:
                continue
            sc, reward, passed = await _score(problem, txt)
            rows.append(
                dict(
                    problem=problem,
                    messages=messages,
                    text=txt,
                    extracted_code=self._extract_c_code(txt),
                    score=sc,
                    reward=reward,
                    passed=passed,
                )
            )

        best_scores: list[Score] = []
        for problem in batch:
            problem_rows = [r for r in rows if r["problem"] is problem]
            if problem_rows:
                best_row = max(problem_rows, key=lambda r: float(r["reward"]))
                best_scores.append(best_row["score"])
                self._update_task_stats(problem, problem_rows)
            else:
                best_scores.append(
                    Score(compiled=False, passed=0, total=0, error="no generation")
                )

        batch_stats = self._build_batch_stats(batch=batch, rows=rows)
        return rows, best_scores, batch_stats

    def _dummy_dpo_step(
        self,
        *,
        rows: list[dict],
        train_model,
        optimizer,
    ) -> float:
        selected = self._select_best_rows(rows)
        if not selected:
            optimizer.zero_grad(set_to_none=True)
            return 0.0

        actual_adapters = list(getattr(train_model.model, "peft_config", {}).keys())
        active_name = (
            self.cfg.student_adapter_name
            if self.cfg.student_adapter_name in actual_adapters
            else (actual_adapters[0] if actual_adapters else "default")
        )
        train_model.set_active_lora_adapter(active_name)

        def loss_fn_batch(
            batch_log_probs: Tensor,
            batch_mask: Tensor,
            hidden_batch=None,
        ) -> Tensor:
            mask = batch_mask.to(batch_log_probs.device).float()
            lengths = mask.sum(dim=1).clamp(min=1.0)
            return (-((batch_log_probs * mask).sum(dim=1) / lengths)).mean()

        optimizer.zero_grad(set_to_none=True)
        bp = train_model.backward(
            messages=[row["messages"] for row in selected],
            completion_texts=[row["text"] for row in selected],
            loss_fn=loss_fn_batch,
            loss_scale=1.0 / max(1, len(selected)),
        )
        torch.nn.utils.clip_grad_norm_(
            [p for p in train_model.model.parameters() if p.requires_grad],
            max_norm=1.0,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return float(bp.get("loss", 0.0))

    def _save_checkpoint(
        self,
        *,
        step: int,
        train_model,
        optimizer,
        loader: CurriculumLoader,
    ) -> Path:
        cfg = self.cfg
        save_dir = Path(cfg.run_dir) / "student_lora" / f"step_{step}"
        tmp_dir = save_dir.parent / f".tmp_step_{step}"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        try:
            train_model.model.eval()
            train_model.save_lora_adapter("default", str(tmp_dir))
            train_model.model.train()

            with (tmp_dir / "meta.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "step": step,
                        "adapter_name": cfg.student_adapter_name,
                    },
                    f,
                )

            torch.save(optimizer.state_dict(), tmp_dir / "optimizer.pt")
            torch.save(self._loader_state_dict(loader), tmp_dir / "curriculum.pt")
            torch.save(
                {
                    "python": random.getstate(),
                    "torch": torch.random.get_rng_state(),
                    "cuda": (
                        torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available()
                        else None
                    ),
                },
                tmp_dir / "rng.pt",
            )

            if save_dir.exists():
                shutil.rmtree(save_dir)
            tmp_dir.rename(save_dir)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

        return save_dir

    async def _ensure_adapters(self, engine: BaseEngine) -> None:
        if self._adapters_initialized:
            return

        self._adapters_initialized = True
        cfg = self.cfg

        if cfg.student_adapter_path:
            student_path = Path(cfg.student_adapter_path).expanduser().resolve()
            if student_path.exists():
                await engine.swap_lora_adapter(  # type: ignore[attr-defined]
                    cfg.student_adapter_name,
                    str(student_path),
                    load_inplace=False,
                )
                self._lora_in_vllm = True
                self._active_engine_adapter = cfg.student_adapter_name
                tqdm.write(
                    f"[init] vLLM loaded {cfg.student_adapter_name} from {student_path.name}"
                )
            else:
                tqdm.write(f"[init] student adapter path not found: {student_path}")
        else:
            tqdm.write("[init] no student_adapter_path, using base model")

    async def _engine_init(self, engine: BaseEngine) -> None:
        try:
            await engine.init()
            tqdm.write("engine: connected to existing server")
        except Exception:
            try:
                await engine.start()
                tqdm.write("engine: started local server")
            except Exception as exc:
                raise RuntimeError(
                    f"Engine startup failed - try lowering engine_gpu_memory_utilization "
                    f"(currently {self.cfg.engine_gpu_memory_utilization})"
                ) from exc

        try:
            if await engine.is_sleeping():
                await engine.wake()
                tqdm.write("engine: woken from sleep")
        except Exception as exc:
            tqdm.write(f"engine: sleep/wake status unavailable: {exc}")

    async def _engine_shutdown(self, engine: BaseEngine) -> None:
        try:
            await engine.sleep(level=1)
            tqdm.write("engine: sleeping")
        except Exception as exc:
            tqdm.write(f"engine: sleep unavailable: {exc}")
        await engine.shutdown()
        tqdm.write("engine: shutdown complete")

    def _loader_state_dict(self, loader: CurriculumLoader) -> dict:
        state_dict = getattr(loader, "state_dict", None)
        if callable(state_dict):
            return state_dict()

        return {
            "distribution": loader.distribution.export(),
            "problem_states": [asdict(s) for s in loader.problem_states.values()],
            "history": {k: list(v) for k, v in loader._history.items()},
            "stop_exhausted": bool(loader._stop_exhausted),
            "rolling_window": int(loader.rolling_window),
        }

    def _loader_load_state_dict(self, loader: CurriculumLoader, state: dict) -> None:
        load_state_dict = getattr(loader, "load_state_dict", None)
        if callable(load_state_dict):
            load_state_dict(state)
            return

        if "distribution" in state:
            loader.distribution.load(state["distribution"])

        loader.problem_states = {}
        for row in state.get("problem_states", []):
            ps = ProblemState(**row)
            loader.problem_states[ps.id] = ps

        loader._history = {}
        for pid, arr in state.get("history", {}).items():
            dq = deque(maxlen=int(state.get("rolling_window", loader.rolling_window)))
            for value in arr:
                dq.append(int(value))
            loader._history[pid] = dq

        loader._stop_exhausted = bool(state.get("stop_exhausted", False))

    def _select_best_rows(self, rows: list[dict]) -> list[dict]:
        by_problem: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_problem[str(row["problem"].id)].append(row)

        selected: list[dict] = []
        for problem_rows in by_problem.values():
            selected.append(max(problem_rows, key=lambda r: float(r["reward"])))
        return selected

    def _build_batch_stats(
        self, *, batch: list[Problem], rows: list[dict]
    ) -> dict[str, float]:
        selected = self._select_best_rows(rows)
        out: dict[str, float] = {
            "n_problems": float(len(batch)),
            "n_attempts": float(len(rows)),
            "n_selected_for_dummy_dpo": float(len(selected)),
        }
        if rows:
            out["combined_pass_rate"] = sum(1 for r in rows if r["passed"]) / len(rows)
            out["mean_reward"] = sum(float(r["reward"]) for r in rows) / len(rows)
        else:
            out["combined_pass_rate"] = 0.0
            out["mean_reward"] = 0.0
        return out

    def _update_task_stats(self, problem: Problem, rows: list[dict]) -> None:
        pid = str(problem.id)
        last_passed = sum(1 for r in rows if r["passed"])
        last_total = len(rows)
        stats = self._task_stats.setdefault(
            pid, {"total_attempts": 0, "total_passed": 0}
        )
        stats["total_attempts"] += last_total
        stats["total_passed"] += last_passed
        stats["overall_pass_rate"] = round(
            stats["total_passed"] / max(1, stats["total_attempts"]), 3
        )
        stats["last_pass_rate"] = round(last_passed / max(1, last_total), 3)

    def _write_task_stats(self) -> None:
        stats_path = Path(self.cfg.run_dir) / "task_stats.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(self._task_stats, f, indent=2)

    def _write_step_artifacts(
        self,
        *,
        step: int,
        batch: list[Problem],
        rows: list[dict],
        batch_stats: dict[str, float],
        dummy_loss: float,
    ) -> None:
        out_dir = Path(self.cfg.run_dir) / "trace_logs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"step_{step}.json"

        data = {
            "step": step,
            "batch_problem_ids": [str(p.id) for p in batch],
            "batch_stats": {**batch_stats, "dummy_dpo_loss": dummy_loss},
            "attempts": [
                {
                    "problem_id": str(row["problem"].id),
                    "messages": row["messages"],
                    "text": row["text"],
                    "extracted_code": row["extracted_code"],
                    "reward": row["reward"],
                    "passed": row["passed"],
                    "score": {
                        "compiled": row["score"].compiled,
                        "passed": row["score"].passed,
                        "total": row["score"].total,
                        "error": row["score"].error,
                        "details": row["score"].details,
                    },
                }
                for row in rows
            ],
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _extract_c_code(self, text: str) -> str:
        match = re.search(r"```c\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return text.strip()

    def _is_clean_logs(self, compile_logs: str, stdout: str = "") -> bool:
        combined = f"{compile_logs or ''} {stdout or ''}".lower()
        if not combined.strip():
            return True
        return not any(
            keyword in combined
            for keyword in [
                "warning:",
                "error:",
                "note:",
                "implicit",
                "addresssanitizer",
                "undefined behavior",
                "segmentation fault",
                "segfault",
                "abort",
            ]
        )
