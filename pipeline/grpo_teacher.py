from __future__ import annotations

import asyncio
import importlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch
from torch import Tensor
from tqdm.auto import tqdm

from algo.grpo import GRPOAlgo
from client.base import BaseClient
from inference.vllm_engine import VLLMEngine
from taskset.base import Problem, Score
from taskset import BucketDistribution, CurriculumLoader
from taskset.codeforces import CodeforcesVerifier


# ---------------------------------------------------------------------------
# Model profiles — per-model engine + inference settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _ModelProfile:
    module: str
    cls_name: str
    reasoning_parser: Optional[str]
    tool_call_parser: str
    teacher_extra_body: dict
    gen_extra_payload: dict


_MODEL_PROFILES: dict[str, _ModelProfile] = {
    "qwen3": _ModelProfile(
        module="model.qwen3",
        cls_name="Qwen3Model",
        reasoning_parser="qwen3",
        tool_call_parser="hermes",
        teacher_extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        gen_extra_payload={"chat_template_kwargs": {"enable_thinking": False}},
    ),
    "qwen3_5": _ModelProfile(
        module="model.qwen3_5",
        cls_name="Qwen3_5Model",
        reasoning_parser="qwen3",
        tool_call_parser="qwen3_coder",
        teacher_extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        gen_extra_payload={"chat_template_kwargs": {"enable_thinking": False}},
    ),
    "gptoss": _ModelProfile(
        module="model.gptoss",
        cls_name="GptOssModel",
        reasoning_parser=None,
        tool_call_parser="openai",
        teacher_extra_body={},
        gen_extra_payload={"include_reasoning": False},
    ),
}


def _get_profile(model_type: str) -> _ModelProfile:
    if model_type not in _MODEL_PROFILES:
        raise ValueError(
            f"Unknown model_type {model_type!r}. "
            f"Choose from: {list(_MODEL_PROFILES)}"
        )
    return _MODEL_PROFILES[model_type]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # --- model ---
    model_path: str
    model_type: str = "qwen3"  # "qwen3" | "qwen3_5" | "gptoss"
    lora_targets: list[str] = field(default_factory=lambda: ["gate_proj", "up_proj", "down_proj"])
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
    engine_semaphore_limit: int = 32

    # --- rollouts ---
    n_rollouts: int = 8
    temperature: float = 0.7
    system_prompt: str = "Solve the problem in C. Return only one ```c``` code block."
    max_tokens: int = 16000

    # --- teacher ---
    teacher_temperature: float = 0.2
    max_hint_attempts: int = 2
    hint_reward_discount: float = 0.7
    teacher_max_turns: int = 6

    # --- LoRA adapters (optional) ---
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


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class GRPOPipeline:
    """
    Owns the full training lifecycle.

    GPU schedule per step:
      generate (vLLM awake)
        → teacher refine (vLLM awake)
          → sleep vLLM
            → GRPO backprop (accumulate grads)
            → SFT backprop on solved teacher fixes (accumulate grads)
          → wake vLLM
        → optimizer.step()
        → LoRA swap to vLLM (if configured)
        → curriculum update
      → next step
    """

    def __init__(self, config: TrainConfig) -> None:
        self.cfg = config
        self.profile = _get_profile(config.model_type)
        self._prev_failed_rollouts: list[dict] = []

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def train(self, teacher_client: Optional[BaseClient] = None) -> None:
        cfg = self.cfg
        profile = self.profile

        # -- build training model --
        from model.config import ModelConfig
        from algo import AlgoConfig

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
        train_model = ModelCls(cfg.model_path, model_cfg)
        trainable_params = [p for p in train_model.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr)

        # -- build inference engine --
        has_runtime_lora = bool(cfg.student_adapter_path) or bool(cfg.ref_adapter_path)
        engine_kwargs: dict = {
            "base_url": cfg.engine_base_url,
            "api_key": cfg.engine_api_key,
            "model_name": cfg.model_path,
            "save_vllm_logs": True,
            "enable_auto_tool_choice": True,
            "tool_call_parser": profile.tool_call_parser,
            "enable_lora": has_runtime_lora,
            "enable_runtime_lora_updating": has_runtime_lora,
            "max_loras": 8,
            "max_lora_rank": 128,
            "gpu_memory_utilization": cfg.engine_gpu_memory_utilization,
        }
        if profile.reasoning_parser is not None:
            engine_kwargs["reasoning_parser"] = profile.reasoning_parser
        engine = VLLMEngine(model_path=cfg.model_path, engine_kwargs=engine_kwargs)
        await self._engine_init(engine)

        # -- teacher client --
        if teacher_client is None:
            from client.agent import AgentClient
            teacher_client = AgentClient(
                base_url=cfg.engine_base_url,
                api_key=cfg.engine_api_key,
                temperature=cfg.teacher_temperature,
                max_output_tokens=cfg.max_tokens,
                system_prompt=(
                    "You are a coding tutor. Provide concise guidance only. "
                    "Do not explicitly provide the final correct answer."
                ),
                model=cfg.model_path,
                tools=[],
                max_turns=cfg.teacher_max_turns,
                extra_body=profile.teacher_extra_body,
            )

        # -- verifier --
        verifier = CodeforcesVerifier(timeout=cfg.verifier_timeout, n_workers=cfg.verifier_workers)
        verifier.check_dependencies()

        # -- GRPO algo --
        algo = GRPOAlgo(AlgoConfig(
            kl_coeff=cfg.kl_coeff,
            clip_ratio_low=cfg.clip_ratio_low,
            clip_ratio_high=cfg.clip_ratio_high,
            norm_advantages=cfg.norm_advantages,
            loss_agg="token_mean",
        ))

        # -- curriculum --
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

        # -- training loop --
        optimizer.zero_grad(set_to_none=True)
        step = 0
        step_bar = tqdm(total=cfg.train_steps, desc="train-steps", leave=True)
        try:
            while not loader.should_stop(step):
                batch = loader.sample(step=step)
                if not batch:
                    break

                sampled_buckets = dict(Counter([p.bucket for p in batch]))
                stats = await self.run_batch(
                    batch=batch,
                    engine=engine,
                    train_model=train_model,
                    algo=algo,
                    tokenizer=train_model.tokenizer,
                    verifier=verifier.verify,
                    teacher_client=teacher_client,
                    semaphore_limit=cfg.engine_semaphore_limit,
                )

                # feed scores back to curriculum
                problem_ids = list(stats.get("problem_ids", []))
                problem_scores = list(stats.get("problem_scores", []))
                if len(problem_ids) == len(problem_scores) and problem_ids:
                    loader.update(problem_ids=problem_ids, scores=problem_scores, step=step)

                # optimizer step
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=cfg.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # push updated LoRA weights into vLLM
                if cfg.student_adapter_path:
                    adapter_save_path = str(
                        Path(cfg.student_adapter_path) / f"step_{step}"
                    )
                    
                    # Save from training model
                    train_model.save_lora_adapter(
                        cfg.student_adapter_name, adapter_save_path
                    )
                    print(f"[lora-save] ✓ saved {cfg.student_adapter_name} to {adapter_save_path}")
                    
                    # Hot-swap into vLLM
                    swap_result = await engine.swap_lora_adapter(
                        cfg.student_adapter_name, adapter_save_path
                    )
                    print(f"[lora-swap] ✓ hot-swapped into vLLM for step={step}")
                    
                    # Verify it's active
                    try:
                        models = await engine._request_json("GET", "/models")
                        active_adapters = [
                            m.get("id", "") for m in models.get("data", [])
                            if cfg.student_adapter_name in str(m.get("id", ""))
                        ]
                        if active_adapters:
                            print(f"[lora-verify] ✓ vLLM reports active: {active_adapters}")
                        else:
                            print(f"[lora-verify] ⚠ adapter not visible in /models endpoint")
                    except Exception as e:
                        print(f"[lora-verify] ! verification failed: {e}")

                print(
                    f"step={step} "
                    f"sampled={sampled_buckets} "
                    f"mean_bucket={loader.distribution.mean:.2f} "
                    f"pass={float(stats.get('combined_pass_rate', 0.0)):.3f} "
                    f"grpo_loss={float(stats.get('grpo_bp_loss', 0.0)):.4f} "
                    f"sft_loss={float(stats.get('sft_loss', 0.0)):.4f} "
                    f"hints={int(stats.get('n_hints_given', 0.0))}"
                )
                step += 1
                step_bar.update(1)
        finally:
            step_bar.close()
            await self._engine_shutdown(engine)

    # ------------------------------------------------------------------
    # Single batch
    # ------------------------------------------------------------------

    async def run_batch(
        self,
        batch: list[Problem],
        *,
        engine: VLLMEngine,
        train_model,
        algo: GRPOAlgo,
        tokenizer,
        verifier: Callable[[Problem, str], Score],
        teacher_client: Optional[BaseClient] = None,
        semaphore_limit: int = 32,
        refine_mode: str = "previous",
    ) -> dict:
        cfg = self.cfg
        profile = self.profile

        await self._ensure_adapters(engine, train_model, cfg)

        sem = asyncio.Semaphore(max(1, semaphore_limit))

        async def _generate(messages: list[dict]) -> str:
            payload: dict = {
                "model": cfg.model_path,
                "messages": messages,
                "temperature": cfg.temperature,
                "max_tokens": cfg.max_tokens,
                "stream": False,
            }
            payload.update(profile.gen_extra_payload)
            if cfg.student_adapter_path:
                payload["lora_name"] = cfg.student_adapter_name
            async with sem:
                resp = await engine._request_json("POST", "/chat/completions", payload)
            msg = ((resp.get("choices") or [{}])[0].get("message") or {})
            return str(msg.get("content") or "").strip()

        async def _score(problem: Problem, text: str) -> tuple[Score, float, bool]:
            sc = verifier(problem, text)
            reward = (float(sc.passed) / float(sc.total)) if sc.total > 0 else 0.0
            if sc.compiled:
                reward += 0.1
            return sc, reward, bool(sc.total > 0 and sc.passed == sc.total)

        # ── phase 1: rollouts (vLLM awake) ────────────────────────────
        gen_tasks = []
        for problem in batch:
            messages = [
                {"role": "system", "content": cfg.system_prompt},
                {"role": "user", "content": str(problem.statement)},
            ]
            for _ in range(cfg.n_rollouts):
                gen_tasks.append((problem, messages, _generate(messages)))

        current: list[dict] = []
        with tqdm(total=len(gen_tasks), desc="grpo-gen", leave=False) as pbar:
            results = await asyncio.gather(*[t[2] for t in gen_tasks])
            for (problem, messages, _), txt in zip(gen_tasks, results):
                pbar.update(1)
                sc, reward, passed = await _score(problem, txt)
                current.append(dict(problem=problem, messages=messages, text=txt,
                                    score=sc, reward=reward, passed=passed))

        _log_rollout_stats("grpo-rollouts", current)

        # ── phase 2: teacher correction (vLLM still awake) ────────────
        refine_candidates = {
            "current":  [r for r in current if not r["passed"]],
            "previous": self._prev_failed_rollouts,
            "none":     [],
        }.get(refine_mode, [])

        refined: list[dict] = []
        hints_given = 0
        passed_after_hint = 0

        if teacher_client is not None and cfg.max_hint_attempts > 0 and refine_candidates:
            refined, hints_given, passed_after_hint = await self._teacher_refine(
                refine_candidates, _generate, _score, teacher_client
            )
            _log_rollout_stats("teacher-fixes", refined, extra=f"hints={hints_given}")
            
            # teacher fix stats
            n_input_failures = len(refine_candidates)
            n_fixed = passed_after_hint
            fix_rate = 100 * n_fixed / n_input_failures if n_input_failures > 0 else 0
            print(f"[teacher] fixed {n_fixed}/{n_input_failures} ({fix_rate:.1f}%)")

        # ── sleep vLLM — all generation done ──────────────────────────
        try:
            await engine.sleep(level=1)
            print("[vllm] sleeping for backprop")
        except Exception as exc:
            print(f"[vllm] sleep unavailable: {exc}")

        # ── phase 3a: GRPO backprop ───────────────────────────────────
        device = next(train_model.model.parameters()).device
        if cfg.student_adapter_path:
            train_model.set_active_lora_adapter(cfg.student_adapter_name)

        grpo_stats: list[dict] = []
        for problem in tqdm(batch, desc="grpo-backprop", leave=False):
            rows = [r for r in current if r["problem"] is problem]
            if not rows:
                continue
            grpo_stats.append(self._grpo_step(
                rows=rows,
                algo=algo,
                train_model=train_model,
                tokenizer=tokenizer,
                device=device,
                n_problems=len(batch),
                cfg=cfg,
            ))
        print(f"[grpo-backprop] completed={len(grpo_stats)}/{len(batch)}")

        # ── phase 3b: SFT backprop (ONLY if teacher solved it) ────────
        sft_loss = 0.0
        # Filter: Only train on completions that actually passed all tests
        sft_candidates = [r for r in refined if r["passed"]] 
        
        if sft_candidates:
            sft_loss = self._sft_step(
                refined=sft_candidates,
                train_model=train_model,
            )
            print(f"[sft-backprop] samples={len(sft_candidates)} loss={sft_loss:.4f}")

        # ── wake vLLM — ready for optimizer step + next rollouts ──────
        try:
            await engine.wake()
            print("[vllm] awake for next step")
        except Exception as exc:
            print(f"[vllm] wake unavailable: {exc}")

        # ── bookkeeping ───────────────────────────────────────────────
        self._prev_failed_rollouts = self._build_failed_cache(current)

        return self._build_stats(batch, current, refined, grpo_stats,
                                 hints_given, passed_after_hint, sft_loss)

    # ------------------------------------------------------------------
    # GRPO gradient step for one problem group
    # ------------------------------------------------------------------

    def _grpo_step(
        self,
        *,
        rows: list[dict],
        algo: GRPOAlgo,
        train_model,
        tokenizer,
        device: torch.device,
        n_problems: int,
        cfg: TrainConfig,
    ) -> dict:
        base_messages = rows[0]["messages"]
        completions = [r["text"] for r in rows]
        scores = [r["score"] for r in rows]
        rewards = [float(r["reward"]) for r in rows]

        prompt_text = tokenizer.apply_chat_template(
            base_messages, tokenize=False, add_generation_prompt=True,
        )
        prompt_tok = tokenizer(prompt_text, return_tensors="pt")
        prompt_ids = prompt_tok["input_ids"].to(device, non_blocking=True)
        prompt_mask = prompt_tok["attention_mask"].to(device, non_blocking=True)

        comp_tok = tokenizer(completions, return_tensors="pt",
                             padding=True, add_special_tokens=False)
        completion_ids = comp_tok["input_ids"].to(device, non_blocking=True)
        completion_mask = comp_tok["attention_mask"].to(device, dtype=torch.float32, non_blocking=True)

        g = completion_ids.shape[0]
        if prompt_ids.shape[0] == 1 and g > 1:
            prompt_ids = prompt_ids.expand(g, -1)
            prompt_mask = prompt_mask.expand(g, -1)

        full_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        full_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        comp_len = completion_ids.shape[1]

        # old log-probs (frozen forward)
        with torch.inference_mode():
            h_pre, pos = train_model._forward_prefix(full_ids, full_mask)
            h_suf = train_model._forward_suffix(h_pre, pos, full_mask)
            h_comp = h_suf[:, -comp_len:, :]
            old_lp = train_model._token_logprobs_chunked(h_comp, completion_ids).detach()
        algo.bind_old_logprobs(old_lp.cpu())

        # optional ref log-probs
        if cfg.ref_adapter_path and hasattr(algo, "bind_ref_logprobs"):
            train_model.set_active_lora_adapter(cfg.ref_adapter_name)
            with torch.inference_mode():
                h_pre, pos = train_model._forward_prefix(full_ids, full_mask)
                h_suf = train_model._forward_suffix(h_pre, pos, full_mask)
                h_comp = h_suf[:, -comp_len:, :]
                ref_lp = train_model._token_logprobs_chunked(h_comp, completion_ids).detach()
            algo.bind_ref_logprobs(ref_lp.cpu())
            if cfg.student_adapter_path:
                train_model.set_active_lora_adapter(cfg.student_adapter_name)

        algo_out = algo.process_rollouts(
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            scores=scores,
            rewards=rewards,
            feedback=[(s.error or "") for s in scores],
        )
        bp = train_model.backward(
            messages=[base_messages] * len(completions),
            completion_texts=completions,
            loss_fn=algo_out.loss_fn,
            loss_scale=1.0 / max(1, n_problems),
        )
        return {**algo_out.stats, **{f"bp_{k}": float(v) for k, v in bp.items()}}

    # ------------------------------------------------------------------
    # SFT gradient step on teacher-corrected completions
    # ------------------------------------------------------------------

    def _sft_step(self, *, refined: list[dict], train_model) -> float:
        def _sft_loss_batch(
            batch_log_probs: Tensor,
            batch_mask: Tensor,
            hidden_comp=None,
        ) -> Tensor:
            mask = batch_mask.to(batch_log_probs.device).float()
            lengths = mask.sum(dim=1).clamp(min=1.0)
            return (-((batch_log_probs * mask).sum(dim=1) / lengths)).mean()

        def _sft_loss(log_probs: Tensor, gen_idx: int, hidden_comp=None) -> Tensor:
            return -log_probs.mean()

        setattr(_sft_loss, "loss_fn_batch", _sft_loss_batch)

        bp = train_model.backward(
            messages=[r["messages"] for r in refined],
            completion_texts=[r["text"] for r in refined],
            loss_fn=_sft_loss,
            loss_scale=1.0 / max(1, len(refined)),
        )
        return float(bp.get("loss", 0.0))

    # ------------------------------------------------------------------
    # Teacher refinement (concurrent)
    # ------------------------------------------------------------------

    async def _teacher_refine(
        self,
        candidates: list[dict],
        generate_fn,
        score_fn,
        teacher_client,
    ) -> tuple[list[dict], int, int]:
        cfg = self.cfg
        sem = asyncio.Semaphore(max(1, cfg.engine_semaphore_limit))
        pbar = tqdm(total=len(candidates) * cfg.max_hint_attempts, desc="teacher-refine", leave=False)

        # import once, not per-iteration
        from client.agent import AgentClient

        async def _refine_one(rec: dict) -> tuple[dict, int, int]:
            problem = rec["problem"]
            best_text = str(rec["text"])
            best_score = rec["score"]
            best_reward = float(rec["reward"])
            solved = False
            local_hints = 0
            local_passed = 0

            for attempt_idx in range(cfg.max_hint_attempts):
                _status = "did not compile" if not best_score.compiled else (
                    f"compiled, passed {best_score.passed}/{best_score.total} tests"
                )
                _details = ""
                if best_score.details:
                    _details = f"\nTest details:\n{best_score.details}"
                teacher_prompt = (
                    "You are a coding tutor. Give a targeted hint only. "
                    "Do NOT explicitly provide the right answer.\n\n"
                    f"Problem:\n{problem.statement}\n\n"
                    f"Student attempt:\n{best_text}\n\n"
                    f"Verifier result: {_status}\n"
                    f"Error: {best_score.error or 'none'}"
                    f"{_details}\n\n"
                    "Return one concise guidance hint."
                )

                # inject reference solution so the teacher can give targeted hints
                ref_answer = (problem.metadata or {}).get("answer", "")
                if ref_answer:
                    teacher_prompt += (
                        "\n\n[Reference solution — for YOUR guidance ONLY, "
                        "do NOT reveal it to the student]:\n"
                        f"{ref_answer}"
                    )

                # fresh client per candidate — no shared history across coroutines
                local_teacher = AgentClient(
                    base_url=teacher_client.llm.openai_api_base,
                    api_key=teacher_client.llm.openai_api_key,
                    temperature=teacher_client.llm.temperature,
                    max_output_tokens=teacher_client.llm.max_tokens,
                    system_prompt=teacher_client.system_prompt,
                    model=teacher_client.llm.model_name,
                    tools=[],
                    max_turns=teacher_client.max_turns,
                    extra_body=self.profile.teacher_extra_body,
                )

                async with sem:
                    _, teacher_out = await local_teacher.run(prompt=teacher_prompt)
                local_hints += 1

                retry_messages = [
                    {"role": "system", "content": cfg.system_prompt},
                    {"role": "user", "content": (
                        f"{problem.statement}\n\n"
                        "Your previous answer was incorrect.\n"
                        f"Hint:\n{teacher_out}\n\n"
                        "Try again with a corrected answer."
                    )},
                ]
                retry_txt = await generate_fn(retry_messages)
                pbar.update(1)

                retry_sc, retry_reward, retry_passed = await score_fn(problem, retry_txt)
                retry_reward *= cfg.hint_reward_discount

                if retry_reward > best_reward:
                    best_text, best_score, best_reward = retry_txt, retry_sc, retry_reward
                if retry_passed:
                    local_passed += 1
                    solved = True
                    pbar.update(cfg.max_hint_attempts - attempt_idx - 1)
                    break

            return dict(
                problem=problem,
                messages=[
                    {"role": "system", "content": cfg.system_prompt},
                    {"role": "user", "content": str(problem.statement)},
                ],
                text=best_text,
                score=best_score,
                reward=best_reward,
                passed=solved,
            ), local_hints, local_passed

        tasks = [_refine_one(rec) for rec in candidates]
        results = await asyncio.gather(*tasks)
        pbar.close()

        refined = []
        hints_given = 0
        passed_after_hint = 0
        for result, h, p in results:
            refined.append(result)
            hints_given += h
            passed_after_hint += p

        return refined, hints_given, passed_after_hint

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _ensure_adapters(self, engine, train_model, cfg: TrainConfig) -> None:
        """Lazy-load LoRA adapters on first call."""
        if not hasattr(self, "_loaded_engine_adapters"):
            self._loaded_engine_adapters: set[str] = set()
            self._loaded_train_adapters: set[str] = set()

        if cfg.student_adapter_path:
            if cfg.student_adapter_name not in self._loaded_engine_adapters:
                await engine.swap_lora_adapter(cfg.student_adapter_name, cfg.student_adapter_path)
                self._loaded_engine_adapters.add(cfg.student_adapter_name)
            if cfg.student_adapter_name not in self._loaded_train_adapters:
                train_model.load_lora_adapter(
                    cfg.student_adapter_name, cfg.student_adapter_path, is_trainable=True
                )
                self._loaded_train_adapters.add(cfg.student_adapter_name)

        if cfg.ref_adapter_path:
            if cfg.ref_adapter_name not in self._loaded_train_adapters:
                train_model.load_lora_adapter(
                    cfg.ref_adapter_name, cfg.ref_adapter_path, is_trainable=False
                )
                self._loaded_train_adapters.add(cfg.ref_adapter_name)

    def _build_failed_cache(self, current: list[dict]) -> list[dict]:
        """Collect failed rollouts; attach best peer solution for next step's teacher."""
        by_problem: dict[int, list[dict]] = {}
        for r in current:
            by_problem.setdefault(id(r["problem"]), []).append(r)

        cache: list[dict] = []
        for rows in by_problem.values():
            passed = [r for r in rows if r["passed"]]
            peer = str(max(passed, key=lambda x: x["reward"])["text"]) if passed else None
            for r in rows:
                if not r["passed"]:
                    cache.append(dict(
                        problem=r["problem"], text=r["text"],
                        score=r["score"], reward=r["reward"], peer_solution=peer,
                    ))
        return cache

    def _build_stats(
        self,
        batch: list[Problem],
        current: list[dict],
        refined: list[dict],
        grpo_stats: list[dict],
        hints_given: int,
        passed_after_hint: int,
        sft_loss: float,
    ) -> dict:
        out: dict = {
            "n_problems": float(len(batch)),
            "n_current_rollouts": float(len(current)),
            "n_refined": float(len(refined)),
            "n_hints_given": float(hints_given),
            "n_passed_after_hint": float(passed_after_hint),
            "n_pending_prev_failed": float(len(self._prev_failed_rollouts)),
            "sft_loss": float(sft_loss),
        }

        if grpo_stats:
            for k in set().union(*[s.keys() for s in grpo_stats]):
                vals = [float(s[k]) for s in grpo_stats if k in s]
                if vals:
                    out[f"grpo_{k}"] = sum(vals) / len(vals)

        all_entries = current + refined
        if all_entries:
            out["combined_pass_rate"] = sum(1 for r in all_entries if r["passed"]) / len(all_entries)

        def _best_score(pid: str) -> Score:
            rows = [r for r in current if str(r["problem"].id) == pid]
            if not rows:
                return Score(compiled=False, passed=0, total=0, error="no rollout")
            return max(
                rows,
                key=lambda r: float(r["score"].passed) / float(r["score"].total)
                if r["score"].total > 0 else 0.0,
            )["score"]

        out["problem_ids"] = [str(p.id) for p in batch]
        out["problem_scores"] = [_best_score(str(p.id)) for p in batch]
        return out

    # ------------------------------------------------------------------
    # Engine lifecycle
    # ------------------------------------------------------------------

    async def _engine_init(self, engine: VLLMEngine) -> None:
        try:
            await engine.init()
            print("vLLM: connected to existing server")
        except Exception:
            try:
                await engine.start()
                print("vLLM: started local server")
            except Exception as exc:
                raise RuntimeError(
                    f"vLLM startup failed — try lowering engine_gpu_memory_utilization "
                    f"(currently {self.cfg.engine_gpu_memory_utilization})"
                ) from exc
        try:
            if await engine.is_sleeping():
                await engine.wake()
                print("vLLM: woken from sleep")
        except Exception as exc:
            print(f"vLLM: sleep/wake status unavailable: {exc}")

    async def _engine_shutdown(self, engine: VLLMEngine) -> None:
        try:
            await engine.sleep(level=1)
            print("vLLM: sleeping")
        except Exception as exc:
            print(f"vLLM: sleep unavailable: {exc}")
        await engine.shutdown()
        print("vLLM: shutdown complete")


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def _log_rollout_stats(tag: str, rows: list[dict], extra: str = "") -> None:
    if not rows:
        return
    ratios = [
        (float(r["score"].passed) / float(r["score"].total)) if r["score"].total > 0 else 0.0
        for r in rows
    ]
    n_pass = sum(1 for r in rows if r["passed"])
    parts = [
        f"[{tag}]",
        f"n={len(rows)}",
        f"mean_score={sum(ratios) / max(1, len(ratios)):.3f}",
        f"passed={n_pass}/{len(rows)}",
    ]
    if extra:
        parts.append(extra)
    print(" ".join(parts))
