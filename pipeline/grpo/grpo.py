# region Imports

# What this import does?
from __future__ import annotations

import asyncio
import importlib
import json
import os
import random
import re
import shutil

# 3rd party
import time
from collections import Counter
from itertools import count
from pathlib import Path
from pprint import pprint
from typing import Callable, Optional

import torch
from tqdm.auto import tqdm

# Within Repo
from algo import AlgoConfig
from algo.grpo import GRPOAlgo
from client.base import BaseClient
from inference import VLLMEngine
from model.config import ModelConfig
from taskset import BucketDistribution, CurriculumLoader
from taskset.base import Problem, Score
from taskset.moove.verify import RemoteDockerVerifier

from .config import TrainConfig
from .teacher import build_failed_cache, init_teacher_client, sft_step, teacher_refine
from .utils import (
    FunctionCallAnalyzer,
    _get_profile,
    _log_rollout_stats,
    write_generation_log,
)

# endregion Imports
# region Pipeline


class GRPOPipeline:
    """
    # TODO: Write docstring with correct flow
    """

    def __init__(self, config: TrainConfig) -> None:

        # Configs
        self.cfg = config
        self.profile = _get_profile(config.model_type)

        # For teacher, we need to keep track of previous failed rollouts so they can be SFT'ed next time
        self._prev_failed_rollouts: list[dict] = []

        # Flags for vllm engine
        self._adapters_initialized = False
        self._lora_in_vllm = False
        self._active_engine_adapter: str = config.student_adapter_name

        # logs and counters
        self._gen_log_counter = count()
        self._task_stats: dict[str, dict] = {}

        stats_path = Path(config.run_dir) / "task_stats.json"
        if stats_path.exists():
            try:
                with stats_path.open() as f:
                    self._task_stats = json.load(f)
                tqdm.write(
                    f"[resume] loaded task_stats with {len(self._task_stats)} entries"
                )
            except Exception:
                pass

        # the vllm engine
        self.engine = None  # Will be set in train()

    # TODO: clean it up
    def _build_prompt_text(self, messages: list[dict]) -> str:
        """Helper to reconstruct the prompt text from messages."""
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # Before vLLM had proper support gemma lora, we had to temperarily roll back to HF generation, so new helper
    # functions were introduced to cover for raw token outputs by hf
    # TODO: clean up
    # region HF gen specific functions
    def _split_reasoning_and_content(self, text: str) -> tuple[str, str]:
        """
        Splits Gemma model output into reasoning and content.
        Strips all <eos> and <turn|> special tokens from content.
        Returns: (reasoning, content)
        """
        text = text.strip()

        # Defensive: trim any junk before the Gemma reasoning marker
        for marker in ("<|channel|>thought",):
            if marker in text:
                text = text[text.index(marker) :]
                break

        # Gemma4 format:
        #   <|channel|>thought\n ... \n<channel|>\n(content)
        match = re.search(
            r"<\|channel\|>thought\n(.*?)<channel\|>\n?(.*)",
            text,
            re.DOTALL,
        )
        if match:
            reasoning = match.group(1).strip()
            content = match.group(2).strip()
            # Remove ALL special tokens from content
            content = content.replace("<eos>", "").replace("<turn|>", "").strip()
            return reasoning, content

        # Fallback: Gemma emitted no reasoning block
        if "<|channel|>" in text:
            tqdm.write("[warning] Gemma reasoning marker present but parsing failed")

        # Clean fallback content too - remove all special tokens
        content = text.replace("<eos>", "").replace("<turn|>", "").strip()
        return "", content

        # THIS helper function was part of run_batch method, moved here for cleanliness
        # async def _batch_generate_hf(
        #     batch_problems: list[Problem], n_rollouts: int
        # ) -> list[dict]:
        #     """Direct HF batched generation with reasoning extraction."""
        #     model_family = cfg.model_type
        #     # 1. Build prompts
        #     prompts = []
        #     problem_map = []
        #     for prob in batch_problems:
        #         messages = [
        #             {"role": "system", "content": "You are a helpful assistant. <|think|>"},
        #             {"role": "user", "content": str(prob.statement)},
        #         ]
        #         prompt_text = train_model.tokenizer.apply_chat_template(
        #             messages, tokenize=False, add_generation_prompt=True
        #         )
        #         for _ in range(n_rollouts):
        #             prompts.append(prompt_text)
        #             problem_map.append(prob)
        #     # 2. Chunked generation (avoids OOM)
        #     chunk_size = getattr(cfg, "gen_batch_size", 8)
        #     all_decoded = []
        #     device = next(train_model.model.parameters()).device
        #     for i in range(0, len(prompts), chunk_size):
        #         chunk_prompts = prompts[i : i + chunk_size]
        #         batch = train_model.tokenizer(
        #             chunk_prompts, return_tensors="pt", padding=True
        #         ).to(device)
        #         with torch.inference_mode():
        #             output_ids = train_model.model.generate(
        #                 input_ids=batch["input_ids"],
        #                 attention_mask=batch["attention_mask"],
        #                 max_new_tokens=2048,
        #                 do_sample=True,
        #                 temperature=cfg.temperature,
        #                 pad_token_id=train_model.tokenizer.pad_token_id
        #                     or train_model.tokenizer.eos_token_id,
        #                 eos_token_id=train_model.tokenizer.eos_token_id,
        #             )
        #         for idx, (prompt, out_ids) in enumerate(zip(chunk_prompts, output_ids)):
        #             prompt_len = batch["attention_mask"][idx].sum().item()
        #             new_text = train_model.tokenizer.decode(
        #                 out_ids[prompt_len:], skip_special_tokens=False
        #             )
        #             all_decoded.append(new_text.strip())
        #         torch.cuda.empty_cache()  # CRITICAL for stable multi-step training
        #     # 3. Parse reasoning + answer, store canonically
        #     results = []
        #     for full_output, prob in zip(all_decoded, problem_map):
        #         reasoning, answer = self._split_reasoning_and_content(full_output)
        #         results.append(
        #             {
        #                 "problem": prob,
        #                 "messages": [
        #                     {"role": "system", "content": "You are a helpful assistant. <|think|>"},
        #                     {"role": "user", "content": str(prob.statement)},
        #                 ],
        #                 "raw_completion": full_output,
        #                 "reasoning": reasoning,
        #                 "answer": answer,
        #             }
        #         )
        #     return results

    # endregion HF gen specific functions

    # Training loop

    async def train(self, teacher_client: Optional[BaseClient] = None) -> None:

        cfg = self.cfg
        profile = self.profile

        ### Model + optimizers ###

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
        train_model = ModelCls(
            cfg.model_path,
            model_cfg,
            lora_path=cfg.student_adapter_path
            or None,  # Either continuation of a SFT'ed Lora or start fresh
        )

        self.tokenizer = train_model.tokenizer
        trainable_params = [
            p for p in train_model.model.parameters() if p.requires_grad
        ]
        optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr)
        sft_optimizer = torch.optim.AdamW(
            trainable_params, lr=cfg.sft_lr
        )  # TODO: Search for A better optimizer? or an LR scheduler?

        ### Curriculum ###

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

        ### Resume from adapter if it was saved during this training loop ###
        step = 0
        if cfg.student_adapter_path:
            adapter_path = Path(cfg.student_adapter_path).expanduser().resolve()

            # We save our own meta.json, if it exists, it means we are resuming RL training
            meta_path = adapter_path / "meta.json"
            if meta_path.exists():
                # Full training checkpoint resume
                with meta_path.open("r") as f:
                    meta = json.load(f)
                step = int(meta["step"]) + 1
                optimizer.load_state_dict(
                    torch.load(
                        adapter_path / "optimizer.pt",
                        map_location="cpu",
                        weights_only=True,
                    )
                )
                sft_optimizer.load_state_dict(
                    torch.load(
                        adapter_path / "sft_optimizer.pt",
                        map_location="cpu",
                        weights_only=True,
                    )
                )
                loader.load_state_dict(
                    torch.load(adapter_path / "curriculum.pt", map_location="cpu")
                )

                # setting random's, torch's random state for true continuation
                # TODO: Investigate if it really matters? since RL is fundamentally sampling process
                rng_path = adapter_path / "rng.pt"
                if rng_path.exists():
                    rng = torch.load(rng_path, map_location="cpu")
                    random.setstate(rng["python"])
                    torch.random.set_rng_state(rng["torch"])
                    if torch.cuda.is_available() and rng.get("cuda") is not None:
                        torch.cuda.set_rng_state_all(rng["cuda"])
                tqdm.write(
                    f"[resume] training checkpoint – resuming from step {step} "
                    f"(last completed: {step - 1})"
                )
            else:
                # SFT primer / bare LoRA – weights already loaded by ModelCls,
                # just start training from step 0 with fresh optimizer state.
                tqdm.write(
                    f"[resume] SFT primer detected (no meta.json) – starting fresh "
                    f"from step 0: {adapter_path.name}"
                )

        optimizer.zero_grad(set_to_none=True)

        # TODO: Needs to be cleaned up, many hardcodings
        # --- vLLM engine ---
        self.engine = VLLMEngine(
            model_path=cfg.model_path,
            engine_kwargs={
                "base_url": cfg.engine_base_url,
                "api_key": cfg.engine_api_key,
                "model_name": cfg.model_path,
                "gpu_memory_utilization": cfg.engine_gpu_memory_utilization,
                "semaphore_limit": cfg.engine_semaphore_limit,
                "reasoning_parser": profile.reasoning_parser,
                "tool_call_parser": profile.tool_call_parser,
                "max_lora_rank": 64,
            },
        )

        # TODO: Bring this back for actual run
        await self._engine_init(self.engine)
        await self.engine.wake()

        ### LoRA buffer for on-policy vLLM ###
        # Buffer location for the CURRENT on-policy LoRA (swapped every step).
        # This is deleted/replaced each step to avoid disk bloat.
        # Permanent checkpoints are only saved every save_lora_every steps.
        self._lora_buffer_dir = Path(cfg.run_dir) / "lora_buffer"
        self._lora_buffer_dir.mkdir(parents=True, exist_ok=True)

        teacher_client = await init_teacher_client(self, teacher_client)
        # Verifier for code
        verifier = RemoteDockerVerifier(timeout=cfg.verifier_timeout)
        verifier.check_dependencies()

        # Algorithm
        # TODO: Current pipeline is very coupled with algo, need to make it decoupled a lil bit for slightly diff variants of grpo
        algo = GRPOAlgo(
            AlgoConfig(
                kl_coeff=cfg.kl_coeff,
                clip_ratio_low=cfg.clip_ratio_low,
                clip_ratio_high=cfg.clip_ratio_high,
                norm_advantages=cfg.norm_advantages,
                loss_agg="token_mean",
            )
        )

        # tqdm init
        step_bar = tqdm(total=cfg.train_steps, initial=step, desc="train-steps")

        ### Training loop ###
        try:

            # Curriculum loader can be stopped if all the buckets are showing satisfactory results, RL convergance
            while not loader.should_stop(step):

                # total time for each step
                step_t0 = time.time()

                batch = loader.sample(step=step)
                if not batch:
                    break

                # TODO: Bring this back for teacher refinement,
                # A non empty refined means some responses were corrected by teacher model
                # current, refined, _, _ = await self.run_batch(
                gen_start = time.time()

                # Generation step for this batch of problems
                # TODO: unnecessary dep injection
                current, _, _, _ = await self.run_batch(
                    batch=batch,
                    step=step,
                    train_model=train_model,
                    tokenizer=train_model.tokenizer,
                    verifier=verifier.verify,
                    teacher_client=teacher_client,
                    semaphore_limit=cfg.engine_semaphore_limit,
                )
                gen_end = time.time()

                # TODO: Activate for bigger models,
                # Currently we are dealing with smaller models only, but for larger models we have to sleep them at time of backpropagation
                # await self.engine.sleep(level=1)

                backprop_start = time.time()

                # Backward pass
                # TODO: unnecessary dep injection
                # TODO: Add comment for each parameter
                grpo_stats, hint_samples, direct_samples, _ = self._backward_pass(
                    batch=batch,
                    current=current,
                    # TODO: Bring this back for teacher refinement
                    # refined=refined,
                    refined=[],
                    algo=algo,
                    train_model=train_model,
                    tokenizer=train_model.tokenizer,
                    cfg=cfg,
                )
                backprop_end = time.time()

                # GRPO loss aggregation and mean
                grpo_losses = [
                    float(s["bp_loss"])
                    for s in grpo_stats
                    if isinstance(s, dict) and "bp_loss" in s
                ]
                grpo_bp_loss = (
                    sum(grpo_losses) / len(grpo_losses) if grpo_losses else 0.0
                )

                # Gradient clipping
                # TODO: Research what it does, how it affects process
                torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip)

                # TODO: Will it help to hold of few steps to accumulate gradients first? will it make process resilient to degradation
                # Update weights
                optimizer.step()

                # Set gradients to zero
                optimizer.zero_grad(set_to_none=True)

                # For teacher refinement we will have a seperate optimizer
                # TODO: Research if its ok to have seperate optim, what LR it should have, or should it be part of RL?
                # Since RL and SFT are fundamentally different processes mathematically, even if the goals align
                # Streamline these two, unnecessary breakdown of SFT Steps
                if hint_samples:
                    sft_optimizer.zero_grad(set_to_none=True)
                    hint_loss = sft_step(
                        samples=hint_samples,
                        train_model=train_model,
                        sft_optimizer=sft_optimizer,
                    )
                    sft_optimizer.step()
                    sft_optimizer.zero_grad(set_to_none=True)

                if direct_samples:
                    sft_optimizer.zero_grad(set_to_none=True)
                    direct_loss = sft_step(
                        samples=direct_samples,
                        train_model=train_model,
                        sft_optimizer=sft_optimizer,
                    )
                    sft_optimizer.step()
                    sft_optimizer.zero_grad(set_to_none=True)

                # Save adapter + state
                # FIX: guard with `step > 0` so we don't waste a save at step 0
                # before any gradient update has occurred.
                if step > 0:
                    # Buffer location for current on-policy LoRA (swapped every step)
                    # this could be simplified, but its easy to see current step without logging in tmux every time
                    buffer_dir = self._lora_buffer_dir / f"step_{step}"

                    # Remove old buffer to save disk
                    for old_buf in self._lora_buffer_dir.glob("step_*"):
                        if old_buf != buffer_dir:
                            shutil.rmtree(old_buf, ignore_errors=True)

                    buffer_dir.mkdir(parents=True, exist_ok=True)
                    try:

                        # Switch to eval mode before saving
                        # TODO: Why
                        # TODO: make helper function of saving lora adapters with metadata since we may just continue training from buffer lora
                        train_model.model.eval()
                        train_model.save_lora_adapter("default", str(buffer_dir))
                        train_model.model.train()

                        # Swap vLLM to the new on-policy LoRA immediately.
                        # This keeps the generation model synchronized with training.
                        try:
                            await self.engine.swap_lora_adapter(
                                cfg.student_adapter_name,
                                str(buffer_dir),
                                load_inplace=False,  # Inplace loading is deprecated since vllm with inplace loading of lora was making generation slow
                                # TODO: Report this bug to vLLM
                            )
                            self._active_engine_adapter = cfg.student_adapter_name
                            self._lora_in_vllm = True
                        except Exception as exc:
                            tqdm.write(
                                f"[lora-swap] failed to swap LoRA in vLLM: {exc}"
                            )
                            tqdm.write(
                                "[lora-swap] continuing with base model for generation"
                            )
                            self._lora_in_vllm = False

                        # Every save_lora_every steps, also persist a permanent checkpoint
                        # TODO: make helper function for this saving with metadata etc, use that for above
                        if step % cfg.save_lora_every == 0:
                            save_dir = (
                                Path(cfg.run_dir) / "student_lora" / f"step_{step}"
                            )
                            if save_dir.exists():
                                shutil.rmtree(save_dir)
                            shutil.copytree(buffer_dir, save_dir)

                            with (save_dir / "meta.json").open("w") as f:
                                json.dump(
                                    {
                                        "step": step,
                                        "adapter_name": cfg.student_adapter_name,
                                    },
                                    f,
                                )
                            torch.save(
                                optimizer.state_dict(), save_dir / "optimizer.pt"
                            )
                            torch.save(
                                sft_optimizer.state_dict(),
                                save_dir / "sft_optimizer.pt",
                            )
                            torch.save(loader.state_dict(), save_dir / "curriculum.pt")
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
                                save_dir / "rng.pt",
                            )
                            tqdm.write(
                                f"[checkpoint] saved permanent LoRA at step {step}"
                            )

                    except Exception:
                        shutil.rmtree(buffer_dir, ignore_errors=True)
                        raise

                # Step-level logging (ALWAYS print GRPO, even if no SFT)
                combined_pass_rate = (
                    sum(1 for r in current if r["passed"]) / len(current)
                    if current
                    else 0.0
                )

                tqdm.write("")
                tqdm.write(
                    f"\nstep={step} "
                    f"mean_bucket={loader.distribution.mean:.2f} "
                    f"pass={combined_pass_rate:.3f} "
                    f"grpo_loss={grpo_bp_loss:.6f} "
                    # f"sft_hint_loss={(0.0 if not hint_samples else hint_loss):.6f} "
                    # f"sft_direct_loss={(0.0 if not direct_samples else direct_loss):.6f} "
                    f"gen={gen_end - gen_start:.1f}s "
                    f"verify={backprop_start - gen_end:.1f}s "
                    f"backprop={backprop_end - backprop_start:.1f}s "
                    f"total={time.time() - step_t0:.1f}s"
                )

                per_task_rates = [
                    round(self._task_stats[str(p.id)]["last_pass_rate"], 2)
                    for p in batch
                    if str(p.id) in self._task_stats
                ]
                per_task_rewards = [
                    round(
                        sum(r["reward"] for r in current if r["problem"] is p)
                        / len([r for r in current if r["problem"] is p]),
                        3,
                    )
                    for p in batch
                    if any(r["problem"] is p for r in current)
                ]

                # How many of the generations per task passed (100% pass)
                tqdm.write(f"per-task passes: {per_task_rates}")

                # What was the average reward for each task?
                tqdm.write(f"per-task avg reward: {per_task_rewards}")

                stats_path = Path(cfg.run_dir) / "task_stats.json"
                stats_path.parent.mkdir(parents=True, exist_ok=True)
                with stats_path.open("w") as f:
                    json.dump(self._task_stats, f, indent=2)

                step += 1
                step_bar.update(1)

        finally:
            step_bar.close()
            await self._engine_shutdown(self.engine)

    # TODO: Remove unnecessary dep injection
    def _backward_pass(
        self,
        *,
        batch: list[Problem],
        current: list[dict],
        refined: list[dict],
        algo: GRPOAlgo,
        train_model,
        tokenizer,
        cfg: TrainConfig,
    ) -> tuple[list[dict], list[tuple], list[tuple], str]:
        """
        Returns
            grpo_stats    : per-problem GRPO backward stats
            hint_samples  : (messages, completion) for hint-context SFT pass
            direct_samples: (messages, completion) for clean-prompt SFT pass
            active_name   : PEFT adapter name used
        """

        # TODO: Use this device param to expand training process to two gpus, one for generation, other for backprop
        # TODO: Why we need a next here? Why an active lora adapters, maybe for reference logprobs, but investigate still
        device = next(train_model.model.parameters()).device
        active_name = "default"
        train_model.set_active_lora_adapter(active_name)

        # GRPO backward pass
        # TODO: this abstraction seems useless, passing training model, rows etc to it for each batch? seems redudant, find better optim for it
        grpo_stats: list[dict] = []
        for problem in tqdm(batch, desc="grpo-backprop", leave=False):
            rows = [r for r in current if r["problem"] is problem]
            if not rows:
                continue
            grpo_stats.append(
                self._grpo_step(
                    rows=rows,
                    algo=algo,
                    train_model=train_model,
                    tokenizer=tokenizer,
                    device=device,
                    n_problems=len(batch),
                    cfg=cfg,
                )
            )

        # Compute per-problem pass ratios
        # TODO: This block up until the end seems useless? what its doing here?
        pass_ratio_by_problem: dict[str, float] = {}
        for problem in batch:
            rows = [r for r in current if r["problem"] is problem]
            if rows:
                pass_ratio_by_problem[str(problem.id)] = sum(
                    1 for r in rows if r["passed"]
                ) / len(rows)

        PASS_RATIO_THRESHOLD = 0.25

        # SFT sample collection with gating
        # TODO: Bring this back for teacher refinement
        # hint_samples: list[tuple[list[dict], str]] = [
        #     (r["sft_hint_messages"], r["sft_hint_text"])
        #     for r in refined
        # ]
        # direct_samples: list[tuple[list[dict], str]] = [
        #     (r["sft_direct_messages"], r["sft_direct_text"])
        #     for r in refined
        #     if r.get("sft_direct_messages") and r.get("sft_direct_text")
        # ]
        hint_samples = []
        direct_samples = []
        return grpo_stats, hint_samples, direct_samples, active_name

    # Make sure vllm engine inits with lora adapters and we have reference lora for training model too
    async def _ensure_adapters(self, train_model, cfg: TrainConfig) -> None:
        if self._adapters_initialized:
            return
        self._adapters_initialized = True

        if cfg.student_adapter_path:
            student_path = Path(cfg.student_adapter_path).expanduser().resolve()
            if student_path.exists():
                try:
                    await self.engine.swap_lora_adapter(
                        cfg.student_adapter_name, str(student_path)
                    )
                    self._lora_in_vllm = True
                    self._active_engine_adapter = cfg.student_adapter_name
                    tqdm.write(
                        f"[init] LoRA adapter loaded into vLLM from {student_path.name}"
                    )
                except Exception as exc:
                    tqdm.write(f"[init] failed to load LoRA adapter into vLLM: {exc}")
                    tqdm.write(
                        "[init] continuing without vLLM LoRA (will use buffer swaps)"
                    )
                    self._lora_in_vllm = False
            else:
                tqdm.write(f"[init] student adapter path not found: {student_path}")
        else:
            tqdm.write("[init] no student_adapter_path, using base model")

        if cfg.ref_adapter_path:
            ref_path = Path(cfg.ref_adapter_path)
            if not ref_path.exists():
                raise FileNotFoundError(f"ref_adapter_path not found: {ref_path}")
            train_model.load_lora_adapter(cfg.ref_adapter_name, str(ref_path))
            tqdm.write(f"[init] reference adapter loaded from {ref_path.name}")
        else:
            tqdm.write("[init] no ref_adapter_path - KL disabled")

    # generation step
    async def run_batch(
        self,
        batch: list[Problem],
        *,
        step: int,
        train_model,
        tokenizer,
        verifier: Callable[[Problem, str], Score],
        teacher_client: Optional[BaseClient] = None,
        semaphore_limit: int = 32,
        refine_mode: str = "current",
    ) -> tuple[list[dict], list[dict], int, int]:
        cfg = self.cfg
        profile = self.profile

        await self._ensure_adapters(train_model, cfg)

        sem = asyncio.Semaphore(max(1, semaphore_limit))
        use_lora = self._lora_in_vllm

        # make generationi requests to vllm
        # TODO: Move this function to utils for easy management
        async def _generate(messages: list[dict]) -> Optional[str]:
            model_name = (
                self._active_engine_adapter if use_lora else cfg.model_path
            )  # vllm uses lora name as model name for distinction
            payload = {
                "model": model_name,
                "messages": messages,
                "temperature": cfg.temperature,
                "max_tokens": cfg.max_tokens,
                "stream": False,
            }
            payload.update(
                profile.gen_extra_payload
            )  # extra payload to use thinking mode, etc

            # make a request with semaphore
            async with sem:
                resp = await self.engine._request_json(
                    "POST", "/chat/completions", payload
                )
            choices = resp.get("choices") or []
            if not choices:
                return None
            raw_msg = choices[0].get("message") or {}

            # get reasoning and main content seperately, then add gemma formatting later
            # TODO: make this model agnostic everywhere
            reasoning = str(raw_msg.get("reasoning") or "").strip()
            content = str(raw_msg.get("content") or "").strip()
            if reasoning:
                return f"<|channel|>thought\n{reasoning}\n<channel|>\n{content}"
            else:
                return f"{content}<turn|>"

        # TODO: move this to utils to reduce lines
        def _is_clean_logs(compile_logs: str, stdout: str = "") -> bool:
            combined = f"{compile_logs or ''} {stdout or ''}".lower()
            if not combined.strip():
                return True
            return not any(
                kw in combined
                for kw in [
                    "warning:",
                    "error:",
                    "note:",
                    "implicit",  # compiler
                    "addresssanitizer",
                    "undefined behavior",  # sanitizers
                    "segmentation fault",
                    "segfault",
                    "abort",  # runtime crashes
                ]
            )

        # Commented-out old scoring logic (kept for reference)
        # async def _score(problem: Problem, text: str):
        #     m = re.search(r"```c\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        #     extracted = m.group(1).strip() if m else text
        #     sc = verifier(problem, extracted)
        #     details = sc.details or {}
        #     total_external = details.get("total_external_functions_executed")
        #     total_correct = details.get("total_correct_functions_executed")
        #     compile_logs = details.get("compile_logs", "") or ""
        #     stdout = details.get("stdout", "") or ""
        #     clean = _is_clean_logs(compile_logs, stdout)
        #     if sc.compiled:
        #         reward = 1.0
        #         reward += 0.2 if clean else -0.2
        #         if total_external is not None and total_correct is not None:
        #             if total_external > 0:
        #                 fn_ratio = float(total_correct) / total_external
        #                 if fn_ratio < 1.0:
        #                     reward = reward * fn_ratio
        #             else:
        #                 reward = 0.0
        #     else:
        #         reward = 0.0
        #         if total_external is not None and total_correct is not None:
        #             if total_external > 0:
        #                 reward = 0.5 * (float(total_correct) / total_external)
        #     required_funcs = problem.metadata["required"]
        #     analyzer = FunctionCallAnalyzer()
        #     called = analyzer.extract_called_functions(extracted)
        #     present = [fn for fn in required_funcs if fn in called]
        #     fn_ratio_required = len(present) / len(required_funcs)
        #     reward = reward * fn_ratio_required
        #     compiled_success = bool(sc.compiled)
        #     return sc, reward, compiled_success

        # TODO: make this score function more coherent and proper
        # Also, move this into verifier logic to make the pipeline task agnostic
        async def _score(problem: Problem, text: str):
            m = re.search(r"```c\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
            extracted = m.group(1).strip() if m else text
            sc = verifier(problem, extracted)
            details = sc.details or {}
            total_external = details.get("total_external_functions_executed")
            total_correct = details.get("total_correct_functions_executed")
            compile_logs = details.get("compile_logs", "") or ""
            stdout = details.get("stdout", "") or ""
            clean = _is_clean_logs(compile_logs, stdout)
            compiled_success = bool(sc.compiled)

            if compiled_success:
                reward = 1.0
                reward += 0.2 if clean else -0.2
            else:
                reward = 0.0

            required_funcs = problem.metadata["required"]
            local_analyzer = FunctionCallAnalyzer()
            called = local_analyzer.extract_called_functions(extracted)
            present = [fn for fn in required_funcs if fn in called]
            if len(present) != len(required_funcs):
                reward = 0.0
                compiled_success = False
                sc.passed = 0
                sc.compiled = False

            return sc, reward, compiled_success

        # Rollout generation + scoring
        # TODO: De-Activate later when vllm fixes lora loading for gemma4
        gen_tasks = []
        for problem in batch:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. <|think|>",
                },  # TODO: Make this model agnostic
                {"role": "user", "content": str(problem.statement)},
            ]
            for _ in range(cfg.n_rollouts):
                # This makes a problem object, with its message list, and a coroutine that can be gathered using asyncio
                gen_tasks.append((problem, messages, _generate(messages)))

        current: list[dict] = []

        # Process results as soon as they are available, rather than waiting
        # for all generations to complete. This overlaps verification with generation.
        tasks = []
        task_to_metadata = {}
        for problem, messages, coro in gen_tasks:
            task = asyncio.create_task(coro)
            tasks.append(task)
            task_to_metadata[task] = (problem, messages)

        # Wait for tasks to complete one by one
        # While task list is non empty
        while tasks:
            done, pending_tasks = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                tasks.remove(task)  # remove it as its done
                txt = task.result()
                problem, messages = task_to_metadata[task]
                if txt is None:
                    continue
                sc, reward, passed = await _score(problem, txt)

                # TODO: Handle logging better, rather than each in seperate file
                write_generation_log(
                    self,
                    step=step,
                    model_name=(
                        self._active_engine_adapter if use_lora else cfg.model_path
                    ),
                    messages=messages,
                    output=txt,
                    score=sc,
                )
                current.append(
                    dict(
                        problem=problem,
                        messages=messages,
                        text=txt,
                        score=sc,
                        reward=reward,
                        passed=passed,
                    )
                )
            tasks = list(pending_tasks)

        _log_rollout_stats("grpo-rollouts", current)

        # TODO: Remove task stats management, we are not using it
        for problem in batch:
            rows = [r for r in current if r["problem"] is problem]
            if not rows:
                continue
            pid = str(problem.id)
            last_passed = sum(1 for r in rows if r["passed"])
            last_total = len(rows)
            last_reward = sum(r["reward"] for r in rows)
            s = self._task_stats.setdefault(
                pid, {"total_attempts": 0, "total_passed": 0, "total_reward": 0.0}
            )
            s["total_attempts"] += last_total
            s["total_passed"] += last_passed
            s["total_reward"] = s.get("total_reward", 0.0) + last_reward
            s["overall_pass_rate"] = round(s["total_passed"] / s["total_attempts"], 3)
            s["last_pass_rate"] = round(last_passed / last_total, 3)

        # Calculating pass ratio for each task, if it crosses a threshold,
        # will skip teacher refinement as GRPO signal is enough for moving gradients for that problem
        pass_ratio_by_problem: dict[str, float] = {}
        for problem in batch:
            rows = [r for r in current if r["problem"] is problem]
            if rows:
                pass_ratio_by_problem[str(problem.id)] = sum(
                    1 for r in rows if r["passed"]
                ) / len(rows)

        PASS_RATIO_THRESHOLD = 0.25

        refine_candidates_all = {
            "current": [r for r in current if not r["passed"]],
            "previous": self._prev_failed_rollouts,
            "none": [],
        }.get(refine_mode, [])

        REFINE_THRESHOLD = 0.25
        refine_candidates = [
            r
            for r in refine_candidates_all
            if pass_ratio_by_problem.get(str(r["problem"].id), 0.0) < REFINE_THRESHOLD
        ]

        refined: list[dict] = []
        hints_given = 0
        passed_after_hint = 0

        if cfg.enable_teacher and cfg.max_hint_attempts > 0 and refine_candidates:
            refined, hints_given, passed_after_hint = await teacher_refine(
                self, refine_candidates, _generate, _score, teacher_client, sem, step
            )
        # _log_rollout_stats("teacher-fixes", refined, extra=f"hints={hints_given}")

        self._prev_failed_rollouts = build_failed_cache(current)
        return current, refined, hints_given, passed_after_hint

    # TODO: understand this properly with algo
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

        # Filter empty completions
        valid = [
            (c, s, r)
            for c, s, r in zip(completions, scores, rewards)
            if c and c.strip()
        ]
        if not valid:
            return {"bp_loss": 0.0}

        if len(valid) < len(completions):
            tqdm.write(f"[grpo-step] dropped {len(completions) - len(valid)} empty")

        completions, scores, rewards = zip(*valid)
        completions = list(completions)
        scores = list(scores)
        rewards = list(rewards)

        # Tokenize prompt + completions
        prompt_text = tokenizer.apply_chat_template(
            base_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_tok = tokenizer(prompt_text, return_tensors="pt")
        prompt_ids = prompt_tok["input_ids"].to(device, non_blocking=True)
        prompt_mask = prompt_tok["attention_mask"].to(device, non_blocking=True)

        comp_tok = tokenizer(
            completions, return_tensors="pt", padding=True, add_special_tokens=False
        )
        completion_ids = comp_tok["input_ids"].to(device, non_blocking=True)
        completion_mask = comp_tok["attention_mask"].to(
            device, dtype=torch.float32, non_blocking=True
        )

        g = completion_ids.shape[0]
        if prompt_ids.shape[0] == 1 and g > 1:
            prompt_ids = prompt_ids.expand(g, -1)
            prompt_mask = prompt_mask.expand(g, -1)

        full_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        full_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        comp_len = completion_ids.shape[1]

        # --- old_lp: from train_model BEFORE weight update ---
        # At step 0: train_model = vLLM = SFT primer -> old_lp ~ gen_lp -> ratio ~ 1
        # This is correct on-policy behavior. Loss is small because advantages
        # cluster around 0 (normalized across group). After optimizer.step(),
        # train_model diverges -> ratio becomes meaningful.
        train_model.set_active_lora_adapter("default")
        with torch.inference_mode():
            h_pre, pos, shared_kv_states, per_layer_inputs, *_ = (
                train_model._forward_prefix(full_ids, full_mask)
            )
            h_suf = train_model._forward_suffix(
                h_pre, pos, full_mask, shared_kv_states, per_layer_inputs
            )
            h_comp = h_suf[:, -comp_len:, :]
            old_lp = train_model._token_logprobs_chunked(
                h_comp, completion_ids
            ).detach()
        algo.bind_old_logprobs(old_lp.cpu())

        # --- ref_lp: for KL penalty ---
        if cfg.ref_adapter_path and hasattr(algo, "bind_ref_logprobs"):
            actual_adapters = list(train_model.model.peft_config.keys())
            if cfg.ref_adapter_name in actual_adapters:
                train_model.set_active_lora_adapter(cfg.ref_adapter_name)
                with torch.inference_mode():
                    h_pre, pos, shared_kv_states, per_layer_inputs, *_ = (
                        train_model._forward_prefix(full_ids, full_mask)
                    )
                    h_suf = train_model._forward_suffix(
                        h_pre, pos, full_mask, shared_kv_states, per_layer_inputs
                    )
                    h_comp = h_suf[:, -comp_len:, :]
                    ref_lp = train_model._token_logprobs_chunked(
                        h_comp, completion_ids
                    ).detach()
                    algo.bind_ref_logprobs(ref_lp.cpu())
                train_model.set_active_lora_adapter("default")

        # --- GRPO algorithm ---
        algo_out = algo.process_rollouts(
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            scores=scores,
            rewards=rewards,
            feedback=[(s.error or "") for s in scores],
        )

        # --- backward: computes new_lp from current train_model weights ---
        bp = train_model.backward(
            messages=[base_messages] * len(completions),
            completion_texts=completions,
            loss_fn=algo_out.loss_fn,
            loss_scale=1.0 / max(1, n_problems),
        )

        return {**algo_out.stats, **{f"bp_{k}": float(v) for k, v in bp.items()}}

    async def _engine_init(self, engine) -> None:
        try:
            await engine.init()
            tqdm.write("engine: connected to existing server")
        except Exception:
            try:
                # TODO: Revert this back
                await engine.start()
                tqdm.write("engine: started local server")
            except Exception as exc:
                raise RuntimeError(
                    f"Engine startup failed – try lowering engine_gpu_memory_utilization "
                    f"(currently {self.cfg.engine_gpu_memory_utilization})"
                ) from exc
        try:
            if await engine.is_sleeping():
                await engine.wake()
                tqdm.write("engine: woken from sleep")
        except Exception as exc:
            tqdm.write(f"engine: sleep/wake status unavailable: {exc}")

    async def _engine_shutdown(self, engine) -> None:
        try:
            await engine.sleep(level=1)
            tqdm.write("engine: sleeping")
        except Exception as exc:
            tqdm.write(f"engine: sleep unavailable: {exc}")
        await engine.shutdown()
        tqdm.write("engine: shutdown complete")


# endregion Pipeline
