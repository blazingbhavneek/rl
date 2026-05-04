# region Imports

# What this import does?
from __future__ import annotations

# 3rd party
import time
import json
import random
import shutil
import re
import os
import json
import torch
import random
import asyncio
import importlib
from torch import Tensor
from pathlib import Path
from pprint import pprint
from tqdm.auto import tqdm
from itertools import count
from collections import Counter
from typing import Callable, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, field
from langchain_openai import ChatOpenAI
from tree_sitter import Parser, Language
from langchain_core.messages import HumanMessage
from tree_sitter_c import language as c_language
from langchain_mcp_adapters.client import MultiServerMCPClient


# Within Repo
from algo import AlgoConfig
from algo.grpo import GRPOAlgo
from inference import VLLMEngine
from client.base import BaseClient
from client.agent import AgentClient
from model.config import ModelConfig
from taskset.base import Problem, Score
from client.tools import build_markdown_rag_tool
from taskset import BucketDistribution, CurriculumLoader
from taskset.moove.verify import RemoteDockerVerifier

# endregion Imports

# region Utils

# This parses C code to check which functions were called in code, written to check whether required function were called or not
# otherwise model learns to game the system by either calling no functions or calling random other functions that are easy to compile
class FunctionCallAnalyzer:

    def __init__(self):

        # Initialize parser
        self.parser = Parser(Language(c_language()))

    # Make a set of functions that were called in the given code
    def extract_called_functions(self, code: str) -> set[str]:
        tree = self.parser.parse(code.encode("utf8")) # encode it incase of euc_jp encoding?
        called = set()

        # Recursive function that goes through all entities
        def visit(node):
            if node.type == "call_expression": # call expression is node for when a function is called
                fn = node.child_by_field_name("function")
                if fn:
                    name = self._extract_name(fn) # for handling multiple types of function calls
                    if name:
                        called.add(name)
            for c in node.children:
                visit(c)

        visit(tree.root_node)
        return called

    # Extracting name of function called, either direct calls (ret = foo(x)) which have identifiers, or calling methods (ret = obj.foo(x))
    def _extract_name(self, node):
        if node.type == "identifier":
            return node.text.decode() # nodes have encoded code, we need to decode it to get string
        if node.type == "field_expression":
            f = node.child_by_field_name("field") 
            if f:
                return f.text.decode()
        return None


analyzer = FunctionCallAnalyzer()

# Final correct response made with teacher's help needs to have its own Chain of thought
COT_SYSTEM_PROMPT = """\
You are generating chain-of-thought data for training a reasoning model.
Write the internal reasoning an expert C programmer would have when solving
the task, based on programming theory, API knowledge, and implementation planning.

Rules:
- Do NOT mention tools, verification, or compilation.
- Do NOT refer to any external process.
- Do NOT invent behavior not present in the code.
- Keep the reasoning technical, neutral, and educational.
- The output should be long and informative
- Use the specialist to ask questions properly first, dont include any information
  in the output that wasnt retrieved or checked by specialist
- Research properly, about the code and theory to make a correct and technically
  sound reasoning
- Answer should be simple text, no formatting, no codeblocks, no markdown or code
- Answer like a person thinking, not announcements

Bad output:
- An expert C programmer analyzing the task to write a minimal program using...
- When approaching the task of writing a minimal C program that uses mpf_mfs_stuprqbf,
  an expert C programmer would begin by examining the function...
- I need to first understand what this function does and how it fits into the PMF
  library architecture ...

Good output:
- I need to first understand what users want. The user wants to write C code ...
  know that this function ...

Dont let the reasoning show that you dont know, the model knows it already. The
reasoning should always sound confident.
Give long reasoning traces rich with information.
"""



@dataclass(frozen=True)
class _ModelProfile:
    module: str # which class from the model module we need to import for this, which are subclasses of BaseModel
    cls_name: str # Name of the subclass  
    reasoning_parser: Optional[str] # reasoning parser used by vllm
    tool_call_parser: str # tool call parser used by vllm
    teacher_extra_body: dict # some calls may need extra kwargs to enable thinking from server side
    gen_extra_payload: dict


_MODEL_PROFILES: dict[str, _ModelProfile] = {
    "qwen3": _ModelProfile(
        module="model.qwen3",
        cls_name="Qwen3Model",
        reasoning_parser="qwen3",
        tool_call_parser="hermes",
        teacher_extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        gen_extra_payload={"chat_template_kwargs": {"enable_thinking": True}},
    ),
    "qwen3_5": _ModelProfile(
        module="model.qwen3_5",
        cls_name="Qwen3_5Model",
        reasoning_parser="qwen3",
        tool_call_parser="qwen3_coder",
        teacher_extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        gen_extra_payload={"chat_template_kwargs": {"enable_thinking": True}},
    ),
    "gptoss": _ModelProfile(
        module="model.gptoss",
        cls_name="GptOssModel",
        reasoning_parser=None,
        tool_call_parser="openai",
        teacher_extra_body={},
        gen_extra_payload={"include_reasoning": True},
    ),
    "gemma": _ModelProfile(
        module="model.gemma4",
        cls_name="Gemma4Model",
        # match vLLM launch flags
        reasoning_parser="gemma4",
        tool_call_parser="gemma4",
        teacher_extra_body={},       # do NOT force enable_thinking
        gen_extra_payload={},        # let reasoning be server-controlled
    ),
}

# TODO: clean it up
def _get_profile(model_type: str) -> _ModelProfile:
    if model_type not in _MODEL_PROFILES:
        raise ValueError(
            f"Unknown model_type {model_type!r}. Choose from: {list(_MODEL_PROFILES)}"
        )
    return _MODEL_PROFILES[model_type]


# Get one valid answer if any, so teacher can have a reference when suggesting fix for wrong ones?
# TODO: clean it up, called only once
def _sample_ref_answer(problem) -> Optional[str]:
    raw = (problem.metadata or {}).get("answer", None)
    if not raw:
        return None
    if isinstance(raw, list):
        valid = [s for s in raw if s and isinstance(s, str)]
        return random.choice(valid) if valid else None
    if isinstance(raw, str):
        return raw.strip() or None
    return None

# endregion Utils

# region Training config

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
                text = text[text.index(marker):]
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


    def _write_generation_log(
        self,
        *,
        step: int,
        model_name: str,
        messages: list[dict],
        output: Optional[str],
        score=None,
        finish_reason: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        log_dir = Path("gen_logs") / f"step_{step}"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        idx = next(self._gen_log_counter)
        path = log_dir / f"gen_{ts}_{idx:06d}.md"

        with path.open("w", encoding="utf-8") as f:
            f.write("# Student Generation\n\n")
            f.write(f"- step: `{step}`\n")
            f.write(f"- model: `{model_name}`\n")
            if finish_reason is not None:
                f.write(f"- finish_reason: `{finish_reason}`\n")
            if error is not None:
                f.write(f"- error: `{error}`\n")
            f.write("\n")

            f.write("## Message History\n\n")
            for i, m in enumerate(messages, 1):
                role = m.get("role", "unknown")
                f.write(f"### {i}. {role}\n\n")
                content = m.get("content", "")
                if isinstance(content, str):
                    f.write(f"{content}\n\n")
                else:
                    f.write("```json\n")
                    f.write(json.dumps(content, ensure_ascii=False, indent=2))
                    f.write("\n```\n\n")

            f.write("## Full Prompt\n\n")
            prompt_text = self._build_prompt_text(messages)
            f.write(f"```text\n{prompt_text}\n```\n\n")

            f.write("## Response\n\n")
            if output is not None:
                reasoning, content = self._split_reasoning_and_content(output)
                f.write("### Reasoning\n\n")
                f.write(f"```text\n{reasoning}\n```\n\n")
                f.write("### Content\n\n")
                f.write(f"```text\n{content}\n```\n\n")
            else:
                f.write("_No output returned._\n")

            # Add compile logs if available
            if score is not None and hasattr(score, "details") and score.details:
                details = score.details or {}
                compile_logs = details.get("compile_logs", "") or ""
                stdout = details.get("stdout", "") or ""
                if compile_logs or stdout:
                    f.write("## Compile Logs\n\n")
                    f.write("```\n")
                    if compile_logs:
                        f.write(compile_logs)
                    if stdout:
                        if compile_logs:
                            f.write("\n")
                        f.write(stdout)
                    f.write("\n```\n")

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
            lora_path=cfg.student_adapter_path or None, # Either continuation of a SFT'ed Lora or start fresh
        )

        self.tokenizer = train_model.tokenizer
        trainable_params = [
            p for p in train_model.model.parameters() if p.requires_grad
        ]
        optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr)
        sft_optimizer = torch.optim.AdamW(trainable_params, lr=cfg.sft_lr) # TODO: Search for A better optimizer? or an LR scheduler?


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

        # TODO: Decouple teacher code completely from GRPO
        # Tools for Teacher
        mcp_config = {
            "moove": {
                "transport": "http",
                "url": "http://10.160.152.38:9001/mcp",
            },
        }
        mcp_client = MultiServerMCPClient(mcp_config)
        mcp_tools = await mcp_client.get_tools()
        rag_tool = build_markdown_rag_tool(
            docs_folder=cfg.docs_folder,
            persist_directory="logs/chroma_intel_rag",
            embedding_backend=cfg.embedding_backend,
            embedding_base_url=cfg.embedding_base_url,
            embedding_api_key=cfg.embedding_api_key,
            embedding_model=cfg.embedding_model,
        )

        tools = [rag_tool, *mcp_tools]

        AgentClient(
            base_url=cfg.teacher_base_url or cfg.engine_base_url,
            api_key=cfg.teacher_api_key or cfg.engine_api_key,
            temperature=cfg.teacher_temperature,
            max_output_tokens=cfg.teacher_max_tokens,
            system_prompt=(
                "You are a coding tutor helping a student fix their C code. "
                "Each turn you will see the student's latest attempt and its "
                "compiler/test output. Give concise targeted hints. "
                "Tell it how to fix its errors. "
                "Explain the API reference/theory to it that you understand from the reference material. "
                "Tell the student model to add the explanations it understood in the comments. "
                "If a Macro is missing, tell it to define the macro in the code itself, search the documentation for a good default value. "
                "The student does not have access to any information or reference material, it's your job to give it all information it needs to fix its mistakes. "
                "Make the student write everything it learns in comments (No CPP style comments) so it remembers what it saw. "
                "Dont give it final answer directly, guide it to what it needs to change to get good answer, suggest changes and instruct it to add theory in comments. "
                "The suggestion should be minimal, dont tell it everything, just enough to fix error"
            ),
            model=cfg.teacher_model_name or cfg.model_path,
            tools=tools,
            max_turns=cfg.teacher_max_turns,
            extra_body=profile.teacher_extra_body,
        )


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
                    hint_loss = self._sft_step(
                        samples=hint_samples,
                        train_model=train_model,
                        sft_optimizer=sft_optimizer,
                    )
                    sft_optimizer.step()
                    sft_optimizer.zero_grad(set_to_none=True)

                if direct_samples:
                    sft_optimizer.zero_grad(set_to_none=True)
                    direct_loss = self._sft_step(
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
                                load_inplace=False, # Inplace loading is deprecated since vllm with inplace loading of lora was making generation slow
                                # TODO: Report this bug to vLLM
                            )
                            self._active_engine_adapter = cfg.student_adapter_name
                            self._lora_in_vllm = True
                        except Exception as exc:
                            tqdm.write(f"[lora-swap] failed to swap LoRA in vLLM: {exc}")
                            tqdm.write("[lora-swap] continuing with base model for generation")
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
                            torch.save(optimizer.state_dict(), save_dir / "optimizer.pt")
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
                            tqdm.write(f"[checkpoint] saved permanent LoRA at step {step}")

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
                pass_ratio_by_problem[str(problem.id)] = (
                    sum(1 for r in rows if r["passed"]) / len(rows)
                )

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
                    tqdm.write("[init] continuing without vLLM LoRA (will use buffer swaps)")
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
            model_name = self._active_engine_adapter if use_lora else cfg.model_path # vllm uses lora name as model name for distinction
            payload = {
                "model": model_name,
                "messages": messages,
                "temperature": cfg.temperature,
                "max_tokens": cfg.max_tokens,
                "stream": False,
            }
            payload.update(profile.gen_extra_payload) # extra payload to use thinking mode, etc

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
                    "implicit",          # compiler
                    "addresssanitizer",
                    "undefined behavior",  # sanitizers
                    "segmentation fault",
                    "segfault",
                    "abort",             # runtime crashes
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
                {"role": "system", "content": "You are a helpful assistant. <|think|>"}, # TODO: Make this model agnostic
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
                tasks.remove(task) # remove it as its done
                txt = task.result()
                problem, messages = task_to_metadata[task]
                if txt is None:
                    continue
                sc, reward, passed = await _score(problem, txt)

                # TODO: Handle logging better, rather than each in seperate file
                self._write_generation_log(
                    step=step,
                    model_name=self._active_engine_adapter if use_lora else cfg.model_path,
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
                pass_ratio_by_problem[str(problem.id)] = (
                    sum(1 for r in rows if r["passed"]) / len(rows)
                )

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

        # TODO: Bring this back for teacher refinement
        # if cfg.max_hint_attempts > 0 and refine_candidates:
        #     refined, hints_given, passed_after_hint = await self._teacher_refine(
        #         refine_candidates, _generate, _score, teacher_client, sem, step
        #     )
        # _log_rollout_stats("teacher-fixes", refined, extra=f"hints={hints_given}")

        self._prev_failed_rollouts = self._build_failed_cache(current)
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


    def _sft_step(
        self,
        *,
        samples: list[tuple[list[dict], str]],
        train_model,
        sft_optimizer,
    ) -> float:
        if not samples:
            return 0.0

        formatted_messages: list[list[dict]] = []
        formatted_completions: list[str] = []
        n_skipped = 0

        for messages, completion_text in samples:
            text = completion_text.strip()
            if "<|im_start|>" in text or text.lstrip().startswith("assistant"):
                n_skipped += 1
                continue
            if not text.endswith("<|im_end|>"):
                text = text + "\n<|im_end|>"
            formatted_messages.append(messages)
            formatted_completions.append(text)

        if n_skipped > 0:
            tqdm.write(f"[sft] skipped {n_skipped}/{len(samples)} malformed samples")

        if not formatted_messages:
            return 0.0

        def loss_fn_batch(
            batch_log_probs: Tensor,
            batch_mask: Tensor,
            hidden_batch=None,
        ) -> Tensor:
            mask = batch_mask.to(batch_log_probs.device).float()
            lengths = mask.sum(dim=1).clamp(min=1.0)
            return (-((batch_log_probs * mask).sum(dim=1) / lengths)).mean()

        bp = train_model.backward(
            messages=formatted_messages,
            completion_texts=formatted_completions,
            loss_fn=loss_fn_batch,
            loss_scale=1.0,
        )
        return float(bp.get("loss", 0.0))


    async def _teacher_refine(
        self,
        candidates: list[dict],
        generate_fn,
        score_fn,
        teacher_client,
        sem: asyncio.Semaphore,
        step: int,
    ) -> tuple[list[dict], int, int]:
        cfg = self.cfg
        pbar = tqdm(
            total=len(candidates) * cfg.max_hint_attempts,
            desc="teacher-refine",
            leave=False,
        )

        async def _refine_one(rec: dict) -> tuple[dict, int, int]:
            MAX_PREV_ATTEMPT_CHARS = 5000
            MAX_ERROR_CHARS = 1500

            problem = rec["problem"]
            current_text = str(rec["text"])
            current_score = rec["score"]
            best_text, best_score, best_reward = (
                current_text,
                current_score,
                float(rec["reward"]),
            )
            solved = False
            local_hints = 0

            # FIX 2+5: Restored best_hint_messages / best_hint_text tracking.
            # These are needed to build sft_hint_messages and sft_hint_text
            # for the hint-context SFT pass.
            best_hint_messages: Optional[list[dict]] = None
            best_hint_text: Optional[str] = None
            ref_answer = _sample_ref_answer(problem)

            system_prompt = (
                "You are a coding tutor helping a student fix their C code. "
                "Each turn you will see the student's latest attempt and its "
                "compiler/test output. Give concise targeted hints. "
                "Tell it how to fix its errors. "
                "Explain the API reference/theory to it that you understand from the reference material. "
                "Tell the student model to add the explanations it understood in the comments. "
                "If a Macro is missing, tell it to define the macro in the code itself, search the documentation for a good default value. "
                "The student does not have access to any information or reference material, it's your job to give it all information it needs to fix its mistakes. "
                "Make the student write everything it learns in comments (No CPP style comments) so it remembers what it saw. "
                "Dont give it final answer directly, guide it to what it needs to change to get good answer, suggest changes and instruct it to add theory in comments. "
                "The suggestion should be minimal, dont tell it everything, just enough to fix error"
            )

            if ref_answer:
                system_prompt += (
                    "\n\n[Reference solution – for YOUR guidance ONLY, "
                    "do NOT reveal to the student]:\n"
                    f"{ref_answer}"
                )

            if cfg.teacher_base_url:
                teacher_base_url = cfg.teacher_base_url
                teacher_api_key = cfg.teacher_api_key or cfg.engine_api_key
                teacher_model = cfg.teacher_model_name
            else:
                teacher_base_url = cfg.engine_base_url
                teacher_api_key = cfg.engine_api_key
                teacher_model = cfg.model_path

            local_teacher = AgentClient(
                base_url=teacher_base_url,
                api_key=teacher_api_key,
                temperature=cfg.teacher_temperature,
                max_output_tokens=cfg.teacher_max_tokens,
                system_prompt=system_prompt,
                model=teacher_model,
                tools=teacher_client.tools,
                max_turns=cfg.teacher_max_turns,
                extra_body=self.profile.teacher_extra_body,
            )

            for attempt in range(cfg.max_hint_attempts):
                if not current_score.compiled:
                    status = "did not compile"
                else:
                    status = (
                        f"compiled, passed {current_score.passed}/"
                        f"{current_score.total} tests"
                    )

                error_block = ""
                if current_score.error:
                    error_block += f"\nError: {current_score.error}"
                if current_score.details:
                    error_block += f"\nTest details:\n{current_score.details}"
                error_block = (
                    error_block[:MAX_ERROR_CHARS] + error_block[-MAX_ERROR_CHARS:]
                )

                current_text_truncated = current_text[-MAX_PREV_ATTEMPT_CHARS:]
                teacher_prompt = (
                    f"Problem:\n{problem.statement}\n\n"
                    f"Student attempt (try {attempt + 1}):\n{current_text_truncated}\n\n"
                    f"Verifier result: {status}"
                    f"{error_block}\n\n"
                    "Give one concise hint to fix this."
                )

                async with sem:
                    _, hint = await local_teacher.run(prompt=teacher_prompt)
                local_hints += 1

                retry_messages = [
                    {
                        "role": "user",
                        "content": (
                            f"{problem.statement}\n\n"
                            f"Your previous attempt:\n{current_text_truncated}\n\n"
                            f"Verifier result: {status}"
                            f"{error_block}\n\n"
                            f"Tutor hint:\n{hint}\n\n"
                            "Fix your solution."
                        ),
                    },
                ]

                retry_text = await generate_fn(retry_messages)
                pbar.update(1)

                if retry_text is None:
                    tqdm.write(
                        f"[teacher-refine] generation failed on attempt {attempt + 1}, skipping"
                    )
                    continue

                retry_sc, retry_reward, retry_passed = await score_fn(
                    problem, retry_text
                )
                self._write_generation_log(
                    step=step,
                    model_name=(
                        self._active_engine_adapter
                        if self._lora_in_vllm
                        else self.cfg.model_path
                    ),
                    messages=retry_messages,
                    output=retry_text,
                    score=retry_sc,
                )

                retry_reward *= cfg.hint_reward_discount
                if retry_reward > best_reward:
                    best_text, best_score, best_reward = (
                        retry_text,
                        retry_sc,
                        retry_reward,
                    )
                    # FIX 5: track the messages+answer of the best hint attempt
                    # so we can build the SFT hint pair.
                    best_hint_messages = retry_messages
                    best_hint_text = retry_text

                current_text = retry_text
                current_score = retry_sc

                if retry_passed:
                    solved = True
                    # Ensure fields set even when first attempt was best
                    if best_hint_messages is None:
                        best_hint_messages = retry_messages
                        best_hint_text = retry_text
                    pbar.update(cfg.max_hint_attempts - attempt - 1)
                    break

            # --- SFT targets ---
            sft_target: Optional[str] = None
            sft_direct_reasoning: Optional[str] = None

            def _strip_think(text: str) -> str:
                return re.sub(
                    r"<think>.*?</think>\s*", "", text, flags=re.DOTALL
                ).strip()

            def _as_c_block(text: Optional[str]) -> Optional[str]:
                if not text:
                    return None
                text = _strip_think(text)
                m = re.search(r"```c\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
                code = m.group(1).strip() if m else text.strip()
                if not code:
                    return None
                return f"```c\n{code}\n```"

            if solved:
                if ref_answer:
                    sft_target = f"```c\n{ref_answer.strip()}\n```"
                else:
                    sft_target = _as_c_block(best_text)

                if sft_target:
                    sft_direct_reasoning = await self._generate_cot_reasoning(
                        problem=problem,
                        answer_code=sft_target,
                        teacher_client=local_teacher,
                        sem=sem,
                    )

                if sft_direct_reasoning:
                    self._write_cot_file(
                        step=step,
                        reasoning=sft_direct_reasoning,
                    )

            # --- Format helper ---
            def _fmt_completion(
                code: Optional[str], reasoning: Optional[str]
            ) -> Optional[str]:
                if not code:
                    return None
                if reasoning:
                    return f"<think>\n{reasoning}\n</think>\n\n{code}\n<|im_end|>"
                return f"<think>\n </think>\n\n{code}\n<|im_end|>"

            # FIX 2 (Critical): Produce sft_hint_messages + sft_hint_text.
            # hint-context pair: the student saw (problem + prev attempt + hint)
            # and produced best_hint_text. We train on that full context -> answer.
            sft_hint_completion = _fmt_completion(
                _strip_think(best_hint_text) if best_hint_text else None, None
            )
            sft_direct_completion = _fmt_completion(sft_target, sft_direct_reasoning)

            sft_messages_direct: Optional[list[dict]] = None
            if sft_direct_completion:
                sft_messages_direct = [
                    {"role": "user", "content": str(problem.statement)},
                ]

            return (
                dict(
                    problem=problem,
                    messages=[
                        {"role": "system", "content": cfg.system_prompt},
                        {"role": "user", "content": str(problem.statement)},
                    ],
                    text=best_text,
                    score=best_score,
                    reward=best_reward,
                    passed=solved,
                    # FIX 2 (Critical): restored sft_hint_* fields
                    sft_hint_messages=best_hint_messages,
                    sft_hint_text=sft_hint_completion,
                    # direct pair unchanged
                    sft_direct_messages=sft_messages_direct,
                    sft_direct_text=sft_direct_completion,
                ),
                local_hints,
                int(solved),
            )

        results = await asyncio.gather(*[_refine_one(rec) for rec in candidates])
        pbar.close()

        refined, hints_given, passed_after_hint = [], 0, 0
        for result, h, p in results:
            refined.append(result)
            hints_given += h
            passed_after_hint += p
        return refined, hints_given, passed_after_hint


    def _build_failed_cache(self, current: list[dict]) -> list[dict]:
        by_problem: dict[int, list[dict]] = {}
        for r in current:
            by_problem.setdefault(id(r["problem"]), []).append(r)

        cache: list[dict] = []
        for rows in by_problem.values():
            passed = [r for r in rows if r["passed"]]
            peer = (
                str(max(passed, key=lambda x: x["reward"])["text"])
                if passed
                else None
            )
            for r in rows:
                if not r["passed"]:
                    cache.append(
                        dict(
                            problem=r["problem"],
                            text=r["text"],
                            score=r["score"],
                            reward=r["reward"],
                            peer_solution=peer,
                        )
                    )
        return cache


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

    async def _generate_cot_reasoning(
        self,
        *,
        problem,
        answer_code: str,
        teacher_client,
        sem: asyncio.Semaphore,
    ) -> Optional[str]:
        cfg = self.cfg
        profile = self.profile

        cot_client = AgentClient(
            base_url=cfg.teacher_base_url or cfg.engine_base_url,
            api_key=cfg.teacher_api_key or cfg.engine_api_key,
            temperature=cfg.teacher_temperature,
            max_output_tokens=cfg.teacher_max_tokens,
            system_prompt=COT_SYSTEM_PROMPT,
            model=cfg.teacher_model_name or cfg.model_path,
            tools=teacher_client.tools,
            max_turns=cfg.teacher_max_turns,
            extra_body=profile.teacher_extra_body,
        )

        cot_prompt = (
            f"Problem:\n{problem.statement}\n\n"
            f"Correct C solution:\n{answer_code}\n\n"
            "Using your tools, research the APIs, theory, and implementation "
            "details relevant to this solution. Then write the internal reasoning "
            "an expert would have had when arriving at this exact solution. "
            "Plain text only, no code blocks, no markdown. "
            "You have the tools, dont make guesses, be deterministic, but no need to "
            "understand everything, dont write about something you dont know. "
            "Bad example (DONT DO THIS!): 'The function is likely part of PMF (process management library)' "
            "– this means you dont know whats happening, better to not write this. "
            "Minimally understand about the functions you are working with and write. "
            "Bad example: 'The expert programmer begins by analyzing...' "
            "Good example: 'The user asked for ...(details in brief)... so i need to use ... function and do ...' "
            "It should be very short, a paragraph with a few lines is enough, be concise and minimal, "
            "around 100-200 words is enough"
        )

        try:
            async with sem:
                _, reasoning = await cot_client.run(prompt=cot_prompt)
            if not reasoning:
                return None
            reasoning = reasoning.strip()

            # If >1500 chars, summarize using LangChain OpenAI with same URL/key/model
            if len(reasoning) > 1500:
                summarizer = ChatOpenAI(
                    model=cfg.teacher_model_name or cfg.model_path,
                    base_url=cfg.teacher_base_url or cfg.engine_base_url,
                    api_key=cfg.teacher_api_key or cfg.engine_api_key,
                    temperature=0.5,
                )
                summary_prompt = (
                    "Summarize the following reasoning in under 1500 characters "
                    "keeping the key details, plain text only:\n\n" + reasoning
                )
                resp = summarizer.invoke([HumanMessage(content=summary_prompt)])
                summary = resp.content.strip() if resp else None
                if not summary or len(summary) > 1500:
                    return None
                return summary

            return reasoning

        except Exception as exc:
            tqdm.write(f"[cot-gen] warning: failed to generate COT reasoning: {exc}")
            return None

    def _write_cot_file(self, *, step: int, reasoning: str) -> None:
        log_dir = Path("gen_logs") / f"step_{step}"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        path = log_dir / f"cot_{ts}.md"
        with path.open("w", encoding="utf-8") as f:
            f.write("<think>\n")
            f.write(reasoning.strip())
            f.write("\n</think>\n")

# endregion Pipeline

# TODO: Move this to util
def _log_rollout_stats(tag: str, rows: list[dict], extra: str = "") -> None:
    if not rows:
        return
    ratios = [
        (float(r["score"].passed) / float(r["score"].total))
        if r["score"].total > 0
        else 0.0
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
    tqdm.write(" ".join(parts))
