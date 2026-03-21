from __future__ import annotations

import logging
from typing import Dict, List

from inference.base import SamplingParams, WeightSwapMode
from tasksets.base import Problem

from .base import BaseClient, ClientContext, ClientResult

log = logging.getLogger(__name__)


class SimpleTurnClient(BaseClient):
    def __init__(
        self,
        engine,
        tokenizer,
        system_prompt: str,
        default_max_new_tokens: int = 2048,
        default_temperature: float = 0.8,
        default_top_p: float = 0.95,
        max_context_tokens: int = 8192,
        min_new_tokens: int = 16,
        context_safety_margin: int = 64,
    ) -> None:
        super().__init__(engine=engine, tokenizer=tokenizer)
        self.system_prompt = system_prompt
        self.default_max_new_tokens = int(default_max_new_tokens)
        self.default_temperature = float(default_temperature)
        self.default_top_p = float(default_top_p)
        self.max_context_tokens = int(max_context_tokens)
        self.min_new_tokens = int(min_new_tokens)
        self.context_safety_margin = int(max(0, context_safety_margin))

    @staticmethod
    def _is_context_limit_error(exc: Exception) -> bool:
        s = str(exc).lower()
        return (
            "maximum context length" in s
            or "requested token count exceeds" in s
            or "context length" in s
        )

    def build_messages(self, problem: Problem, context: ClientContext) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": problem.statement},
        ]

        if context.pass_number >= 2:
            if context.best_code:
                messages.append({"role": "assistant", "content": context.best_code})
            refine_msg = context.error_context or "Refine your previous answer and fix all issues."
            messages.append({"role": "user", "content": refine_msg})

        return messages

    def run(self, problem: Problem, context: ClientContext, n: int) -> ClientResult:
        if context.lora_path:
            self.engine.swap_weights(context.lora_path, WeightSwapMode.LORA)

        messages = self.build_messages(problem, context)
        prompt = self.apply_chat_template(messages)

        requested_max_new_tokens = int(
            context.extra.get("max_new_tokens", self.default_max_new_tokens)
        )
        temperature = float(context.extra.get("temperature", self.default_temperature))
        top_p = float(context.extra.get("top_p", self.default_top_p))
        stop = list(context.extra.get("stop", []))
        prompt_token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        # Auto-clamp completion length to fit model context window:
        # prompt_tokens + max_new_tokens <= max_context_tokens.
        available_for_completion = (
            self.max_context_tokens - len(prompt_token_ids) - self.context_safety_margin
        )
        max_new_tokens = max(1, min(requested_max_new_tokens, available_for_completion))
        if max_new_tokens < requested_max_new_tokens:
            log.warning(
                "Clamping max_new_tokens from %s to %s (prompt_tokens=%s, max_context=%s, margin=%s)",
                requested_max_new_tokens,
                max_new_tokens,
                len(prompt_token_ids),
                self.max_context_tokens,
                self.context_safety_margin,
            )
        if max_new_tokens < self.min_new_tokens:
            log.warning(
                "Very low completion budget (%s tokens) after context clamp; prompt may be too long.",
                max_new_tokens,
            )

        outputs = None
        cur_max_new = int(max_new_tokens)
        retries = 0
        while True:
            params = SamplingParams(
                max_new_tokens=cur_max_new,
                temperature=temperature,
                n=n,
                top_p=top_p,
                stop=stop,
            )
            try:
                outputs = self.engine.generate_batch([prompt], params)[0]
                break
            except Exception as exc:
                if not self._is_context_limit_error(exc):
                    raise
                retries += 1
                next_max = max(1, cur_max_new - max(32, cur_max_new // 10))
                if next_max >= cur_max_new or next_max <= 1:
                    raise
                log.warning(
                    "Context-limit rejection from engine; reducing max_new_tokens %s -> %s (retry=%s)",
                    cur_max_new,
                    next_max,
                    retries,
                )
                cur_max_new = next_max

        completions = [o.text for o in outputs]
        token_ids = [list(o.token_ids) for o in outputs]

        metadata = {
            "client": "simple",
            "pass_number": context.pass_number,
            "n": n,
            "generation_params": {
                "max_new_tokens": max_new_tokens,
                "max_new_tokens_effective": cur_max_new,
                "temperature": temperature,
                "top_p": top_p,
                "stop": stop,
            },
            "prompt_tokens": len(prompt_token_ids),
        }

        return ClientResult(
            completions=completions,
            token_ids=token_ids,
            prompt_text=prompt,
            prompt_token_ids=prompt_token_ids,
            metadata=metadata,
        )
