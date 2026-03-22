from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from inference.base import WeightSwapMode
from tasksets.base import Problem

from .base import BaseClient, ClientContext, ClientResult
from .tools.base import BaseTool, ToolResult


class AgentClient(BaseClient):
    def __init__(
        self,
        engine,
        tokenizer,
        system_prompt: str,
        tools: List[BaseTool],
        max_turns: int = 10,
        server_url: str = "http://localhost:30000",
        model_name: str = "default",
        default_max_completion_tokens: int = 4096,
        default_reasoning_effort: str | None = None,
    ) -> None:
        super().__init__(engine=engine, tokenizer=tokenizer)
        self.system_prompt = system_prompt
        self.tools = list(tools)
        self.max_turns = int(max_turns)
        self.server_url = server_url.rstrip("/")
        self.model_name = model_name
        self.default_max_completion_tokens = int(default_max_completion_tokens)
        self.default_reasoning_effort = default_reasoning_effort

        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "AgentClient requires the 'openai' Python SDK. "
                "Install with: pip install openai"
            ) from exc

        self.oai = OpenAI(base_url=f"{self.server_url}/v1", api_key="none")
        self.tools_schema = [t.to_openai_schema() for t in self.tools]
        self.tools_map = {t.name: t for t in self.tools}

    def _resolve_generation_options(self, extra: Dict[str, Any] | None) -> Dict[str, Any]:
        extra = extra or {}
        raw_max = extra.get("max_completion_tokens", extra.get("max_new_tokens", self.default_max_completion_tokens))
        max_completion_tokens = int(raw_max) if raw_max is not None else self.default_max_completion_tokens
        max_completion_tokens = max(1, max_completion_tokens)
        reasoning_effort = extra.get("reasoning_effort", self.default_reasoning_effort)
        return {
            "max_completion_tokens": max_completion_tokens,
            "reasoning_effort": reasoning_effort,
        }

    def _create_chat_completion(self, messages: List[Dict[str, Any]], generation_options: Dict[str, Any]):
        base_payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "tools": self.tools_schema,
            "tool_choice": "auto",
        }
        if generation_options.get("max_completion_tokens") is not None:
            base_payload["max_completion_tokens"] = int(generation_options["max_completion_tokens"])
        if generation_options.get("reasoning_effort"):
            base_payload["reasoning_effort"] = generation_options["reasoning_effort"]

        variants: List[Dict[str, Any]] = []
        variants.append(dict(base_payload))
        if "reasoning_effort" in base_payload:
            v = dict(base_payload)
            v.pop("reasoning_effort", None)
            variants.append(v)
        if "max_completion_tokens" in base_payload:
            v = dict(base_payload)
            v.pop("max_completion_tokens", None)
            variants.append(v)
        if "reasoning_effort" in base_payload and "max_completion_tokens" in base_payload:
            v = dict(base_payload)
            v.pop("reasoning_effort", None)
            v.pop("max_completion_tokens", None)
            variants.append(v)

        last_exc = None
        seen_keys = set()
        for payload in variants:
            key = tuple(sorted(payload.keys()))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            try:
                return self.oai.chat.completions.create(**payload)
            except Exception as exc:
                last_exc = exc
                continue
        raise last_exc  # type: ignore[misc]

    def build_messages(self, problem: Problem, context: ClientContext) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": problem.statement},
        ]

        if context.pass_number >= 2:
            if context.best_code:
                messages.append({"role": "assistant", "content": context.best_code})
            refine_msg = context.error_context or "Refine your previous answer and fix all issues."
            messages.append({"role": "user", "content": refine_msg})

        return messages

    def _execute_tool(self, name: str, arguments_json: str) -> ToolResult:
        tool = self.tools_map.get(name)
        if tool is None:
            return ToolResult(success=False, output=f"Unknown tool: {name}", metadata={"tool": name})

        try:
            args = json.loads(arguments_json or "{}")
            if not isinstance(args, dict):
                return ToolResult(
                    success=False,
                    output="Tool arguments must decode to a JSON object.",
                    metadata={"tool": name},
                )
        except Exception as exc:
            return ToolResult(
                success=False,
                output=f"Invalid JSON arguments: {exc}",
                metadata={"tool": name},
            )

        try:
            return tool.execute(args)
        except Exception as exc:
            return ToolResult(
                success=False,
                output=f"Tool execution failed: {exc}",
                metadata={"tool": name},
            )

    def _run_single_trajectory_from_messages(
        self,
        messages: List[Dict[str, Any]],
        generation_options: Dict[str, Any] | None = None,
    ) -> Tuple[str, List[int], Dict[str, Any], str, List[int]]:
        trajectory_log: List[Dict[str, Any]] = []
        prompt_text = self.apply_chat_template(messages)
        prompt_token_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        generation_options = generation_options or {}

        msg = None
        for turn in range(self.max_turns):
            response = self._create_chat_completion(messages, generation_options)
            msg = response.choices[0].message

            tool_calls = getattr(msg, "tool_calls", None) or []
            if tool_calls:
                assistant_message = {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
                messages.append(assistant_message)

                for tc in tool_calls:
                    result = self._execute_tool(tc.function.name, tc.function.arguments)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result.output,
                        }
                    )
                    trajectory_log.append(
                        {
                            "turn": turn,
                            "tool": tc.function.name,
                            "args": tc.function.arguments,
                            "success": result.success,
                            "tool_metadata": result.metadata,
                        }
                    )
                continue

            final_text = msg.content or ""
            token_ids = self.tokenizer.encode(final_text, add_special_tokens=False)
            metadata = {
                "turns": turn + 1,
                "trajectory": trajectory_log,
                "truncated": False,
            }
            return final_text, token_ids, metadata, prompt_text, prompt_token_ids

        final_text = ""
        if msg is not None:
            final_text = msg.content or ""
        token_ids = self.tokenizer.encode(final_text, add_special_tokens=False)
        metadata = {
            "turns": self.max_turns,
            "trajectory": trajectory_log,
            "truncated": True,
        }
        return final_text, token_ids, metadata, prompt_text, prompt_token_ids

    def _run_single_trajectory(
        self,
        problem: Problem,
        context: ClientContext,
    ) -> Tuple[str, List[int], Dict[str, Any], str, List[int]]:
        messages: List[Dict[str, Any]] = self.build_messages(problem, context)
        generation_options = self._resolve_generation_options(context.extra)
        return self._run_single_trajectory_from_messages(messages, generation_options=generation_options)

    def run_messages(
        self,
        messages: List[Dict[str, Any]],
        n: int = 1,
        generation_options: Dict[str, Any] | None = None,
    ) -> ClientResult:
        completions: List[str] = []
        token_ids: List[List[int]] = []
        trajectories: List[Dict[str, Any]] = []
        generation_options = generation_options or self._resolve_generation_options({})

        prompt_text = ""
        prompt_token_ids: List[int] = []
        for _ in range(n):
            text, toks, meta, ptxt, ptoks = self._run_single_trajectory_from_messages(
                list(messages),
                generation_options=generation_options,
            )
            completions.append(text)
            token_ids.append(toks)
            trajectories.append(meta)
            prompt_text = ptxt
            prompt_token_ids = ptoks

        return ClientResult(
            completions=completions,
            token_ids=token_ids,
            prompt_text=prompt_text,
            prompt_token_ids=prompt_token_ids,
            metadata={
                "client": "agent",
                "n": n,
                "max_turns": self.max_turns,
                "generation_options": generation_options,
                "trajectories": trajectories,
                "tool_names": [t.name for t in self.tools],
            },
        )

    def run(self, problem: Problem, context: ClientContext, n: int) -> ClientResult:
        if context.lora_path:
            self.engine.swap_weights(context.lora_path, WeightSwapMode.LORA)

        completions: List[str] = []
        token_ids: List[List[int]] = []
        trajectories: List[Dict[str, Any]] = []
        generation_options = self._resolve_generation_options(context.extra)

        prompt_text = ""
        prompt_token_ids: List[int] = []
        for _ in range(n):
            text, toks, meta, ptxt, ptoks = self._run_single_trajectory(problem, context)
            completions.append(text)
            token_ids.append(toks)
            trajectories.append(meta)
            prompt_text = ptxt
            prompt_token_ids = ptoks

        return ClientResult(
            completions=completions,
            token_ids=token_ids,
            prompt_text=prompt_text,
            prompt_token_ids=prompt_token_ids,
            metadata={
                "client": "agent",
                "n": n,
                "max_turns": self.max_turns,
                "generation_options": generation_options,
                "trajectories": trajectories,
                "tool_names": [t.name for t in self.tools],
            },
        )
