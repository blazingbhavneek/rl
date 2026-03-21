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
    ) -> None:
        super().__init__(engine=engine, tokenizer=tokenizer)
        self.system_prompt = system_prompt
        self.tools = list(tools)
        self.max_turns = int(max_turns)
        self.server_url = server_url.rstrip("/")
        self.model_name = model_name

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
    ) -> Tuple[str, List[int], Dict[str, Any], str, List[int]]:
        trajectory_log: List[Dict[str, Any]] = []
        prompt_text = self.apply_chat_template(messages)
        prompt_token_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)

        msg = None
        for turn in range(self.max_turns):
            response = self.oai.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=self.tools_schema,
                tool_choice="auto",
            )
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
        return self._run_single_trajectory_from_messages(messages)

    def run_messages(self, messages: List[Dict[str, Any]], n: int = 1) -> ClientResult:
        completions: List[str] = []
        token_ids: List[List[int]] = []
        trajectories: List[Dict[str, Any]] = []

        prompt_text = ""
        prompt_token_ids: List[int] = []
        for _ in range(n):
            text, toks, meta, ptxt, ptoks = self._run_single_trajectory_from_messages(list(messages))
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
                "trajectories": trajectories,
                "tool_names": [t.name for t in self.tools],
            },
        )
