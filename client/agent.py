from __future__ import annotations

import json
from typing import Any, Optional, Type

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool as LangChainTool
from pydantic import BaseModel

from .chat import ChatClient


OutputModel = Optional[Type[BaseModel]]
RunOutput = tuple[str, str] | tuple[str, BaseModel]


class AgentClient(ChatClient):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        temperature: float,
        max_output_tokens: int,
        system_prompt: str,
        model: str,
        tools: list[LangChainTool],
        max_turns: int = 8,
    ) -> None:
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            system_prompt=system_prompt,
            model=model,
        )
        self.langchain_tools = list(tools)
        self.tool_map = {tool.name: tool for tool in self.langchain_tools}
        self.tools: list[dict[str, Any]] = []
        for tool in self.langchain_tools:
            args_schema = getattr(tool, "args_schema", None)
            if args_schema is not None and hasattr(args_schema, "model_json_schema"):
                params_schema = args_schema.model_json_schema()
            else:
                params_schema = {"type": "object", "properties": {}}
            self.tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": params_schema,
                    },
                }
            )
        self.max_turns = int(max_turns)

    async def run(
        self,
        prompt: str,
        output_model: OutputModel = None,
    ) -> RunOutput:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        self.message_history.append(HumanMessage(content=prompt))
        final_reasoning = ""

        for _ in range(self.max_turns):
            payload_messages: list[dict[str, Any]] = []
            for m in self.message_history:
                if isinstance(m, SystemMessage):
                    payload_messages.append({"role": "system", "content": str(m.content)})
                elif isinstance(m, HumanMessage):
                    payload_messages.append({"role": "user", "content": str(m.content)})
                elif isinstance(m, ToolMessage):
                    payload_messages.append(
                        {
                            "role": "tool",
                            "content": str(m.content),
                            "tool_call_id": m.tool_call_id,
                        }
                    )
                elif isinstance(m, AIMessage):
                    item: dict[str, Any] = {"role": "assistant", "content": str(m.content)}
                    tool_calls = m.additional_kwargs.get("tool_calls") if isinstance(m.additional_kwargs, dict) else None
                    if tool_calls:
                        item["tool_calls"] = tool_calls
                    payload_messages.append(item)

            response = await self.llm.async_client.create(
                model=self.model,
                messages=payload_messages,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
                stream=False,
                tools=self.tools or None,
                tool_choice="auto" if self.tools else None,
                extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            )

            choice = (getattr(response, "choices", None) or [None])[0]
            message = getattr(choice, "message", None)
            assistant_text = str(getattr(message, "content", "") or "")
            final_reasoning = str(getattr(message, "reasoning", "") or "")
            if not final_reasoning:
                final_reasoning = str(getattr(message, "reasoning_content", "") or "")
            if not final_reasoning:
                final_reasoning = self._extract_reasoning(
                    AIMessage(
                        content=assistant_text,
                        additional_kwargs=getattr(message, "model_extra", {}) or {},
                    )
                )

            raw_tool_calls = getattr(message, "tool_calls", None) or []
            if raw_tool_calls:
                self.message_history.append(
                    AIMessage(
                        content=assistant_text,
                        additional_kwargs={
                            "tool_calls": [
                                tc.model_dump() if hasattr(tc, "model_dump") else dict(tc) for tc in raw_tool_calls
                            ]
                        },
                    )
                )
                for tc in raw_tool_calls:
                    tc_id = getattr(tc, "id", "") or ""
                    fn = getattr(tc, "function", None)
                    fn_name = str(getattr(fn, "name", "") or "")
                    fn_args_raw = str(getattr(fn, "arguments", "") or "{}")

                    try:
                        fn_args = json.loads(fn_args_raw)
                        if not isinstance(fn_args, dict):
                            fn_args = {}
                    except Exception:
                        fn_args = {}

                    tool = self.tool_map.get(fn_name)
                    if tool is None:
                        tool_output = f"Unknown tool: {fn_name}"
                    else:
                        try:
                            result = (
                                await tool.ainvoke(fn_args)
                                if hasattr(tool, "ainvoke")
                                else tool.invoke(fn_args)
                            )
                            if isinstance(result, (dict, list)):
                                tool_output = json.dumps(result, ensure_ascii=False)
                            else:
                                tool_output = str(result)
                        except Exception as exc:
                            tool_output = f"Tool execution failed: {exc}"

                    self.message_history.append(ToolMessage(content=tool_output, tool_call_id=tc_id))
                continue

            self.message_history.append(AIMessage(content=assistant_text))
            if output_model is not None:
                try:
                    parsed = output_model.model_validate_json(assistant_text)
                except Exception:
                    parsed = output_model.model_validate(json.loads(assistant_text))
                return final_reasoning, parsed
            return final_reasoning, assistant_text

        self.message_history.append(AIMessage(content=""))
        if output_model is not None:
            raise RuntimeError("agent exceeded max_turns without final structured response")
        return final_reasoning, ""
