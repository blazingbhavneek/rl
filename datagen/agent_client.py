from __future__ import annotations

import re
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool as LangChainTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

RunOutput = tuple[str, any]


class AgentClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        temperature: float,
        max_output_tokens: Optional[int],
        system_prompt: str,
        model: str,
        tools: list[LangChainTool],
        max_turns: int = 40,
        extra_body: Optional[dict] = None,
    ) -> None:
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.tool_map = {tool.name: tool for tool in tools}

        self.llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=float(temperature),
            max_tokens=int(max_output_tokens) if max_output_tokens else None,
            extra_body=extra_body or {},
        ).bind_tools(tools)

        self.reset_history()

    def reset_history(self, system_prompt: Optional[str] = None) -> None:
        if system_prompt is not None:
            self.system_prompt = system_prompt
        self.message_history = [SystemMessage(content=self.system_prompt)]

    async def run(
        self, prompt: str, output_model: Optional[type[BaseModel]] = None
    ) -> RunOutput:
        if not prompt or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        self.message_history.append(HumanMessage(content=prompt))

        for _ in range(self.max_turns):
            ai_message = await self.llm.ainvoke(self.message_history)
            self.message_history.append(ai_message)

            if not ai_message.tool_calls:
                content = str(ai_message.content)
                if output_model is not None:
                    cleaned = self._clean_json(content)
                    try:
                        return content, output_model.model_validate_json(cleaned)
                    except Exception:
                        # Ask model to fix its JSON
                        self.message_history.append(
                            HumanMessage(
                                content=(
                                    f"Your output was not valid JSON. "
                                    f"Output ONLY valid JSON matching this schema:\n"
                                    f"{output_model.model_json_schema()}"
                                )
                            )
                        )
                        continue
                return content, content

            for tool_call in ai_message.tool_calls:
                tool_fn = self.tool_map.get(tool_call["name"])
                try:
                    result = (
                        await tool_fn.ainvoke(tool_call["args"])
                        if tool_fn
                        else f"Unknown tool: {tool_call['name']}"
                    )
                except Exception as e:
                    result = f"Tool error: {type(e).__name__}: {e}"

                self.message_history.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call["id"],
                    )
                )

        # Exhausted turns — return last AI content
        last_content = ""
        for m in reversed(self.message_history):
            if isinstance(m, AIMessage) and m.content:
                last_content = str(m.content)
                break
        return last_content, last_content

    @staticmethod
    def _clean_json(text: str) -> str:
        text = text.strip()
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text
