from __future__ import annotations

from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool as LangChainTool
from langchain_openai import ChatOpenAI

from .base import BaseClient, OutputModel, RunOutput


class AgentClient(BaseClient):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        temperature: float,
        max_output_tokens: Optional[int],
        system_prompt: str,
        model: str,
        tools: list[LangChainTool],
        max_turns: int = 8,
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

    def build_messages(self, prompt: str) -> list:
        if not prompt or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        return [*self.message_history, HumanMessage(content=prompt)]

    async def run(self, prompt: str, output_model: OutputModel = None) -> RunOutput:
        self.message_history.append(HumanMessage(content=prompt))

        for _ in range(self.max_turns):
            ai_message = await self.llm.ainvoke(self.message_history)
            self.message_history.append(ai_message)

            if not ai_message.tool_calls:
                content = str(ai_message.content)
                if output_model is not None:
                    return "", output_model.model_validate_json(content)
                return "", content

            for tool_call in ai_message.tool_calls:
                tool = self.tool_map.get(tool_call["name"])
                result = (
                    await tool.ainvoke(tool_call["args"])
                    if tool
                    else f"Unknown tool: {tool_call['name']}"
                )
                self.message_history.append(
                    ToolMessage(content=str(result), tool_call_id=tool_call["id"])
                )

        return "", ""
