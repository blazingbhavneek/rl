from __future__ import annotations

from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from .base import BaseClient, OutputModel, RunOutput


class ChatClient(BaseClient):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        temperature: float,
        max_output_tokens: Optional[int],
        system_prompt: str,
        model: str,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=float(temperature),
            max_tokens=int(max_output_tokens) if max_output_tokens else None,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )
        self.reset_history()

    def reset_history(self, system_prompt: Optional[str] = None) -> None:
        if system_prompt is not None:
            self.system_prompt = system_prompt
        self.message_history = [SystemMessage(content=self.system_prompt)]

    def build_messages(self, prompt: str) -> list:
        if not prompt or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        return [*self.message_history, HumanMessage(content=prompt)]

    async def run(
        self,
        prompt: str,
        output_model: OutputModel = None,
        reasoning_effort: Optional[str] = None,
    ) -> RunOutput:
        messages = self.build_messages(prompt)

        llm = self.llm
        if reasoning_effort:
            llm = llm.bind(reasoning_effort=reasoning_effort)

        if output_model is not None:
            response: BaseModel = await llm.with_structured_output(output_model).ainvoke(messages)
            self.message_history += [messages[-1], AIMessage(content=response.model_dump_json())]
            return "", response

        # ChatOpenAI silently drops reasoning_content during message conversion
        # (langchain-ai/langchain#34706) — use the raw openai client to preserve it
        payload = [
            {"role": "system" if isinstance(m, SystemMessage) else
                     "assistant" if isinstance(m, AIMessage) else "user",
             "content": str(m.content)}
            for m in messages
        ]
        raw = await self.llm.async_client.create(
            model=self.model,
            messages=payload,
            temperature=self.llm.temperature,
            max_tokens=self.llm.max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )
        msg = raw.choices[0].message
        content = str(msg.content or "")
        reasoning = str(getattr(msg, "reasoning_content", "") or
                        getattr(msg, "reasoning", "") or "")

        self.message_history += [messages[-1], AIMessage(content=content)]
        return reasoning, content
