from __future__ import annotations

import json
from typing import Optional, Type

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from .base import BaseClient


OutputModel = Optional[Type[BaseModel]]
RunOutput = tuple[str, str] | tuple[str, BaseModel]


class ChatClient(BaseClient):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        temperature: float,
        max_output_tokens: int,
        system_prompt: str,
        model: str,
    ) -> None:
        self.model = model
        self.temperature = float(temperature)
        self.max_output_tokens = int(max_output_tokens)
        self.system_prompt = system_prompt
        self.llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": True,
                }
            },
        )
        self.message_history: list[BaseMessage] = [SystemMessage(content=system_prompt)]

    def reset_history(self, system_prompt: Optional[str] = None) -> None:
        if system_prompt is not None:
            self.system_prompt = system_prompt
        self.message_history = [SystemMessage(content=self.system_prompt)]

    def build_messages(self, prompt: str) -> list[BaseMessage]:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        return [*self.message_history, HumanMessage(content=prompt)]

    def _extract_reasoning(self, message: AIMessage) -> str:
        def _normalize(value: object) -> str:
            if isinstance(value, str):
                return value.strip()
            if isinstance(value, list):
                parts: list[str] = []
                for item in value:
                    if isinstance(item, str):
                        txt = item.strip()
                        if txt:
                            parts.append(txt)
                    elif isinstance(item, dict):
                        for k in ("reasoning", "reasoning_content", "text", "content"):
                            v = item.get(k)
                            if isinstance(v, str) and v.strip():
                                parts.append(v.strip())
                return "\n".join(parts).strip()
            if isinstance(value, dict):
                for k in ("reasoning", "reasoning_content", "text", "content"):
                    v = value.get(k)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
            return ""

        for payload in (message.additional_kwargs, message.response_metadata):
            if isinstance(payload, dict):
                for key in ("reasoning", "reasoning_content"):
                    txt = _normalize(payload.get(key))
                    if txt:
                        return txt

        return ""

    async def run(
        self,
        prompt: str,
        output_model: OutputModel = None,
    ) -> RunOutput:
        messages = self.build_messages(prompt)
        user_msg = messages[-1]
        payload_messages = []
        for m in messages:
            role = "user"
            if isinstance(m, SystemMessage):
                role = "system"
            elif isinstance(m, AIMessage):
                role = "assistant"
            payload_messages.append({"role": role, "content": str(m.content)})

        response = await self.llm.async_client.create(
            model=self.model,
            messages=payload_messages,
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
            stream=False,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )

        choice = (getattr(response, "choices", None) or [None])[0]
        message = getattr(choice, "message", None)
        assistant_text = str(getattr(message, "content", "") or "")

        reasoning = str(getattr(message, "reasoning", "") or "")
        if not reasoning:
            reasoning = str(getattr(message, "reasoning_content", "") or "")
        if not reasoning:
            extra = getattr(message, "model_extra", None) or {}
            if isinstance(extra, dict):
                reasoning = str(extra.get("reasoning") or extra.get("reasoning_content") or "")

        if output_model is not None:
            parsed: BaseModel
            try:
                parsed = output_model.model_validate_json(assistant_text)
            except Exception:
                parsed = output_model.model_validate(json.loads(assistant_text))
            self.message_history.append(user_msg)
            self.message_history.append(AIMessage(content=assistant_text))
            return reasoning, parsed

        self.message_history.append(user_msg)
        self.message_history.append(AIMessage(content=assistant_text))
        return reasoning, assistant_text
