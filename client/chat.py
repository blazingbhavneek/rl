from __future__ import annotations

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
        max_output_tokens: Optional[int],
        system_prompt: str,
        model: str,
    ) -> None:
        self.model = model
        self.temperature = float(temperature)
        self.max_output_tokens = (
            int(max_output_tokens)
            if max_output_tokens is not None and int(max_output_tokens) > 0
            else None
        )
        self.system_prompt = system_prompt
        llm_kwargs: dict[str, object] = {
            "model": model,
            "api_key": api_key,
            "base_url": base_url,
            "temperature": self.temperature,
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": True,
                }
            },
        }
        if self.max_output_tokens is not None:
            llm_kwargs["max_tokens"] = self.max_output_tokens
        self.llm = ChatOpenAI(**llm_kwargs)
        self.message_history: list[BaseMessage] = [SystemMessage(content=system_prompt)]

    def reset_history(self, system_prompt: Optional[str] = None) -> None:
        if system_prompt is not None:
            self.system_prompt = system_prompt
        self.message_history = [SystemMessage(content=self.system_prompt)]

    def build_messages(self, prompt: str) -> list[BaseMessage]:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        return [*self.message_history, HumanMessage(content=prompt)]

    def _extract_reasoning(self, message: object) -> str:
        if message is None:
            return ""

        def _from_dict(d: object) -> str:
            if not isinstance(d, dict):
                return ""
            val = d.get("reasoning") or d.get("reasoning_content")
            return val.strip() if isinstance(val, str) else ""

        for key in ("reasoning", "reasoning_content"):
            val = getattr(message, key, None)
            if isinstance(val, str) and val.strip():
                return val.strip()
        for payload in (
            getattr(message, "additional_kwargs", None),
            getattr(message, "response_metadata", None),
            getattr(message, "model_extra", None),
        ):
            txt = _from_dict(payload)
            if txt:
                return txt
        return ""

    async def run(
        self,
        prompt: str,
        output_model: OutputModel = None,
        reasoning_effort: Optional[str] = None,
    ) -> RunOutput:
        messages = self.build_messages(prompt)
        user_msg = messages[-1]

        extra_body: dict[str, object] = {"chat_template_kwargs": {"enable_thinking": True}}
        request_kwargs: dict[str, object] = {}
        if reasoning_effort:
            chat_kwargs = extra_body["chat_template_kwargs"]
            if isinstance(chat_kwargs, dict):
                chat_kwargs["reasoning_effort"] = str(reasoning_effort)
            request_kwargs["reasoning_effort"] = str(reasoning_effort)
        llm_call = self.llm.bind(extra_body=extra_body)
        if reasoning_effort:
            llm_call = llm_call.bind(reasoning_effort=str(reasoning_effort))

        if output_model is not None:
            structured_llm = llm_call.with_structured_output(output_model)
            parsed = await structured_llm.ainvoke(messages)
            if not isinstance(parsed, BaseModel):
                raise ValueError("structured output parsing failed")
            self.message_history.append(user_msg)
            self.message_history.append(AIMessage(content=parsed.model_dump_json()))
            return "", parsed

        payload_messages = []
        for m in messages:
            role = "user"
            if isinstance(m, SystemMessage):
                role = "system"
            elif isinstance(m, AIMessage):
                role = "assistant"
            payload_messages.append({"role": role, "content": str(m.content)})

        request: dict[str, object] = {
            "model": self.model,
            "messages": payload_messages,
            "temperature": self.temperature,
            "stream": False,
            "extra_body": extra_body,
        }
        if self.max_output_tokens is not None:
            request["max_tokens"] = self.max_output_tokens
        request.update(request_kwargs)
        response = await self.llm.async_client.create(**request)
        choice = (getattr(response, "choices", None) or [None])[0]
        response_message = getattr(choice, "message", None)
        assistant_text = str(getattr(response_message, "content", "") or "")
        reasoning = self._extract_reasoning(response_message)

        self.message_history.append(user_msg)
        self.message_history.append(AIMessage(content=assistant_text))
        return reasoning, assistant_text
