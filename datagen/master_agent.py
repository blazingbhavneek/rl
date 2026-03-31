from __future__ import annotations
import asyncio
import json
import re
from typing import Optional
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool, BaseTool as LangChainTool
from langchain_openai import ChatOpenAI
from .agent_client import AgentClient
from .prompts import SPECIALIST_PROMPT


class MasterAgent:
    """
    Master agent that delegates all library lookups to disposable specialist
    sub-agents. Never touches MCP/RAG directly.

    Tool set exposed to the master LLM:
      - ask_specialists(questions: list[str]) — parallel specialist fan-out
      - any extra_tools passed in (e.g. verify_code)

    Optional sliding-window summarization for long-running masters (b3/b4).
    """

    def __init__(
        self,
        system_prompt: str,
        model: str,
        base_url: str,
        api_key: str,
        temperature: float = 0.7,
        max_output_tokens: Optional[int] = None,
        max_turns: int = 30,
        # Specialist config
        specialist_tools: list[LangChainTool] | None = None,
        specialist_model: str | None = None,
        specialist_base_url: str | None = None,
        specialist_api_key: str | None = None,
        specialist_temperature: float = 0.3,
        specialist_max_turns: int = 40,
        # Extra tools given directly to master (e.g. verify_code)
        extra_tools: list[LangChainTool] | None = None,
        # Summarization
        enable_summarization: bool = False,
        summarize_token_limit: int = 48000,
        summarize_target: int = 16000,
        extra_body: Optional[dict] = None,
    ):
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.enable_summarization = enable_summarization
        self.summarize_token_limit = summarize_token_limit
        self.summarize_target = summarize_target
        self.extra_body = extra_body or {}

        # Specialist config stored for sub-agent construction
        self.specialist_tools = specialist_tools or []
        self.specialist_model = specialist_model or model
        self.specialist_base_url = specialist_base_url or base_url
        self.specialist_api_key = specialist_api_key or api_key
        self.specialist_temperature = specialist_temperature
        self.specialist_max_turns = specialist_max_turns

        # ── Build ask_specialists tool as closure over self ───────────
        @tool
        async def ask_specialists(questions: list[str]) -> str:
            """Ask a list of questions to specialist agents who have full access
            to the library MCP server, RAG, and documentation.
            All questions are answered in parallel by separate specialists.
            Each specialist is thorough — expect detailed, structured answers.
            Use this whenever you need function signatures, error codes, types,
            preconditions, cleanup requirements, or any library detail.

            Args:
                questions: List of specific questions about the library.
                           Be precise — one focused question per item.
            """
            return await self._run_specialists(questions)

        self._ask_specialists_tool = ask_specialists

        master_tools: list[LangChainTool] = [ask_specialists]
        if extra_tools:
            master_tools.extend(extra_tools)

        self.tool_map = {t.name: t for t in master_tools}

        self.llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=float(temperature),
            max_tokens=int(max_output_tokens) if max_output_tokens else None,
            extra_body=self.extra_body,
        ).bind_tools(master_tools)

        # Separate LLM for summarization — no tools, low temp
        self.summarize_llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0.1,
            max_tokens=4096,
            extra_body=self.extra_body,
        )

        self.message_history: list = []
        self.reset_history()

    # ── History ────────────────────────────────────────────────────────

    def reset_history(self, system_prompt: Optional[str] = None) -> None:
        if system_prompt is not None:
            self.system_prompt = system_prompt
        self.message_history = [SystemMessage(content=self.system_prompt)]

    def _estimate_tokens(self) -> int:
        total = 0
        for m in self.message_history:
            content = m.content if isinstance(m.content, str) else json.dumps(m.content)
            total += len(content) // 3
        return total

    # ── Optional summarization ─────────────────────────────────────────

    async def maybe_summarize(self) -> None:
        if not self.enable_summarization:
            return
        if self._estimate_tokens() < self.summarize_token_limit:
            return

        system = self.message_history[0]
        rest = self.message_history[1:]

        keep_recent = 6
        if len(rest) <= keep_recent:
            return

        to_summarize = rest[:-keep_recent]
        recent = rest[-keep_recent:]

        formatted = []
        for m in to_summarize:
            role = type(m).__name__.replace("Message", "")
            content = m.content if isinstance(m.content, str) else json.dumps(m.content)
            if len(content) > 3000:
                content = content[:3000] + "\n...[truncated]"
            formatted.append(f"[{role}]: {content}")

        summary_prompt = (
            "Summarize the key findings from this research session.\n"
            "Be DENSE. Preserve ALL:\n"
            "- Function names and exact signatures\n"
            "- Error code names and exact values\n"
            "- Parameter types and correct ordering\n"
            "- Preconditions and cleanup requirements\n"
            "- Warnings, threading constraints, deprecated flags\n"
            "- Any code written and compiler errors encountered\n\n"
            + "\n".join(formatted)
        )

        summary_msg = await self.summarize_llm.ainvoke(
            [HumanMessage(content=summary_prompt)]
        )

        self.message_history = [
            system,
            SystemMessage(content=f"[PRIOR RESEARCH SUMMARY]\n{summary_msg.content}"),
            *recent,
        ]

    # ── Specialist delegation ──────────────────────────────────────────

    async def _run_single_specialist(self, question: str) -> str:
        agent = AgentClient(
            base_url=self.specialist_base_url,
            api_key=self.specialist_api_key,
            temperature=self.specialist_temperature,
            max_output_tokens=None,
            system_prompt=SPECIALIST_PROMPT,
            model=self.specialist_model,
            tools=self.specialist_tools,
            max_turns=self.specialist_max_turns,
            extra_body=self.extra_body,
        )
        try:
            _, answer = await agent.run(question)
            return f"Q: {question}\nA: {answer}"
        except Exception as e:
            return f"Q: {question}\nA: [SPECIALIST ERROR: {type(e).__name__}: {e}]"

    async def _run_specialists(self, questions: list[str]) -> str:
        results = await asyncio.gather(
            *[self._run_single_specialist(q) for q in questions],
            return_exceptions=True,
        )
        combined = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                combined.append(f"Q: {questions[i]}\nA: [ERROR: {result}]")
            else:
                combined.append(str(result))
        return "\n\n---\n\n".join(combined)

    # ── Main run loop ──────────────────────────────────────────────────

    async def run(
        self,
        prompt: str,
        output_model: Optional[type[BaseModel]] = None,
    ) -> tuple[str, any]:
        self.message_history.append(HumanMessage(content=prompt))

        for _ in range(self.max_turns):
            await self.maybe_summarize()

            ai_message = await self.llm.ainvoke(self.message_history)
            self.message_history.append(ai_message)

            if not ai_message.tool_calls:
                content = str(ai_message.content)
                if output_model is not None:
                    cleaned = self._clean_json(content)
                    try:
                        return content, output_model.model_validate_json(cleaned)
                    except Exception:
                        self.message_history.append(
                            HumanMessage(
                                content=(
                                    f"Your output was not valid JSON for the required schema.\n"
                                    f"Output ONLY valid JSON matching:\n"
                                    f"{json.dumps(output_model.model_json_schema(), indent=2)}"
                                )
                            )
                        )
                        continue
                return content, content

            for tc in ai_message.tool_calls:
                tool_fn = self.tool_map.get(tc["name"])
                try:
                    result = (
                        await tool_fn.ainvoke(tc["args"])
                        if tool_fn
                        else f"Unknown tool: {tc['name']}"
                    )
                except Exception as e:
                    result = f"Tool error: {type(e).__name__}: {e}"

                self.message_history.append(
                    ToolMessage(content=str(result), tool_call_id=tc["id"])
                )

        # Exhausted turns
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
