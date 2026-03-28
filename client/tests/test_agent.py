from __future__ import annotations

import asyncio
import os

from langchain_core.messages import ToolMessage

from client import AgentClient
from client.tools import build_markdown_rag_tool
from inference.vllm_engine import VLLMEngine


async def _run() -> None:
    base_url = os.environ.get("BASE_URL", "http://localhost:8000/v1")
    model_path = os.environ.get(
        "MODEL_PATH",
        "/media/blazingbhavneek/Common/Code/sglangServer/Infer/Qwen/Qwen3-1.7B",
    )
    docs_folder = "/media/blazingbhavneek/Common/Code/rl/sft_primer/input/intel/test.md"

    embedding_backend = os.environ.get("EMBEDDING_BACKEND", "huggingface")
    embedding_base_url = os.environ.get("EMBEDDING_BASE_URL", base_url)
    embedding_api_key = os.environ.get("EMBEDDING_API_KEY", "EMPTY")
    embedding_model = os.environ.get(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    print("=== Client Agent Test ===")
    print(f"base_url: {base_url}")
    print(f"model_path: {model_path}")
    print(f"docs_folder: {docs_folder}")

    engine = VLLMEngine(
        model_path=model_path,
        engine_kwargs={
            "base_url": base_url,
            "model_name": model_path,
            "api_key": "EMPTY",
            "save_vllm_logs": True,
            "gpu_memory_utilization": 0.90,
            "reasoning_parser": "qwen3",
            "enable_auto_tool_choice": True,
            "tool_call_parser": "hermes",
        },
    )

    try:
        print("\n[1] Starting engine...")
        await engine.start()
        await engine.init()
        print("Engine ready")

        print("\n[2] Building RAG tool...")
        rag_tool = build_markdown_rag_tool(
            docs_folder=docs_folder,
            persist_directory="logs/chroma_intel_rag",
            embedding_backend=embedding_backend,
            embedding_base_url=embedding_base_url,
            embedding_api_key=embedding_api_key,
            embedding_model=embedding_model,
        )
        print("RAG tool ready")

        agent = AgentClient(
            base_url=base_url,
            api_key="EMPTY",
            temperature=0.0,
            max_output_tokens=512,
            system_prompt=(
                "You are an agent. Always call search_knowledge_base first before answering. "
                "Use retrieved context and keep the final answer concise."
            ),
            model=model_path,
            tools=[rag_tool],
            max_turns=6,
        )

        print("\n[3] Agent run")
        reasoning, output = await agent.run(
            "Based on the intel primer markdown, summarize what it says about oneAPI/SYCL "
            "installation and beginner documentation."
        )
        print(f"reasoning: {repr(reasoning)}")
        print(f"output: {repr(output)}")

        tool_msgs = [m for m in agent.message_history if isinstance(m, ToolMessage)]
        print(f"tool_messages_count: {len(tool_msgs)}")
        if tool_msgs:
            print("last_tool_message_preview:")
            print(str(tool_msgs[-1].content)[:800])

        assert isinstance(output, str)

        print("\nPASS: agent tool-call + rag test")
    finally:
        print("\n[4] Shutting down engine...")
        await engine.shutdown()
        print("Engine shutdown complete")


if __name__ == "__main__":
    asyncio.run(_run())
