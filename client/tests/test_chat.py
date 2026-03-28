from __future__ import annotations

import asyncio
import os

from pydantic import BaseModel

from client import ChatClient
from inference.vllm_engine import VLLMEngine


class ShortAnswer(BaseModel):
    answer: str


async def _run() -> None:
    base_url = os.environ.get("BASE_URL", "http://localhost:8000/v1")
    model_path = os.environ.get(
        "MODEL_PATH",
        "/media/blazingbhavneek/Common/Code/sglangServer/Infer/Qwen/Qwen3-1.7B",
    )

    print("=== Client Chat Test ===")
    print(f"base_url: {base_url}")
    print(f"model_path: {model_path}")

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
            "tool_call_parser": "qwen3_coder",
        },
    )

    try:
        print("\n[1] Starting engine...")
        await engine.start()
        await engine.init()
        print("Engine ready")

        client = ChatClient(
            base_url=base_url,
            api_key="EMPTY",
            temperature=0.0,
            max_output_tokens=256,
            system_prompt="You are a concise assistant.",
            model=model_path,
        )

        print("\n[2] Plain text run")
        reasoning_text, output_text = await client.run(
            "Solve this carefully: A bat and a ball cost $1.10 total. "
            "The bat costs $1.00 more than the ball. "
            "Think step by step first, then give only the ball price in one short line."
        )
        print(f"reasoning: {repr(reasoning_text)}")
        print(f"output: {repr(output_text)}")

        print("\n[3] Structured Pydantic run")
        reasoning_model, output_model = await client.run(
            "Solve carefully: If 3 workers finish a job in 6 days at the same rate, "
            "how many days would 2 workers take? Think first. "
            "Return JSON with key 'answer' only and value as a short string (example: '9 days').",
            output_model=ShortAnswer,
        )
        print(f"reasoning: {repr(reasoning_model)}")
        print(f"output model: {output_model}")

        assert isinstance(output_text, str)
        assert isinstance(output_model, ShortAnswer)

        print("\nPASS: client chat text + structured output")
    finally:
        print("\n[4] Shutting down engine...")
        await engine.shutdown()
        print("Engine shutdown complete")


if __name__ == "__main__":
    asyncio.run(_run())
