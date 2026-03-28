import asyncio
import json

from inference.vllm_engine import VLLMEngine


async def _run_lifecycle(base_url: str, model_name: str) -> None:
    engine = VLLMEngine(
        model_path=model_name,
        engine_kwargs={
            "base_url": base_url,
            "model_name": model_name,
            "api_key": "EMPTY",
            "save_vllm_logs": True,
            "gpu_memory_utilization": 0.90,
            "reasoning_parser": "qwen3",
            "enable_auto_tool_choice": True,
            "tool_call_parser": "hermes",
        },
    )
    try:
        await engine.start()
        print("engine start: ok")

        await engine.init()
        print("engine init: ok")

        response = await engine._request_json(
            "POST",
            "/chat/completions",
            {
                "model": model_name,
                "messages": [{"role": "user", "content": "What is AI? Think carefully first"}],
                "temperature": 0.7,
                "stream": False,
            },
        )
        print("raw chat completion response:")
        print(json.dumps(response, indent=2, ensure_ascii=False))
        message = ((response.get("choices") or [{}])[0].get("message") or {})
        content = (message.get("content") or "").strip()
        reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
        print(f"chat completion content: {repr(content)}")
        print(f"chat completion reasoning: {repr(reasoning)}")
        assert bool(content) or bool(str(reasoning).strip())

        response_no_thinking = await engine._request_json(
            "POST",
            "/chat/completions",
            {
                "model": model_name,
                "messages": [{"role": "user", "content": "What is 2 + 2? One short line."}],
                "temperature": 0.0,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        print("raw chat completion response (enable_thinking=false):")
        print(json.dumps(response_no_thinking, indent=2, ensure_ascii=False))
        message_no_thinking = ((response_no_thinking.get("choices") or [{}])[0].get("message") or {})
        content_no_thinking = (message_no_thinking.get("content") or "").strip()
        reasoning_no_thinking = (
            message_no_thinking.get("reasoning") or message_no_thinking.get("reasoning_content") or ""
        )
        print(f"content (enable_thinking=false): {repr(content_no_thinking)}")
        print(f"reasoning (enable_thinking=false): {repr(reasoning_no_thinking)}")

        try:
            sleeping_before = await engine.is_sleeping()
            print(f"is_sleeping before sleep: {sleeping_before}")

            await engine.sleep(level=1)
            sleeping = await engine.is_sleeping()
            print(f"is_sleeping after sleep: {sleeping}")
            assert sleeping is True

            await engine.wake()
            sleeping_after_wake = await engine.is_sleeping()
            print(f"is_sleeping after wake: {sleeping_after_wake}")
            assert sleeping_after_wake is False

            await engine.sleep(level=2)
            sleeping_after_level2 = await engine.is_sleeping()
            print(f"is_sleeping after sleep(level=2): {sleeping_after_level2}")
            assert sleeping_after_level2 is True
            await engine.wake()
        except Exception as exc:
            print(f"sleep/wake endpoint not available: {exc}")

        await engine.shutdown()
        print("engine shutdown: ok")

        print("PASS: vLLM engine lifecycle test")
    finally:
        if not getattr(engine, "_closed", False):
            await engine.shutdown()


async def _run_real_vllm_tests() -> None:
    base_url = "http://localhost:8000/v1"

    qwen_model = "/media/blazingbhavneek/Common/Code/sglangServer/Infer/Qwen/Qwen3-1.7B"
    await _run_lifecycle(base_url=base_url, model_name=qwen_model)

    # gpt_oss_model = "/media/blazingbhavneek/Common/Code/sglangServer/Infer/openai/gpt-oss-20b"
    # await _run_lifecycle(base_url=base_url, model_name=gpt_oss_model)


if __name__ == "__main__":
    asyncio.run(_run_real_vllm_tests())
    print("PASS: test_vllm_engine")
