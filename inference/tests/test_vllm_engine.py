import asyncio
import json
from pathlib import Path

from inference.vllm_engine import VLLMEngine

BASE_URL = "http://localhost:8000/v1"
MODEL = "/media/blazingbhavneek/Common/Code/sglangServer/Infer/Qwen/Qwen3-1.7B"
LORA_ROOT = (
    "/media/blazingbhavneek/Common/Code/rl/sft_primer/output/intel/lora/"
    "media_blazingbhavneek_Common_Code_sglangServer_Infer_Qwen_Qwen3-1.7B"
)
LORA_ADAPTERS = [
    ("intel_sft_epoch_1", f"{LORA_ROOT}/epoch_1"),
    ("intel_sft_epoch_3", f"{LORA_ROOT}/epoch_3"),
    # ("intel_sft_final",   f"{LORA_ROOT}/final"),
]
LORA_EVAL_PROMPT = "SYCLの簡単な配列初期化サンプルで `constexpr int num` の値はいくつ？数字だけ答えて。"


async def chat(
    engine: VLLMEngine, prompt: str, thinking: bool = True
) -> tuple[str, str]:
    resp = await engine._request_json(
        "POST",
        "/chat/completions",
        {
            "model": engine.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": thinking},
        },
    )
    msg = (resp.get("choices") or [{}])[0].get("message") or {}
    content = (msg.get("content") or "").strip()
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    return content, reasoning


async def test_lifecycle(engine: VLLMEngine) -> None:
    print("\n--- lifecycle ---")

    content, reasoning = await chat(
        engine, "What is AI? Think carefully first.", thinking=True
    )
    print(
        f"[thinking=on]  reasoning={repr(reasoning[:80])}  content={repr(content[:80])}"
    )
    assert content or reasoning

    content, _ = await chat(engine, "What is 2+2? One short line.", thinking=False)
    print(f"[thinking=off] content={repr(content)}")
    assert content

    print("[lifecycle] ok")


async def test_sleep_wake(engine: VLLMEngine) -> None:
    print("\n--- sleep/wake ---")
    try:
        await engine.sleep(level=1)
        assert await engine.is_sleeping()
        print("[sleep=1] ok")

        await engine.wake()
        assert not await engine.is_sleeping()
        print("[wake] ok")

        await engine.sleep(level=2)
        assert await engine.is_sleeping()
        print("[sleep=2] ok")

        await engine.wake()
    except Exception as exc:
        print(f"[sleep/wake] not available: {exc}")


async def test_lora_swap(engine: VLLMEngine) -> None:
    print("\n--- lora swap ---")
    for name, path in LORA_ADAPTERS:
        assert Path(path).is_dir(), f"adapter not found: {path}"

        await engine.swap_lora_adapter(lora_name=name, lora_path=path)
        models = await engine._request_json("GET", "/models")
        ids = [str(r.get("id", "")) for r in (models.get("data") or [])]
        assert any(name in i for i in ids), f"adapter not visible after load: {name}"

        content, _ = await chat(engine, LORA_EVAL_PROMPT, thinking=False)
        print(f"[{name}] {repr(content[:80])}")
        assert content

        await engine.swap_lora_adapter(lora_name=name)
        models = await engine._request_json("GET", "/models")
        ids = [str(r.get("id", "")) for r in (models.get("data") or [])]
        assert not any(
            name in i for i in ids
        ), f"adapter still loaded after unload: {name}"
        print(f"[{name}] unloaded ok")


async def main() -> None:
    engine = VLLMEngine(
        model_path=MODEL,
        engine_kwargs={
            "base_url": BASE_URL,
            "model_name": MODEL,
            "api_key": "EMPTY",
            "save_vllm_logs": True,
            "gpu_memory_utilization": 0.90,
            "reasoning_parser": "qwen3",
            "enable_auto_tool_choice": True,
            "tool_call_parser": "hermes",
            "enable_lora": True,
            "enable_runtime_lora_updating": True,
            "max_loras": 2,
            "max_lora_rank": 256,
            "max_cpu_loras": 4,
        },
    )
    try:
        await engine.start()
        await engine.init()
        print("engine ready")

        await test_lifecycle(engine)
        await test_lora_swap(engine)
        await test_sleep_wake(engine)

        print("\nPASS")
    finally:
        await engine.shutdown()
        print("engine shutdown")


if __name__ == "__main__":
    asyncio.run(main())
