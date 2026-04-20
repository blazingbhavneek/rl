import gc
import json
import tempfile
from pathlib import Path

import torch
from peft import PeftModel
from safetensors import safe_open
from transformers import AutoModelForCausalLM

from model.config import ModelConfig
from model.gemma4 import Gemma4Model

DEFAULT_GEMMA4_MODEL_PATH = (
    "/media/blazingbhavneek/Common/Code/sglangServer/Infer/google/gemma-4-E2B-it"
)
DEFAULT_GEMMA4_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _build_config() -> ModelConfig:
    return ModelConfig(
        lora=DEFAULT_GEMMA4_LORA_TARGETS,
        lora_fraction=0.25,
        lora_rank=128,
        lora_alpha=256,
        chunk_size=64,
        cuda_device_index=0,
        use_grad_checkpoint=False,
    )


def _assert_clean_adapter(adapter_dir: Path) -> None:
    with (adapter_dir / "adapter_config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)

    target_modules = config.get("target_modules")
    assert isinstance(target_modules, list) and target_modules
    assert all("." not in str(target) for target in target_modules)
    assert config.get("layers_to_transform")
    assert config.get("layers_pattern") == "layers"

    with safe_open(
        str(adapter_dir / "adapter_model.safetensors"), framework="pt", device="cpu"
    ) as handle:
        keys = list(handle.keys())
    assert keys
    assert all("base_layer" not in key for key in keys)
    assert all("base_attn" not in key for key in keys)


def _clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_gemma4_lora_adapter_compat(model_path: str = DEFAULT_GEMMA4_MODEL_PATH) -> None:
    with tempfile.TemporaryDirectory(prefix="gemma4_lora_compat_") as tmp:
        adapter_dir = Path(tmp) / "student"

        print("[compat] building fresh Gemma4 LoRA")
        model = Gemma4Model(
            model_path=model_path,
            config=_build_config(),
            lora_adapter_name="student",
        )
        model.save_lora_adapter("student", str(adapter_dir))
        _assert_clean_adapter(adapter_dir)
        del model
        _clear_cuda()

        print("[compat] loading adapter through Gemma4Model constructor")
        loaded = Gemma4Model(
            model_path=model_path,
            config=_build_config(),
            lora_path=str(adapter_dir),
            lora_adapter_name="student",
        )
        adapters = list(getattr(loaded.model, "peft_config", {}).keys())
        assert adapters == ["student"]
        active = getattr(loaded.model, "active_adapter", None)
        if isinstance(active, (list, tuple)):
            active = active[0] if active else None
        assert active == "student"
        assert any(
            param.requires_grad
            for name, param in loaded.model.named_parameters()
            if "lora_" in name
        )
        del loaded
        _clear_cuda()

        print("[compat] loading saved adapter with vanilla PEFT")
        kwargs = {
            "dtype": torch.bfloat16,
            "attn_implementation": "eager",
        }
        if torch.cuda.is_available():
            kwargs["device_map"] = {"": "cuda:0"}
        base_model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        assert list(peft_model.peft_config.keys()) == ["default"]
        del peft_model
        _clear_cuda()


if __name__ == "__main__":
    run_gemma4_lora_adapter_compat()
    print("PASS: gemma4 lora adapter compatibility")
