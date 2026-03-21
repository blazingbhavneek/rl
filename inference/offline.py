from __future__ import annotations

import logging
import argparse
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional

try:
    from .base import BaseEngine, GenerationOutput, SamplingParams, WeightSwapMode
except ImportError:  # pragma: no cover - direct script execution fallback
    from base import BaseEngine, GenerationOutput, SamplingParams, WeightSwapMode

log = logging.getLogger(__name__)


class SGLangOfflineEngine(BaseEngine):
    def __init__(
        self,
        model_path: str,
        *,
        tokenizer_path: Optional[str] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        initial_lora_path: Optional[str] = None,
        health_prompt: str = "ping",
    ) -> None:
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.engine_kwargs = dict(engine_kwargs or {})
        # TODO: remove for other GPU architectures; Blackwell hybrid GDN currently
        # requires triton or trtllm_mha attention backend in SGLang.
        self.engine_kwargs.setdefault("attention_backend", "triton")
        self.health_prompt = health_prompt
        self._active_lora_path: Optional[str] = None
        self._supports_generate_lora_path: Optional[bool] = None
        self._boot_engine(model_path)
        if initial_lora_path:
            self.swap_weights(initial_lora_path, WeightSwapMode.LORA)

    def _boot_engine(
        self,
        model_path: str,
        *,
        engine_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        import sglang as sgl
        from transformers import AutoTokenizer

        kwargs = dict(self.engine_kwargs if engine_kwargs is None else engine_kwargs)
        self.engine = sgl.Engine(model_path=model_path, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.model_path = model_path

    def _reboot_with_static_lora(self, lora_path: str) -> None:
        # Fallback for environments where generate(..., lora_path=...) is unsupported.
        # Recreate engine with LoRA preloaded at startup.
        boot_kwargs = dict(self.engine_kwargs)
        boot_kwargs["enable_lora"] = True
        boot_kwargs["lora_paths"] = [lora_path]

        self.shutdown()
        self._supports_generate_lora_path = False
        self._active_lora_path = lora_path
        self._boot_engine(self.model_path, engine_kwargs=boot_kwargs)

    def _params_dict(self, params: SamplingParams) -> Dict[str, Any]:
        out = asdict(params)
        if not out.get("stop"):
            out.pop("stop", None)
        return out

    def _normalize_single_output(self, raw: Any, prompt: str) -> GenerationOutput:
        if isinstance(raw, dict):
            text = raw.get("text", "")
            token_ids = raw.get("token_ids") or []
        else:
            text = str(raw)
            token_ids = []
        if not token_ids and text:
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        prompt_tokens = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        return GenerationOutput(text=text, token_ids=token_ids, prompt_tokens=prompt_tokens)

    def _kwargs_for_generate(self, params: SamplingParams) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"sampling_params": self._params_dict(params)}
        if self._active_lora_path and self._supports_generate_lora_path:
            kwargs["lora_path"] = self._active_lora_path
        return kwargs

    def generate(self, prompt: str, params: SamplingParams) -> List[GenerationOutput]:
        return self.generate_batch([prompt], params)[0]

    def generate_batch(
        self, prompts: List[str], params: SamplingParams
    ) -> List[List[GenerationOutput]]:
        if not prompts:
            return []

        raw = self.engine.generate(prompts, **self._kwargs_for_generate(params))
        if not isinstance(raw, list):
            raw = [raw]

        # SGLang may return:
        # - List[dict] for single prompt and n outputs
        # - List[List[dict]] for batched prompts
        if raw and isinstance(raw[0], dict) and len(prompts) == 1:
            raw = [raw]

        if len(raw) != len(prompts):
            raise RuntimeError(
                f"Unexpected SGLang batch shape: prompts={len(prompts)} outputs={len(raw)}"
            )

        all_outputs: List[List[GenerationOutput]] = []
        for prompt, prompt_outputs in zip(prompts, raw):
            if not isinstance(prompt_outputs, list):
                prompt_outputs = [prompt_outputs]
            all_outputs.append(
                [self._normalize_single_output(item, prompt) for item in prompt_outputs]
            )
        return all_outputs

    def _probe_lora_path_once(self, lora_path: str) -> bool:
        if self._supports_generate_lora_path is not None:
            return self._supports_generate_lora_path
        try:
            probe = SamplingParams(max_new_tokens=1, temperature=0.0, n=1)
            self.engine.generate(
                "probe",
                sampling_params=self._params_dict(probe),
                lora_path=lora_path,
            )
            self._supports_generate_lora_path = True
        except TypeError as exc:
            if "lora_path" not in str(exc):
                raise
            self._supports_generate_lora_path = False
        except Exception:
            # If probe fails for non-signature reasons, treat support as unknown/unavailable.
            self._supports_generate_lora_path = False
        return self._supports_generate_lora_path

    def swap_weights(self, checkpoint_path: str, mode: WeightSwapMode) -> None:
        if mode == WeightSwapMode.LORA:
            if self._probe_lora_path_once(checkpoint_path):
                self._active_lora_path = checkpoint_path
                return
            log.warning(
                "SGLang generate() does not support runtime lora_path; "
                "falling back to engine reboot with static LoRA preload."
            )
            self._reboot_with_static_lora(checkpoint_path)
            return

        # FULL/COLD_START: prefer native update API, then restart fallback.
        updater = getattr(self.engine, "update_weights", None)
        if callable(updater):
            updater(checkpoint_path)
            self.model_path = checkpoint_path
            self._active_lora_path = None
            return

        self.shutdown()
        self._active_lora_path = None
        self._supports_generate_lora_path = None
        self._boot_engine(checkpoint_path)

    def is_healthy(self) -> bool:
        checker = getattr(self.engine, "is_healthy", None)
        if callable(checker):
            try:
                return bool(checker())
            except Exception:
                return False
        try:
            probe = SamplingParams(max_new_tokens=1, temperature=0.0, n=1)
            _ = self.engine.generate(self.health_prompt, sampling_params=self._params_dict(probe))
            return True
        except Exception:
            return False

    def shutdown(self) -> None:
        try:
            self.engine.shutdown()
        except Exception:
            pass


class VLLMOfflineEngine(BaseEngine):
    def __init__(
        self,
        model_path: str,
        *,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        enable_lora: bool = True,
        initial_lora_path: Optional[str] = None,
        health_prompt: str = "ping",
    ) -> None:
        self.model_path = model_path
        self.engine_kwargs = dict(engine_kwargs or {})
        self.enable_lora = enable_lora
        self.health_prompt = health_prompt
        self._active_lora_path: Optional[str] = None
        self._boot_engine(model_path)
        if initial_lora_path:
            self.swap_weights(initial_lora_path, WeightSwapMode.LORA)

    def _boot_engine(self, model_path: str) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM

        kwargs = dict(self.engine_kwargs)
        if self.enable_lora:
            kwargs.setdefault("enable_lora", True)
        self.llm = LLM(model=model_path, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model_path = model_path

    def _vllm_params(self, params: SamplingParams) -> Any:
        from vllm import SamplingParams as VLLMSamplingParams

        kwargs = {
            "max_tokens": params.max_new_tokens,
            "temperature": params.temperature,
            "n": params.n,
            "top_p": params.top_p,
        }
        if params.stop:
            kwargs["stop"] = params.stop
        return VLLMSamplingParams(**kwargs)

    def _maybe_lora_request(self) -> Optional[Any]:
        if not self._active_lora_path:
            return None
        try:
            from vllm.lora.request import LoRARequest
        except Exception as exc:
            raise RuntimeError("vLLM LoRA support is unavailable in this environment.") from exc
        return LoRARequest("active_adapter", 1, self._active_lora_path)

    def generate(self, prompt: str, params: SamplingParams) -> List[GenerationOutput]:
        return self.generate_batch([prompt], params)[0]

    def generate_batch(
        self, prompts: List[str], params: SamplingParams
    ) -> List[List[GenerationOutput]]:
        if not prompts:
            return []

        outputs = self.llm.generate(
            prompts=prompts,
            sampling_params=self._vllm_params(params),
            lora_request=self._maybe_lora_request(),
        )

        grouped: List[List[GenerationOutput]] = []
        for prompt, req_out in zip(prompts, outputs):
            prompt_tokens = len(req_out.prompt_token_ids or [])
            row: List[GenerationOutput] = []
            for item in req_out.outputs:
                token_ids = list(item.token_ids or [])
                if not token_ids and item.text:
                    token_ids = self.tokenizer.encode(item.text, add_special_tokens=False)
                row.append(
                    GenerationOutput(
                        text=item.text,
                        token_ids=token_ids,
                        prompt_tokens=prompt_tokens,
                    )
                )
            grouped.append(row)
        return grouped

    def swap_weights(self, checkpoint_path: str, mode: WeightSwapMode) -> None:
        if mode == WeightSwapMode.LORA:
            self._active_lora_path = checkpoint_path
            return

        updater = getattr(self.llm, "update_weights", None)
        if callable(updater):
            updater(checkpoint_path)
            self.model_path = checkpoint_path
            self._active_lora_path = None
            return

        self.shutdown()
        self._active_lora_path = None
        self._boot_engine(checkpoint_path)

    def is_healthy(self) -> bool:
        try:
            probe = SamplingParams(max_new_tokens=1, temperature=0.0, n=1)
            _ = self.generate(self.health_prompt, probe)
            return True
        except Exception:
            return False

    def shutdown(self) -> None:
        shutdown = getattr(self.llm, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                pass


def _run_sglang_smoke(model_path: str, lora_path: Optional[str]) -> int:
    print("engine=sglang")
    print(f"model_path_exists={os.path.exists(model_path)}")
    if lora_path:
        print(f"lora_path_exists={os.path.exists(lora_path)}")

    engine: Optional[SGLangOfflineEngine] = None
    try:
        engine = SGLangOfflineEngine(
            model_path=model_path,
            initial_lora_path=lora_path,
            engine_kwargs={"mem_fraction_static": 0.45, "log_level": "error"},
        )
        print("boot_ok=True")
        print(f"healthy={engine.is_healthy()}")

        single = engine.generate(
            "Reply with exactly: OK",
            SamplingParams(max_new_tokens=4, temperature=0.0, n=1),
        )
        print(f"single_count={len(single)}")
        print(f"single_text={single[0].text if single else ''}")

        batch = engine.generate_batch(
            ["2+2=", "Color of sky?", "Say: done"],
            SamplingParams(max_new_tokens=8, temperature=0.0, n=1),
        )
        print(f"batch_prompt_count={len(batch)}")
        print(f"batch_result_sizes={[len(x) for x in batch]}")
        return 0
    except Exception as exc:
        print(f"test_failed={type(exc).__name__}: {exc}")
        return 1
    finally:
        if engine is not None:
            try:
                engine.shutdown()
                print("shutdown_ok=True")
            except Exception as exc:
                print(f"shutdown_failed={type(exc).__name__}: {exc}")


def _run_vllm_smoke(model_path: str, lora_path: Optional[str]) -> int:
    print("engine=vllm")
    print(f"model_path_exists={os.path.exists(model_path)}")
    if lora_path:
        print(f"lora_path_exists={os.path.exists(lora_path)}")

    engine: Optional[VLLMOfflineEngine] = None
    try:
        engine = VLLMOfflineEngine(model_path=model_path, initial_lora_path=lora_path)
        print("boot_ok=True")
        print(f"healthy={engine.is_healthy()}")

        single = engine.generate(
            "Reply with exactly: OK",
            SamplingParams(max_new_tokens=4, temperature=0.0, n=1),
        )
        print(f"single_count={len(single)}")
        print(f"single_text={single[0].text if single else ''}")

        batch = engine.generate_batch(
            ["2+2=", "Color of sky?", "Say: done"],
            SamplingParams(max_new_tokens=8, temperature=0.0, n=1),
        )
        print(f"batch_prompt_count={len(batch)}")
        print(f"batch_result_sizes={[len(x) for x in batch]}")
        return 0
    except Exception as exc:
        print(f"test_failed={type(exc).__name__}: {exc}")
        return 1
    finally:
        if engine is not None:
            try:
                engine.shutdown()
                print("shutdown_ok=True")
            except Exception as exc:
                print(f"shutdown_failed={type(exc).__name__}: {exc}")


def _main() -> int:
    parser = argparse.ArgumentParser(description="Offline engine smoke test runner.")
    parser.add_argument(
        "--engine",
        choices=["sglang", "vllm"],
        default="sglang",
        help="Backend engine to test.",
    )
    parser.add_argument("--model-path", required=True, help="Path or HF model id.")
    parser.add_argument(
        "--lora-path",
        default=None,
        help="Optional LoRA adapter path to activate at boot.",
    )
    args = parser.parse_args()

    if args.engine == "sglang":
        return _run_sglang_smoke(args.model_path, args.lora_path)
    return _run_vllm_smoke(args.model_path, args.lora_path)


if __name__ == "__main__":
    raise SystemExit(_main())
