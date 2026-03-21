from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

try:
    from .base import BaseEngine, GenerationOutput, SamplingParams, WeightSwapMode
except ImportError:  # pragma: no cover - direct script execution fallback
    from base import BaseEngine, GenerationOutput, SamplingParams, WeightSwapMode


class ServerEngine(BaseEngine):
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: Optional[str] = None,
        timeout_s: float = 30.0,
        health_endpoint: str = "/health",
        reload_endpoint: str = "/v1/internal/reload",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = float(timeout_s)
        self.health_endpoint = health_endpoint
        self.reload_endpoint = reload_endpoint

    def _request(
        self, method: str, path: str, payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        body = None
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(url=url, data=body, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            raw = resp.read()
            if not raw:
                return {}
            return json.loads(raw.decode("utf-8"))

    def generate(self, prompt: str, params: SamplingParams) -> List[GenerationOutput]:
        return self.generate_batch([prompt], params)[0]

    def generate_batch(
        self, prompts: List[str], params: SamplingParams
    ) -> List[List[GenerationOutput]]:
        if not prompts:
            return []

        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompts,
            "max_tokens": params.max_new_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "n": params.n,
        }
        if params.stop:
            payload["stop"] = params.stop

        data = self._request("POST", "/v1/completions", payload)
        choices = list(data.get("choices", []))
        usage = data.get("usage", {})
        total_prompt_tokens = int(usage.get("prompt_tokens", 0))

        out: List[List[GenerationOutput]] = [[] for _ in prompts]
        if not choices:
            return out

        # OpenAI-style completions with prompt list generally returns choices in order:
        # prompt0:n, prompt1:n, ...
        for i, choice in enumerate(choices):
            prompt_idx = min(i // max(params.n, 1), len(prompts) - 1)
            token_ids = list(choice.get("token_ids", []))
            prompt_tokens = total_prompt_tokens // len(prompts) if prompts else 0
            out[prompt_idx].append(
                GenerationOutput(
                    text=choice.get("text", ""),
                    token_ids=token_ids,
                    prompt_tokens=prompt_tokens,
                )
            )
        return out

    def swap_weights(self, checkpoint_path: str, mode: WeightSwapMode) -> None:
        payload = {"checkpoint_path": checkpoint_path, "mode": mode.value}
        try:
            _ = self._request("POST", self.reload_endpoint, payload)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise RuntimeError(
                    f"Weight reload endpoint not found: {self.reload_endpoint}"
                ) from exc
            raise

    def is_healthy(self) -> bool:
        try:
            _ = self._request("GET", self.health_endpoint)
            return True
        except Exception:
            pass
        try:
            _ = self._request("GET", "/v1/models")
            return True
        except Exception:
            return False

    def shutdown(self) -> None:
        # Server lifecycle is external; this is intentionally a no-op.
        return None


if __name__ == "__main__":
    # Requires a running OpenAI-compatible server.
    engine = ServerEngine(base_url="http://127.0.0.1:8000", model="dummy")
    print("healthy:", engine.is_healthy())
