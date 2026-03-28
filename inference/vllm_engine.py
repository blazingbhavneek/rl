from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

from .base import BaseEngine


class VLLMEngine(BaseEngine):
    def __init__(
        self,
        model_path: str,
        engine_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        self.base_url = ""
        self.api_key = ""
        self.model_name = model_path
        self._server_proc: Optional[subprocess.Popen[Any]] = None
        self.reasoning_parser: Optional[str] = None
        self.tool_call_parser: Optional[str] = None
        self.tool_parser_plugin: Optional[str] = None
        self.enable_auto_tool_choice: bool = False
        super().__init__(model_path=model_path, engine_kwargs=engine_kwargs)

    def _load_model(self) -> None:
        engine_kwargs = dict(self.engine_kwargs)
        self.base_url = str(engine_kwargs.get("base_url", "http://localhost:8000/v1")).rstrip("/")
        self.api_key = str(engine_kwargs["api_key"]) if engine_kwargs.get("api_key") is not None else ""
        self.model_name = str(engine_kwargs.get("model_name", self.model_path))
        self.reasoning_parser = (
            str(engine_kwargs["reasoning_parser"]) if engine_kwargs.get("reasoning_parser") else None
        )
        self.tool_call_parser = (
            str(engine_kwargs["tool_call_parser"]) if engine_kwargs.get("tool_call_parser") else None
        )
        self.tool_parser_plugin = (
            str(engine_kwargs["tool_parser_plugin"]) if engine_kwargs.get("tool_parser_plugin") else None
        )
        self.enable_auto_tool_choice = bool(engine_kwargs.get("enable_auto_tool_choice", False))
        self.is_awake = True

    async def _request_json(self, method: str, path: str, payload: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        admin_paths = {"/sleep", "/wake_up", "/is_sleeping", "/shutdown"}
        if path in admin_paths or path.startswith("/sleep?"):
            parsed = urllib.parse.urlparse(self.base_url)
            root_base = f"{parsed.scheme}://{parsed.netloc}"
            url = f"{root_base}{path}"
        else:
            url = f"{self.base_url}{path}"
        data: bytes | None = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())

        def _send() -> dict[str, Any]:
            with urllib.request.urlopen(req, timeout=180) as resp:
                raw = resp.read().decode("utf-8")
                if not raw:
                    return {}
                return json.loads(raw)

        try:
            return await asyncio.to_thread(_send)
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8")
            except Exception:
                pass
            raise RuntimeError(f"vLLM server HTTP {exc.code} for {path}: {detail}") from exc

    async def init(self) -> None:
        if self._closed:
            raise RuntimeError("engine is shut down")
        await self._request_json("GET", "/models")
        self.is_awake = True

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("engine is shut down")

        if self._server_proc is None or self._server_proc.poll() is not None:
            parsed = urllib.parse.urlparse(self.base_url)
            host = parsed.hostname or "127.0.0.1"
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            max_model_len = self.engine_kwargs.get("max_model_len")
            gpu_memory_utilization = self.engine_kwargs.get("gpu_memory_utilization")

            server_cmd = [
                sys.executable,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                self.model_path,
                "--host",
                host,
                "--port",
                str(port),
                "--api-key",
                self.api_key,
                "--enable-sleep-mode",
            ]
            if max_model_len is not None:
                server_cmd.extend(["--max-model-len", str(max_model_len)])
            if gpu_memory_utilization is not None:
                server_cmd.extend(["--gpu-memory-utilization", str(gpu_memory_utilization)])
            if self.reasoning_parser:
                server_cmd.extend(["--reasoning-parser", self.reasoning_parser])
            if self.enable_auto_tool_choice:
                server_cmd.append("--enable-auto-tool-choice")
            if self.tool_call_parser:
                server_cmd.extend(["--tool-call-parser", self.tool_call_parser])
            if self.tool_parser_plugin:
                server_cmd.extend(["--tool-parser-plugin", self.tool_parser_plugin])
            server_env = dict(os.environ)
            server_env["VLLM_SERVER_DEV_MODE"] = "1"
            save_vllm_logs = bool(self.engine_kwargs.get("save_vllm_logs", False))
            if save_vllm_logs:
                os.makedirs("logs", exist_ok=True)
                log_path = str(self.engine_kwargs.get("vllm_log_path", "logs/vllm_server.log"))
                log_file = open(log_path, "a", encoding="utf-8")
                setattr(self, "_server_log_file", log_file)
                self._server_proc = subprocess.Popen(
                    server_cmd,
                    env=server_env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
            else:
                self._server_proc = subprocess.Popen(
                    server_cmd,
                    env=server_env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

        max_wait_seconds = int(self.engine_kwargs.get("startup_timeout_s", 180))
        retry_interval_seconds = float(self.engine_kwargs.get("startup_retry_interval_s", 2.0))
        deadline = asyncio.get_event_loop().time() + max_wait_seconds
        last_exc: Exception | None = None

        while asyncio.get_event_loop().time() < deadline:
            if self._server_proc is not None and self._server_proc.poll() is not None:
                raise RuntimeError(f"vLLM server exited early with code {self._server_proc.returncode}")
            try:
                await self.init()
                return
            except Exception as exc:
                last_exc = exc
                await asyncio.sleep(retry_interval_seconds)

        if last_exc is not None:
            raise RuntimeError(f"vLLM server did not become ready within {max_wait_seconds}s") from last_exc
        raise RuntimeError(f"vLLM server did not become ready within {max_wait_seconds}s")

    async def sleep(self, level: int = 1) -> None:
        if self._closed or not self.is_awake:
            return
        lvl = 1 if level not in (1, 2) else int(level)
        await self._request_json("POST", f"/sleep?level={lvl}")
        self.is_awake = False

    async def wake(self) -> None:
        if self._closed or self.is_awake:
            return
        await self._request_json("POST", "/wake_up")
        self.is_awake = True

    async def is_sleeping(self) -> bool:
        if self._closed:
            return True
        data = await self._request_json("GET", "/is_sleeping")
        value = data.get("is_sleeping")
        if isinstance(value, bool):
            return value
        return bool(value)

    async def kill(self) -> None:
        if self._closed:
            return
        if self._server_proc is not None and self._server_proc.poll() is None:
            self._server_proc.terminate()
            try:
                self._server_proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self._server_proc.kill()
        self.is_awake = False
        self._closed = True

    async def shutdown(self) -> None:
        try:
            await self.kill()
        except Exception:
            try:
                await self._request_json("POST", "/sleep?level=2")
            except Exception:
                pass
            self.is_awake = False
            self._closed = True
