from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .base import BaseTool, ToolResult


class MCPToolset:
    def __init__(
        self,
        server_url: str,
        server_name: str,
        include_tools: Optional[List[str]] = None,
        exclude_tools: Optional[List[str]] = None,
    ) -> None:
        self.server_url = server_url
        self.server_name = server_name
        self.include_tools = set(include_tools or [])
        self.exclude_tools = set(exclude_tools or [])

    def get_tools(self) -> List[BaseTool]:
        try:
            import mcp  # noqa: F401
        except Exception as exc:
            raise RuntimeError("MCP toolset requires the 'mcp' package. Install with: pip install mcp") from exc

        # Keep interface stable while avoiding hard dependency on evolving SDK
        # object model at import-time. If manifest fetch fails, return an empty list.
        manifest = self._fetch_manifest_safe()
        tools: List[BaseTool] = []
        for tool_def in manifest:
            name = tool_def.get("name", "")
            if self.include_tools and name not in self.include_tools:
                continue
            if name in self.exclude_tools:
                continue
            tools.append(MCPTool(self.server_url, self.server_name, tool_def))
        return tools

    def _fetch_manifest_safe(self) -> List[Dict[str, Any]]:
        try:
            return asyncio.run(self._fetch_manifest())
        except Exception:
            return []

    async def _fetch_manifest(self) -> List[Dict[str, Any]]:
        # Best-effort protocol call. If SDK APIs differ, caller still gets [] and can proceed.
        try:
            from mcp.client.sse import sse_client
            from mcp.client.session import ClientSession

            async with sse_client(self.server_url) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    res = await session.list_tools()
                    raw_tools = getattr(res, "tools", None) or []
                    out: List[Dict[str, Any]] = []
                    for t in raw_tools:
                        out.append(
                            {
                                "name": getattr(t, "name", ""),
                                "description": getattr(t, "description", ""),
                                "inputSchema": getattr(t, "inputSchema", {}) or {},
                            }
                        )
                    return out
        except Exception:
            return []


class MCPTool(BaseTool):
    def __init__(self, server_url: str, server_name: str, tool_def: Dict[str, Any]) -> None:
        self.server_url = server_url
        self.server_name = server_name
        self.tool_def = dict(tool_def)

    @property
    def name(self) -> str:
        return self.tool_def.get("name", "mcp_tool")

    @property
    def description(self) -> str:
        return self.tool_def.get("description", f"MCP tool from {self.server_name}")

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        schema = self.tool_def.get("inputSchema")
        return schema if isinstance(schema, dict) and schema else {"type": "object", "properties": {}}

    def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        try:
            out = asyncio.run(self._execute_async(arguments))
            return ToolResult(success=True, output=out, metadata={"tool": self.name})
        except Exception as exc:
            return ToolResult(success=False, output=f"MCP tool call failed: {exc}", metadata={"tool": self.name})

    async def _execute_async(self, arguments: Dict[str, Any]) -> str:
        from mcp.client.sse import sse_client
        from mcp.client.session import ClientSession

        async with sse_client(self.server_url) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                res = await session.call_tool(self.name, arguments)
                # Normalize the common response forms.
                content = getattr(res, "content", None)
                if isinstance(content, list):
                    parts: List[str] = []
                    for item in content:
                        txt = getattr(item, "text", None)
                        if txt is not None:
                            parts.append(str(txt))
                        else:
                            parts.append(str(item))
                    return "\n".join(parts)
                if content is not None:
                    return str(content)
                return str(res)
