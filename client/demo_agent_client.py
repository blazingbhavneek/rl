#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import sys
import tempfile
import types
from pathlib import Path
from typing import Any, Dict, List

# Allow running via: python client/demo_agent_client.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from client.agent import AgentClient
from client.base import ClientContext
from client.tools.rag import RAGTool
from tasksets.base import Problem
from tasksets.codeforces.tools import get_tools


log = logging.getLogger("demo.agent")


class DemoTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        del add_special_tokens
        return [ord(c) % 251 for c in text]

    def apply_chat_template(self, messages: List[Dict[str, str]], tokenize: bool = False, add_generation_prompt: bool = True) -> str:
        del tokenize, add_generation_prompt
        return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)


class DemoEngine:
    def swap_weights(self, checkpoint_path: str, mode: Any) -> None:
        del checkpoint_path, mode
        return None


class SimpleNode:
    def __init__(self, text: str, metadata: Dict[str, Any]):
        self.text = text
        self.metadata = metadata


class SimpleMarkdownRetriever:
    def __init__(self, docs_folder: Path, top_k: int):
        self.top_k = int(top_k)
        self.nodes: List[SimpleNode] = []
        for p in sorted(docs_folder.rglob("*.md")):
            txt = p.read_text(encoding="utf-8", errors="ignore")
            self.nodes.append(SimpleNode(text=txt, metadata={"file_name": p.name}))

    def retrieve(self, query: str):
        q = set(query.lower().split())

        def score(node: SimpleNode) -> int:
            words = set(node.text.lower().split())
            return len(q.intersection(words))

        ranked = sorted(self.nodes, key=score, reverse=True)
        return ranked[: self.top_k]


class FakeToolCall:
    def __init__(self, call_id: str, name: str, arguments: str):
        self.id = call_id
        self.function = types.SimpleNamespace(name=name, arguments=arguments)


class FakeMessage:
    def __init__(self, content: str, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class FakeResponse:
    def __init__(self, message: FakeMessage):
        self.choices = [types.SimpleNamespace(message=message)]


class FakeChatCompletions:
    def __init__(self):
        self.turn = 0

    def create(self, **kwargs):
        self.turn += 1
        messages = kwargs.get("messages", [])
        log.info("[openai-mock] turn=%s total_messages=%s", self.turn, len(messages))

        if self.turn == 1:
            return FakeResponse(
                FakeMessage(
                    content="I should query docs first.",
                    tool_calls=[
                        FakeToolCall("tc1", "search_knowledge_base", json.dumps({"query": "bfs shortest path"}))
                    ],
                )
            )
        if self.turn == 2:
            return FakeResponse(
                FakeMessage(
                    content="I should run a quick C snippet.",
                    tool_calls=[
                        FakeToolCall(
                            "tc2",
                            "execute_code",
                            json.dumps(
                                {
                                    "code": "#include <stdio.h>\nint main(){printf(\"42\\n\");return 0;}",
                                    "stdin": "",
                                }
                            ),
                        )
                    ],
                )
            )
        return FakeResponse(
            FakeMessage(
                content=(
                    "```c\n"
                    "#include <stdio.h>\n"
                    "int main(){printf(\"42\\n\"); return 0;}\n"
                    "```\n"
                ),
                tool_calls=[],
            )
        )


class FakeOpenAIClient:
    def __init__(self, base_url: str, api_key: str):
        del base_url, api_key
        self.chat = types.SimpleNamespace(completions=FakeChatCompletions())


def _patch_openai_module() -> None:
    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = FakeOpenAIClient
    sys.modules["openai"] = fake_openai
    log.info("Injected mock openai.OpenAI.")


def _prepare_demo_docs(root: Path) -> Path:
    docs = root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "graph_bfs.md").write_text(
        "# BFS Notes\nUse queue-based BFS for unweighted shortest path.\n",
        encoding="utf-8",
    )
    (docs / "dp_intro.md").write_text(
        "# DP Notes\nDefine state, transition, and base case clearly.\n",
        encoding="utf-8",
    )
    return docs


def _build_rag_tool(docs_folder: Path) -> RAGTool:
    rag = RAGTool(
        docs_folder=str(docs_folder),
        embedding_model_url="http://unused.local",
        top_k=2,
        rebuild=True,
    )
    if not getattr(rag, "_ready", False):
        log.warning("RAG deps unavailable, using local markdown fallback retriever.")
        rag._retriever = SimpleMarkdownRetriever(docs_folder, top_k=2)  # type: ignore[attr-defined]
        rag._ready = True  # type: ignore[attr-defined]
        rag._init_error = None  # type: ignore[attr-defined]
    return rag


def run_demo() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _patch_openai_module()

    with tempfile.TemporaryDirectory(prefix="agent_demo_") as tmp:
        root = Path(tmp)
        docs = _prepare_demo_docs(root)

        tools = get_tools(docs_folder=str(docs), embedding_model_url="http://unused.local")
        tools = [t for t in tools if t.name == "execute_code"] + [_build_rag_tool(docs)]
        log.info("Using tools: %s", [t.name for t in tools])

        client = AgentClient(
            engine=DemoEngine(),
            tokenizer=DemoTokenizer(),
            system_prompt="You are a coding assistant.",
            tools=tools,
            max_turns=6,
            server_url="http://localhost:30000",
            model_name="default",
        )

        problem = Problem(
            id="demo_problem",
            statement="Print number 42.",
            bucket=0,
            difficulty_label="b0",
            metadata={},
        )
        result = client.run(problem=problem, context=ClientContext(pass_number=1), n=1)

        print("\n[FINAL COMPLETION]")
        print(result.completions[0])
        print("\n[METADATA]")
        print(json.dumps(result.metadata, indent=2))


if __name__ == "__main__":
    run_demo()

