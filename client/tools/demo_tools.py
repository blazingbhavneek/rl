#!/usr/bin/env python3
from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

# Allow running via: python client/tools/demo_tools.py
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from client.tools.rag import RAGTool
from tasksets.codeforces.tools import get_tools


log = logging.getLogger("demo.tools")


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
    (docs / "io_tips.md").write_text(
        "# C I/O Tips\nUse scanf/printf and check bounds for arrays.\n",
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
        log.warning("RAG dependencies unavailable, using local markdown fallback retriever.")
        rag._retriever = SimpleMarkdownRetriever(docs_folder, top_k=2)  # type: ignore[attr-defined]
        rag._ready = True  # type: ignore[attr-defined]
        rag._init_error = None  # type: ignore[attr-defined]
    return rag


def run_demo() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    with tempfile.TemporaryDirectory(prefix="tools_demo_") as tmp:
        root = Path(tmp)
        docs = _prepare_demo_docs(root)
        log.info("Created docs at %s", docs)

        rag_tool = _build_rag_tool(docs)
        tools = get_tools(docs_folder=str(docs), embedding_model_url="http://unused.local")
        log.info("Tools from taskset/codeforces/tools.py: %s", [t.name for t in tools])

        rag_out = rag_tool.execute({"query": "shortest path bfs"})
        print("\n[RAG OUTPUT]")
        print(rag_out.output[:700])
        log.info("RAG success=%s metadata=%s", rag_out.success, rag_out.metadata)

        code_tool = next(t for t in tools if t.name == "execute_code")
        code_out = code_tool.execute(
            {"code": "#include <stdio.h>\nint main(){printf(\"tool-ok\\n\");return 0;}"}
        )
        print("\n[CODE TOOL OUTPUT]")
        print(code_out.output.strip())
        log.info("Code tool success=%s metadata=%s", code_out.success, code_out.metadata)


if __name__ == "__main__":
    run_demo()

