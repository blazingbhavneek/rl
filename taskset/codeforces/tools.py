from __future__ import annotations

from typing import List, Optional

from client.tools.base import BaseTool
from client.tools.code_execution import CodeExecutionTool
from client.tools.rag import RAGTool


def get_tools(
    docs_folder: Optional[str] = None, embedding_model_url: Optional[str] = None
) -> List[BaseTool]:
    tools: List[BaseTool] = [CodeExecutionTool(language="c", timeout=5.0)]

    if docs_folder and embedding_model_url:
        tools.append(
            RAGTool(
                docs_folder=docs_folder,
                embedding_model_url=embedding_model_url,
            )
        )

    return tools
