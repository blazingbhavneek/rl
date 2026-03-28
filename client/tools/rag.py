from __future__ import annotations

from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_core.tools import StructuredTool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from pydantic import BaseModel, Field


class RAGQueryInput(BaseModel):
    query: str = Field(..., description="Query to search in indexed markdown files.")
    k: int = Field(default=4, ge=1, le=20, description="Number of chunks to retrieve.")


def build_markdown_rag_tool(
    docs_folder: str,
    persist_directory: str,
    embedding_base_url: Optional[str] = None,
    embedding_api_key: Optional[str] = None,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    embedding_backend: str = "huggingface",
    collection_name: str = "markdown_rag",
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> StructuredTool:
    def _decode_markdown(path: Path) -> str:
        raw = path.read_bytes()
        if raw.startswith(b"\xef\xbb\xbf"):
            return raw.decode("utf-8-sig", errors="replace")
        candidates = [
            "utf-8",
            "utf-8-sig",
            "cp932",
            "shift_jis",
            "euc_jp",
            "iso2022_jp",
        ]
        for enc in candidates:
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="replace")

    root = Path(docs_folder)
    if root.is_file():
        md_files = [root] if root.suffix.lower() == ".md" else []
    else:
        md_files = sorted(root.rglob("*.md"))
    if not md_files:
        raise ValueError(f"No markdown files found at: {docs_folder}")

    docs: list[Document] = []
    for path in md_files:
        text = _decode_markdown(path)
        if text.strip():
            docs.append(Document(page_content=text, metadata={"source": str(path)}))
    if not docs:
        raise ValueError(f"Markdown files are empty at: {docs_folder}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(chunk_size),
        chunk_overlap=int(chunk_overlap),
    )
    chunks = splitter.split_documents(docs)
    if not chunks:
        raise ValueError(f"No text chunks produced from markdown files at: {docs_folder}")

    if embedding_backend == "huggingface":
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
        )
    else:
        embeddings = OpenAIEmbeddings(
            model=embedding_model,
            base_url=embedding_base_url,
            api_key=embedding_api_key,
        )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    def _query(query: str, k: int = 4) -> str:
        local_retriever = vectorstore.as_retriever(search_kwargs={"k": int(k)})
        hits = local_retriever.invoke(query)
        if not hits:
            return "No relevant context found."
        lines: list[str] = []
        for idx, hit in enumerate(hits, start=1):
            source = hit.metadata.get("source", "unknown")
            lines.append(f"[{idx}] source: {source}\n{hit.page_content}")
        return "\n\n---\n\n".join(lines)

    return StructuredTool.from_function(
        func=_query,
        name="search_knowledge_base",
        description="Search indexed markdown knowledge base and return relevant chunks.",
        args_schema=RAGQueryInput,
    )
