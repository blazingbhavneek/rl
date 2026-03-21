from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseTool, ToolResult


class RAGTool(BaseTool):
    def __init__(
        self,
        docs_folder: str,
        embedding_model_url: str,
        collection_name: str = "rag",
        top_k: int = 5,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        rebuild: bool = False,
        embedding_backend: str = "openai",
    ) -> None:
        self.docs_folder = Path(docs_folder)
        self.embedding_model_url = embedding_model_url
        self.collection_name = collection_name
        self.top_k = int(top_k)
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)
        self.rebuild = bool(rebuild)
        self.embedding_backend = embedding_backend

        self._ready = False
        self._retriever = None
        self._init_error: Optional[str] = None
        self._initialize_index()

    @property
    def name(self) -> str:
        return "search_knowledge_base"

    @property
    def description(self) -> str:
        return "Search documentation for relevant information."

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        }

    def _initialize_index(self) -> None:
        try:
            from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
            from llama_index.core.node_parser import SentenceSplitter
            from llama_index.vector_stores.chroma import ChromaVectorStore
            import chromadb

            if self.embedding_backend == "huggingface":
                from llama_index.embeddings.huggingface import HuggingFaceEmbedding

                embed_model = HuggingFaceEmbedding(model_name=self.embedding_model_url)
            else:
                from llama_index.embeddings.openai import OpenAIEmbedding

                embed_model = OpenAIEmbedding(api_base=self.embedding_model_url, api_key="none")

            Settings.embed_model = embed_model
            Settings.node_parser = SentenceSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

            persist_path = self.docs_folder / ".chroma"
            persist_path.mkdir(parents=True, exist_ok=True)

            db = chromadb.PersistentClient(path=str(persist_path))
            collection_names = {c.name for c in db.list_collections()}
            has_collection = self.collection_name in collection_names

            collection = db.get_or_create_collection(self.collection_name)
            vector_store = ChromaVectorStore(chroma_collection=collection)
            storage = StorageContext.from_defaults(vector_store=vector_store)

            if has_collection and not self.rebuild:
                index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
            else:
                docs = SimpleDirectoryReader(str(self.docs_folder), recursive=True).load_data()
                index = VectorStoreIndex.from_documents(docs, storage_context=storage)

            self._retriever = index.as_retriever(similarity_top_k=self.top_k)
            self._ready = True
        except Exception as exc:
            self._ready = False
            self._init_error = str(exc)

    def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            return ToolResult(False, "Missing required non-empty 'query' string", {})

        if not self._ready or self._retriever is None:
            return ToolResult(
                False,
                f"RAG index is not available: {self._init_error or 'not initialized'}",
                {"docs_folder": str(self.docs_folder)},
            )

        try:
            nodes = self._retriever.retrieve(query)
        except Exception as exc:
            return ToolResult(False, f"RAG retrieval failed: {exc}", {})

        chunks: List[str] = []
        for node in nodes:
            text = getattr(node, "text", "")
            md = getattr(node, "metadata", {}) or {}
            source = md.get("file_name") or md.get("source") or "unknown"
            heading = md.get("header_path") or md.get("section")
            if heading:
                header = f"[Source: {source} > {heading}]"
            else:
                header = f"[Source: {source}]"
            chunks.append(f"{header}\n{text}\n---")

        return ToolResult(
            success=True,
            output="\n".join(chunks) if chunks else "No relevant context found.",
            metadata={"hits": len(chunks), "collection": self.collection_name},
        )
