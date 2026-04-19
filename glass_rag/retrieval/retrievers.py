"""Retrieval services for keyword, vector, and hybrid search."""

from __future__ import annotations

from langchain_core.documents import Document

from glass_rag.config import AppConfig
from glass_rag.governance import AccessController
from glass_rag.schemas import RetrievalMode, UserContext
from glass_rag.storage.mongo import MongoGateway


class RetrievalService:
    """Runs the retrieval strategy selected by the query router."""

    def __init__(
        self,
        config: AppConfig,
        mongo: MongoGateway,
        access_controller: AccessController,
        embeddings,
    ) -> None:
        self.config = config
        self.mongo = mongo
        self.access_controller = access_controller
        self.embeddings = embeddings

    def retrieve(
        self,
        *,
        question: str,
        mode: RetrievalMode,
        user: UserContext,
        top_k: int,
    ) -> list[Document]:
        if mode == RetrievalMode.KEYWORD:
            return self.keyword_search(question, user, top_k)
        if mode == RetrievalMode.VECTOR:
            return self.vector_search(question, user, top_k)
        return self.hybrid_search(question, user, top_k)

    def keyword_search(self, question: str, user: UserContext, top_k: int) -> list[Document]:
        access_filter = self.access_controller.build_filter(user)
        results = self.mongo.keyword_search(question, access_filter, top_k)
        for item in results:
            item.metadata["retrieval_mode"] = RetrievalMode.KEYWORD.value
        return results

    def vector_search(self, question: str, user: UserContext, top_k: int) -> list[Document]:
        access_filter = self.access_controller.build_filter(user)
        vector_store = self.mongo.vector_store(self.embeddings)
        results = vector_store.similarity_search(
            question,
            k=top_k,
            pre_filter=access_filter,
        )
        for item in results:
            item.metadata["retrieval_mode"] = RetrievalMode.VECTOR.value
        return results

    def hybrid_search(self, question: str, user: UserContext, top_k: int) -> list[Document]:
        keyword_results = self.keyword_search(question, user, top_k)
        vector_results = self.vector_search(question, user, top_k)

        merged: list[Document] = []
        seen: set[str] = set()
        for result in keyword_results + vector_results:
            chunk_id = result.metadata.get("chunk_id")
            if chunk_id and chunk_id not in seen:
                seen.add(chunk_id)
                result.metadata["retrieval_mode"] = RetrievalMode.HYBRID.value
                merged.append(result)
        return merged[:top_k]
