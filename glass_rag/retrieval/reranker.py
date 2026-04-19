"""LLM-based reranking for retrieved chunks."""

from __future__ import annotations

from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate


class RankedChunk(BaseModel):
    """One reranked chunk returned by the reranker."""

    chunk_id: str
    score: float = Field(ge=0, le=1)
    reason: str


class RankedChunkList(BaseModel):
    """Structured reranker response."""

    rankings: list[RankedChunk]


class LLMReranker:
    """Uses a configurable chat model to score retrieved chunks."""

    def __init__(self, rerank_llm, top_n: int) -> None:
        self.rerank_llm = rerank_llm
        self.top_n = top_n
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are the GLASS-RAG reranker. Score documents only by relevance, evidence quality, and terminology match.",
                ),
                (
                    "human",
                    "Question: {question}\n\nCandidates:\n{candidates}\n\n"
                    "Return structured rankings with a score between 0 and 1 for each chunk_id.",
                ),
            ]
        )

    def rerank(self, question: str, documents: list[Document], callbacks=None) -> list[Document]:
        if not documents:
            return []
        if len(documents) <= self.top_n:
            return documents

        candidates = []
        for index, document in enumerate(documents, start=1):
            candidates.append(
                "\n".join(
                    [
                        f"Index: {index}",
                        f"chunk_id: {document.metadata.get('chunk_id', '')}",
                        f"title: {document.metadata.get('title', '')}",
                        f"page: {document.metadata.get('page_number', '')}",
                        f"content: {document.page_content[:600]}",
                    ]
                )
            )

        chain = self.prompt | self.rerank_llm.with_structured_output(RankedChunkList)
        try:
            result = chain.invoke(
                {"question": question, "candidates": "\n\n".join(candidates)},
                config={"callbacks": callbacks or []},
            )
        except Exception:
            return documents[: self.top_n]

        ranked_map = {item.chunk_id: item.score for item in result.rankings}
        ordered = sorted(
            documents,
            key=lambda item: ranked_map.get(item.metadata.get("chunk_id", ""), 0.0),
            reverse=True,
        )
        for item in ordered:
            item.metadata["rerank_score"] = ranked_map.get(item.metadata.get("chunk_id", ""), 0.0)
        return ordered[: self.top_n]
