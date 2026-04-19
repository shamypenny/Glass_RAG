"""MongoDB persistence layer for documents, chunks, audit, and chat history."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import ASCENDING, MongoClient, TEXT

from glass_rag.config import AppConfig
from glass_rag.schemas import AuditRecord, ParsedDocument, SessionTurn


class MongoGateway:
    """Centralizes MongoDB access used by ingestion and query pipelines."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.client = MongoClient(config.mongodb.uri)
        self.db = self.client[config.mongodb.database]

    @property
    def documents(self):
        return self.db[self.config.mongodb.documents_collection]

    @property
    def chunks(self):
        return self.db[self.config.mongodb.chunks_collection]

    @property
    def audits(self):
        return self.db[self.config.mongodb.audit_collection]

    @property
    def sessions(self):
        return self.db[self.config.mongodb.sessions_collection]

    def ensure_indexes(self) -> None:
        """Creates standard MongoDB indexes required by this service."""

        self.documents.create_index([("document_id", ASCENDING)], unique=True)
        self.documents.create_index([("metadata.source_type", ASCENDING)])
        self.documents.create_index([("metadata.secrecy_level", ASCENDING)])
        self.chunks.create_index([("metadata.document_id", ASCENDING)])
        self.chunks.create_index([("metadata.project", ASCENDING)])
        self.chunks.create_index([("metadata.department", ASCENDING)])
        self.chunks.create_index([("page_content", TEXT)])
        self.sessions.create_index([("session_id", ASCENDING), ("created_at", ASCENDING)])
        self.audits.create_index([("actor_id", ASCENDING), ("created_at", ASCENDING)])

    def vector_store(self, embeddings) -> MongoDBAtlasVectorSearch:
        """Returns the LangChain vector store bound to the chunk collection."""

        return MongoDBAtlasVectorSearch(
            collection=self.chunks,
            embedding=embeddings,
            index_name=self.config.mongodb.vector_index_name,
            text_key=self.config.mongodb.text_key,
            embedding_key=self.config.mongodb.embedding_key,
        )

    def save_document(self, parsed_document: ParsedDocument) -> None:
        payload = parsed_document.model_dump(mode="json")
        self.documents.replace_one(
            {"document_id": parsed_document.document_id},
            payload,
            upsert=True,
        )

    def save_audit(self, record: AuditRecord) -> None:
        self.audits.insert_one(record.model_dump(mode="json"))

    def append_session_turn(self, turn: SessionTurn) -> None:
        self.sessions.insert_one(turn.model_dump(mode="json"))

    def load_session_turns(self, session_id: str, limit: int = 6) -> list[SessionTurn]:
        cursor = (
            self.sessions.find({"session_id": session_id})
            .sort("created_at", -1)
            .limit(limit)
        )
        items = [SessionTurn.model_validate(item) for item in cursor]
        return list(reversed(items))

    def keyword_search(self, query: str, pre_filter: dict[str, Any], top_k: int) -> list[Document]:
        """Fallback keyword search based on a standard Mongo text index."""

        mongo_query: dict[str, Any] = {"$text": {"$search": query}}
        if pre_filter:
            mongo_query = {"$and": [pre_filter, mongo_query]}

        cursor = (
            self.chunks.find(
                mongo_query,
                {"score": {"$meta": "textScore"}, "page_content": 1, "metadata": 1},
            )
            .sort([("score", {"$meta": "textScore"})])
            .limit(top_k)
        )

        results: list[Document] = []
        for item in cursor:
            metadata = item.get("metadata", {})
            metadata["keyword_score"] = item.get("score")
            results.append(Document(page_content=item["page_content"], metadata=metadata))
        return results

