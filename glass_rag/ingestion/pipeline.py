"""Knowledge ingestion pipeline for GLASS-RAG."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from pydantic import BaseModel

from glass_rag.governance import AuditFactory
from glass_rag.observability import ObservabilityManager
from glass_rag.schemas import DocumentMetadata
from glass_rag.storage.mongo import MongoGateway


class IngestionResult(BaseModel):
    """Response returned after a document is ingested."""

    document_id: str
    chunk_count: int
    stored_path: str
    trace_id: str | None = None


class KnowledgeIngestionPipeline:
    """Executes parse, enrichment, chunking, and vector indexing."""

    def __init__(
        self,
        mongo: MongoGateway,
        parser,
        chunker,
        embeddings,
        observability: ObservabilityManager,
    ) -> None:
        self.mongo = mongo
        self.parser = parser
        self.chunker = chunker
        self.embeddings = embeddings
        self.observability = observability

    def ingest_file(self, file_path: str, metadata: DocumentMetadata, actor_id: str) -> IngestionResult:
        self.mongo.ensure_indexes()
        with self.observability.trace(
            name="document_ingestion",
            user_id=actor_id,
            input_payload={"file_path": file_path, "title": metadata.title},
        ) as trace_id:
            callbacks = self.observability.callbacks()
            parsed_document = self.parser.parse(file_path, metadata, callbacks=callbacks)
            chunks = self.chunker.chunk_document(parsed_document)

            vector_store = self.mongo.vector_store(self.embeddings)
            documents = [
                Document(
                    page_content=chunk.page_content,
                    metadata={
                        **chunk.metadata,
                        "chunk_id": chunk.chunk_id,
                    },
                )
                for chunk in chunks
            ]

            self.mongo.save_document(parsed_document)
            if documents:
                vector_store.add_documents(documents)

            self.mongo.save_audit(
                AuditFactory.build(
                    event_type="document_ingested",
                    actor_id=actor_id,
                    payload={
                        "document_id": parsed_document.document_id,
                        "chunk_count": len(documents),
                        "file_name": Path(file_path).name,
                    },
                )
            )

            return IngestionResult(
                document_id=parsed_document.document_id,
                chunk_count=len(documents),
                stored_path=file_path,
                trace_id=trace_id,
            )
