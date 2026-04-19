"""Shared data models for the GLASS-RAG pipelines."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class DataSource(str, Enum):
    """Supported upstream knowledge source categories."""

    INTERNAL_DOCUMENT = "internal_document"
    INDUSTRY_REPORT = "industry_report"
    PATENT = "patent"
    TECHNICAL_REPORT = "technical_report"
    SAMPLE_DATA = "sample_data"
    USER_UPLOAD = "user_upload"


class SecrecyLevel(str, Enum):
    """Classification labels used for access control."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class QueryIntent(str, Enum):
    """Intent categories extracted from the product spec."""

    RETRIEVAL_QA = "retrieval_qa"
    SAMPLE_QUERY = "sample_query"
    SUMMARY = "summary"
    COMPARISON = "comparison"
    KNOWLEDGE_LOOKUP = "knowledge_lookup"


class RetrievalMode(str, Enum):
    """Available retrieval strategies."""

    KEYWORD = "keyword"
    VECTOR = "vector"
    HYBRID = "hybrid"


class UserContext(BaseModel):
    """Identity and authorization attributes for the current user."""

    user_id: str
    department: str | None = None
    projects: list[str] = Field(default_factory=list)
    allowed_levels: list[SecrecyLevel] = Field(
        default_factory=lambda: [
            SecrecyLevel.PUBLIC,
            SecrecyLevel.INTERNAL,
        ]
    )


class DocumentMetadata(BaseModel):
    """Business metadata attached to a source document."""

    source_id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    source_type: DataSource
    source_uri: str | None = None
    tags: list[str] = Field(default_factory=list)
    business_domain: str | None = None
    secrecy_level: SecrecyLevel = SecrecyLevel.INTERNAL
    project: str | None = None
    department: str | None = None
    author: str | None = None
    published_at: datetime | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    extra: dict[str, Any] = Field(default_factory=dict)


class StructuredSection(BaseModel):
    """Normalized parsed section before chunking."""

    section_id: str = Field(default_factory=lambda: str(uuid4()))
    heading: str | None = None
    content: str
    section_type: str = "text"
    page_number: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    """Parsed representation of one physical file."""

    document_id: str = Field(default_factory=lambda: str(uuid4()))
    file_path: str
    metadata: DocumentMetadata
    sections: list[StructuredSection]
    extracted_entities: dict[str, Any] = Field(default_factory=dict)

    @property
    def suffix(self) -> str:
        return Path(self.file_path).suffix.lower()


class ChunkRecord(BaseModel):
    """Persistable chunk structure for MongoDB and vector indexing."""

    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    page_content: str
    metadata: dict[str, Any]


class QueryRequest(BaseModel):
    """User query payload."""

    question: str
    session_id: str
    user: UserContext
    top_k: int | None = None
    intent_hint: QueryIntent | None = None
    preferred_mode: RetrievalMode | None = None


class Citation(BaseModel):
    """Answer citation returned to the client."""

    chunk_id: str
    title: str
    source_uri: str | None = None
    score: float | None = None
    page_number: int | None = None
    quote: str


class QueryResponse(BaseModel):
    """Final answer envelope."""

    answer: str
    intent: QueryIntent
    retrieval_mode: RetrievalMode
    citations: list[Citation]
    trace_id: str | None = None


class AuditRecord(BaseModel):
    """MongoDB audit record."""

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: str
    actor_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    payload: dict[str, Any] = Field(default_factory=dict)


class SessionTurn(BaseModel):
    """Stored conversation turn for follow-up queries."""

    session_id: str
    role: str
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
