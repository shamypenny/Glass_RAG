"""Chunking strategies for parsed documents."""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from glass_rag.config import AppConfig
from glass_rag.schemas import ChunkRecord, ParsedDocument


class SmartChunker:
    """Builds semantically reasonable chunks while keeping metadata intact."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.pipeline.chunk_size,
            chunk_overlap=config.pipeline.chunk_overlap,
            separators=["\n## ", "\n### ", "\n\n", "\n", ". ", "; ", ", ", " "],
        )

    def chunk_document(self, parsed_document: ParsedDocument) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        base_metadata = {
            "document_id": parsed_document.document_id,
            "title": parsed_document.metadata.title,
            "source_type": parsed_document.metadata.source_type.value,
            "source_uri": parsed_document.metadata.source_uri,
            "tags": parsed_document.metadata.tags,
            "business_domain": parsed_document.metadata.business_domain,
            "secrecy_level": parsed_document.metadata.secrecy_level.value,
            "project": parsed_document.metadata.project,
            "department": parsed_document.metadata.department,
            "author": parsed_document.metadata.author,
            "source_id": parsed_document.metadata.source_id,
            "extracted_entities": parsed_document.extracted_entities,
        }

        for section in parsed_document.sections:
            parts = [section.content] if section.section_type == "table" else self.splitter.split_text(section.content)
            for index, part in enumerate(parts, start=1):
                if not part.strip():
                    continue
                chunk_metadata = {
                    **base_metadata,
                    "heading": section.heading,
                    "section_type": section.section_type,
                    "page_number": section.page_number,
                    "section_id": section.section_id,
                    "chunk_order": index,
                }
                chunks.append(
                    ChunkRecord(
                        document_id=parsed_document.document_id,
                        page_content=part.strip(),
                        metadata=chunk_metadata,
                    )
                )
        return chunks

