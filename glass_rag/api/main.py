"""FastAPI entrypoint for GLASS-RAG."""

from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, UploadFile

from glass_rag.schemas import DataSource, DocumentMetadata, QueryRequest, SecrecyLevel
from glass_rag.services.application import get_application

app = FastAPI(title="GLASS-RAG API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/documents/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    actor_id: str = Form(...),
    title: str = Form(...),
    source_type: DataSource = Form(...),
    source_uri: str | None = Form(default=None),
    business_domain: str | None = Form(default=None),
    secrecy_level: SecrecyLevel = Form(default=SecrecyLevel.INTERNAL),
    project: str | None = Form(default=None),
    department: str | None = Form(default=None),
    author: str | None = Form(default=None),
    tags: str | None = Form(default=None),
) -> dict:
    application = get_application()
    upload_path = application.config.storage.upload_path / f"{uuid4()}_{file.filename}"
    with upload_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    metadata = DocumentMetadata(
        title=title,
        source_type=source_type,
        source_uri=source_uri,
        business_domain=business_domain,
        secrecy_level=secrecy_level,
        project=project,
        department=department,
        author=author,
        tags=[item.strip() for item in (tags or "").split(",") if item.strip()],
    )
    result = application.ingestion_pipeline.ingest_file(str(upload_path), metadata, actor_id=actor_id)
    return result.model_dump()


@app.post("/query")
def query(payload: QueryRequest) -> dict:
    application = get_application()
    response = application.query_pipeline.answer(payload)
    return response.model_dump()
