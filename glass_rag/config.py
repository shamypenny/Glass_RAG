"""Configuration loading for GLASS-RAG."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator


class MongoConfig(BaseModel):
    """MongoDB persistence settings."""

    uri_env: str = "MONGODB_URI"
    database: str = "glass_rag"
    documents_collection: str = "documents"
    chunks_collection: str = "chunks"
    audit_collection: str = "audit_logs"
    sessions_collection: str = "sessions"
    vector_index_name: str = "chunk_vector_index"
    text_search_index_name: str = "chunk_text_search"
    embedding_key: str = "embedding"
    text_key: str = "page_content"

    @property
    def uri(self) -> str:
        value = os.getenv(self.uri_env)
        if not value:
            raise ValueError(f"MongoDB URI environment variable '{self.uri_env}' is not set.")
        return value


class LangfuseConfig(BaseModel):
    """Langfuse tracing configuration."""

    enabled: bool = True
    public_key_env: str = "LANGFUSE_PUBLIC_KEY"
    secret_key_env: str = "LANGFUSE_SECRET_KEY"
    host_env: str = "LANGFUSE_BASE_URL"

    @property
    def public_key(self) -> str | None:
        return os.getenv(self.public_key_env)

    @property
    def secret_key(self) -> str | None:
        return os.getenv(self.secret_key_env)

    @property
    def host(self) -> str | None:
        return os.getenv(self.host_env)


class StorageConfig(BaseModel):
    """Local storage settings."""

    upload_dir: str = "./data/uploads"

    @property
    def upload_path(self) -> Path:
        path = Path(os.path.expandvars(self.upload_dir)).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path


class SecurityConfig(BaseModel):
    """Security and compliance filters."""

    sensitive_patterns: list[str] = Field(default_factory=list)


class PipelineConfig(BaseModel):
    """Runtime parameters for chunking and retrieval."""

    default_top_k: int = 8
    rerank_top_n: int = 4
    chunk_size: int = 1200
    chunk_overlap: int = 200
    title_max_length: int = 120
    max_context_chars: int = 12000


class LLMProfile(BaseModel):
    """Configurable chat model definition."""

    provider: str
    model: str
    temperature: float = 0.0
    max_tokens: int | None = None
    timeout: int | None = None
    kwargs: dict[str, Any] = Field(default_factory=dict)


class EmbeddingProfile(BaseModel):
    """Configurable embedding model definition."""

    provider: str
    model: str
    kwargs: dict[str, Any] = Field(default_factory=dict)


class ComponentBindings(BaseModel):
    """Maps pipeline components to named model profiles."""

    router_llm: str = "router"
    answer_llm: str = "answer"
    parser_llm: str = "multimodal_parser"
    extractor_llm: str = "metadata_extractor"
    reranker_llm: str = "reranker"
    embedding: str = "default"


class AppConfig(BaseModel):
    """Top-level application configuration."""

    app_name: str = "GLASS-RAG"
    environment: str = "dev"
    mongodb: MongoConfig = Field(default_factory=MongoConfig)
    langfuse: LangfuseConfig = Field(default_factory=LangfuseConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    llm_profiles: dict[str, LLMProfile]
    embedding_profiles: dict[str, EmbeddingProfile]
    component_bindings: ComponentBindings = Field(default_factory=ComponentBindings)

    @model_validator(mode="after")
    def validate_bindings(self) -> "AppConfig":
        bindings = self.component_bindings
        for name in [
            bindings.router_llm,
            bindings.answer_llm,
            bindings.parser_llm,
            bindings.extractor_llm,
            bindings.reranker_llm,
        ]:
            if name not in self.llm_profiles:
                raise ValueError(f"Missing llm profile '{name}' in llm_profiles.")
        if bindings.embedding not in self.embedding_profiles:
            raise ValueError(f"Missing embedding profile '{bindings.embedding}' in embedding_profiles.")
        return self

    def llm_profile(self, profile_name: str) -> LLMProfile:
        return self.llm_profiles[profile_name]

    def embedding_profile(self, profile_name: str) -> EmbeddingProfile:
        return self.embedding_profiles[profile_name]


def _expand_env(data: Any) -> Any:
    """Recursively expands environment variables inside YAML values."""

    if isinstance(data, dict):
        return {key: _expand_env(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_expand_env(item) for item in data]
    if isinstance(data, str):
        return os.path.expandvars(data)
    return data


def load_config(config_path: str | None = None) -> AppConfig:
    """Loads YAML configuration and environment variables."""

    load_dotenv()
    path = Path(config_path or os.getenv("GLASS_RAG_CONFIG", "./config/app.example.yaml")).resolve()
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    return AppConfig.model_validate(_expand_env(payload))
