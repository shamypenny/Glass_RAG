"""Application wiring for GLASS-RAG."""

from __future__ import annotations

from functools import lru_cache

from glass_rag.config import AppConfig, load_config
from glass_rag.factories import ModelFactory
from glass_rag.governance import AccessController, ComplianceService
from glass_rag.ingestion.chunker import SmartChunker
from glass_rag.ingestion.parser import DocumentParser
from glass_rag.ingestion.pipeline import KnowledgeIngestionPipeline
from glass_rag.observability import ObservabilityManager
from glass_rag.pipelines.query_pipeline import QueryPipeline
from glass_rag.retrieval.reranker import LLMReranker
from glass_rag.retrieval.retrievers import RetrievalService
from glass_rag.storage.mongo import MongoGateway


class GlassRAGApplication:
    """Builds and exposes the main application services."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.observability = ObservabilityManager(config)
        self.mongo = MongoGateway(config)
        self.model_factory = ModelFactory(config)
        self.access_controller = AccessController()
        self.compliance = ComplianceService(config)

        router_llm = self.model_factory.create_chat_model(config.component_bindings.router_llm)
        answer_llm = self.model_factory.create_chat_model(config.component_bindings.answer_llm)
        parser_llm = self.model_factory.create_chat_model(config.component_bindings.parser_llm)
        extractor_llm = self.model_factory.create_chat_model(config.component_bindings.extractor_llm)
        reranker_llm = self.model_factory.create_chat_model(config.component_bindings.reranker_llm)
        embeddings = self.model_factory.create_embedding_model(config.component_bindings.embedding)

        parser = DocumentParser(
            parser_llm=parser_llm,
            extractor_llm=extractor_llm,
            title_max_length=config.pipeline.title_max_length,
        )
        chunker = SmartChunker(config)
        retrieval_service = RetrievalService(
            config=config,
            mongo=self.mongo,
            access_controller=self.access_controller,
            embeddings=embeddings,
        )
        reranker = LLMReranker(rerank_llm=reranker_llm, top_n=config.pipeline.rerank_top_n)

        self.ingestion_pipeline = KnowledgeIngestionPipeline(
            mongo=self.mongo,
            parser=parser,
            chunker=chunker,
            embeddings=embeddings,
            observability=self.observability,
        )
        self.query_pipeline = QueryPipeline(
            config=config,
            mongo=self.mongo,
            retrieval_service=retrieval_service,
            reranker=reranker,
            router_llm=router_llm,
            answer_llm=answer_llm,
            observability=self.observability,
            compliance=self.compliance,
        )


@lru_cache(maxsize=1)
def get_application() -> GlassRAGApplication:
    return GlassRAGApplication(load_config())

