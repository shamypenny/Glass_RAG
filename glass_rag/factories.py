"""Factories for configurable chat models and embedding models."""

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from .config import AppConfig, EmbeddingProfile, LLMProfile


class ModelFactory:
    """Creates models based on named profiles from YAML configuration."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def create_chat_model(self, profile_name: str) -> BaseChatModel:
        profile = self.config.llm_profile(profile_name)
        provider = profile.provider.lower()

        if provider in {"openai", "azure_openai"}:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=profile.model,
                temperature=profile.temperature,
                max_tokens=profile.max_tokens,
                timeout=profile.timeout,
                **profile.kwargs,
            )

        if provider == "ollama":
            from langchain_ollama import ChatOllama

            return ChatOllama(
                model=profile.model,
                temperature=profile.temperature,
                **profile.kwargs,
            )

        raise ValueError(
            f"Unsupported chat model provider '{profile.provider}'. "
            "Extend ModelFactory.create_chat_model to add support."
        )

    def create_embedding_model(self, profile_name: str) -> Embeddings:
        profile = self.config.embedding_profile(profile_name)
        provider = profile.provider.lower()

        if provider in {"openai", "azure_openai"}:
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(model=profile.model, **profile.kwargs)

        if provider == "ollama":
            from langchain_ollama import OllamaEmbeddings

            return OllamaEmbeddings(model=profile.model, **profile.kwargs)

        if provider == "huggingface":
            from langchain_huggingface import HuggingFaceEmbeddings

            return HuggingFaceEmbeddings(model_name=profile.model, model_kwargs=profile.kwargs)

        raise ValueError(
            f"Unsupported embedding provider '{profile.provider}'. "
            "Extend ModelFactory.create_embedding_model to add support."
        )

