"""Langfuse helpers used across pipelines."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler

from .config import AppConfig


class ObservabilityManager:
    """Builds Langfuse callbacks and spans when Langfuse is configured."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def enabled(self) -> bool:
        settings = self.config.langfuse
        return bool(settings.enabled and settings.public_key and settings.secret_key)

    def callbacks(self) -> list[CallbackHandler]:
        if not self.enabled():
            return []
        return [CallbackHandler()]

    @contextmanager
    def trace(
        self,
        *,
        name: str,
        user_id: str | None = None,
        session_id: str | None = None,
        input_payload: dict[str, Any] | None = None,
    ) -> Iterator[str | None]:
        """Wraps application logic in a Langfuse span and yields the trace id."""

        if not self.enabled():
            yield None
            return

        langfuse = get_client()
        with langfuse.start_as_current_observation(as_type="span", name=name) as span:
            with propagate_attributes(trace_name=name, user_id=user_id, session_id=session_id):
                if input_payload:
                    span.update(input=input_payload)
                yield span.trace_id

    def flush(self) -> None:
        if self.enabled():
            get_client().flush()

