"""Access control, compliance filtering, and audit helpers."""

from __future__ import annotations

import re
from typing import Any

from .config import AppConfig
from .schemas import AuditRecord, UserContext


class AccessController:
    """Builds MongoDB filters according to user, project, and secrecy level."""

    def build_filter(self, user: UserContext) -> dict[str, Any]:
        clauses: list[dict[str, Any]] = [
            {"metadata.secrecy_level": {"$in": [level.value for level in user.allowed_levels]}},
            {
                "$or": [
                    {"metadata.department": None},
                    {"metadata.department": user.department},
                ]
            },
        ]

        if user.projects:
            clauses.append(
                {
                    "$or": [
                        {"metadata.project": None},
                        {"metadata.project": {"$in": user.projects}},
                    ]
                }
            )
        else:
            clauses.append({"metadata.project": None})

        return {"$and": clauses}


class ComplianceService:
    """Applies simple configurable redaction rules."""

    def __init__(self, config: AppConfig) -> None:
        self.patterns = [re.compile(pattern) for pattern in config.security.sensitive_patterns]

    def redact(self, text: str) -> str:
        result = text
        for pattern in self.patterns:
            result = pattern.sub("[REDACTED]", result)
        return result


class AuditFactory:
    """Creates normalized audit records."""

    @staticmethod
    def build(event_type: str, actor_id: str, payload: dict[str, Any]) -> AuditRecord:
        return AuditRecord(event_type=event_type, actor_id=actor_id, payload=payload)
