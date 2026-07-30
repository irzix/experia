"""Shared validation for framework-produced experience candidates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from experia.core.logging import (
    EVENT_SCHEMA_VERSION,
    IntegrationEvent,
    IntegrationName,
    IntegrationOutcome,
    emit_integration_event,
)


class ExperienceCandidate(BaseModel):
    """Internal candidate matching the current ``Learner.record`` inputs."""

    model_config = ConfigDict(extra="ignore")

    task: str
    action: str
    result: str
    context: dict[str, Any] = Field(default_factory=dict)

    @field_validator("task", "action", "result")
    @classmethod
    def require_non_blank_record_input(cls, value: str) -> str:
        """Reject empty and whitespace-only record inputs without rewriting them."""

        if not value.strip():
            raise ValueError("experience record inputs must not be blank")
        return value


def finalize_experience_candidate(
    value: object,
    *,
    integration: IntegrationName,
    run_id: UUID | None = None,
    default_context: Mapping[str, Any] | None = None,
) -> ExperienceCandidate | None:
    """Validate one candidate or emit its single safe no-experience outcome."""

    try:
        if isinstance(value, ExperienceCandidate):
            payload: dict[str, Any] = value.model_dump()
        elif isinstance(value, Mapping):
            payload = dict(value)
        else:
            raise TypeError("experience candidate must be a mapping")

        if default_context is not None:
            supplied_context = payload.get("context", {})
            if not isinstance(supplied_context, Mapping):
                raise TypeError("experience candidate context must be a mapping")
            payload["context"] = {
                **dict(supplied_context),
                **dict(default_context),
            }

        return ExperienceCandidate.model_validate(payload)
    except (TypeError, ValueError, ValidationError):
        emit_integration_event(
            IntegrationEvent(
                schema_version=EVENT_SCHEMA_VERSION,
                integration=integration,
                outcome=IntegrationOutcome.NO_EXPERIENCE,
                run_id=run_id,
            )
        )
        return None


__all__ = ["ExperienceCandidate", "finalize_experience_candidate"]
