"""Host-neutral package logging and allow-listed observability models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar
from uuid import UUID

EVENT_SCHEMA_VERSION = 1


class OperationType(str, Enum):
    """Background operations that may produce lifecycle events."""

    RECORD = "record"
    EVALUATION = "evaluation"
    EMBEDDING = "embedding"
    RULE_GENERATION = "rule_generation"
    REFLECTION = "reflection"
    INDEX_REBUILD = "index_rebuild"


class TerminalState(str, Enum):
    """The complete set of terminal background-job states."""

    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLATION = "cancellation"


class RetrievalDiagnosticCode(str, Enum):
    """Safe retrieval conditions that can be exposed to observers."""

    DIMENSION_MISMATCH = "dimension_mismatch"
    INDEX_FALLBACK = "index_fallback"


class IntegrationName(str, Enum):
    """Implemented integration sources that can emit outcomes."""

    LANGCHAIN = "langchain"
    LANGGRAPH = "langgraph"


class IntegrationOutcome(str, Enum):
    """Allow-listed outcomes from framework extraction and persistence."""

    RECORDED = "recorded"
    NO_EXPERIENCE = "no_experience"
    FAILED = "failed"
    ORPHAN = "orphan"


_EnumType = TypeVar("_EnumType", bound=Enum)


def _normalize_enum(value: object, enum_type: type[_EnumType], field: str) -> _EnumType:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} is not an allow-listed value") from error


def _validate_schema_version(value: object) -> None:
    if type(value) is not int or value != EVENT_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be the integer {EVENT_SCHEMA_VERSION}")


def _validate_dimension(value: object, field: str) -> None:
    if value is not None and (type(value) is not int or value < 0):
        raise ValueError(f"{field} must be a non-negative integer or None")


@dataclass(frozen=True, slots=True)
class LifecycleEvent:
    """Safe terminal-job event containing operational identifiers only."""

    schema_version: int
    job_id: UUID
    operation: OperationType
    terminal_state: TerminalState
    duration_ms: int

    def __post_init__(self) -> None:
        _validate_schema_version(self.schema_version)
        if not isinstance(self.job_id, UUID):
            raise TypeError("job_id must be a UUID")
        object.__setattr__(
            self,
            "operation",
            _normalize_enum(self.operation, OperationType, "operation"),
        )
        object.__setattr__(
            self,
            "terminal_state",
            _normalize_enum(self.terminal_state, TerminalState, "terminal_state"),
        )
        if type(self.duration_ms) is not int or self.duration_ms < 0:
            raise ValueError("duration_ms must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class RetrievalDiagnostic:
    """Safe diagnostic for retrieval fallback and incompatible vectors."""

    code: RetrievalDiagnosticCode
    memory_id: UUID | None
    stored_dimension: int | None
    query_dimension: int | None

    def __post_init__(self) -> None:
        code = _normalize_enum(self.code, RetrievalDiagnosticCode, "code")
        object.__setattr__(self, "code", code)
        if self.memory_id is not None and not isinstance(self.memory_id, UUID):
            raise TypeError("memory_id must be a UUID or None")
        _validate_dimension(self.stored_dimension, "stored_dimension")
        _validate_dimension(self.query_dimension, "query_dimension")

        if code is RetrievalDiagnosticCode.DIMENSION_MISMATCH:
            if (
                self.memory_id is None
                or self.stored_dimension is None
                or self.query_dimension is None
            ):
                raise ValueError(
                    "dimension_mismatch requires a memory identifier and both dimensions"
                )
        elif any(
            value is not None
            for value in (
                self.memory_id,
                self.stored_dimension,
                self.query_dimension,
            )
        ):
            raise ValueError("index_fallback does not accept record-specific context")


@dataclass(frozen=True, slots=True)
class IntegrationEvent:
    """Safe integration outcome without extracted record or exception payloads."""

    schema_version: int
    integration: IntegrationName
    outcome: IntegrationOutcome
    run_id: UUID | None = None

    def __post_init__(self) -> None:
        _validate_schema_version(self.schema_version)
        object.__setattr__(
            self,
            "integration",
            _normalize_enum(self.integration, IntegrationName, "integration"),
        )
        object.__setattr__(
            self,
            "outcome",
            _normalize_enum(self.outcome, IntegrationOutcome, "outcome"),
        )
        if self.run_id is not None and not isinstance(self.run_id, UUID):
            raise TypeError("run_id must be a UUID or None")


logger = logging.getLogger("experia")
if not any(isinstance(handler, logging.NullHandler) for handler in logger.handlers):
    logger.addHandler(logging.NullHandler())


def emit_lifecycle_event(event: LifecycleEvent) -> None:
    """Emit one structured lifecycle event without expanding its safe payload."""

    if not isinstance(event, LifecycleEvent):
        raise TypeError("event must be a LifecycleEvent")
    logger.info(
        "Background job reached a terminal state.",
        extra={"experia_lifecycle_event": event},
    )


def emit_integration_event(event: IntegrationEvent) -> None:
    """Emit one safe framework-integration outcome event."""

    if not isinstance(event, IntegrationEvent):
        raise TypeError("event must be an IntegrationEvent")
    logger.info(
        "Framework integration reached an outcome.",
        extra={"experia_integration_event": event},
    )


def emit_observer_failure_diagnostic() -> None:
    """Report observer failure without exception or job payloads."""

    logger.warning(
        "Lifecycle event observer failed.",
        extra={"experia_diagnostic": "lifecycle_observer_failure"},
    )


__all__ = [
    "EVENT_SCHEMA_VERSION",
    "emit_integration_event",
    "emit_lifecycle_event",
    "emit_observer_failure_diagnostic",
    "IntegrationEvent",
    "IntegrationName",
    "IntegrationOutcome",
    "LifecycleEvent",
    "OperationType",
    "RetrievalDiagnostic",
    "RetrievalDiagnosticCode",
    "TerminalState",
    "logger",
]
