"""Public, contextual exception types for Experia.

Exception context is deliberately limited to operational identifiers. Original
payloads and dependency exception text must not be stored in these attributes;
callers can use normal exception chaining when local debugging context is needed.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypeAlias
from uuid import UUID

PathComponent: TypeAlias = str | int
RecordIdentifier: TypeAlias = str | UUID


class ExperiaError(Exception):
    """Base exception for all Experia errors."""


class ConfigurationError(ExperiaError):
    """Raised when Experia is misconfigured.

    Existing positional exception arguments remain supported. New call sites can
    attach the safe identifiers ``feature``, ``parameter``, and ``extra``.
    """

    def __init__(
        self,
        *args: object,
        feature: str | None = None,
        parameter: str | None = None,
        extra: str | None = None,
    ) -> None:
        super().__init__(*(args or ("Experia configuration is invalid.",)))
        self.feature = feature
        self.parameter = parameter
        self.extra = extra


class LifecycleError(ExperiaError):
    """Raised when an operation is invalid for the current lifecycle state."""

    def __init__(
        self,
        message: str | None = None,
        *,
        state: str,
        operation: str,
    ) -> None:
        super().__init__(
            message or "Operation is unavailable in the current lifecycle state."
        )
        self.state = state
        self.operation = operation


class EvaluationError(ExperiaError):
    """Raised when an experience evaluation fails."""


@dataclass(frozen=True)
class FailureDetail:
    """Non-sensitive context for one failed background operation."""

    job_id: UUID
    operation: str
    experience_id: UUID | None = None
    error_type: str = "evaluation_failure"


class EvaluationFailure(EvaluationError):
    """A contextual failure from evaluation or related background work."""

    def __init__(
        self,
        message: str | None = None,
        *,
        job_id: UUID,
        operation: str,
        experience_id: UUID | None = None,
        failures: Iterable[FailureDetail] = (),
    ) -> None:
        super().__init__(message or "Background evaluation work failed.")
        self.job_id = job_id
        self.operation = operation
        self.experience_id = experience_id
        self.failures = tuple(failures)


class StorageError(ExperiaError):
    """Raised when a storage operation fails.

    Existing positional exception arguments remain supported. Structured context
    contains identifiers only and never retains a record payload or raw cause.
    """

    def __init__(
        self,
        *args: object,
        operation: str | None = None,
        table: str | None = None,
        record_ids: Iterable[RecordIdentifier] | RecordIdentifier = (),
        migration: str | None = None,
        field: str | None = None,
    ) -> None:
        super().__init__(*(args or ("Storage operation failed.",)))
        self.operation = operation
        self.table = table
        if isinstance(record_ids, (str, UUID)):
            record_ids = (record_ids,)
        self.record_ids = tuple(str(record_id) for record_id in record_ids)
        self.migration = migration
        self.field = field


class SanitizationError(ExperiaError):
    """Raised when data protection fails before an external sink is reached."""

    def __init__(
        self,
        message: str | None = None,
        *,
        path: Iterable[PathComponent] = (),
        operation: str | None = None,
    ) -> None:
        super().__init__(message or "Data sanitization failed.")
        self.path = tuple(path)
        self.operation = operation


class UnavailableFeatureError(ConfigurationError):
    """Raised when an importable feature is not operational in this release."""

    def __init__(
        self,
        feature: str,
        *,
        status: str = "planned",
        message: str | None = None,
    ) -> None:
        super().__init__(
            message or "The requested Experia feature is unavailable.",
            feature=feature,
        )
        self.status = status


__all__ = [
    "ConfigurationError",
    "EvaluationError",
    "EvaluationFailure",
    "ExperiaError",
    "FailureDetail",
    "LifecycleError",
    "SanitizationError",
    "StorageError",
    "UnavailableFeatureError",
]
