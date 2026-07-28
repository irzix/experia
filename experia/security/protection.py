"""Non-mutating protection for values crossing Experia data boundaries."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Protocol

from pydantic import BaseModel

from experia.core.exceptions import PathComponent, SanitizationError


class Sanitizer(Protocol):
    """Transforms one copied leaf value at its location in a payload."""

    def sanitize(
        self,
        value: Any,
        *,
        path: tuple[PathComponent, ...],
    ) -> Any:
        """Return a protected replacement for ``value``."""


class DataProtectionLayer:
    """Copy and optionally sanitize payload values before they reach a sink.

    Mapping keys are retained as schema or control data. Mapping values,
    sequence members, set members, and Pydantic fields are traversed
    recursively. A sanitizer only receives copied leaf values, so even a
    sanitizer that mutates its argument cannot mutate caller-owned input.
    """

    def __init__(self, sanitizer: Sanitizer | None = None) -> None:
        self._sanitizer = sanitizer

    def protect_external(self, fields: Mapping[str, Any]) -> dict[str, Any]:
        """Return a protected copy of fields for an external request."""

        return self._protect(fields, operation="external_request")

    def protect_metadata(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        """Return a protected copy of metadata for logging or observation."""

        return self._protect(metadata, operation="log_metadata")

    def protect_sink(
        self,
        fields: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Atomically prepare one external request and its log metadata.

        Both structures are copied, recursively protected, and proven JSON
        serializable before either can reach a side-effecting sink. This keeps
        request transmission and associated metadata emission fail-closed even
        when a sanitizer returns an unsupported value.
        """

        try:
            protected_fields = self.protect_external(fields)
            protected_metadata = self.protect_metadata(metadata)
            self._validate_serializable(
                protected_fields,
                operation="external_request",
            )
            self._validate_serializable(
                protected_metadata,
                operation="log_metadata",
            )
        except SanitizationError:
            raise
        except Exception:
            raise SanitizationError(
                "Data protection serialization failed.",
                operation="external_request",
            ) from None
        return protected_fields, protected_metadata

    @staticmethod
    def _validate_serializable(values: Mapping[str, Any], *, operation: str) -> None:
        try:
            json.dumps(
                values,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
            )
        except (TypeError, ValueError, OverflowError):
            raise SanitizationError(
                "Data protection serialization failed.",
                operation=operation,
            ) from None

    def _protect(
        self,
        values: Mapping[str, Any],
        *,
        operation: str,
    ) -> dict[str, Any]:
        return {
            key: self._copy_value(
                value,
                path=(self._path_component(key),),
                operation=operation,
            )
            for key, value in values.items()
        }

    def _copy_value(
        self,
        value: Any,
        *,
        path: tuple[PathComponent, ...],
        operation: str,
    ) -> Any:
        if isinstance(value, BaseModel):
            return self._copy_model(value, path=path, operation=operation)
        if isinstance(value, Mapping):
            return {
                key: self._copy_value(
                    nested,
                    path=path + (self._path_component(key),),
                    operation=operation,
                )
                for key, nested in value.items()
            }
        if isinstance(value, list):
            return [
                self._copy_value(
                    nested,
                    path=path + (index,),
                    operation=operation,
                )
                for index, nested in enumerate(value)
            ]
        if isinstance(value, tuple):
            copied = tuple(
                self._copy_value(
                    nested,
                    path=path + (index,),
                    operation=operation,
                )
                for index, nested in enumerate(value)
            )
            if hasattr(value, "_fields"):
                return type(value)(*copied)
            return copied
        if isinstance(value, set):
            return {
                self._copy_value(
                    nested,
                    path=path + (index,),
                    operation=operation,
                )
                for index, nested in enumerate(value)
            }
        if isinstance(value, frozenset):
            return frozenset(
                self._copy_value(
                    nested,
                    path=path + (index,),
                    operation=operation,
                )
                for index, nested in enumerate(value)
            )

        copied = deepcopy(value)
        if self._sanitizer is None:
            return copied

        succeeded, protected = self._sanitize(copied, path=path)
        if not succeeded:
            raise SanitizationError(path=path, operation=operation)
        return protected

    def _copy_model(
        self,
        model: BaseModel,
        *,
        path: tuple[PathComponent, ...],
        operation: str,
    ) -> BaseModel:
        updates = {
            field_name: self._copy_value(
                getattr(model, field_name),
                path=path + (field_name,),
                operation=operation,
            )
            for field_name in type(model).model_fields
        }
        if model.model_extra:
            updates.update(
                {
                    field_name: self._copy_value(
                        field_value,
                        path=path + (self._path_component(field_name),),
                        operation=operation,
                    )
                    for field_name, field_value in model.model_extra.items()
                }
            )
        return model.model_copy(update=updates, deep=True)

    def _sanitize(
        self,
        value: Any,
        *,
        path: tuple[PathComponent, ...],
    ) -> tuple[bool, Any]:
        """Call the sanitizer without retaining its possibly sensitive error."""

        try:
            return True, self._sanitizer.sanitize(value, path=path)  # type: ignore[union-attr]
        except Exception:
            # Deliberately discard the original exception. It may contain the raw
            # value in its message or attributes, so it must not be chained to the
            # safe boundary error raised by the caller.
            return False, None

    @staticmethod
    def _path_component(key: object) -> PathComponent:
        if isinstance(key, str):
            return key
        if isinstance(key, int):
            return int(key)
        return str(key)


__all__ = ["DataProtectionLayer", "Sanitizer"]
