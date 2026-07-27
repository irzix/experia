"""Typed, strict serialization boundary for SQLite storage."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence
from uuid import UUID

from experia.core.exceptions import StorageError
from experia.experience.models import ExperienceRecord, Lesson
from experia.memory.models import Memory, MemoryType


@dataclass(frozen=True)
class EncodedExperience:
    """Database-ready values for an :class:`ExperienceRecord`."""

    id: str
    task: str
    action: str
    result: str
    agent_role: str
    context: str
    created_at: str

    def values(self) -> tuple[object, ...]:
        return (
            self.id,
            self.task,
            self.action,
            self.result,
            self.agent_role,
            self.context,
            self.created_at,
        )


@dataclass(frozen=True)
class EncodedLesson:
    """Database-ready values for a :class:`Lesson`."""

    id: str
    experience_id: str
    content: str
    agent_role: str
    root_cause: str | None
    confidence: float
    created_at: str

    def values(self) -> tuple[object, ...]:
        return (
            self.id,
            self.experience_id,
            self.content,
            self.agent_role,
            self.root_cause,
            self.confidence,
            self.created_at,
        )


@dataclass(frozen=True)
class EncodedMemory:
    """Database-ready values for a :class:`Memory`."""

    id: str
    content: str
    type: str
    agent_role: str
    confidence: float
    importance: float
    source: str | None
    metadata: str
    embedding: str | None
    reinforcement_count: int
    success_count: int
    created_at: str
    updated_at: str
    expires_at: str | None

    @property
    def embedding_dimension(self) -> int | None:
        """Return the dimension of the already-validated encoded embedding."""
        if self.embedding is None:
            return None
        decoded = json.loads(self.embedding)
        return len(decoded)

    def values(self) -> tuple[object, ...]:
        return (
            self.id,
            self.content,
            self.type,
            self.agent_role,
            self.confidence,
            self.importance,
            self.source,
            self.metadata,
            self.embedding,
            self.reinforcement_count,
            self.success_count,
            self.created_at,
            self.updated_at,
            self.expires_at,
        )


class StorageSerializer:
    """Convert public models to and from typed SQLite values.

    Encoding is deliberately completed before callers enter a transaction. JSON
    is canonical and rejects non-standard numbers and lossy Python containers.
    Decoding never writes and reports only safe storage identifiers.
    """

    def encode_experience(self, value: ExperienceRecord) -> EncodedExperience:
        table = "experiences"
        record_id = value.id
        return EncodedExperience(
            id=str(value.id),
            task=value.task,
            action=value.action,
            result=value.result,
            agent_role=value.agent_role,
            context=self._encode_json(
                value.context,
                table=table,
                record_id=record_id,
                field="context",
            ),
            created_at=self._encode_timestamp(
                value.created_at,
                table=table,
                record_id=record_id,
                field="created_at",
            ),
        )

    def decode_experience(self, row: Sequence[Any]) -> ExperienceRecord:
        table = "experiences"
        record_id = self._row_identifier(row)
        try:
            return ExperienceRecord(
                id=self._decode_uuid(
                    row[0], table=table, record_id=record_id, field="id"
                ),
                task=row[1],
                action=row[2],
                result=row[3],
                agent_role=row[4],
                context=self._decode_json(
                    row[5],
                    table=table,
                    record_id=record_id,
                    field="context",
                    expected="mapping",
                    legacy_default={},
                ),
                created_at=self._decode_timestamp(
                    row[6],
                    table=table,
                    record_id=record_id,
                    field="created_at",
                ),
            )
        except StorageError:
            raise
        except Exception as exc:
            raise self._decode_error(table, record_id) from exc

    def encode_lesson(self, value: Lesson) -> EncodedLesson:
        table = "lessons"
        record_id = value.id
        return EncodedLesson(
            id=str(value.id),
            experience_id=str(value.experience_id),
            content=value.content,
            agent_role=value.agent_role,
            root_cause=value.root_cause,
            confidence=value.confidence,
            created_at=self._encode_timestamp(
                value.created_at,
                table=table,
                record_id=record_id,
                field="created_at",
            ),
        )

    def decode_lesson(self, row: Sequence[Any]) -> Lesson:
        table = "lessons"
        record_id = self._row_identifier(row)
        try:
            return Lesson(
                id=self._decode_uuid(
                    row[0], table=table, record_id=record_id, field="id"
                ),
                experience_id=self._decode_uuid(
                    row[1],
                    table=table,
                    record_id=record_id,
                    field="experience_id",
                ),
                content=row[2],
                agent_role=row[3],
                root_cause=row[4],
                confidence=row[5],
                created_at=self._decode_timestamp(
                    row[6],
                    table=table,
                    record_id=record_id,
                    field="created_at",
                ),
            )
        except StorageError:
            raise
        except Exception as exc:
            raise self._decode_error(table, record_id) from exc

    def encode_memory(self, value: Memory) -> EncodedMemory:
        table = "memories"
        record_id = value.id
        return EncodedMemory(
            id=str(value.id),
            content=value.content,
            type=value.type.value,
            agent_role=value.agent_role,
            confidence=value.confidence,
            importance=value.importance,
            source=value.source,
            metadata=self._encode_json(
                value.metadata,
                table=table,
                record_id=record_id,
                field="metadata",
            ),
            embedding=(
                None
                if value.embedding is None
                else self._encode_json(
                    value.embedding,
                    table=table,
                    record_id=record_id,
                    field="embedding",
                )
            ),
            reinforcement_count=value.reinforcement_count,
            success_count=value.success_count,
            created_at=self._encode_timestamp(
                value.created_at,
                table=table,
                record_id=record_id,
                field="created_at",
            ),
            updated_at=self._encode_timestamp(
                value.updated_at,
                table=table,
                record_id=record_id,
                field="updated_at",
            ),
            expires_at=(
                None
                if value.expires_at is None
                else self._encode_timestamp(
                    value.expires_at,
                    table=table,
                    record_id=record_id,
                    field="expires_at",
                )
            ),
        )

    def decode_memory(self, row: Sequence[Any]) -> Memory:
        table = "memories"
        record_id = self._row_identifier(row)
        try:
            return Memory(
                id=self._decode_uuid(
                    row[0], table=table, record_id=record_id, field="id"
                ),
                content=row[1],
                type=self._decode_memory_type(row[2], table=table, record_id=record_id),
                agent_role=row[3],
                confidence=row[4],
                importance=row[5],
                source=row[6],
                metadata=self._decode_json(
                    row[7],
                    table=table,
                    record_id=record_id,
                    field="metadata",
                    expected="mapping",
                    legacy_default={},
                ),
                embedding=self._decode_json(
                    row[8],
                    table=table,
                    record_id=record_id,
                    field="embedding",
                    expected="embedding",
                    legacy_default=None,
                ),
                reinforcement_count=row[9] or 0,
                success_count=row[10] or 0,
                created_at=self._decode_timestamp(
                    row[11],
                    table=table,
                    record_id=record_id,
                    field="created_at",
                ),
                updated_at=self._decode_timestamp(
                    row[12],
                    table=table,
                    record_id=record_id,
                    field="updated_at",
                ),
                expires_at=(
                    None
                    if row[13] is None
                    else self._decode_timestamp(
                        row[13],
                        table=table,
                        record_id=record_id,
                        field="expires_at",
                    )
                ),
            )
        except StorageError:
            raise
        except Exception as exc:
            raise self._decode_error(table, record_id) from exc

    @classmethod
    def _encode_json(
        cls,
        value: Any,
        *,
        table: str,
        record_id: UUID,
        field: str,
    ) -> str:
        try:
            cls._validate_json_value(value, active_containers=set())
            return json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        except Exception as exc:
            raise StorageError(
                "Record could not be encoded for storage.",
                operation="encode",
                table=table,
                record_ids=(record_id,),
                field=field,
            ) from exc

    @classmethod
    def _validate_json_value(cls, value: Any, *, active_containers: set[int]) -> None:
        if value is None or type(value) in (bool, int, str):
            return
        if type(value) is float:
            if not math.isfinite(value):
                raise ValueError("JSON numbers must be finite")
            return
        if type(value) is list:
            identity = id(value)
            if identity in active_containers:
                raise ValueError("JSON values must not contain cycles")
            active_containers.add(identity)
            try:
                for item in value:
                    cls._validate_json_value(item, active_containers=active_containers)
            finally:
                active_containers.remove(identity)
            return
        if type(value) is dict:
            identity = id(value)
            if identity in active_containers:
                raise ValueError("JSON values must not contain cycles")
            active_containers.add(identity)
            try:
                for key, item in value.items():
                    if type(key) is not str:
                        raise TypeError("JSON object keys must be strings")
                    cls._validate_json_value(item, active_containers=active_containers)
            finally:
                active_containers.remove(identity)
            return
        raise TypeError("Value is not strict JSON data")

    @classmethod
    def _decode_json(
        cls,
        value: Any,
        *,
        table: str,
        record_id: tuple[str, ...],
        field: str,
        expected: str,
        legacy_default: Any,
    ) -> Any:
        if value is None:
            return legacy_default
        try:
            if not isinstance(value, str):
                raise TypeError("Stored JSON must be text")
            decoded = json.loads(
                value,
                parse_constant=cls._reject_json_constant,
                object_pairs_hook=cls._object_without_duplicate_keys,
            )
            cls._validate_json_value(decoded, active_containers=set())
            if expected == "mapping" and decoded is not None:
                if type(decoded) is not dict:
                    raise TypeError("Stored JSON must contain an object or null")
            elif expected == "embedding" and decoded is not None:
                if type(decoded) is not list or any(
                    type(item) is not float or not math.isfinite(item)
                    for item in decoded
                ):
                    raise TypeError("Stored embedding must contain finite floats")
            return decoded
        except Exception as exc:
            raise StorageError(
                "Stored value could not be decoded.",
                operation="decode",
                table=table,
                record_ids=record_id,
                field=field,
            ) from exc

    @staticmethod
    def _reject_json_constant(value: str) -> None:
        raise ValueError("Non-standard JSON constant")

    @staticmethod
    def _object_without_duplicate_keys(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON object key")
            result[key] = value
        return result

    @staticmethod
    def _encode_timestamp(
        value: datetime,
        *,
        table: str,
        record_id: UUID,
        field: str,
    ) -> str:
        try:
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("Timestamp must be timezone-aware")
            return value.isoformat()
        except Exception as exc:
            raise StorageError(
                "Record could not be encoded for storage.",
                operation="encode",
                table=table,
                record_ids=(record_id,),
                field=field,
            ) from exc

    @staticmethod
    def _decode_timestamp(
        value: Any,
        *,
        table: str,
        record_id: tuple[str, ...],
        field: str,
    ) -> datetime:
        try:
            if not isinstance(value, str):
                raise TypeError("Stored timestamp must be text")
            decoded = datetime.fromisoformat(value)
            if decoded.tzinfo is None or decoded.utcoffset() is None:
                raise ValueError("Stored timestamp must be timezone-aware")
            return decoded
        except Exception as exc:
            raise StorageError(
                "Stored value could not be decoded.",
                operation="decode",
                table=table,
                record_ids=record_id,
                field=field,
            ) from exc

    @staticmethod
    def _decode_uuid(
        value: Any,
        *,
        table: str,
        record_id: tuple[str, ...],
        field: str,
    ) -> UUID:
        try:
            if not isinstance(value, str):
                raise TypeError("Stored UUID must be text")
            return UUID(value)
        except Exception as exc:
            raise StorageError(
                "Stored value could not be decoded.",
                operation="decode",
                table=table,
                record_ids=record_id,
                field=field,
            ) from exc

    @staticmethod
    def _decode_memory_type(
        value: Any,
        *,
        table: str,
        record_id: tuple[str, ...],
    ) -> MemoryType:
        try:
            if not isinstance(value, str):
                raise TypeError("Stored memory type must be text")
            return MemoryType(value)
        except Exception as exc:
            raise StorageError(
                "Stored value could not be decoded.",
                operation="decode",
                table=table,
                record_ids=record_id,
                field="type",
            ) from exc

    @staticmethod
    def _row_identifier(row: Sequence[Any]) -> tuple[str, ...]:
        try:
            return () if row[0] is None else (str(row[0]),)
        except Exception:
            return ()

    @staticmethod
    def _decode_error(table: str, record_id: tuple[str, ...]) -> StorageError:
        return StorageError(
            "Stored record could not be decoded.",
            operation="decode",
            table=table,
            record_ids=record_id,
        )


__all__ = [
    "EncodedExperience",
    "EncodedLesson",
    "EncodedMemory",
    "StorageSerializer",
]
