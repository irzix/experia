from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

from experia.core.exceptions import StorageError
from experia.experience.models import ExperienceRecord, Lesson
from experia.memory.models import Memory, MemoryType
from experia.memory.serialization import StorageSerializer
from experia.memory.store import SQLiteStore


def test_experience_round_trip_uses_canonical_json_and_preserves_offset():
    serializer = StorageSerializer()
    created_at = datetime(
        2024, 6, 1, 12, 30, 45, 123456, tzinfo=timezone(timedelta(hours=5, minutes=45))
    )
    experience = ExperienceRecord(
        task="deploy",
        action="restart",
        result="ok",
        context={"z": [3, {"active": True}], "a": "café"},
        created_at=created_at,
    )

    encoded = serializer.encode_experience(experience)
    decoded = serializer.decode_experience(encoded.values())

    assert encoded.context == '{"a":"café","z":[3,{"active":true}]}'
    assert decoded == experience
    assert isinstance(decoded.id, UUID)
    assert decoded.created_at.utcoffset() == timedelta(hours=5, minutes=45)
    assert decoded.created_at.astimezone(timezone.utc) == created_at.astimezone(
        timezone.utc
    )


def test_memory_round_trip_preserves_types_nested_metadata_and_embedding_order():
    serializer = StorageSerializer()
    created_at = datetime(2024, 2, 2, 8, 0, tzinfo=timezone(timedelta(hours=-4)))
    updated_at = datetime(
        2024, 2, 3, 9, 15, tzinfo=timezone(timedelta(hours=9, minutes=30))
    )
    expires_at = datetime(2024, 3, 1, tzinfo=timezone.utc)
    memory = Memory(
        content="Check logs first",
        type=MemoryType.LESSON,
        metadata={"nested": {"b": 2, "a": [None, "value"]}},
        embedding=[0.25, -1.5, 3.0, 0.0],
        created_at=created_at,
        updated_at=updated_at,
        expires_at=expires_at,
    )

    encoded = serializer.encode_memory(memory)
    decoded = serializer.decode_memory(encoded.values())

    assert encoded.metadata == '{"nested":{"a":[null,"value"],"b":2}}'
    assert encoded.embedding == "[0.25,-1.5,3.0,0.0]"
    assert decoded == memory
    assert isinstance(decoded.id, UUID)
    assert isinstance(decoded.type, MemoryType)
    assert decoded.embedding == [0.25, -1.5, 3.0, 0.0]
    assert decoded.created_at.utcoffset() == timedelta(hours=-4)
    assert decoded.updated_at.utcoffset() == timedelta(hours=9, minutes=30)


def test_lesson_round_trip_preserves_public_model_types():
    serializer = StorageSerializer()
    lesson = Lesson(
        experience_id=UUID("3e7064be-36b7-42bd-bd1a-10d7eb122eb9"),
        content="Inspect the bound port",
        root_cause="Port already in use",
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )

    decoded = serializer.decode_lesson(serializer.encode_lesson(lesson).values())

    assert decoded == lesson
    assert isinstance(decoded.id, UUID)
    assert isinstance(decoded.experience_id, UUID)


def test_encode_rejects_non_json_values_with_context():
    serializer = StorageSerializer()
    experience = ExperienceRecord(
        task="task",
        action="action",
        result="result",
        context={"unsupported": {1, 2}},
    )

    with pytest.raises(StorageError) as raised:
        serializer.encode_experience(experience)

    error = raised.value
    assert error.operation == "encode"
    assert error.table == "experiences"
    assert error.record_ids == (str(experience.id),)
    assert error.field == "context"


def test_encode_rejects_naive_timestamps_with_context():
    serializer = StorageSerializer()
    memory = Memory(
        content="memory",
        type=MemoryType.FACT,
        created_at=datetime(2024, 1, 1),
    )

    with pytest.raises(StorageError) as raised:
        serializer.encode_memory(memory)

    error = raised.value
    assert error.operation == "encode"
    assert error.table == "memories"
    assert error.record_ids == (str(memory.id),)
    assert error.field == "created_at"


def test_decode_rejects_malformed_uuid_json_enum_and_timestamp_contextually():
    serializer = StorageSerializer()
    experience = ExperienceRecord(task="task", action="action", result="result")
    memory = Memory(content="memory", type=MemoryType.FACT)
    experience_values = list(serializer.encode_experience(experience).values())
    memory_values = list(serializer.encode_memory(memory).values())

    cases = [
        (serializer.decode_experience, experience_values, 0, "not-a-uuid", "id"),
        (serializer.decode_experience, experience_values, 5, '{"a":NaN}', "context"),
        (
            serializer.decode_experience,
            experience_values,
            6,
            "not-a-time",
            "created_at",
        ),
        (serializer.decode_memory, memory_values, 2, "not-a-memory-type", "type"),
    ]

    for decoder, original, index, malformed, field in cases:
        row = original.copy()
        row[index] = malformed
        with pytest.raises(StorageError) as raised:
            decoder(row)
        error = raised.value
        assert error.operation == "decode"
        assert error.field == field
        assert error.table in {"experiences", "memories"}
        assert error.record_ids


@pytest.mark.asyncio
async def test_serialization_failure_occurs_before_database_write(tmp_path):
    store = SQLiteStore(str(tmp_path / "serializer.db"))
    await store.initialize()
    try:
        baseline = ExperienceRecord(task="valid", action="save", result="ok")
        await store.save_experience(baseline)
        invalid = ExperienceRecord(
            task="invalid",
            action="save",
            result="must fail",
            context={"value": float("nan")},
        )

        with pytest.raises(StorageError) as raised:
            await store.save_experience(invalid)

        assert raised.value.operation == "encode"
        conn = store._require_conn()
        cursor = await conn.execute("SELECT id FROM experiences ORDER BY id")
        assert await cursor.fetchall() == [(str(baseline.id),)]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_decode_failure_does_not_modify_malformed_stored_value(tmp_path):
    store = SQLiteStore(str(tmp_path / "malformed.db"))
    await store.initialize()
    try:
        experience = ExperienceRecord(task="task", action="action", result="result")
        await store.save_experience(experience)
        malformed = '{"duplicate":1,"duplicate":2}'
        conn = store._require_conn()
        await conn.execute(
            "UPDATE experiences SET context = ? WHERE id = ?",
            (malformed, str(experience.id)),
        )
        await conn.commit()

        with pytest.raises(StorageError) as raised:
            await store.get_experience(experience.id)

        error = raised.value
        assert error.operation == "decode"
        assert error.table == "experiences"
        assert error.record_ids == (str(experience.id),)
        assert error.field == "context"
        cursor = await conn.execute(
            "SELECT context FROM experiences WHERE id = ?", (str(experience.id),)
        )
        assert await cursor.fetchone() == (malformed,)
    finally:
        await store.close()
