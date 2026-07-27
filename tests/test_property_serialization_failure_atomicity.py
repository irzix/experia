"""Property tests for storage serialization-failure atomicity."""

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from experia.core.exceptions import StorageError
from experia.experience.models import ExperienceRecord, Lesson
from experia.memory.models import Memory, MemoryType
from experia.memory.store import SQLiteStore
from tests.quality_profiles import SERIALIZER_MIN_EXAMPLES

_SERIALIZABLE_FIELDS = ("task", "action", "result", "context", "metadata")
_FIXED_TIME = datetime(2024, 1, 1, tzinfo=timezone.utc)

_JSON_SCALARS = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2**31), max_value=2**31 - 1),
    st.floats(allow_nan=False, allow_infinity=False, width=32),
    st.text(max_size=20),
)
_JSON_VALUES = st.recursive(
    _JSON_SCALARS,
    lambda children: st.one_of(
        st.lists(children, max_size=3),
        st.dictionaries(st.text(max_size=10), children, max_size=3),
    ),
    max_leaves=8,
)
_JSON_OBJECTS = st.dictionaries(st.text(max_size=10), _JSON_VALUES, max_size=4)
_UNSERIALIZABLE_VALUES = st.one_of(
    st.sets(st.integers(min_value=-10, max_value=10), max_size=4),
    st.tuples(
        st.integers(min_value=-10, max_value=10),
        st.integers(min_value=-10, max_value=10),
    ),
)


async def _logical_database_snapshot(store: SQLiteStore) -> tuple[Any, ...]:
    """Return application rows independent of SQLite's physical file state."""
    connection = store._require_conn()
    snapshot = []
    for table in ("experiences", "lessons", "memories"):
        cursor = await connection.execute(f"SELECT * FROM {table} ORDER BY id")
        snapshot.append((table, tuple(await cursor.fetchall())))
    return tuple(snapshot)


async def _seed_database(store: SQLiteStore) -> None:
    experience = ExperienceRecord(
        id=UUID("00000000-0000-4000-8000-000000000001"),
        task="baseline task",
        action="baseline action",
        result="baseline result",
        context={"baseline": True},
        created_at=_FIXED_TIME,
    )
    lesson = Lesson(
        id=UUID("00000000-0000-4000-8000-000000000002"),
        experience_id=experience.id,
        content="baseline lesson",
        created_at=_FIXED_TIME,
    )
    memory = Memory(
        id=UUID("00000000-0000-4000-8000-000000000003"),
        content="baseline memory",
        type=MemoryType.FACT,
        metadata={"baseline": True},
        created_at=_FIXED_TIME,
        updated_at=_FIXED_TIME,
    )
    await store.save_experience(experience)
    await store.save_lesson(lesson)
    await store.save_memory(memory)


# Feature: open-source-project-improvements, Property 10: Serialization failure has no partial effect
# **Validates: Requirements 2.8**
@pytest.mark.asyncio
@settings(max_examples=SERIALIZER_MIN_EXAMPLES, deadline=None)
@given(
    task=st.text(max_size=30),
    action=st.text(max_size=30),
    result=st.text(max_size=30),
    context=_JSON_OBJECTS,
    metadata=_JSON_OBJECTS,
    invalid_values=st.tuples(*[_UNSERIALIZABLE_VALUES for _ in _SERIALIZABLE_FIELDS]),
)
async def test_serialization_failure_has_no_partial_effect(
    task: str,
    action: str,
    result: str,
    context: dict[str, Any],
    metadata: dict[str, Any],
    invalid_values: tuple[Any, ...],
) -> None:
    store = SQLiteStore(":memory:")
    await store.initialize()
    try:
        await _seed_database(store)
        valid_experience = ExperienceRecord(
            id=UUID("00000000-0000-4000-8000-000000000004"),
            task=task,
            action=action,
            result=result,
            context=context,
            created_at=_FIXED_TIME,
        )
        valid_memory = Memory(
            id=UUID("00000000-0000-4000-8000-000000000005"),
            content="generated memory",
            type=MemoryType.FACT,
            metadata=metadata,
            created_at=_FIXED_TIME,
            updated_at=_FIXED_TIME,
        )

        for field, invalid_value in zip(
            _SERIALIZABLE_FIELDS, invalid_values, strict=True
        ):
            before = await _logical_database_snapshot(store)
            if field == "metadata":
                record = valid_memory.model_copy(
                    update={"metadata": {"unserializable": invalid_value}}
                )
                save = store.save_memory(record)
                expected_table = "memories"
                expected_id = record.id
            else:
                replacement = (
                    {"unserializable": invalid_value}
                    if field == "context"
                    else invalid_value
                )
                record = valid_experience.model_copy(update={field: replacement})
                save = store.save_experience(record)
                expected_table = "experiences"
                expected_id = record.id

            with pytest.raises(StorageError) as raised:
                await save

            assert raised.value.table == expected_table
            assert raised.value.record_ids == (str(expected_id),)
            assert await _logical_database_snapshot(store) == before
    finally:
        await store.close()
