"""Property tests for atomic lesson and derived-memory persistence."""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from uuid import UUID

import aiosqlite
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from experia.core.exceptions import StorageError
from experia.experience.models import ExperienceRecord, Lesson
from experia.memory.models import Memory, MemoryType
from experia.memory.serialization import StorageSerializer
from experia.memory.store import SQLiteStore

_SAFE_TEXT = st.text(
    alphabet=st.characters(blacklist_categories=("Cs", "Cc")),
    min_size=1,
    max_size=48,
)
_AWARE_DATETIMES = st.datetimes(
    min_value=datetime(2000, 1, 1),
    max_value=datetime(2035, 12, 31, 23, 59, 59, 999999),
    timezones=st.just(timezone.utc),
)
_UNIT_FLOATS = st.floats(
    min_value=0.0,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
)
_FAILURE_POINTS = ("success", "lesson_insert", "memory_insert", "commit")


@dataclass(frozen=True)
class LessonMemoryPair:
    """A valid experience and its corresponding lesson/derived-memory pair."""

    experience: ExperienceRecord
    lesson: Lesson
    memory: Memory


@st.composite
def lesson_memory_pairs(draw: st.DrawFn) -> LessonMemoryPair:
    experience_id = draw(st.uuids())
    lesson_id = draw(st.uuids())
    content = draw(_SAFE_TEXT)
    agent_role = draw(_SAFE_TEXT)
    root_cause = draw(st.one_of(st.none(), _SAFE_TEXT))
    confidence = draw(_UNIT_FLOATS)
    created_at = draw(_AWARE_DATETIMES)
    experience = ExperienceRecord(
        id=experience_id,
        task=draw(_SAFE_TEXT),
        action=draw(_SAFE_TEXT),
        result=draw(_SAFE_TEXT),
        agent_role=agent_role,
        created_at=created_at,
    )
    lesson = Lesson(
        id=lesson_id,
        experience_id=experience_id,
        content=content,
        agent_role=agent_role,
        root_cause=root_cause,
        confidence=confidence,
        created_at=created_at,
    )
    memory = Memory(
        id=draw(st.uuids()),
        content=content,
        type=MemoryType.LESSON,
        agent_role=agent_role,
        confidence=confidence,
        importance=draw(_UNIT_FLOATS),
        source=f"experience_{experience_id}",
        metadata={"root_cause": root_cause} if root_cause else {},
        embedding=draw(
            st.one_of(
                st.none(),
                st.lists(
                    st.floats(
                        min_value=-1_000.0,
                        max_value=1_000.0,
                        allow_nan=False,
                        allow_infinity=False,
                    ),
                    min_size=1,
                    max_size=6,
                ),
            )
        ),
        created_at=created_at,
        updated_at=created_at,
    )
    return LessonMemoryPair(experience=experience, lesson=lesson, memory=memory)


def _example_pair() -> LessonMemoryPair:
    created_at = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    experience = ExperienceRecord(
        id=UUID("11111111-1111-1111-1111-111111111111"),
        task="Diagnose deployment",
        action="Inspect logs",
        result="Found an occupied port",
        agent_role="operator",
        created_at=created_at,
    )
    lesson = Lesson(
        id=UUID("22222222-2222-2222-2222-222222222222"),
        experience_id=experience.id,
        content="Check port availability before deployment",
        agent_role=experience.agent_role,
        root_cause="The configured port was occupied",
        confidence=0.75,
        created_at=created_at,
    )
    memory = Memory(
        id=UUID("33333333-3333-3333-3333-333333333333"),
        content=lesson.content,
        type=MemoryType.LESSON,
        agent_role=lesson.agent_role,
        confidence=lesson.confidence,
        importance=0.7,
        source=f"experience_{experience.id}",
        metadata={"root_cause": lesson.root_cause},
        embedding=[0.25, -0.5, 1.0],
        created_at=created_at,
        updated_at=created_at,
    )
    return LessonMemoryPair(experience=experience, lesson=lesson, memory=memory)


async def _visible_rows(
    db_path: Path, lesson_id: UUID, memory_id: UUID
) -> tuple[tuple | None, tuple | None]:
    """Read only committed state through a connection independent of the store."""
    connection = await aiosqlite.connect(db_path)
    try:
        lesson_cursor = await connection.execute(
            "SELECT id, experience_id, content, agent_role, root_cause, "
            "confidence, created_at FROM lessons WHERE id = ?",
            (str(lesson_id),),
        )
        memory_cursor = await connection.execute(
            "SELECT * FROM memories WHERE id = ?", (str(memory_id),)
        )
        return await lesson_cursor.fetchone(), await memory_cursor.fetchone()
    finally:
        await connection.close()


async def _save_with_injected_failure(
    store: SQLiteStore,
    pair: LessonMemoryPair,
    failure_point: str,
) -> StorageError:
    connection = store._require_conn()
    if failure_point == "commit":

        async def fail_commit() -> None:
            raise aiosqlite.OperationalError("injected pair commit failure")

        with patch.object(connection, "commit", new=fail_commit):
            with pytest.raises(StorageError) as raised:
                await store.save_lesson_and_memory(pair.lesson, pair.memory)
        return raised.value

    original_execute = connection.execute
    failure_prefix = {
        "lesson_insert": "INSERT INTO LESSONS",
        "memory_insert": "INSERT OR REPLACE INTO MEMORIES",
    }[failure_point]

    async def execute_with_failure(sql, *args, **kwargs):
        normalized = " ".join(sql.split()).upper()
        if normalized.startswith(failure_prefix):
            raise aiosqlite.OperationalError(f"injected {failure_point} failure")
        return await original_execute(sql, *args, **kwargs)

    with patch.object(connection, "execute", new=execute_with_failure):
        with pytest.raises(StorageError) as raised:
            await store.save_lesson_and_memory(pair.lesson, pair.memory)
    return raised.value


async def _exercise_atomic_save(pair: LessonMemoryPair, failure_point: str) -> None:
    with TemporaryDirectory(prefix="experia-lesson-memory-atomicity-") as directory:
        db_path = Path(directory) / "atomicity.db"
        store = SQLiteStore(str(db_path))
        await store.initialize()
        try:
            await store.save_experience(pair.experience)

            if failure_point == "success":
                await store.save_lesson_and_memory(pair.lesson, pair.memory)
            else:
                error = await _save_with_injected_failure(store, pair, failure_point)
                assert error.operation == "save"
                assert error.table == "lessons,memories"
                assert error.record_ids == (
                    str(pair.lesson.id),
                    str(pair.memory.id),
                )
                assert isinstance(error.__cause__, aiosqlite.OperationalError)

            lesson_row, memory_row = await _visible_rows(
                db_path, pair.lesson.id, pair.memory.id
            )
            assert (lesson_row is not None) == (memory_row is not None)

            if failure_point == "success":
                serializer = StorageSerializer()
                assert serializer.decode_lesson(lesson_row) == pair.lesson
                assert serializer.decode_memory(memory_row) == pair.memory
            else:
                assert lesson_row is None
                assert memory_row is None
        finally:
            await store.close()


# Feature: open-source-project-improvements, Property 12: Lesson and memory save as one atomic unit
# **Validates: Requirements 3.4, 3.5**
@settings(max_examples=100, deadline=None)
@given(
    pair=lesson_memory_pairs(),
    failure_point=st.sampled_from(_FAILURE_POINTS),
)
@example(pair=_example_pair(), failure_point="success")
@example(pair=_example_pair(), failure_point="lesson_insert")
@example(pair=_example_pair(), failure_point="memory_insert")
@example(pair=_example_pair(), failure_point="commit")
def test_lesson_and_memory_save_as_one_atomic_unit(
    pair: LessonMemoryPair, failure_point: str
) -> None:
    asyncio.run(_exercise_atomic_save(pair, failure_point))
