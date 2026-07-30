import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import aiosqlite
import pytest

from experia.core.exceptions import StorageError
from experia.experience.models import ExperienceRecord, Lesson
from experia.memory.models import Memory, MemoryType
from experia.memory.store import SQLiteStore


async def _memory_snapshot(db_path) -> list[tuple[Any, ...]]:
    conn = await aiosqlite.connect(db_path)
    try:
        cursor = await conn.execute("SELECT * FROM memories ORDER BY id")
        return await cursor.fetchall()
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_concurrent_close_callers_share_one_completion(tmp_path, monkeypatch):
    store = SQLiteStore(str(tmp_path / "concurrent-close.db"))
    await store.initialize()
    conn = store._require_conn()
    original_close = conn.close
    original_coordinate = store._coordinate_close
    physical_close_started = asyncio.Event()
    release_physical_close = asyncio.Event()
    all_callers_coordinated = asyncio.Event()
    physical_close_calls = 0
    coordinated_callers = 0
    caller_count = 8

    async def controlled_connection_close() -> None:
        nonlocal physical_close_calls
        physical_close_calls += 1
        physical_close_started.set()
        await release_physical_close.wait()
        await original_close()

    async def observed_coordinate_close():
        nonlocal coordinated_callers
        task = await original_coordinate()
        coordinated_callers += 1
        if coordinated_callers == caller_count:
            all_callers_coordinated.set()
        return task

    monkeypatch.setattr(conn, "close", controlled_connection_close)
    monkeypatch.setattr(store, "_coordinate_close", observed_coordinate_close)

    callers = [asyncio.create_task(store.close()) for _ in range(caller_count)]
    try:
        await physical_close_started.wait()
        await all_callers_coordinated.wait()
        assert all(not caller.done() for caller in callers)
    finally:
        release_physical_close.set()

    assert await asyncio.gather(*callers) == [None] * caller_count
    assert physical_close_calls == 1
    assert store._conn is None


@pytest.mark.asyncio
async def test_every_operation_after_close_raises_lifecycle_storage_error(tmp_path):
    store = SQLiteStore(str(tmp_path / "post-close.db"))
    await store.initialize()
    await store.close()

    experience = ExperienceRecord(task="task", action="action", result="result")
    lesson = Lesson(experience_id=experience.id, content="lesson")
    memory = Memory(content="memory", type=MemoryType.FACT)
    operations: list[tuple[str, Callable[[], Awaitable[Any]]]] = [
        ("initialize", store.initialize),
        ("save_experience", lambda: store.save_experience(experience)),
        ("get_experience", lambda: store.get_experience(experience.id)),
        ("get_recent_experiences", store.get_recent_experiences),
        ("save_lesson", lambda: store.save_lesson(lesson)),
        (
            "save_lesson_and_memory",
            lambda: store.save_lesson_and_memory(lesson, memory),
        ),
        ("save_memory", lambda: store.save_memory(memory)),
        ("get_memory", lambda: store.get_memory(memory.id)),
        ("search_memories", store.search_memories),
        ("find_similar_memory", lambda: store.find_similar_memory([])),
        (
            "update_memory_feedback",
            lambda: store.update_memory_feedback(memory.id, success=True),
        ),
        ("prune_expired", store.prune_expired),
    ]

    for operation_name, operation in operations:
        with pytest.raises(StorageError) as raised:
            await operation()
        assert raised.value.operation == "lifecycle", operation_name


@pytest.mark.asyncio
async def test_repeated_close_leaves_persisted_data_unchanged(tmp_path):
    db_path = tmp_path / "stable-close.db"
    store = SQLiteStore(str(db_path))
    await store.initialize()
    memory = Memory(
        content="Persist through repeated close",
        type=MemoryType.LESSON,
        metadata={"nested": {"stable": True}},
        embedding=[0.25, 0.5, 0.75],
    )
    await store.save_memory(memory)

    await store.close()
    after_first_close = await _memory_snapshot(db_path)
    await asyncio.gather(*(store.close() for _ in range(6)))
    after_repeated_close = await _memory_snapshot(db_path)

    assert after_repeated_close == after_first_close

    reopened = SQLiteStore(str(db_path))
    await reopened.initialize()
    try:
        persisted = await reopened.get_memory(memory.id)
        assert persisted == memory
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_close_drains_accepted_operation_and_rejects_later_operation(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "close-operation-race.db"
    store = SQLiteStore(str(db_path))
    await store.initialize()
    memory = Memory(content="Accepted before close", type=MemoryType.FACT)
    original_upsert = store._upsert_memory
    original_close_when_drained = store._close_when_drained
    write_entered = asyncio.Event()
    release_write = asyncio.Event()
    close_started = asyncio.Event()

    async def controlled_upsert(conn, encoded_memory) -> None:
        write_entered.set()
        await release_write.wait()
        await original_upsert(conn, encoded_memory)

    async def observed_close_when_drained() -> None:
        close_started.set()
        await original_close_when_drained()

    monkeypatch.setattr(store, "_upsert_memory", controlled_upsert)
    monkeypatch.setattr(store, "_close_when_drained", observed_close_when_drained)

    save_task = asyncio.create_task(store.save_memory(memory))
    await write_entered.wait()
    close_task = asyncio.create_task(store.close())
    await close_started.wait()

    with pytest.raises(StorageError) as raised:
        await store.get_memory(memory.id)
    assert raised.value.operation == "lifecycle"
    assert not close_task.done()

    release_write.set()
    await save_task
    await close_task

    reopened = SQLiteStore(str(db_path))
    await reopened.initialize()
    try:
        assert await reopened.get_memory(memory.id) == memory
    finally:
        await reopened.close()
