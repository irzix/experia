import os

import aiosqlite
import pytest

from experia.core.exceptions import StorageError
from experia.experience.models import ExperienceRecord, Lesson
from experia.memory.models import Memory, MemoryType
from experia.memory.store import SQLiteStore


@pytest.fixture
async def store():
    db_path = "test_experia_store.db"
    store = SQLiteStore(db_path=db_path)
    await store.initialize()
    yield store
    # Cleanup after test
    await store.close()
    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(db_path + suffix):
            os.remove(db_path + suffix)


async def _lesson_memory_pair(store: SQLiteStore) -> tuple[Lesson, Memory]:
    experience = ExperienceRecord(
        task="Diagnose a deployment failure",
        action="Inspect the service logs",
        result="Found an occupied port",
    )
    await store.save_experience(experience)
    lesson = Lesson(
        experience_id=experience.id,
        content="Check port availability before deployment",
        root_cause="The configured port was already occupied",
    )
    memory = Memory(
        content=lesson.content,
        type=MemoryType.LESSON,
        source=str(lesson.id),
    )
    return lesson, memory


@pytest.mark.asyncio
async def test_save_lesson_and_memory_encodes_before_one_transaction(
    store, monkeypatch
):
    lesson, memory = await _lesson_memory_pair(store)
    conn = store._require_conn()
    events: list[str] = []
    await conn.set_trace_callback(events.append)
    original_encode_lesson = store._serializer.encode_lesson
    original_encode_memory = store._serializer.encode_memory

    def encode_lesson(value):
        events.append("encode_lesson")
        return original_encode_lesson(value)

    def encode_memory(value):
        events.append("encode_memory")
        return original_encode_memory(value)

    monkeypatch.setattr(store._serializer, "encode_lesson", encode_lesson)
    monkeypatch.setattr(store._serializer, "encode_memory", encode_memory)

    await store.save_lesson_and_memory(lesson, memory)

    normalized = [" ".join(event.split()).upper() for event in events]
    lesson_encode_index = normalized.index("ENCODE_LESSON")
    memory_encode_index = normalized.index("ENCODE_MEMORY")
    begin_index = normalized.index("BEGIN IMMEDIATE")
    lesson_insert_index = next(
        index
        for index, statement in enumerate(normalized)
        if statement.startswith("INSERT INTO LESSONS")
    )
    memory_insert_index = next(
        index
        for index, statement in enumerate(normalized)
        if statement.startswith("INSERT OR REPLACE INTO MEMORIES")
    )
    commit_index = normalized.index("COMMIT")

    assert lesson_encode_index < begin_index
    assert memory_encode_index < begin_index
    assert begin_index < lesson_insert_index < memory_insert_index < commit_index
    assert normalized.count("BEGIN IMMEDIATE") == 1
    assert normalized.count("COMMIT") == 1

    lesson_cursor = await conn.execute(
        "SELECT experience_id, content FROM lessons WHERE id = ?", (str(lesson.id),)
    )
    memory_cursor = await conn.execute(
        "SELECT content, type, source FROM memories WHERE id = ?", (str(memory.id),)
    )
    assert await lesson_cursor.fetchone() == (str(lesson.experience_id), lesson.content)
    assert await memory_cursor.fetchone() == (
        memory.content,
        memory.type.value,
        memory.source,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_table", ["lessons", "memories"])
async def test_save_lesson_and_memory_rolls_back_insertion_failure(
    store, monkeypatch, failed_table
):
    lesson, memory = await _lesson_memory_pair(store)
    conn = store._require_conn()
    original_execute = conn.execute
    failure_prefix = {
        "lessons": "INSERT INTO LESSONS",
        "memories": "INSERT OR REPLACE INTO MEMORIES",
    }[failed_table]

    async def execute_with_failure(sql, *args, **kwargs):
        if " ".join(sql.split()).upper().startswith(failure_prefix):
            raise aiosqlite.OperationalError(
                f"injected {failed_table} insertion failure"
            )
        return await original_execute(sql, *args, **kwargs)

    monkeypatch.setattr(conn, "execute", execute_with_failure)

    with pytest.raises(StorageError) as raised:
        await store.save_lesson_and_memory(lesson, memory)

    error = raised.value
    assert error.operation == "save"
    assert error.table == "lessons,memories"
    assert error.record_ids == (str(lesson.id), str(memory.id))
    assert isinstance(error.__cause__, aiosqlite.OperationalError)
    lesson_cursor = await conn.execute(
        "SELECT id FROM lessons WHERE id = ?", (str(lesson.id),)
    )
    memory_cursor = await conn.execute(
        "SELECT id FROM memories WHERE id = ?", (str(memory.id),)
    )
    assert await lesson_cursor.fetchone() is None
    assert await memory_cursor.fetchone() is None


@pytest.mark.asyncio
async def test_save_lesson_and_memory_rolls_back_commit_failure(store, monkeypatch):
    lesson, memory = await _lesson_memory_pair(store)
    conn = store._require_conn()

    async def fail_commit():
        raise aiosqlite.OperationalError("injected pair commit failure")

    monkeypatch.setattr(conn, "commit", fail_commit)

    with pytest.raises(StorageError) as raised:
        await store.save_lesson_and_memory(lesson, memory)

    error = raised.value
    assert error.operation == "save"
    assert error.table == "lessons,memories"
    assert error.record_ids == (str(lesson.id), str(memory.id))
    assert isinstance(error.__cause__, aiosqlite.OperationalError)
    lesson_cursor = await conn.execute(
        "SELECT id FROM lessons WHERE id = ?", (str(lesson.id),)
    )
    memory_cursor = await conn.execute(
        "SELECT id FROM memories WHERE id = ?", (str(memory.id),)
    )
    assert await lesson_cursor.fetchone() is None
    assert await memory_cursor.fetchone() is None


@pytest.mark.asyncio
async def test_save_and_get_experience(store):
    exp = ExperienceRecord(
        task="Test task",
        action="Test action",
        result="Success",
        context={"key": "value"},
    )

    await store.save_experience(exp)

    retrieved = await store.get_experience(exp.id)
    assert retrieved is not None
    assert retrieved.id == exp.id
    assert retrieved.task == "Test task"
    assert retrieved.context == {"key": "value"}


@pytest.mark.asyncio
async def test_save_and_search_memory(store):
    mem1 = Memory(
        content="User likes python",
        type=MemoryType.PREFERENCE,
        confidence=0.9,
        importance=0.8,
    )
    mem2 = Memory(
        content="Always check logs on failure",
        type=MemoryType.LESSON,
        confidence=0.8,
        importance=0.5,
    )

    await store.save_memory(mem1)
    await store.save_memory(mem2)

    # Search all
    results = await store.search_memories()
    assert len(results) == 2

    # Search by type
    pref_results = await store.search_memories(memory_type=MemoryType.PREFERENCE)
    assert len(pref_results) == 1
    assert pref_results[0].id == mem1.id

    # Search by text
    text_results = await store.search_memories(query="logs")
    assert len(text_results) == 1
    assert text_results[0].content == "Always check logs on failure"


@pytest.mark.asyncio
async def test_semantic_search_ranks_by_similarity(store):
    # Two memories with hand-crafted 3-dim embeddings.
    close = Memory(
        content="Restart the database when connections hang",
        type=MemoryType.LESSON,
        embedding=[1.0, 0.0, 0.0],
    )
    far = Memory(
        content="Prefer tabs over spaces",
        type=MemoryType.PREFERENCE,
        embedding=[0.0, 1.0, 0.0],
    )
    await store.save_memory(close)
    await store.save_memory(far)

    results = await store.search_memories(query_embedding=[0.9, 0.1, 0.0], limit=2)
    assert results[0].id == close.id  # most similar ranked first


@pytest.mark.asyncio
async def test_update_memory_feedback_moves_confidence(store):
    mem = Memory(content="Check logs first", type=MemoryType.LESSON, confidence=0.5)
    await store.save_memory(mem)

    updated = await store.update_memory_feedback(mem.id, success=True)
    assert updated is not None
    assert updated.confidence > 0.5
    assert updated.reinforcement_count == 1
    assert updated.success_count == 1

    weakened = await store.update_memory_feedback(mem.id, success=False)
    assert weakened is not None
    assert weakened.confidence < updated.confidence
    assert weakened.reinforcement_count == 2
    assert weakened.success_count == 1


@pytest.mark.asyncio
async def test_concurrent_feedback_preserves_exact_counts_and_bounded_confidence(store):
    import asyncio

    memory = Memory(
        content="Check logs first",
        type=MemoryType.LESSON,
        confidence=0.5,
        reinforcement_count=7,
        success_count=3,
    )
    await store.save_memory(memory)
    outcomes = [index % 3 == 0 for index in range(24)]
    start = asyncio.Event()
    all_ready = asyncio.Event()
    ready_count = 0

    async def apply_feedback(success):
        nonlocal ready_count
        ready_count += 1
        if ready_count == len(outcomes):
            all_ready.set()
        await start.wait()
        return await store.update_memory_feedback(memory.id, success=success)

    operations = [asyncio.create_task(apply_feedback(success)) for success in outcomes]
    await all_ready.wait()
    start.set()
    results = await asyncio.gather(*operations)

    updated = await store.get_memory(memory.id)
    assert updated is not None
    assert all(result is not None for result in results)
    assert updated.reinforcement_count == memory.reinforcement_count + len(outcomes)
    assert updated.success_count == memory.success_count + sum(outcomes)
    assert 0.0 <= updated.confidence <= 1.0


@pytest.mark.asyncio
async def test_update_memory_feedback_clamps_confidence_to_closed_unit_interval(store):
    memory = Memory(content="Bounded confidence", type=MemoryType.FACT, confidence=0.5)
    await store.save_memory(memory)

    upper = await store.update_memory_feedback(memory.id, success=True, alpha=100.0)
    lower = await store.update_memory_feedback(memory.id, success=False, alpha=100.0)

    assert upper is not None
    assert upper.confidence == 1.0
    assert lower is not None
    assert lower.confidence == 0.0
    assert lower.reinforcement_count == 2
    assert lower.success_count == 1


@pytest.mark.asyncio
async def test_update_memory_feedback_falls_back_without_returning_in_transaction(
    store, monkeypatch
):
    import aiosqlite

    memory = Memory(content="Portable feedback", type=MemoryType.FACT, confidence=0.5)
    await store.save_memory(memory)
    connection = store._require_conn()
    original_execute = connection.execute
    statements = []
    await connection.set_trace_callback(statements.append)

    async def execute_without_returning(sql, *args, **kwargs):
        if "RETURNING" in sql.upper():
            raise aiosqlite.OperationalError('near "RETURNING": syntax error')
        return await original_execute(sql, *args, **kwargs)

    monkeypatch.setattr(connection, "execute", execute_without_returning)

    updated = await store.update_memory_feedback(memory.id, success=True)

    assert updated is not None
    assert updated.reinforcement_count == 1
    assert updated.success_count == 1
    normalized = [statement.strip().upper() for statement in statements]
    begin_index = normalized.index("BEGIN IMMEDIATE")
    update_index = next(
        index
        for index, statement in enumerate(normalized)
        if statement.startswith("UPDATE MEMORIES")
    )
    select_index = next(
        index
        for index, statement in enumerate(normalized)
        if statement.startswith("SELECT ID, CONTENT")
        and "FROM MEMORIES WHERE ID" in statement
    )
    commit_index = normalized.index("COMMIT")
    assert begin_index < update_index < select_index < commit_index
    assert sum(statement.startswith("UPDATE MEMORIES") for statement in normalized) == 1


@pytest.mark.asyncio
async def test_update_memory_feedback_returns_none_for_unknown_memory(store):
    from uuid import uuid4

    assert await store.update_memory_feedback(uuid4(), success=True) is None


@pytest.mark.asyncio
async def test_find_similar_memory_dedup(store):
    mem = Memory(
        content="Always verify port availability before starting the server",
        type=MemoryType.LESSON,
        embedding=[0.1, 0.2, 0.3],
    )
    await store.save_memory(mem)

    hit = await store.find_similar_memory(
        embedding=[0.1, 0.2, 0.3], memory_type=MemoryType.LESSON, threshold=0.95
    )
    assert hit is not None
    assert hit.id == mem.id

    miss = await store.find_similar_memory(
        embedding=[1.0, 0.0, 0.0], memory_type=MemoryType.LESSON, threshold=0.95
    )
    assert miss is None


@pytest.mark.asyncio
async def test_prune_expired_removes_only_expired(store):
    from datetime import datetime, timedelta, timezone

    expired = Memory(
        content="ephemeral",
        type=MemoryType.FACT,
        expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
    )
    live = Memory(content="permanent", type=MemoryType.FACT)
    await store.save_memory(expired)
    await store.save_memory(live)

    removed = await store.prune_expired()
    assert removed == 1

    remaining = await store.search_memories()
    assert len(remaining) == 1
    assert remaining[0].content == "permanent"
