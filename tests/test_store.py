import os

import pytest

from experia.experience.models import ExperienceRecord
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
    assert weakened.confidence < updated.confidence
    assert weakened.reinforcement_count == 2
    assert weakened.success_count == 1


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
