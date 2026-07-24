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
    if os.path.exists(db_path):
        os.remove(db_path)


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
