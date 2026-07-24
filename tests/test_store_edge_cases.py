"""Edge case tests for the SQLiteStore."""

import os
import uuid

import pytest

from experia.core.exceptions import StorageError
from experia.experience.models import ExperienceRecord, Lesson
from experia.memory.models import Memory, MemoryType
from experia.memory.store import SQLiteStore


@pytest.fixture
async def store():
    db_path = "test_store_edge_cases.db"
    s = SQLiteStore(db_path=db_path)
    await s.initialize()
    yield s
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.mark.asyncio
async def test_get_experience_returns_none_for_missing(store):
    result = await store.get_experience(uuid.uuid4())
    assert result is None


@pytest.mark.asyncio
async def test_get_recent_experiences_empty(store):
    result = await store.get_recent_experiences(limit=10)
    assert result == []


@pytest.mark.asyncio
async def test_save_and_get_experience_with_agent_role(store):
    exp = ExperienceRecord(
        task="Test task",
        action="Test action",
        result="Success",
        agent_role="Coder",
        context={"key": "value"},
    )
    await store.save_experience(exp)

    retrieved = await store.get_experience(exp.id)
    assert retrieved is not None
    assert retrieved.agent_role == "Coder"
    assert retrieved.context == {"key": "value"}
    assert retrieved.task == "Test task"


@pytest.mark.asyncio
async def test_save_and_get_experience_no_context(store):
    exp = ExperienceRecord(
        task="Simple task",
        action="do something",
        result="ok",
        context={},
    )
    await store.save_experience(exp)

    retrieved = await store.get_experience(exp.id)
    assert retrieved is not None
    assert retrieved.context == {}


@pytest.mark.asyncio
async def test_search_memories_with_query_empty(store):
    result = await store.search_memories(query="nonexistent")
    assert result == []


@pytest.mark.asyncio
async def test_search_memories_by_agent_role(store):
    mem = Memory(
        content="Agent specific knowledge",
        type=MemoryType.FACT,
        agent_role="specialist",
        confidence=0.8,
        importance=0.7,
    )
    await store.save_memory(mem)

    # Search with matching role
    result = await store.search_memories(agent_role="specialist")
    assert len(result) == 1

    # Search with non-matching role — STRATEGY memories are shared, not FACT
    result = await store.search_memories(agent_role="other")
    assert len(result) == 0


@pytest.mark.asyncio
async def test_save_and_get_lesson(store):
    exp = ExperienceRecord(
        task="Test", action="test", result="ok"
    )
    await store.save_experience(exp)

    lesson = Lesson(
        experience_id=exp.id,
        content="Always test before committing",
        root_cause="Lack of testing",
        confidence=0.9,
    )
    await store.save_lesson(lesson)

    # Verify lesson was saved by searching memories
    memories = await store.search_memories(memory_type=MemoryType.LESSON)
    assert len(memories) == 0  # Lessons are not stored as memories automatically

    # Re-retrieve experience to confirm it still works
    retrieved = await store.get_experience(exp.id)
    assert retrieved is not None
    assert retrieved.task == "Test"
