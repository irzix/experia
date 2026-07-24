import os

import pytest

from experia.core.learner import Learner
from experia.memory.models import MemoryType
from experia.memory.store import SQLiteStore


@pytest.fixture
def learner():
    db_path = "test_learner.db"
    store = SQLiteStore(db_path=db_path)
    learner_instance = Learner(store=store)

    yield learner_instance

    # Cleanup after test
    if os.path.exists(db_path):
        os.remove(db_path)


def test_learner_record_failure(learner):
    """Test that a failure experience automatically generates a lesson."""

    # Act: Record a failure
    exp = learner.record(
        task="deploy application",
        action="restart nginx",
        result="failed with error 500",
    )

    # Assert experience is saved
    saved_exp = learner.store.get_experience(exp.id)
    assert saved_exp is not None
    assert saved_exp.action == "restart nginx"

    # Assert lesson was extracted and saved as memory
    memories = learner.store.search_memories(memory_type=MemoryType.LESSON)
    assert len(memories) == 1

    memory = memories[0]
    assert "failed" in memory.content
    assert "restart nginx" in memory.content


def test_learner_record_success(learner):
    """Test that a success experience generates a positive lesson."""

    learner.record(
        task="fix memory leak",
        action="increase max heap size",
        result="success, app is stable",
    )

    memories = learner.store.search_memories(memory_type=MemoryType.LESSON)
    assert len(memories) == 1
    assert "successful" in memories[0].content


def test_learner_manual_memory_and_context(learner):
    """Test that manual memories can be added and retrieved via context builder."""

    # Manually add a preference
    learner.remember("User prefers detailed explanations", MemoryType.PREFERENCE)

    # Add a failure experience
    learner.record(
        task="write code",
        action="use generic variable names",
        result="failed code review",
    )

    # Retrieve context
    context = learner.retrieve_context()

    # Check if context string is built properly
    assert "--- User Context & Learned Experience ---" in context
    assert "[PREFERENCE]" in context
    assert "User prefers detailed explanations" in context
    assert "[LESSON]" in context
    assert "failed" in context
