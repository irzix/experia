"""Edge case tests for the Learner."""

import os

import pytest

from experia.core.exceptions import ConfigurationError
from experia.core.learner import Learner
from experia.memory.models import MemoryType
from experia.memory.store import SQLiteStore


class DummyEvaluator:
    """Minimal evaluator that does nothing."""

    async def evaluate(self, experience):
        return None


@pytest.mark.asyncio
async def test_learner_raises_without_store():
    with pytest.raises(ConfigurationError):
        Learner(store=None, evaluator=DummyEvaluator())


@pytest.mark.asyncio
async def test_learner_raises_without_evaluator():
    store = SQLiteStore(":memory:")
    await store.initialize()
    with pytest.raises(ConfigurationError):
        Learner(store=store, evaluator=None)


@pytest.mark.asyncio
async def test_retrieve_context_empty():
    db_path = "test_empty_context.db"
    store = SQLiteStore(db_path)
    await store.initialize()
    try:
        evaluator = DummyEvaluator()
        learner = Learner(store=store, evaluator=evaluator)
        context = await learner.retrieve_context()
        assert context == ""
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)
