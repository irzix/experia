import asyncio
import os
import uuid

import pytest

from experia.core.learner import Learner
from experia.experience.evaluator import SimpleHeuristicEvaluator
from experia.memory.models import Memory, MemoryType
from experia.memory.store import SQLiteStore


@pytest.mark.asyncio
async def test_shared_experience_multi_agent():
    db_path = "test_shared_experience.db"
    store = SQLiteStore(db_path)
    await store.initialize()

    try:
        evaluator = SimpleHeuristicEvaluator()

        # Instantiate two agents pointing to the SAME store
        coder = Learner(store=store, evaluator=evaluator, agent_role="Coder")
        researcher = Learner(store=store, evaluator=evaluator, agent_role="Researcher")

        # Coder fails to compile
        await coder.record(
            task="Write Python script",
            action="Run python main.py",
            result="ERROR: SyntaxError missing colon",
        )

        # Allow async recording to finish
        await asyncio.sleep(0.1)

        # Ensure Coder context includes the failure lesson (heuristic evaluator generates
        # a generic message containing the action name, not the raw error text)
        coder_context = await coder.retrieve_context(query="Python script")
        assert "python main.py" in coder_context

        # Ensure Researcher context DOES NOT include Coder's lesson
        # (because the simple heuristic evaluator saves lessons per-agent_role)
        researcher_context = await researcher.retrieve_context(query="Python script")
        assert "python main.py" not in researcher_context

        # Now let's simulate a global Strategy memory being created (e.g., via ReflectionEngine)
        # We manually insert a STRATEGY memory (agent_role="default", but type=MemoryType.STRATEGY)
        strategy_memory = Memory(
            id=uuid.uuid4(),
            content="Always write unit tests before executing main scripts.",
            type=MemoryType.STRATEGY,
            agent_role="Supervisor",  # Supervisor generated it
            confidence=0.9,
            importance=1.0,
            source="Reflection",
        )
        await store.save_memory(strategy_memory)

        # Now BOTH Coder and Researcher should retrieve the strategy!
        coder_context_2 = await coder.retrieve_context(query="unit tests")
        researcher_context_2 = await researcher.retrieve_context(query="unit tests")

        assert "Always write unit tests" in coder_context_2
        assert "Always write unit tests" in researcher_context_2

    finally:
        if os.path.exists(db_path):
            os.remove(db_path)
