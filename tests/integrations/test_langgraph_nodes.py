import asyncio
import os

import pytest

from experia.core.learner import Learner
from experia.experience.evaluator import SimpleHeuristicEvaluator
from experia.integrations.langgraph.nodes import ExperiaContextNode, ExperiaLearningNode
from experia.memory.store import SQLiteStore


class MockHumanMessage:
    def __init__(self, content):
        self.type = "human"
        self.content = content


class MockAIMessage:
    def __init__(self, tool_calls):
        self.type = "ai"
        self.tool_calls = tool_calls


class MockToolMessage:
    def __init__(self, content):
        self.type = "tool"
        self.content = content


@pytest.mark.asyncio
async def test_langgraph_nodes():
    db_path = "test_langgraph_nodes.db"
    store = SQLiteStore(db_path)
    await store.initialize()

    try:
        evaluator = SimpleHeuristicEvaluator()
        agent = Learner(store=store, evaluator=evaluator)

        # We simulate a memory being present
        # This will test ExperiaContextNode
        # ... Wait, retrieve_context handles formatting, if we put a lesson in, it will show up.

        # First, test Learning Node
        learning_node = ExperiaLearningNode(agent=agent)

        # Simulate LangGraph MessagesState
        state = {
            "messages": [
                MockHumanMessage("Fix the database connection"),
                MockAIMessage(tool_calls=[{"name": "SqlConnect", "args": {}}]),
                MockToolMessage("ERROR: Connection refused on port 5432"),
            ]
        }

        # Node returns {} (side-effect only)
        result_state = await learning_node(state)
        assert result_state == {}

        # Allow async record to finish
        await asyncio.sleep(0.1)

        recent = await store.get_recent_experiences(limit=5)
        assert len(recent) == 1
        exp = recent[0]
        assert exp.task == "Fix the database connection"
        assert "SqlConnect" in exp.action
        assert "Connection refused" in exp.result

        # Because of the heuristic evaluator, it should have created a lesson for the failure
        # Let's use the Context Node to retrieve it!
        context_node = ExperiaContextNode(agent=agent)

        new_state = {"messages": [MockHumanMessage("Fix the database connection")]}

        update = await context_node(new_state)

        assert "messages" in update
        sys_msg = update["messages"][0]
        assert (
            getattr(sys_msg, "type", type(sys_msg).__name__.lower()) == "system"
            or getattr(sys_msg, "type", type(sys_msg).__name__.lower()) == "systemmessage"
        )
        assert "Connection refused" in sys_msg.content or "failed" in sys_msg.content

    finally:
        if os.path.exists(db_path):
            os.remove(db_path)
