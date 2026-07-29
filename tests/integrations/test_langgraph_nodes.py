import asyncio
import logging
import os

import pytest

from experia.core.learner import Learner
from experia.core.logging import (
    IntegrationEvent,
    IntegrationName,
    IntegrationOutcome,
)
from experia.experience.evaluator import SimpleHeuristicEvaluator
from experia.integrations.langgraph.nodes import ExperiaContextNode, ExperiaLearningNode
from experia.integrations.langgraph.utils import default_messages_state_extractor
from experia.memory.store import SQLiteStore


class MockHumanMessage:
    def __init__(self, content):
        self.type = "human"
        self.content = content


class MockAIMessage:
    def __init__(self, tool_calls):
        self.type = "ai"
        self.tool_calls = tool_calls


_MISSING_ID = object()


class MockToolMessage:
    def __init__(self, content, tool_call_id=_MISSING_ID):
        self.type = "tool"
        self.content = content
        if tool_call_id is not _MISSING_ID:
            self.tool_call_id = tool_call_id


def test_extractor_associates_id_results_in_event_order():
    state = {
        "messages": [
            MockHumanMessage("Investigate both services"),
            MockAIMessage(
                tool_calls=[
                    {"id": "call-api", "name": "CheckApi", "args": {}},
                    {"id": "call-db", "name": "CheckDatabase", "args": {}},
                ]
            ),
            MockToolMessage("API check started", "call-api"),
            MockToolMessage("Database is healthy", "call-db"),
            MockToolMessage("API returned 503", "call-api"),
        ]
    }

    extracted = default_messages_state_extractor(state)

    assert extracted is not None
    assert extracted["task"] == "Investigate both services"
    assert extracted["action"].index("CheckApi") < extracted["action"].index(
        "CheckDatabase"
    )
    assert extracted["result"] == (
        "API check started\nDatabase is healthy\nAPI returned 503"
    )


def test_extractor_uses_only_exact_idless_positional_fallback():
    state = {
        "messages": [
            MockHumanMessage("Check services"),
            MockAIMessage(
                tool_calls=[
                    {"name": "CheckApi", "args": {}},
                    {"name": "CheckDatabase", "args": {}},
                ]
            ),
            MockToolMessage("API is healthy"),
            MockToolMessage("Database is healthy"),
        ]
    }

    extracted = default_messages_state_extractor(state)

    assert extracted is not None
    assert extracted["action"].index("CheckApi") < extracted["action"].index(
        "CheckDatabase"
    )
    assert extracted["result"] == "API is healthy\nDatabase is healthy"


@pytest.mark.parametrize(
    "tool_calls,tool_results",
    [
        (
            [
                {"name": "CheckApi", "args": {}},
                {"name": "CheckDatabase", "args": {}},
            ],
            [MockToolMessage("Only one result")],
        ),
        (
            [
                {"id": "call-api", "name": "CheckApi", "args": {}},
                {"name": "CheckDatabase", "args": {}},
            ],
            [
                MockToolMessage("API is healthy", "call-api"),
                MockToolMessage("Database is healthy"),
            ],
        ),
        (
            [{"id": "call-api", "name": "CheckApi", "args": {}}],
            [MockToolMessage("Database is healthy", "call-db")],
        ),
        (
            [
                {"id": "duplicate", "name": "CheckApi", "args": {}},
                {"id": "duplicate", "name": "CheckDatabase", "args": {}},
            ],
            [MockToolMessage("Ambiguous result", "duplicate")],
        ),
    ],
    ids=[
        "idless-count-mismatch",
        "partial-identifiers",
        "unknown-identifier",
        "duplicate-call-identifiers",
    ],
)
def test_extractor_rejects_incompatible_associations(tool_calls, tool_results):
    state = {
        "messages": [
            MockHumanMessage("Check services"),
            MockAIMessage(tool_calls=tool_calls),
            *tool_results,
        ]
    }

    assert default_messages_state_extractor(state) is None


def test_extractor_does_not_reuse_an_older_run_for_incompatible_latest_results():
    state = {
        "messages": [
            MockHumanMessage("Old task"),
            MockAIMessage(
                tool_calls=[{"id": "old-call", "name": "OldTool", "args": {}}]
            ),
            MockToolMessage("old result", "old-call"),
            MockHumanMessage("Current task"),
            MockAIMessage(
                tool_calls=[{"id": "current-call", "name": "CurrentTool", "args": {}}]
            ),
            MockToolMessage("orphaned result", "other-run-call"),
        ]
    }

    assert default_messages_state_extractor(state) is None


@pytest.mark.asyncio
async def test_incompatible_association_emits_no_experience_without_writing(
    tmp_path, caplog
):
    store = SQLiteStore(str(tmp_path / "langgraph-incompatible.db"))
    await store.initialize()
    learner = Learner(store=store, evaluator=SimpleHeuristicEvaluator())
    node = ExperiaLearningNode(agent=learner, callback_mode="durable")
    state = {
        "messages": [
            MockHumanMessage("Check services"),
            MockAIMessage(
                tool_calls=[
                    {"name": "CheckApi", "args": {}},
                    {"name": "CheckDatabase", "args": {}},
                ]
            ),
            MockToolMessage("Only one result"),
        ]
    }

    try:
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="experia"):
            assert await node(state) == {}

        events = [
            record.experia_integration_event
            for record in caplog.records
            if isinstance(
                getattr(record, "experia_integration_event", None), IntegrationEvent
            )
        ]
        assert events == [
            IntegrationEvent(
                schema_version=1,
                integration=IntegrationName.LANGGRAPH,
                outcome=IntegrationOutcome.NO_EXPERIENCE,
                run_id=None,
            )
        ]
        assert await store.get_recent_experiences() == []
        assert learner._work_manager.jobs == ()
    finally:
        await store.close()


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

        # Managed background persistence and evaluation share one flush boundary.
        await asyncio.wait_for(agent.flush(), timeout=1)

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
            or getattr(sys_msg, "type", type(sys_msg).__name__.lower())
            == "systemmessage"
        )
        assert "Connection refused" in sys_msg.content or "failed" in sys_msg.content

    finally:
        await store.close()
        for suffix in ("", "-wal", "-shm"):
            if os.path.exists(db_path + suffix):
                os.remove(db_path + suffix)
