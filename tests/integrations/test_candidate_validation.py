import logging
from uuid import uuid4

import pytest

from experia.core.learner import Learner
from experia.core.logging import (
    IntegrationEvent,
    IntegrationName,
    IntegrationOutcome,
    logger,
)
from experia.experience.evaluator import SimpleHeuristicEvaluator
from experia.integrations._candidates import finalize_experience_candidate
from experia.integrations.langchain.builder import ExperienceBuilder
from experia.integrations.langgraph.nodes import ExperiaLearningNode
from experia.memory.store import SQLiteStore


class _IntegrationEventHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[IntegrationEvent] = []

    def emit(self, record: logging.LogRecord) -> None:
        event = getattr(record, "experia_integration_event", None)
        if isinstance(event, IntegrationEvent):
            self.events.append(event)


@pytest.fixture
def integration_events():
    handler = _IntegrationEventHandler()
    previous_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        yield handler.events
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


def test_shared_candidate_validation_accepts_non_blank_record_inputs(
    integration_events,
):
    candidate = finalize_experience_candidate(
        {
            "task": " Deploy the service ",
            "action": "Run deploy",
            "result": "success",
            "context": {"request_id": "request-1"},
        },
        integration=IntegrationName.LANGGRAPH,
        default_context={"source": "langgraph"},
    )

    assert candidate is not None
    assert candidate.task == " Deploy the service "
    assert candidate.action == "Run deploy"
    assert candidate.result == "success"
    assert candidate.context == {
        "request_id": "request-1",
        "source": "langgraph",
    }
    assert integration_events == []


@pytest.mark.asyncio
async def test_incomplete_langchain_candidate_finalizes_once_without_writing(
    tmp_path,
    integration_events,
):
    store = SQLiteStore(str(tmp_path / "langchain-invalid-candidate.db"))
    await store.initialize()
    learner = Learner(store=store, evaluator=SimpleHeuristicEvaluator())
    builder = ExperienceBuilder(agent=learner, callback_mode="durable")
    run_id = uuid4()

    try:
        await builder.on_chain_start(
            run_id,
            serialized={"name": "AgentExecutor"},
            inputs={"input": "Deploy the service"},
        )

        # No tool action was observed, so the current record tuple is incomplete.
        await builder.on_chain_end(run_id, outputs={"output": "success"})
        # A duplicate terminal callback cannot finalize or emit the consumed run again.
        await builder.on_chain_end(run_id, outputs={"output": "success"})

        assert await store.get_recent_experiences() == []
        assert learner._work_manager.jobs == ()
        assert run_id not in builder.active_runs
        assert integration_events == [
            IntegrationEvent(
                schema_version=1,
                integration=IntegrationName.LANGCHAIN,
                outcome=IntegrationOutcome.NO_EXPERIENCE,
                run_id=run_id,
            )
        ]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_invalid_custom_extractor_candidate_emits_once_without_writing(
    tmp_path,
    integration_events,
):
    store = SQLiteStore(str(tmp_path / "langgraph-invalid-candidate.db"))
    await store.initialize()
    learner = Learner(store=store, evaluator=SimpleHeuristicEvaluator())

    def invalid_extractor(_state):
        return {
            "task": "Deploy the service",
            "action": "   ",
            "result": "success",
        }

    node = ExperiaLearningNode(
        agent=learner,
        extractor=invalid_extractor,
        callback_mode="durable",
    )

    try:
        assert await node({"messages": []}) == {}

        assert await store.get_recent_experiences() == []
        assert learner._work_manager.jobs == ()
        assert integration_events == [
            IntegrationEvent(
                schema_version=1,
                integration=IntegrationName.LANGGRAPH,
                outcome=IntegrationOutcome.NO_EXPERIENCE,
                run_id=None,
            )
        ]
    finally:
        await store.close()
