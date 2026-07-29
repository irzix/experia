import asyncio
import logging
from collections import Counter
from uuid import uuid4

import pytest

from experia.core.exceptions import StorageError
from experia.core.learner import Learner
from experia.core.logging import (
    IntegrationEvent,
    IntegrationName,
    IntegrationOutcome,
    logger,
)
from experia.experience.evaluator import SimpleHeuristicEvaluator
from experia.integrations.langchain.builder import ExperienceBuilder
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


async def _start_action(
    builder: ExperienceBuilder,
    chain_run_id,
    tool_run_id,
    *,
    task: str,
    tool: str,
) -> None:
    await builder.on_chain_start(
        chain_run_id,
        serialized={"name": "AgentExecutor"},
        inputs={"input": task},
    )
    await builder.on_tool_start(
        tool_run_id,
        chain_run_id,
        serialized={"name": tool},
        inputs={"input": task},
    )


@pytest.mark.asyncio
async def test_interleaved_runs_match_tool_identifiers_and_isolate_orphans(
    tmp_path,
    integration_events,
):
    store = SQLiteStore(str(tmp_path / "langchain-run-isolation.db"))
    await store.initialize()
    learner = Learner(store=store, evaluator=SimpleHeuristicEvaluator())
    builder = ExperienceBuilder(agent=learner, callback_mode="durable")
    chain_a, chain_b = uuid4(), uuid4()
    tool_a, tool_b = uuid4(), uuid4()

    try:
        await _start_action(
            builder,
            chain_a,
            tool_a,
            task="Deploy service A",
            tool="DeployA",
        )
        await _start_action(
            builder,
            chain_b,
            tool_b,
            task="Deploy service B",
            tool="DeployB",
        )

        # A tool identifier cannot be completed through another chain identifier.
        await builder.on_tool_end(tool_a, chain_b, "wrong result")
        assert builder.active_runs[chain_a].tools[tool_a].result is None
        assert builder.active_runs[chain_b].tools[tool_b].result is None

        await builder.on_tool_end(tool_b, chain_b, "result B")
        await builder.on_tool_end(tool_a, chain_a, "result A")
        await asyncio.gather(
            builder.on_chain_end(chain_a, outputs={"output": "chain A"}),
            builder.on_chain_end(chain_b, outputs={"output": "chain B"}),
        )
        await asyncio.wait_for(learner.flush(), timeout=1)

        experiences = await store.get_recent_experiences(limit=10)
        by_chain = {
            experience.context["chain_id"]: experience for experience in experiences
        }
        assert set(by_chain) == {str(chain_a), str(chain_b)}
        assert "DeployA" in by_chain[str(chain_a)].action
        assert "DeployB" not in by_chain[str(chain_a)].action
        assert by_chain[str(chain_a)].result == "result A"
        assert "DeployB" in by_chain[str(chain_b)].action
        assert "DeployA" not in by_chain[str(chain_b)].action
        assert by_chain[str(chain_b)].result == "result B"
        assert builder.active_runs == {}

        assert Counter(event.outcome for event in integration_events) == Counter(
            {
                IntegrationOutcome.ORPHAN: 1,
                IntegrationOutcome.RECORDED: 2,
            }
        )
        orphan = next(
            event
            for event in integration_events
            if event.outcome is IntegrationOutcome.ORPHAN
        )
        assert orphan == IntegrationEvent(
            schema_version=1,
            integration=IntegrationName.LANGCHAIN,
            outcome=IntegrationOutcome.ORPHAN,
            run_id=tool_a,
        )
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_durable_record_runs_outside_registry_lock(tmp_path, integration_events):
    store = SQLiteStore(str(tmp_path / "langchain-short-lock.db"))
    await store.initialize()
    learner = Learner(store=store, evaluator=SimpleHeuristicEvaluator())
    builder = ExperienceBuilder(agent=learner, callback_mode="durable")
    chain_a, chain_b = uuid4(), uuid4()
    tool_a = uuid4()
    save_started = asyncio.Event()
    release_save = asyncio.Event()
    original_save = store.save_experience

    async def blocking_save(experience):
        save_started.set()
        await release_save.wait()
        await original_save(experience)

    store.save_experience = blocking_save

    try:
        await _start_action(
            builder,
            chain_a,
            tool_a,
            task="Deploy service A",
            tool="DeployA",
        )
        await builder.on_tool_end(tool_a, chain_a, "result A")

        finalize_task = asyncio.create_task(
            builder.on_chain_end(chain_a, outputs={"output": "chain A"})
        )
        await asyncio.wait_for(save_started.wait(), timeout=1)

        # A different run can mutate the registry while Learner.record is suspended.
        await asyncio.wait_for(
            builder.on_chain_start(
                chain_b,
                serialized={"name": "AgentExecutor"},
                inputs={"input": "Deploy service B"},
            ),
            timeout=1,
        )
        assert chain_a not in builder.active_runs
        assert chain_b in builder.active_runs

        release_save.set()
        await asyncio.wait_for(finalize_task, timeout=1)
        await asyncio.wait_for(learner.flush(), timeout=1)

        assert integration_events == [
            IntegrationEvent(
                schema_version=1,
                integration=IntegrationName.LANGCHAIN,
                outcome=IntegrationOutcome.RECORDED,
                run_id=chain_a,
            )
        ]
    finally:
        release_save.set()
        await store.close()


@pytest.mark.asyncio
async def test_failed_record_finalizes_once_and_emits_one_failed_outcome(
    tmp_path,
    integration_events,
):
    store = SQLiteStore(str(tmp_path / "langchain-record-failure.db"))
    await store.initialize()
    learner = Learner(store=store, evaluator=SimpleHeuristicEvaluator())
    builder = ExperienceBuilder(agent=learner, callback_mode="durable")
    chain_run_id, tool_run_id = uuid4(), uuid4()
    await _start_action(
        builder,
        chain_run_id,
        tool_run_id,
        task="Deploy service",
        tool="Deploy",
    )
    await builder.on_tool_end(tool_run_id, chain_run_id, "result")
    await store.close()

    with pytest.raises(StorageError):
        await builder.on_chain_end(chain_run_id, outputs={"output": "chain"})

    # Duplicate terminal delivery cannot record or emit an outcome for the consumed run.
    await builder.on_chain_end(chain_run_id, outputs={"output": "chain"})

    assert chain_run_id not in builder.active_runs
    assert integration_events == [
        IntegrationEvent(
            schema_version=1,
            integration=IntegrationName.LANGCHAIN,
            outcome=IntegrationOutcome.FAILED,
            run_id=chain_run_id,
        )
    ]


@pytest.mark.asyncio
async def test_background_record_emits_failed_only_after_persistence_fails(
    tmp_path,
    integration_events,
):
    from experia.core.exceptions import EvaluationFailure

    store = SQLiteStore(str(tmp_path / "langchain-background-failure.db"))
    await store.initialize()
    learner = Learner(store=store, evaluator=SimpleHeuristicEvaluator())
    builder = ExperienceBuilder(agent=learner)
    chain_run_id, tool_run_id = uuid4(), uuid4()
    await _start_action(
        builder,
        chain_run_id,
        tool_run_id,
        task="Deploy service",
        tool="Deploy",
    )
    await builder.on_tool_end(tool_run_id, chain_run_id, "result")
    await store.close()

    await builder.on_chain_end(chain_run_id, outputs={"output": "chain"})

    with pytest.raises(EvaluationFailure):
        await asyncio.wait_for(learner.flush(), timeout=1)

    assert integration_events == [
        IntegrationEvent(
            schema_version=1,
            integration=IntegrationName.LANGCHAIN,
            outcome=IntegrationOutcome.FAILED,
            run_id=chain_run_id,
        )
    ]
