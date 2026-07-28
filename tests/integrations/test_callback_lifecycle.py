import asyncio
from copy import deepcopy
from uuid import uuid4

import pytest

from experia.core.exceptions import (
    ConfigurationError,
    EvaluationFailure,
    LifecycleError,
    StorageError,
)
from experia.core.learner import Learner
from experia.core.logging import OperationType, TerminalState
from experia.core.work import AsyncWorkManager
from experia.experience.evaluator import SimpleHeuristicEvaluator
from experia.integrations.langchain.callbacks import ExperiaCallbackHandler
from experia.integrations.langgraph.nodes import ExperiaLearningNode
from experia.memory.store import SQLiteStore


class ShutdownSignallingManager(AsyncWorkManager):
    def __init__(self):
        super().__init__()
        self.shutdown_captured = asyncio.Event()

    def _capture_shutdown_generation(self, generation):
        records = super()._capture_shutdown_generation(generation)
        self.shutdown_captured.set()
        return records


async def _add_langchain_action(handler, run_id):
    tool_run_id = uuid4()
    await handler.on_chain_start(
        serialized={"name": "AgentExecutor"},
        inputs={"input": "Deploy the service"},
        run_id=run_id,
    )
    await handler.on_tool_start(
        serialized={"name": "Deploy"},
        input_str="deploy",
        run_id=tool_run_id,
        parent_run_id=run_id,
    )
    await handler.on_tool_end(
        output="success",
        run_id=tool_run_id,
        parent_run_id=run_id,
    )


def _langgraph_extractor(state):
    return {
        "task": state["task"],
        "action": "Deploy",
        "result": "success",
    }


@pytest.mark.asyncio
async def test_durable_callbacks_persist_before_return(tmp_path):
    store = SQLiteStore(str(tmp_path / "durable.db"))
    await store.initialize()
    learner = Learner(store=store, evaluator=SimpleHeuristicEvaluator())

    try:
        handler = ExperiaCallbackHandler(agent=learner, callback_mode="durable")
        run_id = uuid4()
        await _add_langchain_action(handler, run_id)
        await handler.on_chain_end(outputs={"output": "success"}, run_id=run_id)

        recent = await store.get_recent_experiences()
        assert len(recent) == 1
        assert recent[0].task == "Deploy the service"

        node = ExperiaLearningNode(
            agent=learner,
            extractor=_langgraph_extractor,
            callback_mode="durable",
        )
        assert await node({"task": "Deploy another service"}) == {}

        recent = await store.get_recent_experiences()
        assert len(recent) == 2
        assert {experience.task for experience in recent} == {
            "Deploy the service",
            "Deploy another service",
        }
        await asyncio.wait_for(learner.flush(), timeout=1)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_background_save_failure_is_reported_by_managed_flush(tmp_path):
    store = SQLiteStore(str(tmp_path / "background-save-failure.db"))
    await store.initialize()
    await store.close()
    learner = Learner(store=store, evaluator=SimpleHeuristicEvaluator())
    handler = ExperiaCallbackHandler(agent=learner)
    run_id = uuid4()
    await _add_langchain_action(handler, run_id)

    await handler.on_chain_end(outputs={"output": "success"}, run_id=run_id)

    with pytest.raises(EvaluationFailure) as raised:
        await asyncio.wait_for(learner.flush(), timeout=1)

    assert raised.value.operation == OperationType.RECORD.value
    assert len(raised.value.failures) == 1
    assert raised.value.failures[0].operation == OperationType.RECORD.value


@pytest.mark.asyncio
async def test_durable_save_failure_is_raised_by_langgraph_callback(tmp_path):
    store = SQLiteStore(str(tmp_path / "durable-save-failure.db"))
    await store.initialize()
    await store.close()
    learner = Learner(store=store, evaluator=SimpleHeuristicEvaluator())
    node = ExperiaLearningNode(
        agent=learner,
        extractor=_langgraph_extractor,
        callback_mode="durable",
    )

    with pytest.raises(StorageError):
        await node({"task": "Deploy the service"})

    assert learner._work_manager.jobs == ()


@pytest.mark.asyncio
async def test_background_evaluation_failure_retains_persisted_callback(tmp_path):
    class FlushSignallingManager(AsyncWorkManager):
        def __init__(self):
            super().__init__()
            self.flush_captured = asyncio.Event()

        def _capture_flush_generation(self, generation, cutoff):
            super()._capture_flush_generation(generation, cutoff)
            self.flush_captured.set()

    class BlockingFailingEvaluator:
        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def evaluate(self, experience):
            self.started.set()
            await self.release.wait()
            raise RuntimeError("private evaluation failure")

    store = SQLiteStore(str(tmp_path / "background-evaluation-failure.db"))
    await store.initialize()
    evaluator = BlockingFailingEvaluator()
    learner = Learner(store=store, evaluator=evaluator)
    manager = FlushSignallingManager()
    learner._work_manager = manager
    node = ExperiaLearningNode(agent=learner, extractor=_langgraph_extractor)

    try:
        assert await node({"task": "Deploy the service"}) == {}
        await asyncio.wait_for(evaluator.started.wait(), timeout=1)
        persisted_before_flush = await store.get_recent_experiences()
        assert len(persisted_before_flush) == 1

        flush_task = asyncio.create_task(learner.flush())
        await asyncio.wait_for(manager.flush_captured.wait(), timeout=1)
        evaluator.release.set()

        with pytest.raises(EvaluationFailure) as raised:
            await asyncio.wait_for(flush_task, timeout=1)

        persisted_after_failure = await store.get_recent_experiences()
        assert persisted_after_failure == persisted_before_flush
        assert raised.value.operation == OperationType.EVALUATION.value
        assert raised.value.experience_id == persisted_after_failure[0].id
    finally:
        evaluator.release.set()
        await store.close()


@pytest.mark.asyncio
async def test_langchain_shutdown_rejection_preserves_active_run(tmp_path):
    store = SQLiteStore(str(tmp_path / "langchain-shutdown.db"))
    await store.initialize()
    learner = Learner(store=store, evaluator=SimpleHeuristicEvaluator())
    handler = ExperiaCallbackHandler(agent=learner)
    run_id = uuid4()
    await _add_langchain_action(handler, run_id)
    active_run = deepcopy(handler.builder.active_runs[run_id])

    try:
        await learner.shutdown()

        with pytest.raises(LifecycleError) as raised:
            await handler.on_chain_end(outputs={"output": "success"}, run_id=run_id)

        assert raised.value.operation == "record"
        assert handler.builder.active_runs[run_id] == active_run
        assert await store.get_recent_experiences() == []
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_langgraph_shutdown_rejection_preserves_input_and_store(tmp_path):
    store = SQLiteStore(str(tmp_path / "langgraph-shutdown.db"))
    await store.initialize()
    learner = Learner(store=store, evaluator=SimpleHeuristicEvaluator())
    node = ExperiaLearningNode(agent=learner, extractor=_langgraph_extractor)
    state = {"task": "Deploy the service"}
    original_state = deepcopy(state)

    try:
        await learner.shutdown()

        with pytest.raises(LifecycleError) as raised:
            await node(state)

        assert raised.value.operation == "record"
        assert state == original_state
        assert await store.get_recent_experiences() == []
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_accepted_background_callback_drains_record_and_evaluation(tmp_path):
    store = SQLiteStore(str(tmp_path / "drain.db"))
    await store.initialize()
    learner = Learner(store=store, evaluator=SimpleHeuristicEvaluator())
    manager = ShutdownSignallingManager()
    learner._work_manager = manager
    save_started = asyncio.Event()
    release_save = asyncio.Event()
    save_experience = store.save_experience

    async def blocking_save(experience):
        save_started.set()
        await release_save.wait()
        await save_experience(experience)

    store.save_experience = blocking_save

    try:
        handler = ExperiaCallbackHandler(agent=learner)
        run_id = uuid4()
        await _add_langchain_action(handler, run_id)
        await handler.on_chain_end(outputs={"output": "success"}, run_id=run_id)
        await asyncio.wait_for(save_started.wait(), timeout=1)

        shutdown_task = asyncio.create_task(learner.shutdown("drain"))
        await asyncio.wait_for(manager.shutdown_captured.wait(), timeout=1)
        release_save.set()
        report = await asyncio.wait_for(shutdown_task, timeout=1)

        assert len(report.job_ids) == 2
        assert tuple(report.terminal_states.values()) == (
            TerminalState.SUCCESS,
            TerminalState.SUCCESS,
        )
        assert len(await store.get_recent_experiences()) == 1
    finally:
        release_save.set()
        await store.close()


@pytest.mark.parametrize(
    ("factory", "feature"),
    [
        (
            lambda: ExperiaCallbackHandler(object(), callback_mode="invalid"),
            "ExperienceBuilder",
        ),
        (
            lambda: ExperiaLearningNode(object(), callback_mode="invalid"),
            "ExperiaLearningNode",
        ),
    ],
)
def test_invalid_callback_mode_fails_before_construction(factory, feature):
    with pytest.raises(ConfigurationError) as raised:
        factory()

    assert raised.value.feature == feature
    assert raised.value.parameter == "callback_mode"
