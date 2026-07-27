import asyncio

import pytest

from experia.core.exceptions import EvaluationFailure, LifecycleError, StorageError
from experia.core.learner import Learner
from experia.core.logging import OperationType, TerminalState
from experia.core.work import AsyncWorkManager, FlushReport, ShutdownReport
from experia.experience.models import Lesson
from experia.memory.models import MemoryType
from experia.memory.store import SQLiteStore


class SnapshotSignallingManager(AsyncWorkManager):
    def __init__(self):
        super().__init__()
        self.snapshot_captured = asyncio.Event()

    def _capture_flush_generation(self, generation, cutoff):
        super()._capture_flush_generation(generation, cutoff)
        self.snapshot_captured.set()


class BlockingEvaluator:
    def __init__(self, *, failure=None, produce_lesson=False):
        self.failure = failure
        self.produce_lesson = produce_lesson
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0

    async def evaluate(self, experience):
        self.calls += 1
        self.started.set()
        await self.release.wait()
        if self.failure is not None:
            raise self.failure
        if self.produce_lesson:
            return Lesson(
                experience_id=experience.id,
                content="Use a deterministic recovery procedure.",
            )
        return None


class CountingEvaluator:
    def __init__(self):
        self.calls = 0

    async def evaluate(self, experience):
        self.calls += 1
        return None


class BlockingFailingEmbedder:
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def embed_one(self, text):
        self.started.set()
        await self.release.wait()
        raise RuntimeError("private embedding failure")


@pytest.mark.asyncio
async def test_record_save_failure_submits_no_evaluation_job(tmp_path):
    store = SQLiteStore(str(tmp_path / "closed.db"))
    await store.initialize()
    await store.close()
    evaluator = CountingEvaluator()
    learner = Learner(store=store, evaluator=evaluator)

    with pytest.raises(StorageError):
        await learner.record("task", "action", "result")

    assert evaluator.calls == 0
    assert learner._work_manager.jobs == ()


@pytest.mark.asyncio
async def test_flush_returns_report_for_managed_evaluation(tmp_path):
    store = SQLiteStore(str(tmp_path / "flush.db"))
    await store.initialize()
    evaluator = BlockingEvaluator()
    learner = Learner(store=store, evaluator=evaluator)
    manager = SnapshotSignallingManager()
    learner._work_manager = manager

    try:
        experience = await learner.record("task", "action", "result")
        await asyncio.wait_for(evaluator.started.wait(), timeout=1)
        flush_task = asyncio.create_task(learner.flush())
        await asyncio.wait_for(manager.snapshot_captured.wait(), timeout=1)
        evaluator.release.set()

        report = await asyncio.wait_for(flush_task, timeout=1)

        assert isinstance(report, FlushReport)
        assert len(report.job_ids) == 1
        assert tuple(report.terminal_states.values()) == (TerminalState.SUCCESS,)
        assert await store.get_experience(experience.id) == experience
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_learner_exposes_lifecycle_observer_for_managed_work(tmp_path):
    store = SQLiteStore(str(tmp_path / "observer.db"))
    await store.initialize()
    observed = []
    learner = Learner(
        store=store,
        evaluator=CountingEvaluator(),
        lifecycle_observer=observed.append,
    )

    try:
        await learner.record("task", "action", "result")
        report = await asyncio.wait_for(learner.flush(), timeout=1)

        assert len(report.job_ids) == 1
        assert len(observed) == 1
        assert observed[0].job_id == report.job_ids[0]
        assert observed[0].operation is OperationType.EVALUATION
        assert observed[0].terminal_state is TerminalState.SUCCESS
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_evaluation_failure_is_typed_and_preserves_experience(tmp_path):
    store = SQLiteStore(str(tmp_path / "evaluation-failure.db"))
    await store.initialize()
    evaluator = BlockingEvaluator(failure=ValueError("private evaluation failure"))
    learner = Learner(store=store, evaluator=evaluator)
    manager = SnapshotSignallingManager()
    learner._work_manager = manager

    try:
        experience = await learner.record("task", "action", "result")
        await asyncio.wait_for(evaluator.started.wait(), timeout=1)
        flush_task = asyncio.create_task(learner.flush())
        await asyncio.wait_for(manager.snapshot_captured.wait(), timeout=1)
        evaluator.release.set()

        with pytest.raises(EvaluationFailure) as raised:
            await asyncio.wait_for(flush_task, timeout=1)

        assert raised.value.operation == OperationType.EVALUATION.value
        assert raised.value.experience_id == experience.id
        assert await store.get_experience(experience.id) == experience
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_embedding_fallback_persists_lesson_and_surfaces_typed_failure(tmp_path):
    store = SQLiteStore(str(tmp_path / "embedding-failure.db"))
    await store.initialize()
    evaluator = BlockingEvaluator(produce_lesson=True)
    embedder = BlockingFailingEmbedder()
    learner = Learner(store=store, evaluator=evaluator, embedder=embedder)
    manager = SnapshotSignallingManager()
    learner._work_manager = manager

    try:
        experience = await learner.record("task", "action", "result")
        flush_task = asyncio.create_task(learner.flush())
        await asyncio.wait_for(manager.snapshot_captured.wait(), timeout=1)
        evaluator.release.set()
        await asyncio.wait_for(embedder.started.wait(), timeout=1)
        embedder.release.set()

        with pytest.raises(EvaluationFailure) as raised:
            await asyncio.wait_for(flush_task, timeout=1)

        assert learner.embedding_failure == "fallback"
        assert raised.value.operation == OperationType.EMBEDDING.value
        assert raised.value.experience_id == experience.id
        assert await store.get_experience(experience.id) == experience
        memories = await store.search_memories(memory_type=MemoryType.LESSON)
        assert len(memories) == 1
        assert memories[0].embedding is None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_shutdown_delegates_policy_and_aclose_preserves_store_ownership(tmp_path):
    store = SQLiteStore(str(tmp_path / "shutdown.db"))
    await store.initialize()
    evaluator = BlockingEvaluator()
    learner = Learner(store=store, evaluator=evaluator)

    experience = await learner.record("task", "action", "result")
    await asyncio.wait_for(evaluator.started.wait(), timeout=1)
    report = await asyncio.wait_for(learner.shutdown("cancel"), timeout=1)

    assert isinstance(report, ShutdownReport)
    assert report.policy.value == "cancel"
    assert tuple(report.terminal_states.values()) == (TerminalState.CANCELLATION,)
    assert await store.get_experience(experience.id) == experience
    with pytest.raises(LifecycleError) as raised:
        await learner.record("later", "action", "result")
    assert raised.value.state == "closed"
    assert raised.value.operation == "record"

    await learner.aclose(close_store=False)
    assert await store.get_recent_experiences() == [experience]
    await learner.aclose(close_store=True)
    with pytest.raises(StorageError):
        await store.get_recent_experiences()
