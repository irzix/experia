"""Property tests for persistence ordering before evaluation submission."""

import asyncio
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any
from uuid import UUID

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from experia.core.exceptions import StorageError
from experia.core.learner import Learner
from experia.core.logging import OperationType
from experia.core.work import AsyncWorkManager, JobHandle
from experia.experience.models import ExperienceRecord

_SAFE_TEXT = st.text(
    alphabet=st.characters(blacklist_categories=("Cs", "Cc")),
    min_size=1,
    max_size=48,
)
_JSON_SCALARS = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2**31), max_value=2**31 - 1),
    st.floats(allow_nan=False, allow_infinity=False, width=32),
    _SAFE_TEXT,
)
_JSON_VALUES = st.recursive(
    _JSON_SCALARS,
    lambda children: st.one_of(
        st.lists(children, max_size=3),
        st.dictionaries(_SAFE_TEXT, children, max_size=3),
    ),
    max_leaves=8,
)
_CONTEXTS = st.one_of(
    st.none(),
    st.dictionaries(_SAFE_TEXT, _JSON_VALUES, max_size=4),
)


class SaveOutcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"


class RecordingStore:
    """Controllable MemoryStore double exposing the save completion boundary."""

    def __init__(self, outcome: SaveOutcome, trace: list[str]) -> None:
        self.outcome = outcome
        self.trace = trace
        self.save_started = asyncio.Event()
        self.release_save = asyncio.Event()
        self.saved_experience: ExperienceRecord | None = None
        self.failure: StorageError | None = None

    async def save_experience(self, experience: ExperienceRecord) -> None:
        self.trace.append("save_started")
        self.save_started.set()
        await self.release_save.wait()
        if self.outcome is SaveOutcome.FAILURE:
            self.failure = StorageError(
                operation="save",
                table="experiences",
                record_ids=(experience.id,),
            )
            self.trace.append("save_failed")
            raise self.failure
        self.saved_experience = experience
        self.trace.append("save_succeeded")


class RecordingEvaluator:
    def __init__(self, trace: list[str]) -> None:
        self.trace = trace
        self.calls: list[ExperienceRecord] = []

    async def evaluate(self, experience: ExperienceRecord) -> None:
        self.trace.append("evaluation_started")
        self.calls.append(experience)
        return None


class RecordingWorkManager(AsyncWorkManager):
    """Observe accepted submissions while retaining real manager behavior."""

    def __init__(self, trace: list[str]) -> None:
        super().__init__()
        self.trace = trace

    def submit(
        self,
        operation: OperationType | str,
        awaitable_factory: Callable[[], Awaitable[Any]],
        *,
        experience_id: UUID | None = None,
        parent_job_id: UUID | None = None,
    ) -> JobHandle:
        handle = super().submit(
            operation,
            awaitable_factory,
            experience_id=experience_id,
            parent_job_id=parent_job_id,
        )
        self.trace.append(f"{handle.operation.value}_submitted")
        return handle


async def _exercise_persistence_ordering(
    *,
    task: str,
    action: str,
    result: str,
    context: dict[str, Any] | None,
    agent_role: str,
    outcome: SaveOutcome,
) -> None:
    trace: list[str] = []
    store = RecordingStore(outcome, trace)
    evaluator = RecordingEvaluator(trace)
    manager = RecordingWorkManager(trace)
    learner = Learner(store=store, evaluator=evaluator, agent_role=agent_role)
    learner._work_manager = manager
    record_task = asyncio.create_task(learner.record(task, action, result, context))

    try:
        await asyncio.wait_for(store.save_started.wait(), timeout=1)

        # Persistence is still in flight: record cannot report success and no
        # evaluation work may be accepted or started before the save completes.
        assert not record_task.done()
        assert trace == ["save_started"]
        assert manager.jobs == ()
        assert evaluator.calls == []

        store.release_save.set()
        if outcome is SaveOutcome.FAILURE:
            with pytest.raises(StorageError) as raised:
                await asyncio.wait_for(record_task, timeout=1)
            report = await learner.flush()

            assert raised.value is store.failure
            assert trace == ["save_started", "save_failed"]
            assert store.saved_experience is None
            assert manager.jobs == ()
            assert report.job_ids == ()
            assert evaluator.calls == []
            return

        recorded = await asyncio.wait_for(record_task, timeout=1)
        trace.append("record_succeeded")
        await asyncio.wait_for(learner.flush(), timeout=1)

        assert store.saved_experience is recorded
        assert recorded.task == task
        assert recorded.action == action
        assert recorded.result == result
        assert recorded.context == (context or {})
        assert recorded.agent_role == agent_role
        assert trace.index("save_succeeded") < trace.index("evaluation_submitted")
        assert trace.index("evaluation_submitted") < trace.index("record_succeeded")
        assert trace.index("evaluation_submitted") < trace.index("evaluation_started")
        assert evaluator.calls == [recorded]
    finally:
        store.release_save.set()
        if not record_task.done():
            record_task.cancel()
        await asyncio.gather(record_task, return_exceptions=True)


# Feature: open-source-project-improvements, Property 1: Persistence precedes evaluation submission
# **Validates: Requirements 1.1, 1.11**
@settings(max_examples=100, deadline=None)
@given(
    task=_SAFE_TEXT,
    action=_SAFE_TEXT,
    result=_SAFE_TEXT,
    context=_CONTEXTS,
    agent_role=_SAFE_TEXT,
    outcome=st.sampled_from(tuple(SaveOutcome)),
)
def test_persistence_precedes_evaluation_submission(
    task: str,
    action: str,
    result: str,
    context: dict[str, Any] | None,
    agent_role: str,
    outcome: SaveOutcome,
) -> None:
    asyncio.run(
        _exercise_persistence_ordering(
            task=task,
            action=action,
            result=result,
            context=context,
            agent_role=agent_role,
            outcome=outcome,
        )
    )
