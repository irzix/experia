"""Property tests for asynchronous job terminal states and lifecycle events."""

import asyncio
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, fields

from hypothesis import given
from hypothesis import strategies as st

from experia.core.logging import EVENT_SCHEMA_VERSION, LifecycleEvent
from experia.core.work import (
    AsyncWorkManager,
    JobState,
    OperationType,
    TerminalState,
)

_EVENT_FIELDS = (
    "schema_version",
    "job_id",
    "operation",
    "terminal_state",
    "duration_ms",
)


@dataclass(frozen=True)
class _JobSchedule:
    outcomes: tuple[TerminalState, ...]
    operations: tuple[OperationType, ...]
    interleaving: tuple[int, ...]


@st.composite
def _job_schedules(draw: st.DrawFn) -> _JobSchedule:
    outcomes = tuple(
        draw(
            st.lists(
                st.sampled_from(tuple(TerminalState)),
                min_size=1,
                max_size=12,
            )
        )
    )
    operations = tuple(
        draw(
            st.lists(
                st.sampled_from(tuple(OperationType)),
                min_size=len(outcomes),
                max_size=len(outcomes),
            )
        )
    )
    interleaving = tuple(draw(st.permutations(tuple(range(len(outcomes))))))
    return _JobSchedule(outcomes, operations, interleaving)


class _LifecycleCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[LifecycleEvent] = []

    def emit(self, record: logging.LogRecord) -> None:
        event = getattr(record, "experia_lifecycle_event", None)
        if event is not None:
            self.events.append(event)


class _TransitionTrackingWorkManager(AsyncWorkManager):
    def __init__(self, observer) -> None:
        super().__init__(observer=observer)
        self.terminal_transitions: list[tuple[object, TerminalState]] = []

    def _transition_terminal(self, record, terminal_state, *, error_type=None):
        was_terminal = record.state.is_terminal
        observer_task = super()._transition_terminal(
            record,
            terminal_state,
            error_type=error_type,
        )
        if not was_terminal and record.state.is_terminal:
            self.terminal_transitions.append((record.handle.job_id, terminal_state))
        return observer_task


class _GeneratedJobFailure(Exception):
    pass


async def _exercise_schedule(schedule: _JobSchedule) -> None:
    observed: list[LifecycleEvent] = []
    manager = _TransitionTrackingWorkManager(observed.append)
    started = [asyncio.Event() for _ in schedule.outcomes]
    releases = [asyncio.Event() for _ in schedule.outcomes]

    async def controlled_job(index: int) -> None:
        started[index].set()
        await releases[index].wait()
        if schedule.outcomes[index] is TerminalState.FAILURE:
            raise _GeneratedJobFailure

    handles = [
        manager.submit(
            operation,
            lambda index=index: controlled_job(index),
        )
        for index, operation in enumerate(schedule.operations)
    ]

    package_logger = logging.getLogger("experia")
    capture = _LifecycleCapture()
    previous_level = package_logger.level
    package_logger.addHandler(capture)
    package_logger.setLevel(logging.INFO)
    try:
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in started)),
            timeout=1,
        )

        for index in schedule.interleaving:
            handle = handles[index]
            if schedule.outcomes[index] is TerminalState.CANCELLATION:
                assert manager.cancel(handle)
            else:
                releases[index].set()

        terminal_states = await asyncio.wait_for(
            asyncio.gather(*(manager.wait(handle) for handle in handles)),
            timeout=1,
        )
    finally:
        package_logger.removeHandler(capture)
        package_logger.setLevel(previous_level)
        for release in releases:
            release.set()
        for handle in handles:
            record = manager.get_record(handle)
            if record is not None and not record.state.is_terminal:
                manager.cancel(handle)
        await asyncio.gather(
            *(
                record.task
                for handle in handles
                if (record := manager.get_record(handle)) is not None
                and record.task is not None
            ),
            return_exceptions=True,
        )

    job_ids = [handle.job_id for handle in handles]
    assert len(job_ids) == len(set(job_ids))
    assert terminal_states == list(schedule.outcomes)

    transition_counts = Counter(
        job_id for job_id, _terminal_state in manager.terminal_transitions
    )
    assert transition_counts == Counter({job_id: 1 for job_id in job_ids})

    observed_by_job = defaultdict(list)
    logged_by_job = defaultdict(list)
    for event in observed:
        observed_by_job[event.job_id].append(event)
    for event in capture.events:
        logged_by_job[event.job_id].append(event)

    assert Counter(event.job_id for event in observed) == Counter(job_ids)
    assert Counter(event.job_id for event in capture.events) == Counter(job_ids)

    for index, handle in enumerate(handles):
        expected_state = schedule.outcomes[index]
        record = manager.get_record(handle)
        assert record is not None
        assert record.state is JobState(expected_state.value)
        assert record.terminal_state is expected_state
        assert record.finished_at_ns is not None

        assert len(observed_by_job[handle.job_id]) == 1
        assert len(logged_by_job[handle.job_id]) == 1
        event = observed_by_job[handle.job_id][0]
        assert event is logged_by_job[handle.job_id][0]
        assert event is record.terminal_event
        assert isinstance(event, LifecycleEvent)
        assert tuple(field.name for field in fields(event)) == _EVENT_FIELDS
        assert event.schema_version == EVENT_SCHEMA_VERSION
        assert event.job_id == handle.job_id
        assert event.operation is handle.operation
        assert event.terminal_state is expected_state
        assert type(event.duration_ms) is int
        assert event.duration_ms >= 0


# Feature: open-source-project-improvements, Property 2: Every job has one terminal state and one event
# **Validates: Requirements 1.2, 1.4, 11.2, 11.4**
@given(schedule=_job_schedules())
def test_every_job_has_one_terminal_state_and_one_event(
    schedule: _JobSchedule,
) -> None:
    asyncio.run(_exercise_schedule(schedule))
