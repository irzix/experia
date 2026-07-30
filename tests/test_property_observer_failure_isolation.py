"""Property tests for lifecycle observer failure isolation."""

import asyncio
from dataclasses import dataclass

from hypothesis import given
from hypothesis import strategies as st

from experia.core.work import (
    AsyncWorkManager,
    JobState,
    OperationType,
    TerminalState,
)

_OBSERVER_EXCEPTION_TYPES = (LookupError, OSError, RuntimeError, ValueError)


class _GeneratedJobFailure(Exception):
    """Generated job failure used to reach the failure terminal state."""


@dataclass(frozen=True)
class _ObserverFailureCase:
    terminal_state: TerminalState
    operation: OperationType
    observer_kind: str
    exception_type: type[Exception]
    exception_message: str
    repeated_waits: int


@st.composite
def _observer_failure_cases(draw: st.DrawFn) -> _ObserverFailureCase:
    return _ObserverFailureCase(
        terminal_state=draw(st.sampled_from(tuple(TerminalState))),
        operation=draw(st.sampled_from(tuple(OperationType))),
        observer_kind=draw(st.sampled_from(("sync", "async"))),
        exception_type=draw(st.sampled_from(_OBSERVER_EXCEPTION_TYPES)),
        exception_message=draw(st.text(max_size=80)),
        repeated_waits=draw(st.integers(min_value=1, max_value=5)),
    )


async def _exercise_observer_failure(case: _ObserverFailureCase) -> None:
    invocations = []
    states_during_observer = []
    raised_errors = []
    manager: AsyncWorkManager

    def record_invocation_and_raise(event) -> None:
        invocations.append(event)
        record = manager.get_record(event.job_id)
        assert record is not None
        states_during_observer.append(
            (record.state, record.terminal_state, record.terminal_event)
        )
        error = case.exception_type(case.exception_message)
        raised_errors.append(error)
        raise error

    if case.observer_kind == "async":

        async def raising_observer(event) -> None:
            record_invocation_and_raise(event)

    else:
        raising_observer = record_invocation_and_raise

    manager = AsyncWorkManager(observer=raising_observer)
    started = asyncio.Event()
    release = asyncio.Event()

    async def controlled_job() -> None:
        started.set()
        await release.wait()
        if case.terminal_state is TerminalState.FAILURE:
            raise _GeneratedJobFailure

    handle = manager.submit(case.operation, controlled_job)
    await asyncio.wait_for(started.wait(), timeout=1)

    if case.terminal_state is TerminalState.CANCELLATION:
        assert manager.cancel(handle)
    else:
        release.set()

    observed_state = await asyncio.wait_for(manager.wait(handle), timeout=1)
    record = manager.get_record(handle)
    assert record is not None

    expected_job_state = JobState(case.terminal_state.value)
    assert observed_state is case.terminal_state
    assert record.state is expected_job_state
    assert record.terminal_state is case.terminal_state
    assert record.terminal_future.result() is case.terminal_state
    assert states_during_observer == [
        (expected_job_state, case.terminal_state, record.terminal_event)
    ]
    assert invocations == [record.terminal_event]
    assert len(raised_errors) == 1
    assert type(raised_errors[0]) is case.exception_type
    assert raised_errors[0].args == (case.exception_message,)

    stable_snapshot = (
        record.state,
        record.terminal_state,
        record.terminal_event,
        record.failure,
        record.finished_at_ns,
    )
    for _ in range(case.repeated_waits):
        assert await manager.wait(handle) is case.terminal_state

    assert len(invocations) == 1
    assert (
        record.state,
        record.terminal_state,
        record.terminal_event,
        record.failure,
        record.finished_at_ns,
    ) == stable_snapshot


# Feature: open-source-project-improvements, Property 27: Observer failure does not change terminal state
# **Validates: Requirements 11.10**
@given(case=_observer_failure_cases())
def test_observer_failure_does_not_change_terminal_state(
    case: _ObserverFailureCase,
) -> None:
    asyncio.run(_exercise_observer_failure(case))
