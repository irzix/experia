"""Property tests for the asynchronous shutdown state machine."""

import asyncio
from contextlib import suppress
from dataclasses import dataclass
from uuid import UUID

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from experia.core.exceptions import EvaluationFailure, LifecycleError
from experia.core.work import (
    AsyncWorkManager,
    OperationType,
    ShutdownPolicy,
    TerminalState,
    WorkManagerState,
)

_BACKGROUND_OPERATIONS = (
    OperationType.EVALUATION,
    OperationType.EMBEDDING,
    OperationType.RULE_GENERATION,
    OperationType.REFLECTION,
)


class GeneratedJobFailure(RuntimeError):
    """Generated downstream failure used to exercise drain reporting."""


@dataclass(frozen=True)
class ShutdownCase:
    """Accepted jobs with an early failure and one delayed successful job."""

    operations: tuple[OperationType, ...]
    experience_ids: tuple[UUID, ...]
    failure_indexes: frozenset[int]
    delayed_success_index: int


@st.composite
def shutdown_cases(draw) -> ShutdownCase:
    job_count = draw(st.integers(min_value=2, max_value=8))
    failure_indexes = frozenset(
        draw(
            st.sets(
                st.integers(min_value=0, max_value=job_count - 1),
                min_size=1,
                max_size=job_count - 1,
            )
        )
    )
    success_indexes = [
        index for index in range(job_count) if index not in failure_indexes
    ]
    return ShutdownCase(
        operations=tuple(
            draw(
                st.lists(
                    st.sampled_from(_BACKGROUND_OPERATIONS),
                    min_size=job_count,
                    max_size=job_count,
                )
            )
        ),
        experience_ids=tuple(
            draw(
                st.lists(
                    st.uuids(),
                    min_size=job_count,
                    max_size=job_count,
                    unique=True,
                )
            )
        ),
        failure_indexes=failure_indexes,
        delayed_success_index=draw(st.sampled_from(success_indexes)),
    )


class ShutdownSignallingManager(AsyncWorkManager):
    """Expose the atomic shutdown cutoff to event-controlled property tests."""

    def __init__(self, observer):
        super().__init__(observer=observer)
        self.shutdown_captured = asyncio.Event()

    def _capture_shutdown_generation(self, generation):
        records = super()._capture_shutdown_generation(generation)
        self.shutdown_captured.set()
        return records


async def _wait_for_events(events: list[asyncio.Event]) -> None:
    await asyncio.wait_for(
        asyncio.gather(*(event.wait() for event in events)),
        timeout=2,
    )


async def _exercise_shutdown(case: ShutdownCase, policy: ShutdownPolicy) -> None:
    observed = []
    manager = ShutdownSignallingManager(observer=observed.append)
    job_count = len(case.operations)
    started = [asyncio.Event() for _ in range(job_count)]
    releases = [asyncio.Event() for _ in range(job_count)]
    successful = [asyncio.Event() for _ in range(job_count)]
    failed = [asyncio.Event() for _ in range(job_count)]
    cancellation_seen = [asyncio.Event() for _ in range(job_count)]
    cancellation_cleanup_finished = [asyncio.Event() for _ in range(job_count)]
    cancellation_cleanup_release = asyncio.Event()
    rejected_factory_called = False

    async def accepted_work(index: int) -> None:
        started[index].set()
        try:
            await releases[index].wait()
        except asyncio.CancelledError:
            cancellation_seen[index].set()
            await cancellation_cleanup_release.wait()
            cancellation_cleanup_finished[index].set()
            raise

        if index in case.failure_indexes:
            failed[index].set()
            raise GeneratedJobFailure("generated private failure text")
        successful[index].set()

    async def rejected_work() -> None:
        nonlocal rejected_factory_called
        rejected_factory_called = True

    handles = tuple(
        manager.submit(
            operation,
            lambda index=index: accepted_work(index),
            experience_id=case.experience_ids[index],
        )
        for index, operation in enumerate(case.operations)
    )
    await _wait_for_events(started)

    shutdown_task = asyncio.create_task(manager.shutdown(policy))
    try:
        await asyncio.wait_for(manager.shutdown_captured.wait(), timeout=2)
        expected_active_state = (
            WorkManagerState.DRAINING
            if policy is ShutdownPolicy.DRAIN
            else WorkManagerState.CANCELLING
        )
        assert manager.state is expected_active_state

        with pytest.raises(LifecycleError) as rejected:
            manager.submit(OperationType.RECORD, rejected_work)
        assert rejected.value.state == expected_active_state.value
        assert rejected.value.operation == "submit"
        assert not rejected_factory_called

        if policy is ShutdownPolicy.DRAIN:
            delayed_index = case.delayed_success_index
            early_indexes = [
                index for index in range(job_count) if index != delayed_index
            ]
            for index in early_indexes:
                releases[index].set()

            early_states = await asyncio.wait_for(
                asyncio.gather(
                    *(manager.wait(handles[index]) for index in early_indexes)
                ),
                timeout=2,
            )
            assert early_states == [
                (
                    TerminalState.FAILURE
                    if index in case.failure_indexes
                    else TerminalState.SUCCESS
                )
                for index in early_indexes
            ]
            assert all(failed[index].is_set() for index in case.failure_indexes)
            assert not shutdown_task.done()
            assert not successful[delayed_index].is_set()

            releases[delayed_index].set()
            with pytest.raises(EvaluationFailure) as raised:
                await asyncio.wait_for(shutdown_task, timeout=2)

            expected_failures = [
                handles[index]
                for index in range(job_count)
                if index in case.failure_indexes
            ]
            assert successful[delayed_index].is_set()
            assert [detail.job_id for detail in raised.value.failures] == [
                handle.job_id for handle in expected_failures
            ]
            assert [detail.operation for detail in raised.value.failures] == [
                handle.operation.value for handle in expected_failures
            ]
            assert [detail.experience_id for detail in raised.value.failures] == [
                handle.experience_id for handle in expected_failures
            ]
            assert all(
                detail.error_type == "GeneratedJobFailure"
                for detail in raised.value.failures
            )
            expected_terminal_states = {
                handle.job_id: (
                    TerminalState.FAILURE
                    if index in case.failure_indexes
                    else TerminalState.SUCCESS
                )
                for index, handle in enumerate(handles)
            }
        else:
            await _wait_for_events(cancellation_seen)
            assert not shutdown_task.done()
            assert not any(event.is_set() for event in cancellation_cleanup_finished)
            assert not any(event.is_set() for event in releases)

            cancellation_cleanup_release.set()
            report = await asyncio.wait_for(shutdown_task, timeout=2)

            assert report.policy is ShutdownPolicy.CANCEL
            assert report.job_ids == tuple(handle.job_id for handle in handles)
            assert report.failures == ()
            assert all(event.is_set() for event in cancellation_cleanup_finished)
            assert not any(event.is_set() for event in successful)
            assert not any(event.is_set() for event in failed)
            expected_terminal_states = {
                handle.job_id: TerminalState.CANCELLATION for handle in handles
            }
            assert report.terminal_states == expected_terminal_states

        assert manager.state is WorkManagerState.CLOSED
        assert len(observed) == job_count
        assert {
            event.job_id: event.terminal_state for event in observed
        } == expected_terminal_states

        with pytest.raises(LifecycleError) as closed_rejection:
            manager.submit(OperationType.RECORD, rejected_work)
        assert closed_rejection.value.state == WorkManagerState.CLOSED.value
        assert closed_rejection.value.operation == "submit"
        assert not rejected_factory_called
    finally:
        for release in releases:
            release.set()
        cancellation_cleanup_release.set()
        if not shutdown_task.done():
            with suppress(BaseException):
                await asyncio.wait_for(shutdown_task, timeout=2)


_DRAIN_EXAMPLE = ShutdownCase(
    operations=(OperationType.EVALUATION, OperationType.EMBEDDING),
    experience_ids=(UUID(int=1), UUID(int=2)),
    failure_indexes=frozenset({0}),
    delayed_success_index=1,
)
_CANCEL_EXAMPLE = ShutdownCase(
    operations=(OperationType.RULE_GENERATION, OperationType.REFLECTION),
    experience_ids=(UUID(int=3), UUID(int=4)),
    failure_indexes=frozenset({0}),
    delayed_success_index=1,
)


# Feature: open-source-project-improvements, Property 5: Shutdown is a closed state machine
# **Validates: Requirements 1.7, 1.8, 1.9, 1.12**
@given(case=shutdown_cases(), policy=st.sampled_from(tuple(ShutdownPolicy)))
@example(case=_DRAIN_EXAMPLE, policy=ShutdownPolicy.DRAIN)
@example(case=_CANCEL_EXAMPLE, policy=ShutdownPolicy.CANCEL)
def test_shutdown_is_a_closed_state_machine(
    case: ShutdownCase,
    policy: ShutdownPolicy,
) -> None:
    asyncio.run(_exercise_shutdown(case, policy))
