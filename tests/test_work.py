import asyncio
import logging
from uuid import uuid4

import pytest

from experia.core.exceptions import EvaluationFailure
from experia.core.work import (
    AsyncWorkManager,
    FlushReport,
    JobState,
    OperationType,
    TerminalState,
)


@pytest.mark.asyncio
async def test_submit_assigns_unique_ordered_handles_and_tracks_parentage():
    release = asyncio.Event()
    started = asyncio.Queue()
    observed = []
    observed_states = []
    manager = None

    async def observer(event):
        observed.append(event)
        record = manager.get_record(event.job_id)
        observed_states.append(record.state if record is not None else None)

    async def work():
        started.put_nowait(None)
        await release.wait()

    manager = AsyncWorkManager(observer=observer)
    experience_id = uuid4()
    parent = manager.submit(
        OperationType.EVALUATION,
        work,
        experience_id=experience_id,
    )
    child = manager.submit(
        OperationType.EMBEDDING,
        work,
        experience_id=experience_id,
        parent_job_id=parent.job_id,
    )

    await asyncio.wait_for(started.get(), timeout=1)
    await asyncio.wait_for(started.get(), timeout=1)

    assert parent.job_id != child.job_id
    assert child.accepted_sequence == parent.accepted_sequence + 1
    assert child.parent_job_id == parent.job_id
    assert manager.descendants_of(parent) == (child,)
    assert manager.get_record(parent).task is not None
    assert manager.get_record(child).task is not None

    release.set()
    assert (
        await asyncio.wait_for(manager.wait(parent), timeout=1) is TerminalState.SUCCESS
    )
    assert (
        await asyncio.wait_for(manager.wait(child), timeout=1) is TerminalState.SUCCESS
    )

    assert [event.job_id for event in observed] == [parent.job_id, child.job_id]
    assert all(state is JobState.SUCCESS for state in observed_states)
    assert all(event.duration_ms >= 0 for event in observed)
    assert all(event.schema_version == 1 for event in observed)


@pytest.mark.asyncio
async def test_failure_and_prestart_cancellation_each_transition_once():
    observed = []
    manager = AsyncWorkManager(observer=observed.append)

    async def fail():
        raise ValueError("private failure text")

    async def never_started():
        raise AssertionError("a pre-start cancelled factory must not run")

    failed = manager.submit(OperationType.RULE_GENERATION, fail)
    cancelled = manager.submit(OperationType.REFLECTION, never_started)
    assert manager.cancel(cancelled)

    assert (
        await asyncio.wait_for(manager.wait(failed), timeout=1) is TerminalState.FAILURE
    )
    assert (
        await asyncio.wait_for(manager.wait(cancelled), timeout=1)
        is TerminalState.CANCELLATION
    )

    failed_record = manager.get_record(failed)
    cancelled_record = manager.get_record(cancelled)
    assert failed_record.state is JobState.FAILURE
    assert failed_record.failure.error_type == "ValueError"
    assert cancelled_record.state is JobState.CANCELLATION
    assert cancelled_record.failure is None
    assert [event.job_id for event in observed].count(failed.job_id) == 1
    assert [event.job_id for event in observed].count(cancelled.job_id) == 1
    assert {event.terminal_state for event in observed} == {
        TerminalState.FAILURE,
        TerminalState.CANCELLATION,
    }
    assert all(not hasattr(event, "exception") for event in observed)


async def _complete_with_terminal_state(manager, terminal_state):
    started = asyncio.Event()
    blocked = asyncio.Event()

    async def work():
        started.set()
        if terminal_state is TerminalState.FAILURE:
            raise ValueError("private-job-failure-text")
        if terminal_state is TerminalState.CANCELLATION:
            await blocked.wait()

    handle = manager.submit(OperationType.INDEX_REBUILD, work)
    await asyncio.wait_for(started.wait(), timeout=1)
    if terminal_state is TerminalState.CANCELLATION:
        assert manager.cancel(handle)

    observed_state = await asyncio.wait_for(manager.wait(handle), timeout=1)
    assert observed_state is terminal_state
    return handle


@pytest.mark.parametrize("terminal_state", list(TerminalState))
@pytest.mark.asyncio
async def test_each_terminal_transition_is_logged_and_observed_exactly_once(
    caplog, terminal_state
):
    observed = []
    manager = AsyncWorkManager(observer=observed.append)
    caplog.set_level(logging.INFO, logger="experia")

    handle = await _complete_with_terminal_state(manager, terminal_state)

    event_records = [
        record
        for record in caplog.records
        if hasattr(record, "experia_lifecycle_event")
    ]
    assert len(observed) == 1
    assert len(event_records) == 1
    assert observed[0] is event_records[0].experia_lifecycle_event
    assert observed[0] is manager.get_record(handle).terminal_event
    assert observed[0].terminal_state is terminal_state
    assert "private-job-failure-text" not in caplog.text


@pytest.mark.parametrize("terminal_state", list(TerminalState))
@pytest.mark.parametrize("observer_kind", ["sync", "async"])
@pytest.mark.asyncio
async def test_observer_failure_cannot_change_or_repeat_terminal_state(
    caplog, terminal_state, observer_kind
):
    invocations = []

    if observer_kind == "async":

        async def raising_observer(event):
            invocations.append(event)
            raise RuntimeError("observer-private-text")

    else:

        def raising_observer(event):
            invocations.append(event)
            raise RuntimeError("observer-private-text")

    manager = AsyncWorkManager(observer=raising_observer)
    caplog.set_level(logging.INFO, logger="experia")

    handle = await _complete_with_terminal_state(manager, terminal_state)

    record = manager.get_record(handle)
    assert record.state is JobState(terminal_state.value)
    assert record.terminal_event is invocations[0]
    assert len(invocations) == 1
    assert [
        entry.experia_diagnostic
        for entry in caplog.records
        if hasattr(entry, "experia_diagnostic")
    ] == ["lifecycle_observer_failure"]
    assert "observer-private-text" not in caplog.text


@pytest.mark.asyncio
async def test_terminal_entry_is_retained_until_flush_generation_releases_it():
    release = asyncio.Event()
    started = asyncio.Event()
    manager = AsyncWorkManager()

    async def work():
        started.set()
        await release.wait()

    handle = manager.submit(OperationType.EVALUATION, work)
    await asyncio.wait_for(started.wait(), timeout=1)
    retained = manager.retain_for_flush(7, (handle,))
    assert retained == (manager.get_record(handle),)

    release.set()
    assert (
        await asyncio.wait_for(manager.wait(handle), timeout=1) is TerminalState.SUCCESS
    )

    manager.prune_terminal()
    assert manager.get_record(handle) is not None

    manager.release_flush_generation(7)
    assert manager.get_record(handle) is None


@pytest.mark.asyncio
async def test_unreferenced_terminal_entries_prune_only_after_task_completion():
    manager = AsyncWorkManager()

    async def work():
        return None

    handle = manager.submit(OperationType.RECORD, work)
    assert (
        await asyncio.wait_for(manager.wait(handle), timeout=1) is TerminalState.SUCCESS
    )

    manager.prune_terminal()
    assert manager.get_record(handle) is None


class SnapshotSignallingManager(AsyncWorkManager):
    """Expose the synchronous cutoff point to barrier-controlled tests."""

    def __init__(self):
        super().__init__()
        self.snapshot_captured = asyncio.Event()

    def _capture_flush_generation(self, generation, cutoff):
        super()._capture_flush_generation(generation, cutoff)
        self.snapshot_captured.set()


@pytest.mark.asyncio
async def test_flush_reports_cutoff_states_without_waiting_for_later_independent_job():
    manager = SnapshotSignallingManager()
    release_success = asyncio.Event()
    release_cancelled = asyncio.Event()
    release_later = asyncio.Event()
    success_started = asyncio.Event()
    cancelled_started = asyncio.Event()
    later_started = asyncio.Event()

    async def wait_for_release(started, release):
        started.set()
        await release.wait()

    successful = manager.submit(
        OperationType.EVALUATION,
        lambda: wait_for_release(success_started, release_success),
    )
    cancelled = manager.submit(
        OperationType.EMBEDDING,
        lambda: wait_for_release(cancelled_started, release_cancelled),
    )
    await asyncio.wait_for(success_started.wait(), timeout=1)
    await asyncio.wait_for(cancelled_started.wait(), timeout=1)

    flush_task = asyncio.create_task(manager.flush())
    await asyncio.wait_for(manager.snapshot_captured.wait(), timeout=1)
    later = manager.submit(
        OperationType.REFLECTION,
        lambda: wait_for_release(later_started, release_later),
    )
    await asyncio.wait_for(later_started.wait(), timeout=1)

    assert manager.cancel(cancelled)
    release_success.set()
    try:
        report = await asyncio.wait_for(flush_task, timeout=1)
    finally:
        release_later.set()
        await asyncio.wait_for(manager.wait(later), timeout=1)

    assert isinstance(report, FlushReport)
    assert report.job_ids == (successful.job_id, cancelled.job_id)
    assert list(report.terminal_states) == [successful.job_id, cancelled.job_id]
    assert report.terminal_states == {
        successful.job_id: TerminalState.SUCCESS,
        cancelled.job_id: TerminalState.CANCELLATION,
    }
    assert later.job_id not in report.terminal_states
    manager.prune_terminal()


@pytest.mark.asyncio
async def test_flush_includes_descendant_submitted_after_cutoff():
    manager = SnapshotSignallingManager()
    parent_started = asyncio.Event()
    create_child = asyncio.Event()
    child_started = asyncio.Event()
    child_submitted = asyncio.Event()
    release_child = asyncio.Event()
    children = []

    async def child_work():
        child_started.set()
        await release_child.wait()

    async def parent_work():
        parent_started.set()
        await create_child.wait()
        children.append(
            manager.submit(
                OperationType.RULE_GENERATION,
                child_work,
                parent_job_id=parent.job_id,
            )
        )
        child_submitted.set()

    parent = manager.submit(OperationType.EVALUATION, parent_work)
    await asyncio.wait_for(parent_started.wait(), timeout=1)
    flush_task = asyncio.create_task(manager.flush())
    await asyncio.wait_for(manager.snapshot_captured.wait(), timeout=1)

    create_child.set()
    await asyncio.wait_for(child_submitted.wait(), timeout=1)
    await asyncio.wait_for(child_started.wait(), timeout=1)
    child = children[0]
    assert child.accepted_sequence > parent.accepted_sequence
    assert not flush_task.done()

    release_child.set()
    report = await asyncio.wait_for(flush_task, timeout=1)

    assert report.job_ids == (parent.job_id, child.job_id)
    assert tuple(report.terminal_states.values()) == (
        TerminalState.SUCCESS,
        TerminalState.SUCCESS,
    )


@pytest.mark.asyncio
async def test_flush_aggregates_failures_in_acceptance_order_after_all_jobs_finish():
    manager = SnapshotSignallingManager()
    releases = [asyncio.Event() for _ in range(3)]
    started = [asyncio.Event() for _ in range(3)]
    successful_finished = asyncio.Event()
    first_experience_id = uuid4()
    second_experience_id = uuid4()

    async def fail(index, error):
        started[index].set()
        await releases[index].wait()
        raise error

    async def succeed():
        started[2].set()
        await releases[2].wait()
        successful_finished.set()

    first = manager.submit(
        OperationType.EVALUATION,
        lambda: fail(0, ValueError("first private text")),
        experience_id=first_experience_id,
    )
    second = manager.submit(
        OperationType.EMBEDDING,
        lambda: fail(1, RuntimeError("second private text")),
        experience_id=second_experience_id,
    )
    manager.submit(OperationType.REFLECTION, succeed)
    for event in started:
        await asyncio.wait_for(event.wait(), timeout=1)

    flush_task = asyncio.create_task(manager.flush())
    await asyncio.wait_for(manager.snapshot_captured.wait(), timeout=1)
    releases[1].set()
    assert (
        await asyncio.wait_for(manager.wait(second), timeout=1) is TerminalState.FAILURE
    )
    releases[0].set()
    releases[2].set()

    with pytest.raises(EvaluationFailure) as raised:
        await asyncio.wait_for(flush_task, timeout=1)

    error = raised.value
    assert error.job_id == first.job_id
    assert error.operation == OperationType.EVALUATION.value
    assert error.experience_id == first_experience_id
    assert [detail.job_id for detail in error.failures] == [
        first.job_id,
        second.job_id,
    ]
    assert [detail.operation for detail in error.failures] == [
        OperationType.EVALUATION.value,
        OperationType.EMBEDDING.value,
    ]
    assert [detail.experience_id for detail in error.failures] == [
        first_experience_id,
        second_experience_id,
    ]
    assert [detail.error_type for detail in error.failures] == [
        "ValueError",
        "RuntimeError",
    ]
    assert successful_finished.is_set()


class ShutdownSignallingManager(AsyncWorkManager):
    """Expose shutdown's atomic submission cutoff to barrier-controlled tests."""

    def __init__(self):
        super().__init__()
        self.shutdown_captured = asyncio.Event()

    def _capture_shutdown_generation(self, generation):
        records = super()._capture_shutdown_generation(generation)
        self.shutdown_captured.set()
        return records


@pytest.mark.asyncio
async def test_drain_shutdown_rejects_work_and_delays_ordered_failures():
    from experia.core.exceptions import LifecycleError

    manager = ShutdownSignallingManager()
    started = [asyncio.Event() for _ in range(3)]
    releases = [asyncio.Event() for _ in range(3)]
    successful_finished = asyncio.Event()

    async def fail(index, error):
        started[index].set()
        await releases[index].wait()
        raise error

    async def succeed():
        started[2].set()
        await releases[2].wait()
        successful_finished.set()

    first = manager.submit(
        OperationType.EVALUATION,
        lambda: fail(0, ValueError("first private text")),
    )
    second = manager.submit(
        OperationType.EMBEDDING,
        lambda: fail(1, RuntimeError("second private text")),
    )
    manager.submit(OperationType.REFLECTION, succeed)
    for event in started:
        await asyncio.wait_for(event.wait(), timeout=1)

    shutdown_task = asyncio.create_task(manager.shutdown("drain"))
    await asyncio.wait_for(manager.shutdown_captured.wait(), timeout=1)

    with pytest.raises(LifecycleError) as rejected:
        manager.submit(OperationType.RECORD, succeed)
    assert rejected.value.state == "draining"
    assert rejected.value.operation == "submit"

    releases[1].set()
    await asyncio.wait_for(manager.wait(second), timeout=1)
    releases[0].set()
    await asyncio.wait_for(manager.wait(first), timeout=1)
    assert not shutdown_task.done()

    releases[2].set()
    with pytest.raises(EvaluationFailure) as raised:
        await asyncio.wait_for(shutdown_task, timeout=1)

    assert successful_finished.is_set()
    assert [failure.job_id for failure in raised.value.failures] == [
        first.job_id,
        second.job_id,
    ]
    assert type(raised.value) is EvaluationFailure
    assert manager.state.value == "closed"


@pytest.mark.asyncio
async def test_cancel_shutdown_is_shared_and_awaits_cancellation_cleanup():
    from experia.core.exceptions import LifecycleError

    manager = ShutdownSignallingManager()
    started = [asyncio.Event(), asyncio.Event()]
    cancellation_seen = [asyncio.Event(), asyncio.Event()]
    cleanup_release = asyncio.Event()

    async def cancellable(index):
        started[index].set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_seen[index].set()
            await cleanup_release.wait()
            raise

    first = manager.submit(OperationType.EVALUATION, lambda: cancellable(0))
    second = manager.submit(OperationType.EMBEDDING, lambda: cancellable(1))
    for event in started:
        await asyncio.wait_for(event.wait(), timeout=1)

    first_caller = asyncio.create_task(manager.shutdown("cancel"))
    await asyncio.wait_for(manager.shutdown_captured.wait(), timeout=1)
    second_caller = asyncio.create_task(manager.shutdown("drain"))
    for event in cancellation_seen:
        await asyncio.wait_for(event.wait(), timeout=1)

    with pytest.raises(LifecycleError) as rejected:
        manager.submit(OperationType.RECORD, lambda: cancellable(0))
    assert rejected.value.state == "cancelling"
    assert rejected.value.operation == "submit"
    assert not first_caller.done()
    assert not second_caller.done()

    cleanup_release.set()
    first_report, second_report = await asyncio.wait_for(
        asyncio.gather(first_caller, second_caller),
        timeout=1,
    )

    assert first_report is second_report
    assert first_report.policy.value == "cancel"
    assert first_report.job_ids == (first.job_id, second.job_id)
    assert tuple(first_report.terminal_states.values()) == (
        TerminalState.CANCELLATION,
        TerminalState.CANCELLATION,
    )
    assert first_report.failures == ()
    assert manager.state.value == "closed"
    assert await manager.shutdown("drain") is first_report
