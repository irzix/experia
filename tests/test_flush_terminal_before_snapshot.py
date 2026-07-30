"""Regression tests for flushing jobs that finish before the cutoff snapshot.

These tests are deterministic and version-independent: they force a background
job to reach a terminal state *before* ``flush()`` takes its synchronous
snapshot by awaiting the job first. On the previously-buggy code the terminal
job was dropped from the cutoff generation, so ``flush()`` returned an empty
report and never surfaced the typed failure. The scheduling here does not rely
on any ``asyncio.wait_for`` internals, so it fails on the old code on every
supported Python version.
"""

import asyncio

import pytest

from experia.core.exceptions import EvaluationFailure
from experia.core.work import (
    AsyncWorkManager,
    OperationType,
    TerminalState,
)


async def _succeeds() -> None:
    return None


async def _fails() -> None:
    raise RuntimeError("private downstream failure")


async def _report_terminal_success_before_snapshot() -> None:
    manager = AsyncWorkManager()
    handle = manager.submit(OperationType.EVALUATION, _succeeds)

    # The job is fully terminal before flush() snapshots the cutoff.
    assert await manager.wait(handle) is TerminalState.SUCCESS
    assert manager.get_record(handle).state.is_terminal

    report = await manager.flush()

    assert report.job_ids == (handle.job_id,)
    assert report.terminal_states[handle.job_id] is TerminalState.SUCCESS


async def _report_terminal_failure_before_snapshot() -> None:
    manager = AsyncWorkManager()
    handle = manager.submit(OperationType.EVALUATION, _fails)

    assert await manager.wait(handle) is TerminalState.FAILURE

    with pytest.raises(EvaluationFailure) as raised:
        await manager.flush()

    assert [detail.job_id for detail in raised.value.failures] == [handle.job_id]
    assert raised.value.failures[0].operation == OperationType.EVALUATION.value


async def _terminal_job_is_reported_by_exactly_one_flush() -> None:
    manager = AsyncWorkManager()
    handle = manager.submit(OperationType.EVALUATION, _succeeds)
    assert await manager.wait(handle) is TerminalState.SUCCESS

    first = await manager.flush()
    assert first.job_ids == (handle.job_id,)

    # A later flush must not re-report a job an earlier flush already released.
    second = await manager.flush()
    assert second.job_ids == ()


def test_flush_reports_job_that_succeeded_before_snapshot() -> None:
    asyncio.run(_report_terminal_success_before_snapshot())


def test_flush_surfaces_failure_that_completed_before_snapshot() -> None:
    asyncio.run(_report_terminal_failure_before_snapshot())


def test_terminal_job_is_reported_by_exactly_one_flush() -> None:
    asyncio.run(_terminal_job_is_reported_by_exactly_one_flush())
