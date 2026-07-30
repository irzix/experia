"""Property tests for stable asynchronous flush cutoffs."""

import asyncio
from dataclasses import dataclass

from hypothesis import example, given, settings
from hypothesis import strategies as st

from experia.core.work import (
    AsyncWorkManager,
    FlushReport,
    JobHandle,
    OperationType,
    TerminalState,
)


@dataclass(frozen=True)
class FlushCutoffCase:
    """A controlled set of jobs submitted on both sides of a flush cutoff."""

    pre_operations: tuple[OperationType, ...]
    pre_cancelled: tuple[bool, ...]
    completion_order: tuple[int, ...]
    post_operations: tuple[OperationType, ...]


@st.composite
def flush_cutoff_cases(draw: st.DrawFn) -> FlushCutoffCase:
    pre_count = draw(st.integers(min_value=1, max_value=8))
    operations = st.sampled_from(tuple(OperationType))
    return FlushCutoffCase(
        pre_operations=tuple(
            draw(st.lists(operations, min_size=pre_count, max_size=pre_count))
        ),
        pre_cancelled=tuple(
            draw(st.lists(st.booleans(), min_size=pre_count, max_size=pre_count))
        ),
        completion_order=tuple(draw(st.permutations(tuple(range(pre_count))))),
        post_operations=tuple(draw(st.lists(operations, min_size=1, max_size=5))),
    )


class CutoffSignallingManager(AsyncWorkManager):
    """Expose the exact synchronous cutoff to event-controlled property tests."""

    def __init__(self) -> None:
        super().__init__()
        self.snapshot_captured = asyncio.Event()

    def _capture_flush_generation(self, generation: int, cutoff: int) -> None:
        super()._capture_flush_generation(generation, cutoff)
        self.snapshot_captured.set()


async def _wait_for_release(started: asyncio.Event, release: asyncio.Event) -> None:
    started.set()
    await release.wait()


async def _await_all_started(events: list[asyncio.Event]) -> None:
    await asyncio.wait_for(
        asyncio.gather(*(event.wait() for event in events)),
        timeout=2,
    )


async def _exercise_flush_cutoff(case: FlushCutoffCase) -> None:
    manager = CutoffSignallingManager()
    pre_started = [asyncio.Event() for _ in case.pre_operations]
    pre_releases = [asyncio.Event() for _ in case.pre_operations]
    post_started = [asyncio.Event() for _ in case.post_operations]
    post_releases = [asyncio.Event() for _ in case.post_operations]
    pre_handles: list[JobHandle] = []
    post_handles: list[JobHandle] = []
    flush_task: asyncio.Task[FlushReport] | None = None

    try:
        for operation, started, release in zip(
            case.pre_operations,
            pre_started,
            pre_releases,
            strict=True,
        ):
            pre_handles.append(
                manager.submit(
                    operation,
                    lambda started=started, release=release: _wait_for_release(
                        started, release
                    ),
                )
            )
        await _await_all_started(pre_started)

        flush_task = asyncio.create_task(manager.flush())
        await asyncio.wait_for(manager.snapshot_captured.wait(), timeout=2)

        for operation, started, release in zip(
            case.post_operations,
            post_started,
            post_releases,
            strict=True,
        ):
            post_handles.append(
                manager.submit(
                    operation,
                    lambda started=started, release=release: _wait_for_release(
                        started, release
                    ),
                )
            )
        await _await_all_started(post_started)

        assert not flush_task.done()
        expected_states: dict = {}
        for position, index in enumerate(case.completion_order):
            handle = pre_handles[index]
            if case.pre_cancelled[index]:
                assert manager.cancel(handle)
                expected_state = TerminalState.CANCELLATION
            else:
                pre_releases[index].set()
                expected_state = TerminalState.SUCCESS

            assert (
                await asyncio.wait_for(manager.wait(handle), timeout=2)
                is expected_state
            )
            expected_states[handle.job_id] = expected_state
            if position < len(pre_handles) - 1:
                assert not flush_task.done()

        report = await asyncio.wait_for(flush_task, timeout=2)
        expected_job_ids = tuple(handle.job_id for handle in pre_handles)
        expected_states = {
            handle.job_id: expected_states[handle.job_id] for handle in pre_handles
        }

        assert report.job_ids == expected_job_ids
        assert dict(report.terminal_states) == expected_states
        assert all(
            handle.job_id not in report.terminal_states for handle in post_handles
        )
        assert all(
            not manager.get_record(handle).state.is_terminal for handle in post_handles
        )
    finally:
        for release in (*pre_releases, *post_releases):
            release.set()
        await asyncio.gather(
            *(manager.wait(handle) for handle in (*pre_handles, *post_handles)),
            return_exceptions=True,
        )
        if flush_task is not None:
            await asyncio.gather(flush_task, return_exceptions=True)
        manager.prune_terminal()


# Feature: open-source-project-improvements, Property 3: Flush cutoff is complete and stable
# **Validates: Requirements 1.3, 1.4**
@settings(max_examples=100, deadline=None)
@given(case=flush_cutoff_cases())
@example(
    case=FlushCutoffCase(
        pre_operations=(OperationType.EVALUATION,),
        pre_cancelled=(False,),
        completion_order=(0,),
        post_operations=(OperationType.EMBEDDING,),
    )
)
@example(
    case=FlushCutoffCase(
        pre_operations=(
            OperationType.EVALUATION,
            OperationType.RULE_GENERATION,
            OperationType.REFLECTION,
        ),
        pre_cancelled=(True, False, True),
        completion_order=(2, 0, 1),
        post_operations=(OperationType.RECORD, OperationType.EMBEDDING),
    )
)
def test_flush_cutoff_is_complete_and_stable(case: FlushCutoffCase) -> None:
    asyncio.run(_exercise_flush_cutoff(case))
