"""Owned asynchronous background-work registry and terminal state machine."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from itertools import count
from types import MappingProxyType
from typing import Any, TypeAlias
from uuid import UUID, uuid4

from experia.core.exceptions import EvaluationFailure, FailureDetail, LifecycleError
from experia.core.logging import (
    EVENT_SCHEMA_VERSION,
    LifecycleEvent,
    OperationType,
    TerminalState,
    emit_lifecycle_event,
    emit_observer_failure_diagnostic,
)


@dataclass(frozen=True)
class FlushReport:
    """Deterministic terminal outcomes for one cutoff generation."""

    generation: int
    job_ids: tuple[UUID, ...]
    terminal_states: Mapping[UUID, TerminalState]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "terminal_states",
            MappingProxyType(dict(self.terminal_states)),
        )


class ShutdownPolicy(str, Enum):
    """Supported handling for work accepted before shutdown."""

    DRAIN = "drain"
    CANCEL = "cancel"


class WorkManagerState(str, Enum):
    """Lifecycle state controlling whether new work may be accepted."""

    OPEN = "open"
    DRAINING = "draining"
    CANCELLING = "cancelling"
    CLOSED = "closed"


@dataclass(frozen=True)
class ShutdownReport:
    """Deterministic terminal outcomes for the shared shutdown operation."""

    policy: ShutdownPolicy
    job_ids: tuple[UUID, ...]
    terminal_states: Mapping[UUID, TerminalState]
    failures: tuple[FailureDetail, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "terminal_states",
            MappingProxyType(dict(self.terminal_states)),
        )
        object.__setattr__(self, "failures", tuple(self.failures))


class JobState(str, Enum):
    """Internal state of a registered background job."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = TerminalState.SUCCESS.value
    FAILURE = TerminalState.FAILURE.value
    CANCELLATION = TerminalState.CANCELLATION.value

    @property
    def is_terminal(self) -> bool:
        return self in {
            JobState.SUCCESS,
            JobState.FAILURE,
            JobState.CANCELLATION,
        }


@dataclass(frozen=True)
class JobHandle:
    """Stable, non-sensitive identity assigned when work is accepted."""

    job_id: UUID
    operation: OperationType
    experience_id: UUID | None
    accepted_sequence: int
    parent_job_id: UUID | None


LifecycleObserver: TypeAlias = Callable[[LifecycleEvent], None | Awaitable[None]]
AwaitableFactory: TypeAlias = Callable[[], Awaitable[Any]]
JobIdentifier: TypeAlias = JobHandle | UUID


@dataclass
class JobRecord:
    """Loop-owned registry entry retaining a strong task reference."""

    handle: JobHandle
    accepted_at_ns: int
    terminal_future: asyncio.Future[TerminalState]
    state: JobState = JobState.PENDING
    task: asyncio.Task[None] | None = None
    started_at_ns: int | None = None
    finished_at_ns: int | None = None
    failure: FailureDetail | None = None
    flush_generations: set[int] = field(default_factory=set)
    terminal_event: LifecycleEvent | None = None
    observer_task: asyncio.Task[None] | None = None
    observer_started: bool = False
    prune_when_unreferenced: bool = field(default=False, repr=False)

    @property
    def terminal_state(self) -> TerminalState | None:
        if not self.state.is_terminal:
            return None
        return TerminalState(self.state.value)


class AsyncWorkManager:
    """Own background tasks and record each accepted job's terminal outcome.

    Registry mutation is synchronous and loop-local: no mutation path suspends
    between checking and changing a state. Observer execution is scheduled only
    after the terminal state and event have been stored.
    """

    def __init__(self, observer: LifecycleObserver | None = None) -> None:
        self._observer = observer
        self._records: dict[UUID, JobRecord] = {}
        self._children: dict[UUID, set[UUID]] = {}
        self._issued_job_ids: set[UUID] = set()
        self._accepted_sequences = count(1)
        self._flush_generations = count(1)
        self._observer_tasks: set[asyncio.Task[None]] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._state = WorkManagerState.OPEN
        self._shutdown_task: asyncio.Task[ShutdownReport] | None = None

    @property
    def state(self) -> WorkManagerState:
        """Return the manager lifecycle state."""

        return self._state

    @property
    def jobs(self) -> tuple[JobHandle, ...]:
        """Return registered handles in acceptance order."""

        records = sorted(
            self._records.values(), key=lambda record: record.handle.accepted_sequence
        )
        return tuple(record.handle for record in records)

    @property
    def records(self) -> Mapping[UUID, JobRecord]:
        """Expose a read-only view of current registry entries."""

        return MappingProxyType(self._records)

    def submit(
        self,
        operation: OperationType | str,
        awaitable_factory: AwaitableFactory,
        *,
        experience_id: UUID | None = None,
        parent_job_id: UUID | None = None,
    ) -> JobHandle:
        """Accept and schedule one job, retaining its task strongly."""

        parent = self._records.get(parent_job_id) if parent_job_id is not None else None
        accepted_drain_descendant = (
            self._state is WorkManagerState.DRAINING
            and parent is not None
            and not parent.state.is_terminal
            and bool(parent.flush_generations)
        )
        if self._state is not WorkManagerState.OPEN and not accepted_drain_descendant:
            raise LifecycleError(state=self._state.value, operation="submit")
        if not callable(awaitable_factory):
            raise TypeError("awaitable_factory must be callable")

        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise RuntimeError("AsyncWorkManager cannot be used across event loops")

        normalized_operation = OperationType(operation)
        job_id = self._new_job_id()
        handle = JobHandle(
            job_id=job_id,
            operation=normalized_operation,
            experience_id=experience_id,
            accepted_sequence=next(self._accepted_sequences),
            parent_job_id=parent_job_id,
        )
        accepted_at_ns = time.perf_counter_ns()
        record = JobRecord(
            handle=handle,
            accepted_at_ns=accepted_at_ns,
            terminal_future=loop.create_future(),
        )
        self._records[job_id] = record
        if parent_job_id is not None:
            self._children.setdefault(parent_job_id, set()).add(job_id)
            parent = self._records.get(parent_job_id)
            if parent is not None:
                record.flush_generations.update(parent.flush_generations)

        task = loop.create_task(
            self._run(record, awaitable_factory),
            name=f"experia-{normalized_operation.value}-{job_id}",
        )
        record.task = task
        task.add_done_callback(lambda completed: self._on_task_done(record, completed))
        return handle

    def get_record(self, job: JobIdentifier) -> JobRecord | None:
        """Return the current registry entry for a handle or identifier."""

        return self._records.get(self._job_id(job))

    def cancel(self, job: JobIdentifier) -> bool:
        """Request cooperative cancellation of a non-terminal job."""

        record = self.get_record(job)
        if record is None or record.state.is_terminal or record.task is None:
            return False
        return record.task.cancel()

    async def wait(self, job: JobIdentifier) -> TerminalState:
        """Wait for one job's terminal transition and observer completion."""

        record = self.get_record(job)
        if record is None:
            raise KeyError(self._job_id(job))

        terminal_state = await asyncio.shield(record.terminal_future)
        observer_task = record.observer_task
        if observer_task is not None:
            await asyncio.shield(observer_task)
        return terminal_state

    def retain_for_flush(
        self, generation: int, jobs: Iterable[JobIdentifier]
    ) -> tuple[JobRecord, ...]:
        """Retain existing registry entries for a flush generation."""

        retained: list[JobRecord] = []
        for job in jobs:
            record = self.get_record(job)
            if record is None:
                continue
            record.flush_generations.add(generation)
            retained.append(record)
        return tuple(retained)

    def release_flush_generation(self, generation: int) -> None:
        """Release one generation and prune newly unreferenced terminal entries."""

        for record in tuple(self._records.values()):
            if generation not in record.flush_generations:
                continue
            record.flush_generations.remove(generation)
            record.prune_when_unreferenced = True
            self._try_prune(record)

    def prune_terminal(self) -> None:
        """Request pruning of terminal entries not retained by a generation."""

        for record in tuple(self._records.values()):
            if record.state.is_terminal:
                record.prune_when_unreferenced = True
                self._try_prune(record)

    def descendants_of(self, job: JobIdentifier) -> tuple[JobHandle, ...]:
        """Return currently registered causal descendants in acceptance order."""

        root_id = self._job_id(job)
        pending = list(self._children.get(root_id, ()))
        seen: set[UUID] = set()
        descendants: list[JobHandle] = []
        while pending:
            child_id = pending.pop()
            if child_id in seen:
                continue
            seen.add(child_id)
            record = self._records.get(child_id)
            if record is not None:
                descendants.append(record.handle)
            pending.extend(self._children.get(child_id, ()))
        descendants.sort(key=lambda handle: handle.accepted_sequence)
        return tuple(descendants)

    async def flush(self) -> FlushReport:
        """Await the non-terminal cutoff snapshot and its causal descendants."""

        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise RuntimeError("AsyncWorkManager cannot be used across event loops")

        generation = next(self._flush_generations)
        cutoff = max(
            (record.handle.accepted_sequence for record in self._records.values()),
            default=0,
        )
        self._capture_flush_generation(generation, cutoff)

        try:
            records = await self._wait_for_flush_generation(generation)
            terminal_states = {
                record.handle.job_id: record.terminal_state
                for record in records
                if record.terminal_state is not None
            }
            report = FlushReport(
                generation=generation,
                job_ids=tuple(record.handle.job_id for record in records),
                terminal_states=terminal_states,
            )
            failures = tuple(
                record.failure for record in records if record.failure is not None
            )
            if failures:
                first = failures[0]
                raise EvaluationFailure(
                    job_id=first.job_id,
                    operation=first.operation,
                    experience_id=first.experience_id,
                    failures=failures,
                )
            return report
        finally:
            self.release_flush_generation(generation)

    async def shutdown(
        self, policy: ShutdownPolicy | str = ShutdownPolicy.DRAIN
    ) -> ShutdownReport:
        """Close submissions and share one drain-or-cancel completion.

        The first caller selects the policy. Concurrent and later callers await
        the same shielded task, so cancelling one caller cannot interrupt the
        manager's terminal-state accounting.
        """

        normalized_policy = ShutdownPolicy(policy)
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise RuntimeError("AsyncWorkManager cannot be used across event loops")

        if self._state is WorkManagerState.OPEN:
            self._state = (
                WorkManagerState.DRAINING
                if normalized_policy is ShutdownPolicy.DRAIN
                else WorkManagerState.CANCELLING
            )
            generation = next(self._flush_generations)
            records = self._capture_shutdown_generation(generation)
            if normalized_policy is ShutdownPolicy.CANCEL:
                self._request_shutdown_cancellation(records)
            self._shutdown_task = loop.create_task(
                self._complete_shutdown(normalized_policy, generation),
                name=f"experia-shutdown-{normalized_policy.value}",
            )

        shutdown_task = self._shutdown_task
        if shutdown_task is None:  # pragma: no cover - lifecycle invariant
            raise RuntimeError("AsyncWorkManager shutdown completion is unavailable")
        return await asyncio.shield(shutdown_task)

    def _capture_shutdown_generation(self, generation: int) -> tuple[JobRecord, ...]:
        """Retain every job that is non-terminal when submissions close."""

        records = tuple(
            sorted(
                (
                    record
                    for record in self._records.values()
                    if not record.state.is_terminal
                ),
                key=lambda record: (
                    record.handle.accepted_sequence,
                    record.handle.job_id.int,
                ),
            )
        )
        for record in records:
            record.flush_generations.add(generation)
        return records

    @staticmethod
    def _request_shutdown_cancellation(records: Iterable[JobRecord]) -> None:
        """Request cancellation for each captured job without suspending."""

        for record in records:
            if record.state.is_terminal or record.task is None:
                continue
            record.task.cancel()

    async def _complete_shutdown(
        self,
        policy: ShutdownPolicy,
        generation: int,
    ) -> ShutdownReport:
        """Await the retained shutdown generation and publish its outcome."""

        try:
            records = await self._wait_for_flush_generation(generation)
            terminal_states = {
                record.handle.job_id: record.terminal_state
                for record in records
                if record.terminal_state is not None
            }
            failures = tuple(
                record.failure for record in records if record.failure is not None
            )
            report = ShutdownReport(
                policy=policy,
                job_ids=tuple(record.handle.job_id for record in records),
                terminal_states=terminal_states,
                failures=failures,
            )
            if policy is ShutdownPolicy.DRAIN and failures:
                first = failures[0]
                raise EvaluationFailure(
                    job_id=first.job_id,
                    operation=first.operation,
                    experience_id=first.experience_id,
                    failures=failures,
                )
            return report
        finally:
            self.release_flush_generation(generation)
            self._state = WorkManagerState.CLOSED

    def _capture_flush_generation(self, generation: int, cutoff: int) -> None:
        """Retain cutoff-eligible roots and their currently known descendants."""

        pending = [
            record.handle.job_id
            for record in self._records.values()
            if (
                record.handle.accepted_sequence <= cutoff
                and not record.state.is_terminal
            )
        ]
        seen: set[UUID] = set()
        while pending:
            job_id = pending.pop()
            if job_id in seen:
                continue
            seen.add(job_id)
            record = self._records.get(job_id)
            if record is None:
                continue
            record.flush_generations.add(generation)
            pending.extend(self._children.get(job_id, ()))

    async def _wait_for_flush_generation(
        self, generation: int
    ) -> tuple[JobRecord, ...]:
        """Wait until one generation stops gaining work and is terminal."""

        while True:
            records = tuple(
                sorted(
                    (
                        record
                        for record in self._records.values()
                        if generation in record.flush_generations
                    ),
                    key=lambda record: (
                        record.handle.accepted_sequence,
                        record.handle.job_id.int,
                    ),
                )
            )
            pending = tuple(
                record
                for record in records
                if (
                    not record.state.is_terminal
                    or (
                        record.observer_task is not None
                        and not record.observer_task.done()
                    )
                )
            )
            if not pending:
                return records
            await asyncio.gather(*(self.wait(record.handle) for record in pending))

    async def _run(
        self, record: JobRecord, awaitable_factory: AwaitableFactory
    ) -> None:
        self._mark_running(record)
        try:
            await awaitable_factory()
        except asyncio.CancelledError:
            observer_task = self._transition_terminal(
                record, TerminalState.CANCELLATION
            )
            await self._await_observer_without_changing_state(observer_task)
            raise
        except BaseException as error:
            observer_task = self._transition_terminal(
                record,
                TerminalState.FAILURE,
                error_type=type(error).__name__,
            )
            await self._await_observer_without_changing_state(observer_task)
            if not isinstance(error, Exception):
                raise
        else:
            observer_task = self._transition_terminal(record, TerminalState.SUCCESS)
            await self._await_observer_without_changing_state(observer_task)

    def _mark_running(self, record: JobRecord) -> None:
        if record.state is not JobState.PENDING:
            return
        record.state = JobState.RUNNING
        record.started_at_ns = time.perf_counter_ns()

    def _transition_terminal(
        self,
        record: JobRecord,
        terminal_state: TerminalState,
        *,
        error_type: str | None = None,
    ) -> asyncio.Task[None] | None:
        if record.state.is_terminal:
            return record.observer_task

        finished_at_ns = time.perf_counter_ns()
        started_at_ns = record.started_at_ns or record.accepted_at_ns
        record.finished_at_ns = finished_at_ns
        record.state = JobState(terminal_state.value)
        if terminal_state is TerminalState.FAILURE:
            record.failure = FailureDetail(
                job_id=record.handle.job_id,
                operation=record.handle.operation.value,
                experience_id=record.handle.experience_id,
                error_type=error_type or "Exception",
            )

        event = LifecycleEvent(
            schema_version=EVENT_SCHEMA_VERSION,
            job_id=record.handle.job_id,
            operation=record.handle.operation,
            terminal_state=terminal_state,
            duration_ms=max(0, (finished_at_ns - started_at_ns) // 1_000_000),
        )
        record.terminal_event = event
        if not record.terminal_future.done():
            record.terminal_future.set_result(terminal_state)
        self._emit_lifecycle_event_without_changing_state(event)
        return self._schedule_observer(record, event)

    @staticmethod
    def _emit_lifecycle_event_without_changing_state(event: LifecycleEvent) -> None:
        try:
            emit_lifecycle_event(event)
        except BaseException:
            # Host logging hooks cannot alter terminal-state accounting.
            pass

    def _schedule_observer(
        self, record: JobRecord, event: LifecycleEvent
    ) -> asyncio.Task[None] | None:
        if self._observer is None or record.observer_started:
            return record.observer_task

        record.observer_started = True
        loop = self._loop
        if loop is None:  # pragma: no cover - submit always binds the loop first
            return None
        observer_task = loop.create_task(
            self._invoke_observer(event),
            name=f"experia-observer-{record.handle.job_id}",
        )
        record.observer_task = observer_task
        self._observer_tasks.add(observer_task)
        observer_task.add_done_callback(
            lambda completed: self._on_observer_done(record, completed)
        )
        return observer_task

    async def _invoke_observer(self, event: LifecycleEvent) -> None:
        observer = self._observer
        if observer is None:
            return
        try:
            result = observer(event)
            if inspect.isawaitable(result):
                await result
        except BaseException:
            # Observer code cannot alter a committed terminal state and is never retried.
            try:
                emit_observer_failure_diagnostic()
            except BaseException:
                # A failing host logging hook is isolated for the same reason.
                pass

    async def _await_observer_without_changing_state(
        self, observer_task: asyncio.Task[None] | None
    ) -> None:
        if observer_task is None:
            return
        try:
            await asyncio.shield(observer_task)
        except asyncio.CancelledError:
            # The observer task remains strongly referenced and continues independently.
            return

    def _on_task_done(self, record: JobRecord, completed: asyncio.Task[None]) -> None:
        if not record.state.is_terminal:
            if completed.cancelled():
                self._transition_terminal(record, TerminalState.CANCELLATION)
            else:
                error = completed.exception()
                if error is None:
                    self._transition_terminal(record, TerminalState.SUCCESS)
                else:
                    self._transition_terminal(
                        record,
                        TerminalState.FAILURE,
                        error_type=type(error).__name__,
                    )
        elif not completed.cancelled():
            # Retrieve a BaseException re-raised by the runner to avoid an
            # unobserved-task warning; ordinary job failures are stored instead.
            completed.exception()
        self._try_prune(record)

    def _on_observer_done(
        self, record: JobRecord, completed: asyncio.Task[None]
    ) -> None:
        self._observer_tasks.discard(completed)
        if not completed.cancelled():
            completed.exception()
        self._try_prune(record)

    def _try_prune(self, record: JobRecord) -> None:
        task_done = record.task is not None and record.task.done()
        observer_done = record.observer_task is None or record.observer_task.done()
        if not (
            record.prune_when_unreferenced
            and record.state.is_terminal
            and not record.flush_generations
            and task_done
            and observer_done
        ):
            return

        job_id = record.handle.job_id
        if self._records.get(job_id) is record:
            del self._records[job_id]
        parent_id = record.handle.parent_job_id
        if parent_id is not None:
            siblings = self._children.get(parent_id)
            if siblings is not None:
                siblings.discard(job_id)
                if not siblings:
                    del self._children[parent_id]

    def _new_job_id(self) -> UUID:
        job_id = uuid4()
        while job_id in self._issued_job_ids:
            job_id = uuid4()
        self._issued_job_ids.add(job_id)
        return job_id

    @staticmethod
    def _job_id(job: JobIdentifier) -> UUID:
        if isinstance(job, JobHandle):
            return job.job_id
        if isinstance(job, UUID):
            return job
        raise TypeError("job must be a JobHandle or UUID")


__all__ = [
    "AsyncWorkManager",
    "FlushReport",
    "JobHandle",
    "JobRecord",
    "JobState",
    "LifecycleEvent",
    "LifecycleObserver",
    "OperationType",
    "ShutdownPolicy",
    "ShutdownReport",
    "TerminalState",
    "WorkManagerState",
]
