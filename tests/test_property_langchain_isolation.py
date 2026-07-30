"""Property coverage for identifier-isolated LangChain run state."""

from __future__ import annotations

import asyncio
import copy
import logging
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any
from uuid import UUID

from hypothesis import given, settings
from hypothesis import strategies as st

from experia.core.logging import (
    EVENT_SCHEMA_VERSION,
    IntegrationEvent,
    IntegrationName,
    IntegrationOutcome,
    logger,
)
from experia.integrations.langchain.builder import ExperienceBuilder


class _RunKind(str, Enum):
    RECORDED = "recorded"
    TOOL_FAILURE = "tool_failure"
    RECORD_FAILURE = "record_failure"
    NO_EXPERIENCE = "no_experience"


class _OperationKind(str, Enum):
    CHAIN_START = "chain_start"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    TOOL_ERROR = "tool_error"
    WRONG_PARENT_TOOL_END = "wrong_parent_tool_end"
    ORPHAN_TOOL_END = "orphan_tool_end"
    CHAIN_END = "chain_end"
    CHAIN_ERROR = "chain_error"
    LATE_TOOL_END = "late_tool_end"


@dataclass(frozen=True)
class _RunSpec:
    index: int
    kind: _RunKind
    duplicate_tool_starts: int
    duplicate_tool_ends: int
    duplicate_chain_ends: int
    chain_fails: bool

    @property
    def chain_id(self) -> UUID:
        return UUID(int=self.index + 1)

    @property
    def tool_id(self) -> UUID:
        return UUID(int=1_000 + self.index)

    @property
    def orphan_tool_id(self) -> UUID:
        return UUID(int=10_000 + self.index)

    @property
    def marker(self) -> str:
        return str(self.index)

    @property
    def has_tool(self) -> bool:
        return self.kind is not _RunKind.NO_EXPERIENCE

    @property
    def expected_result(self) -> str:
        if self.kind is _RunKind.TOOL_FAILURE:
            return f"ERROR: tool-failure-{self.marker}"
        return f"result-{self.marker}"

    @property
    def expected_outcome(self) -> IntegrationOutcome:
        if self.kind is _RunKind.NO_EXPERIENCE:
            return IntegrationOutcome.NO_EXPERIENCE
        if self.kind is _RunKind.RECORD_FAILURE:
            return IntegrationOutcome.FAILED
        return IntegrationOutcome.RECORDED


@dataclass(frozen=True)
class _Operation:
    kind: _OperationKind
    run_index: int


@dataclass(frozen=True)
class _IsolationCase:
    runs: tuple[_RunSpec, ...]
    operations: tuple[_Operation, ...]


class _RecordFailure(RuntimeError):
    pass


class _RecordingLearner:
    """Minimal durable record sink used by the real integration dispatcher."""

    def __init__(self, failing_runs: set[UUID]) -> None:
        self.failing_runs = failing_runs
        self.attempts: Counter[UUID] = Counter()
        self.records: list[dict[str, Any]] = []

    def _assert_accepting(self, operation: str) -> None:
        assert operation == "record"

    async def record(
        self,
        *,
        task: str,
        action: str,
        result: str,
        context: dict[str, Any],
    ) -> None:
        run_id = UUID(context["chain_id"])
        self.attempts[run_id] += 1
        if run_id in self.failing_runs:
            raise _RecordFailure(str(run_id))
        self.records.append(
            {
                "task": task,
                "action": action,
                "result": result,
                "context": dict(context),
            }
        )


class _IntegrationEventHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[IntegrationEvent] = []

    def emit(self, record: logging.LogRecord) -> None:
        event = getattr(record, "experia_integration_event", None)
        if isinstance(event, IntegrationEvent):
            self.events.append(event)


def _operation_queue(spec: _RunSpec) -> list[_Operation]:
    def operation(kind: _OperationKind) -> _Operation:
        return _Operation(kind=kind, run_index=spec.index)

    queue = [operation(_OperationKind.CHAIN_START)]

    if spec.has_tool:
        queue.append(operation(_OperationKind.TOOL_START))
        queue.extend(
            operation(_OperationKind.TOOL_START)
            for _ in range(spec.duplicate_tool_starts)
        )
        queue.append(operation(_OperationKind.WRONG_PARENT_TOOL_END))
        tool_terminal = (
            _OperationKind.TOOL_ERROR
            if spec.kind is _RunKind.TOOL_FAILURE
            else _OperationKind.TOOL_END
        )
        queue.append(operation(tool_terminal))
        queue.extend(operation(tool_terminal) for _ in range(spec.duplicate_tool_ends))

    queue.append(operation(_OperationKind.ORPHAN_TOOL_END))
    chain_terminal = (
        _OperationKind.CHAIN_ERROR if spec.chain_fails else _OperationKind.CHAIN_END
    )
    queue.append(operation(chain_terminal))
    queue.extend(operation(chain_terminal) for _ in range(spec.duplicate_chain_ends))
    if spec.has_tool:
        queue.append(operation(_OperationKind.LATE_TOOL_END))
    return queue


@st.composite
def _isolation_cases(draw: st.DrawFn) -> _IsolationCase:
    run_count = draw(st.integers(min_value=1, max_value=100))
    kinds = draw(
        st.lists(
            st.sampled_from(tuple(_RunKind)),
            min_size=run_count,
            max_size=run_count,
        )
    )
    duplicate_tool_starts = draw(
        st.lists(
            st.integers(min_value=0, max_value=2),
            min_size=run_count,
            max_size=run_count,
        )
    )
    duplicate_tool_ends = draw(
        st.lists(
            st.integers(min_value=0, max_value=2),
            min_size=run_count,
            max_size=run_count,
        )
    )
    duplicate_chain_ends = draw(
        st.lists(
            st.integers(min_value=1, max_value=2),
            min_size=run_count,
            max_size=run_count,
        )
    )
    chain_failures = draw(
        st.lists(
            st.booleans(),
            min_size=run_count,
            max_size=run_count,
        )
    )
    runs = tuple(
        _RunSpec(
            index=index,
            kind=kinds[index],
            duplicate_tool_starts=duplicate_tool_starts[index],
            duplicate_tool_ends=duplicate_tool_ends[index],
            duplicate_chain_ends=duplicate_chain_ends[index],
            chain_fails=chain_failures[index],
        )
        for index in range(run_count)
    )
    queues = [_operation_queue(spec) for spec in runs]
    order = draw(st.permutations(tuple(range(run_count))))
    max_queue_length = max(map(len, queues))
    round_offsets = draw(
        st.lists(
            st.integers(min_value=0, max_value=run_count - 1),
            min_size=max_queue_length,
            max_size=max_queue_length,
        )
    )
    reverse_rounds = draw(
        st.lists(
            st.booleans(),
            min_size=max_queue_length,
            max_size=max_queue_length,
        )
    )

    operations: list[_Operation] = []
    for round_index in range(max_queue_length):
        offset = round_offsets[round_index]
        round_order = order[offset:] + order[:offset]
        if reverse_rounds[round_index]:
            round_order = tuple(reversed(round_order))
        operations.extend(
            queues[run_index][round_index]
            for run_index in round_order
            if round_index < len(queues[run_index])
        )
    return _IsolationCase(runs=runs, operations=tuple(operations))


async def _apply_operation(
    builder: ExperienceBuilder,
    operation: _Operation,
    runs: tuple[_RunSpec, ...],
) -> None:
    spec = runs[operation.run_index]
    if operation.kind is _OperationKind.CHAIN_START:
        await builder.on_chain_start(
            spec.chain_id,
            serialized={"name": "AgentExecutor"},
            inputs={"input": f"task-{spec.marker}"},
        )
    elif operation.kind is _OperationKind.TOOL_START:
        await builder.on_tool_start(
            spec.tool_id,
            spec.chain_id,
            serialized={"name": f"tool-{spec.marker}"},
            inputs={"input": f"action-input-{spec.marker}"},
        )
    elif operation.kind is _OperationKind.TOOL_END:
        await builder.on_tool_end(
            spec.tool_id,
            spec.chain_id,
            f"result-{spec.marker}",
        )
    elif operation.kind is _OperationKind.TOOL_ERROR:
        await builder.on_tool_error(
            spec.tool_id,
            spec.chain_id,
            RuntimeError(f"tool-failure-{spec.marker}"),
        )
    elif operation.kind is _OperationKind.WRONG_PARENT_TOOL_END:
        wrong_parent = (
            runs[(spec.index + 1) % len(runs)].chain_id
            if len(runs) > 1
            else UUID(int=50_000)
        )
        await builder.on_tool_end(spec.tool_id, wrong_parent, "foreign-result")
    elif operation.kind is _OperationKind.ORPHAN_TOOL_END:
        await builder.on_tool_end(
            spec.orphan_tool_id,
            spec.chain_id,
            "orphan-result",
        )
    elif operation.kind is _OperationKind.CHAIN_END:
        await builder.on_chain_end(
            spec.chain_id,
            outputs={"output": f"chain-result-{spec.marker}"},
        )
    elif operation.kind is _OperationKind.CHAIN_ERROR:
        await builder.on_chain_error(
            spec.chain_id,
            RuntimeError(f"chain-failure-{spec.marker}"),
        )
    elif operation.kind is _OperationKind.LATE_TOOL_END:
        await builder.on_tool_end(
            spec.tool_id,
            spec.chain_id,
            "late-result",
        )


def _affected_chain(operation: _Operation, spec: _RunSpec) -> UUID | None:
    if operation.kind in {
        _OperationKind.WRONG_PARENT_TOOL_END,
        _OperationKind.ORPHAN_TOOL_END,
        _OperationKind.LATE_TOOL_END,
    }:
        return None
    return spec.chain_id


def _expected_events(case: _IsolationCase) -> Counter[IntegrationEvent]:
    events: list[IntegrationEvent] = []
    for spec in case.runs:
        events.append(
            IntegrationEvent(
                schema_version=EVENT_SCHEMA_VERSION,
                integration=IntegrationName.LANGCHAIN,
                outcome=spec.expected_outcome,
                run_id=spec.chain_id,
            )
        )
        events.append(
            IntegrationEvent(
                schema_version=EVENT_SCHEMA_VERSION,
                integration=IntegrationName.LANGCHAIN,
                outcome=IntegrationOutcome.ORPHAN,
                run_id=spec.orphan_tool_id,
            )
        )
        if spec.has_tool:
            events.extend(
                IntegrationEvent(
                    schema_version=EVENT_SCHEMA_VERSION,
                    integration=IntegrationName.LANGCHAIN,
                    outcome=IntegrationOutcome.ORPHAN,
                    run_id=spec.tool_id,
                )
                for _ in range(2)
            )
    return Counter(events)


async def _exercise_isolation_case(case: _IsolationCase) -> None:
    failing_runs = {
        spec.chain_id for spec in case.runs if spec.kind is _RunKind.RECORD_FAILURE
    }
    learner = _RecordingLearner(failing_runs)
    builder = ExperienceBuilder(agent=learner, callback_mode="durable")
    event_handler = _IntegrationEventHandler()
    previous_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(event_handler)

    try:
        for operation in case.operations:
            spec = case.runs[operation.run_index]
            before = copy.deepcopy(builder.active_runs)
            try:
                await _apply_operation(builder, operation, case.runs)
            except _RecordFailure:
                assert spec.kind is _RunKind.RECORD_FAILURE
                assert operation.kind in {
                    _OperationKind.CHAIN_END,
                    _OperationKind.CHAIN_ERROR,
                }

            affected_chain = _affected_chain(operation, spec)
            if affected_chain is None:
                assert builder.active_runs == before
            else:
                assert {
                    run_id: state
                    for run_id, state in builder.active_runs.items()
                    if run_id != affected_chain
                } == {
                    run_id: state
                    for run_id, state in before.items()
                    if run_id != affected_chain
                }

        assert builder.active_runs == {}
        assert builder.registry._tool_parents == {}
        assert Counter(event_handler.events) == _expected_events(case)

        expected_attempts = Counter(
            {spec.chain_id: 1 for spec in case.runs if spec.has_tool}
        )
        assert learner.attempts == expected_attempts

        successful_specs = {
            spec.chain_id: spec
            for spec in case.runs
            if spec.has_tool and spec.kind is not _RunKind.RECORD_FAILURE
        }
        assert len(learner.records) == len(successful_specs)
        for record in learner.records:
            run_id = UUID(record["context"]["chain_id"])
            spec = successful_specs.pop(run_id)
            assert record == {
                "task": f"task-{spec.marker}",
                "action": (
                    f"Tool: tool-{spec.marker} | "
                    f"Input: {{'input': 'action-input-{spec.marker}'}}"
                ),
                "result": spec.expected_result,
                "context": {"chain_id": str(spec.chain_id)},
            }
        assert successful_specs == {}
    finally:
        logger.removeHandler(event_handler)
        logger.setLevel(previous_level)


# Feature: open-source-project-improvements, Property 19: LangChain state is run-isolated and finalized at most once
@settings(max_examples=100, deadline=None)
@given(case=_isolation_cases())
def test_langchain_state_is_run_isolated_and_finalized_at_most_once(
    case: _IsolationCase,
) -> None:
    """**Validates: Requirements 6.1, 6.2, 6.8**"""
    asyncio.run(_exercise_isolation_case(case))
