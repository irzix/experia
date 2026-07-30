"""Property tests for terminal lifecycle-event sensitive-data safety."""

import asyncio
import json
import string
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any
from uuid import UUID

from hypothesis import given, settings
from hypothesis import strategies as st

from experia.core.logging import (
    EVENT_SCHEMA_VERSION,
    LifecycleEvent,
    OperationType,
    TerminalState,
)
from experia.core.work import AsyncWorkManager
from experia.experience.models import ExperienceRecord

_ALLOW_LISTED_FIELDS = frozenset(
    {
        "schema_version",
        "job_id",
        "operation",
        "terminal_state",
        "duration_ms",
    }
)
_TOKEN = st.text(
    alphabet=string.ascii_letters + string.digits + "_-",
    min_size=1,
    max_size=24,
)
_OPERATIONS = st.tuples(
    *(st.sampled_from(tuple(OperationType)) for _ in tuple(TerminalState))
)


@dataclass(frozen=True)
class _SensitiveCase:
    record: ExperienceRecord
    secret: str
    exception_text: str


@st.composite
def _sensitive_cases(draw: st.DrawFn) -> _SensitiveCase:
    secret = f"SECRET_CANARY_{draw(_TOKEN)}"
    exception_text = f"EXCEPTION_CANARY_{draw(_TOKEN)}"
    context_key = f"CONTEXT_KEY_{draw(_TOKEN)}"
    context_value = f"CONTEXT_VALUE_{draw(_TOKEN)}"
    record = ExperienceRecord(
        id=draw(st.uuids(version=1)),
        task=f"TASK_CANARY_{draw(_TOKEN)}_{secret}",
        action=f"ACTION_CANARY_{draw(_TOKEN)}",
        result=f"RESULT_CANARY_{draw(_TOKEN)}_{exception_text}",
        agent_role=f"ROLE_CANARY_{draw(_TOKEN)}",
        context={
            context_key: {
                "nested_secret": secret,
                "nested_value": context_value,
            }
        },
    )
    return _SensitiveCase(
        record=record,
        secret=secret,
        exception_text=exception_text,
    )


class _GeneratedFailure(RuntimeError):
    pass


def _serialize_event(event: LifecycleEvent) -> str:
    """Serialize every event field, converting only documented scalar types."""

    payload: dict[str, Any] = {}
    for field in fields(event):
        value = getattr(event, field.name)
        if isinstance(value, Enum):
            value = value.value
        elif isinstance(value, UUID):
            value = str(value)
        payload[field.name] = value
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _context_text_values(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, dict):
        values: list[str] = []
        for key, child in value.items():
            values.append(str(key))
            values.extend(_context_text_values(child))
        return tuple(values)
    if isinstance(value, (list, tuple)):
        return tuple(text for child in value for text in _context_text_values(child))
    return ()


async def _capture_terminal_events(
    case: _SensitiveCase,
    operations: tuple[OperationType, OperationType, OperationType],
) -> tuple[LifecycleEvent, ...]:
    observed: list[LifecycleEvent] = []
    manager = AsyncWorkManager(observer=observed.append)

    for terminal_state, operation in zip(TerminalState, operations, strict=True):
        started = asyncio.Event()
        release = asyncio.Event()

        async def controlled_job(
            outcome: TerminalState = terminal_state,
            gate: asyncio.Event = release,
        ) -> None:
            started.set()
            await gate.wait()
            if outcome is TerminalState.FAILURE:
                raise _GeneratedFailure(case.exception_text)

        handle = manager.submit(
            operation,
            controlled_job,
            experience_id=case.record.id,
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        if terminal_state is TerminalState.CANCELLATION:
            assert manager.cancel(handle)
        else:
            release.set()

        assert await asyncio.wait_for(manager.wait(handle), timeout=1) is terminal_state
        job_record = manager.get_record(handle)
        assert job_record is not None
        assert job_record.task is not None
        await asyncio.gather(job_record.task, return_exceptions=True)

    assert len(observed) == len(tuple(TerminalState))
    return tuple(observed)


# Feature: open-source-project-improvements, Property 25: Terminal events contain no sensitive data
# **Validates: Requirements 11.3**
@settings(max_examples=100, deadline=None)
@given(case=_sensitive_cases(), operations=_OPERATIONS)
def test_terminal_events_contain_no_sensitive_data(
    case: _SensitiveCase,
    operations: tuple[OperationType, OperationType, OperationType],
) -> None:
    events = asyncio.run(_capture_terminal_events(case, operations))
    record = case.record
    sensitive_values = {
        str(record.id),
        record.task,
        record.action,
        record.result,
        record.agent_role,
        record.created_at.isoformat(),
        case.secret,
        case.exception_text,
        *_context_text_values(record.context),
    }

    for event in events:
        serialized = _serialize_event(event)
        payload = json.loads(serialized)

        assert set(payload) == _ALLOW_LISTED_FIELDS
        assert set(ExperienceRecord.model_fields).isdisjoint(payload)
        assert payload == {
            "schema_version": EVENT_SCHEMA_VERSION,
            "job_id": str(event.job_id),
            "operation": event.operation.value,
            "terminal_state": event.terminal_state.value,
            "duration_ms": event.duration_ms,
        }
        assert all(value not in serialized for value in sensitive_values)
