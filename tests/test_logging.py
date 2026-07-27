import logging
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from uuid import uuid4

import pytest

from experia.core.logging import (
    EVENT_SCHEMA_VERSION,
    IntegrationEvent,
    IntegrationName,
    IntegrationOutcome,
    LifecycleEvent,
    OperationType,
    RetrievalDiagnostic,
    RetrievalDiagnosticCode,
    TerminalState,
    emit_lifecycle_event,
)


def test_import_preserves_root_logger_and_installs_only_package_null_handler():
    repository_root = Path(__file__).resolve().parents[1]
    script = """
import importlib
import logging

root = logging.getLogger()
first = logging.StreamHandler()
second = logging.NullHandler()
root.handlers[:] = [first, second]
root.setLevel(logging.ERROR)
root.propagate = False
before = ([id(handler) for handler in root.handlers], root.level, root.propagate)

module = importlib.import_module("experia.core.logging")
module = importlib.reload(module)
after = ([id(handler) for handler in root.handlers], root.level, root.propagate)
package_logger = logging.getLogger("experia")

assert after == before
assert package_logger.level == logging.NOTSET
assert package_logger.propagate is True
assert len(package_logger.handlers) == 1
assert type(package_logger.handlers[0]) is logging.NullHandler
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            (str(repository_root), environment.get("PYTHONPATH", "")),
        )
    )

    subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("terminal_state", list(TerminalState))
def test_lifecycle_event_uses_one_schema_for_every_terminal_state(terminal_state):
    event = LifecycleEvent(
        schema_version=EVENT_SCHEMA_VERSION,
        job_id=uuid4(),
        operation=OperationType.EVALUATION,
        terminal_state=terminal_state,
        duration_ms=0,
    )

    assert event.terminal_state is terminal_state
    assert [field.name for field in fields(event)] == [
        "schema_version",
        "job_id",
        "operation",
        "terminal_state",
        "duration_ms",
    ]


@pytest.mark.parametrize("terminal_state", list(TerminalState))
def test_lifecycle_event_emission_keeps_the_structured_allow_list(
    caplog, terminal_state
):
    event = LifecycleEvent(
        schema_version=EVENT_SCHEMA_VERSION,
        job_id=uuid4(),
        operation=OperationType.EVALUATION,
        terminal_state=terminal_state,
        duration_ms=1,
    )
    caplog.set_level(logging.INFO, logger="experia")

    emit_lifecycle_event(event)

    records = [
        record
        for record in caplog.records
        if hasattr(record, "experia_lifecycle_event")
    ]
    assert len(records) == 1
    assert records[0].experia_lifecycle_event is event
    assert records[0].getMessage() == "Background job reached a terminal state."
    assert records[0].exc_info is None


def test_event_models_are_frozen_and_reject_non_allowlisted_content():
    event = LifecycleEvent(
        schema_version=EVENT_SCHEMA_VERSION,
        job_id=uuid4(),
        operation="evaluation",
        terminal_state="failure",
        duration_ms=1,
    )

    assert event.operation is OperationType.EVALUATION
    assert event.terminal_state is TerminalState.FAILURE
    with pytest.raises(FrozenInstanceError):
        event.duration_ms = 2
    with pytest.raises(TypeError):
        LifecycleEvent(
            schema_version=EVENT_SCHEMA_VERSION,
            job_id=uuid4(),
            operation=OperationType.EVALUATION,
            terminal_state=TerminalState.FAILURE,
            duration_ms=1,
            exception_text="raw dependency failure",
        )
    with pytest.raises(ValueError, match="allow-listed"):
        LifecycleEvent(
            schema_version=EVENT_SCHEMA_VERSION,
            job_id=uuid4(),
            operation="secret-value",
            terminal_state=TerminalState.FAILURE,
            duration_ms=1,
        )
    with pytest.raises(ValueError, match="non-negative"):
        LifecycleEvent(
            schema_version=EVENT_SCHEMA_VERSION,
            job_id=uuid4(),
            operation=OperationType.EVALUATION,
            terminal_state=TerminalState.SUCCESS,
            duration_ms=-1,
        )


def test_retrieval_diagnostics_allow_only_safe_identifier_and_dimensions():
    memory_id = uuid4()
    mismatch = RetrievalDiagnostic(
        code="dimension_mismatch",
        memory_id=memory_id,
        stored_dimension=3,
        query_dimension=4,
    )
    fallback = RetrievalDiagnostic(
        code=RetrievalDiagnosticCode.INDEX_FALLBACK,
        memory_id=None,
        stored_dimension=None,
        query_dimension=None,
    )

    assert mismatch.code is RetrievalDiagnosticCode.DIMENSION_MISMATCH
    assert mismatch.memory_id == memory_id
    assert fallback.code is RetrievalDiagnosticCode.INDEX_FALLBACK
    assert [field.name for field in fields(mismatch)] == [
        "code",
        "memory_id",
        "stored_dimension",
        "query_dimension",
    ]


@pytest.mark.parametrize("outcome", list(IntegrationOutcome))
def test_integration_event_uses_the_same_safe_schema_for_every_outcome(outcome):
    event = IntegrationEvent(
        schema_version=EVENT_SCHEMA_VERSION,
        integration=IntegrationName.LANGCHAIN,
        outcome=outcome,
        run_id=uuid4(),
    )

    assert event.outcome is outcome
    assert [field.name for field in fields(event)] == [
        "schema_version",
        "integration",
        "outcome",
        "run_id",
    ]


def test_diagnostics_reject_unstructured_or_record_specific_values():
    with pytest.raises(ValueError, match="requires"):
        RetrievalDiagnostic(
            code=RetrievalDiagnosticCode.DIMENSION_MISMATCH,
            memory_id=None,
            stored_dimension=3,
            query_dimension=4,
        )
    with pytest.raises(ValueError, match="record-specific"):
        RetrievalDiagnostic(
            code=RetrievalDiagnosticCode.INDEX_FALLBACK,
            memory_id=uuid4(),
            stored_dimension=None,
            query_dimension=None,
        )
    with pytest.raises(ValueError, match="allow-listed"):
        IntegrationEvent(
            schema_version=EVENT_SCHEMA_VERSION,
            integration="langchain",
            outcome="api-key-value",
        )
