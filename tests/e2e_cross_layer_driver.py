"""Self-contained cross-layer end-to-end driver for the installed Experia wheel.

This module is intentionally free of any dependency on the test package so it
can be copied outside the source tree and executed against an installed wheel.
It drives the full pipeline with deterministic fakes and event/timeout
synchronization (never elapsed-time sleeps):

* durable and background framework recording,
* sanitization at the external embedding boundary,
* atomic SQLite persistence and evaluation/flush,
* bounded, role-isolated retrieval and untrusted prompt formatting,
* LangChain and LangGraph experience extraction,
* typed failures, persisted-data retention, shutdown, and repeated close.

Running ``python e2e_cross_layer_driver.py`` prints ``E2E_CROSS_LAYER_OK`` on
success and exits non-zero (with a traceback) on any failed assertion.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

import experia
from experia import (
    ConfigurationError,
    EvaluationFailure,
    Learner,
    LifecycleError,
    Memory,
    MemoryType,
    SimpleHeuristicEvaluator,
    SQLiteStore,
    StorageError,
)
from experia.context.builder import ContextBuilder
from experia.core.logging import (
    IntegrationEvent,
    IntegrationName,
    IntegrationOutcome,
    LifecycleEvent,
)
from experia.core.work import ShutdownReport, TerminalState
from experia.integrations._candidates import finalize_experience_candidate
from experia.integrations.langchain.builder import ExperienceBuilder
from experia.integrations.langgraph.utils import default_messages_state_extractor
from experia.security.protection import DataProtectionLayer

SUCCESS_SENTINEL = "E2E_CROSS_LAYER_OK"
SECRET = "SECRET_TOKEN_XYZ"
REDACTED = "[REDACTED]"
TIMEOUT = 5.0

EXPECTED_PUBLIC_API = {
    "ConfigurationError",
    "Embedder",
    "EvaluationError",
    "EvaluationFailure",
    "ExperiaError",
    "FailureDetail",
    "Learner",
    "LifecycleError",
    "LiteLLMEmbedder",
    "Memory",
    "MemoryType",
    "SanitizationError",
    "SimpleHeuristicEvaluator",
    "SQLiteStore",
    "StorageError",
    "UnavailableFeatureError",
}

LIFECYCLE_EVENT_FIELDS = {
    "schema_version",
    "job_id",
    "operation",
    "terminal_state",
    "duration_ms",
}


class RedactingSanitizer:
    """Deterministic sanitizer that masks a known secret in copied leaves."""

    def sanitize(self, value: Any, *, path: tuple[Any, ...]) -> Any:
        if isinstance(value, str) and SECRET in value:
            return value.replace(SECRET, REDACTED)
        return value


class RecordingEmbedder:
    """Deterministic embedder that records every text it is asked to embed."""

    def __init__(self) -> None:
        self.seen_texts: list[str] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed_one(text) for text in texts]

    async def embed_one(self, text: str) -> list[float]:
        self.seen_texts.append(text)
        # A fixed four-dimension deterministic vector keeps ranking stable and
        # keeps every stored embedding dimension-compatible with queries.
        checksum = sum(bytes(text, "utf-8")) % 97
        return [0.11, 0.23, 0.37, checksum / 97.0]


class BlockingEvaluator:
    """Evaluator that blocks until released so a job can be caught in flight."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def evaluate(self, experience: Any) -> None:
        self.started.set()
        await self.release.wait()
        return None


class FailingEvaluator:
    """Evaluator that always raises to exercise downstream failure retention."""

    async def evaluate(self, experience: Any) -> None:
        raise RuntimeError("private evaluator failure with sensitive detail")


class EventCollector(logging.Handler):
    """Capture allow-listed lifecycle and integration events from the logger."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.integration_events: list[IntegrationEvent] = []
        self.lifecycle_events: list[LifecycleEvent] = []

    def emit(self, record: logging.LogRecord) -> None:
        integration = getattr(record, "experia_integration_event", None)
        if isinstance(integration, IntegrationEvent):
            self.integration_events.append(integration)
        lifecycle = getattr(record, "experia_lifecycle_event", None)
        if isinstance(lifecycle, LifecycleEvent):
            self.lifecycle_events.append(lifecycle)


class HumanMessage:
    def __init__(self, content: str) -> None:
        self.type = "human"
        self.content = content


class AIMessage:
    def __init__(self, tool_calls: list[dict[str, Any]]) -> None:
        self.type = "ai"
        self.tool_calls = tool_calls


class ToolMessage:
    def __init__(self, content: str, tool_call_id: str | None = None) -> None:
        self.type = "tool"
        self.content = content
        if tool_call_id is not None:
            self.tool_call_id = tool_call_id


def assert_safe_lifecycle_event(event: LifecycleEvent) -> None:
    """A terminal event must expose only allow-listed, secret-free fields."""

    field_names = {field.name for field in dataclasses.fields(event)}
    assert field_names == LIFECYCLE_EVENT_FIELDS, field_names
    assert isinstance(event.duration_ms, int) and event.duration_ms >= 0
    rendered = repr(event)
    assert SECRET not in rendered, "lifecycle event leaked a secret"
    assert "deployment" not in rendered.lower(), "lifecycle event leaked record text"


def assert_safe_integration_event(event: IntegrationEvent) -> None:
    """An integration event must never carry extracted record text or secrets."""

    assert isinstance(event.outcome, IntegrationOutcome)
    rendered = repr(event)
    assert SECRET not in rendered, "integration event leaked a secret"


async def drive_langchain_run(
    builder: ExperienceBuilder,
    *,
    chain_run_id: Any,
    tool_run_id: Any,
    task: str,
    tool_output: str,
    chain_output: str,
) -> None:
    """Replay one isolated LangChain chain/tool event stream through the builder."""

    await builder.on_chain_start(
        chain_run_id,
        {"name": "AgentExecutor"},
        {"input": task},
    )
    await builder.on_tool_start(
        tool_run_id,
        chain_run_id,
        {"name": "Deploy"},
        {"input": "deploy"},
    )
    await builder.on_tool_end(tool_run_id, chain_run_id, tool_output)
    await builder.on_chain_end(chain_run_id, {"output": chain_output})


async def phase_primary_pipeline(workdir: Path, collector: EventCollector) -> None:
    """Durable + background recording, sanitization, retrieval, and safe events."""

    observed: list[LifecycleEvent] = []
    store = SQLiteStore(str(workdir / "primary.db"))
    await store.initialize()
    embedder = RecordingEmbedder()
    learner = Learner(
        store=store,
        evaluator=SimpleHeuristicEvaluator(),
        embedder=embedder,
        agent_role="deployer",
        data_protection=DataProtectionLayer(RedactingSanitizer()),
        lifecycle_observer=observed.append,
    )

    try:
        # --- Durable framework recording awaits persistence before returning. ---
        durable_builder = ExperienceBuilder(agent=learner, callback_mode="durable")
        durable_task = f"Deploy billing service using token {SECRET}"
        await asyncio.wait_for(
            drive_langchain_run(
                durable_builder,
                chain_run_id=uuid4(),
                tool_run_id=uuid4(),
                task=durable_task,
                tool_output="deployment success",
                chain_output="deployment success",
            ),
            timeout=TIMEOUT,
        )
        # Persistence already completed; flush completes background evaluation.
        await asyncio.wait_for(learner.flush(), timeout=TIMEOUT)

        persisted = await store.get_recent_experiences(limit=10)
        assert len(persisted) == 1, persisted
        # Storage is documented pass-through: the raw task is retained verbatim.
        assert persisted[0].task == durable_task
        lessons = await store.search_memories(
            memory_type=MemoryType.LESSON,
            agent_role="deployer",
            limit=10,
        )
        assert len(lessons) == 1, lessons

        # Sanitization: the external embedder must never receive the secret.
        assert embedder.seen_texts, "embedder was never called"
        assert all(SECRET not in text for text in embedder.seen_texts)
        assert any(REDACTED in text for text in embedder.seen_texts)

        # --- Background framework recording is managed and awaited via flush. ---
        background_builder = ExperienceBuilder(
            agent=learner, callback_mode="background"
        )
        await asyncio.wait_for(
            drive_langchain_run(
                background_builder,
                chain_run_id=uuid4(),
                tool_run_id=uuid4(),
                task="Restart cache tier",
                tool_output="restart success",
                chain_output="restart success",
            ),
            timeout=TIMEOUT,
        )
        await asyncio.wait_for(learner.flush(), timeout=TIMEOUT)
        assert len(await store.get_recent_experiences(limit=10)) == 2

        # --- LangGraph extraction (valid) records one cohesive experience. ---
        valid_state = {
            "messages": [
                HumanMessage("Investigate the outage"),
                AIMessage([{"id": "call-logs", "name": "CheckLogs", "args": {}}]),
                ToolMessage("error: disk full", "call-logs"),
            ]
        }
        extracted = default_messages_state_extractor(valid_state)
        assert extracted is not None and extracted["task"] == "Investigate the outage"
        candidate = finalize_experience_candidate(
            extracted,
            integration=IntegrationName.LANGGRAPH,
            default_context={"source": "langgraph"},
        )
        assert candidate is not None
        await asyncio.wait_for(
            learner.record(
                candidate.task,
                candidate.action,
                candidate.result,
                candidate.context,
            ),
            timeout=TIMEOUT,
        )
        await asyncio.wait_for(learner.flush(), timeout=TIMEOUT)
        assert len(await store.get_recent_experiences(limit=10)) == 3

        # --- A RECORDED outcome was emitted by the earlier LangChain runs. ---
        # Capture and verify it before clearing the buffer to isolate the
        # no-experience check below; RECORDED events precede the clear.
        recorded_events = [
            event
            for event in collector.integration_events
            if event.outcome is IntegrationOutcome.RECORDED
        ]
        assert recorded_events, "no RECORDED integration outcome observed"
        for event in collector.integration_events:
            assert_safe_integration_event(event)

        # --- LangGraph extraction (invalid) yields one safe no-experience event. ---
        collector.integration_events.clear()
        invalid_state = {
            "messages": [
                HumanMessage("Check services"),
                AIMessage([{"name": "CheckApi"}, {"name": "CheckDatabase"}]),
                ToolMessage("only one result"),
            ]
        }
        assert default_messages_state_extractor(invalid_state) is None
        no_experience = finalize_experience_candidate(
            default_messages_state_extractor(invalid_state),
            integration=IntegrationName.LANGGRAPH,
            default_context={"source": "langgraph"},
        )
        assert no_experience is None
        assert len(await store.get_recent_experiences(limit=10)) == 3
        assert any(
            event.outcome is IntegrationOutcome.NO_EXPERIENCE
            for event in collector.integration_events
        )

        # --- Role isolation across a mixed dataset. ---
        await store.save_memory(
            Memory(content="DEPLOYER_FACT", type=MemoryType.FACT, agent_role="deployer")
        )
        await store.save_memory(
            Memory(content="ANALYST_FACT", type=MemoryType.FACT, agent_role="analyst")
        )
        await store.save_memory(
            Memory(
                content="GLOBAL_STRATEGY",
                type=MemoryType.STRATEGY,
                agent_role="global",
            )
        )
        await store.save_memory(
            Memory(
                content="ANALYST_STRATEGY",
                type=MemoryType.STRATEGY,
                agent_role="analyst",
            )
        )
        role_results = await store.search_memories(agent_role="deployer", limit=100)
        contents = {memory.content for memory in role_results}
        assert "DEPLOYER_FACT" in contents
        assert "GLOBAL_STRATEGY" in contents
        assert "ANALYST_FACT" not in contents
        assert "ANALYST_STRATEGY" not in contents

        # --- Untrusted prompt formatting wraps every memory in safety markers. ---
        prompt = await asyncio.wait_for(
            learner.retrieve_context(limit=5), timeout=TIMEOUT
        )
        assert ContextBuilder.SAFETY_INSTRUCTION in prompt
        assert ContextBuilder.START_MARKER in prompt
        assert ContextBuilder.END_MARKER in prompt

        # --- Every observed and logged terminal event is safe. ---
        assert observed, "no lifecycle events were observed"
        for event in observed:
            assert_safe_lifecycle_event(event)
        for event in collector.lifecycle_events:
            assert_safe_lifecycle_event(event)
        # RECORDED outcomes were captured and asserted before the buffer was
        # cleared above; the remaining events must also be secret-free.
        for event in collector.integration_events:
            assert_safe_integration_event(event)

        # --- Drain shutdown closes submissions; later work is a typed error. ---
        report = await asyncio.wait_for(learner.shutdown("drain"), timeout=TIMEOUT)
        assert isinstance(report, ShutdownReport)
        try:
            await learner.record("post shutdown", "action", "result")
        except LifecycleError as error:
            assert error.state == "closed"
            assert error.operation == "record"
        else:  # pragma: no cover - defensive
            raise AssertionError("record after shutdown must raise LifecycleError")
    finally:
        await store.close()

    # --- Repeated close is idempotent and persisted data survives it. ---
    await store.close()
    try:
        await store.get_recent_experiences()
    except StorageError as error:
        assert error.operation == "lifecycle"
    else:  # pragma: no cover - defensive
        raise AssertionError("operations on a closed store must raise StorageError")

    reopened = SQLiteStore(str(workdir / "primary.db"))
    await reopened.initialize()
    try:
        retained = await reopened.get_recent_experiences(limit=10)
        assert len(retained) == 3, retained
    finally:
        await reopened.close()


async def phase_downstream_failure_retention(workdir: Path) -> None:
    """A failing evaluator raises a typed failure yet retains the experience."""

    store = SQLiteStore(str(workdir / "failure.db"))
    await store.initialize()
    learner = Learner(
        store=store,
        evaluator=FailingEvaluator(),
        agent_role="deployer",
    )
    try:
        experience = await learner.record("task", "action", "result")
        try:
            await asyncio.wait_for(learner.flush(), timeout=TIMEOUT)
        except EvaluationFailure as failure:
            assert failure.experience_id == experience.id
            assert failure.operation == "evaluation"
            assert SECRET not in str(failure)
        else:  # pragma: no cover - defensive
            raise AssertionError("failing evaluation must raise EvaluationFailure")

        # Persisted-data retention: the experience survives downstream failure.
        assert await store.get_experience(experience.id) == experience
        await learner.shutdown("drain")
    finally:
        await store.close()


async def phase_cancel_shutdown_retention(workdir: Path) -> None:
    """Cancel shutdown terminates in-flight work while retaining persisted data."""

    store = SQLiteStore(str(workdir / "cancel.db"))
    await store.initialize()
    evaluator = BlockingEvaluator()
    learner = Learner(store=store, evaluator=evaluator, agent_role="deployer")
    try:
        experience = await learner.record("task", "action", "result")
        await asyncio.wait_for(evaluator.started.wait(), timeout=TIMEOUT)
        report = await asyncio.wait_for(learner.shutdown("cancel"), timeout=TIMEOUT)

        assert isinstance(report, ShutdownReport)
        assert TerminalState.CANCELLATION in report.terminal_states.values()
        assert await store.get_experience(experience.id) == experience
        try:
            await learner.record("later", "action", "result")
        except LifecycleError as error:
            assert error.operation == "record"
        else:  # pragma: no cover - defensive
            raise AssertionError("record after cancel shutdown must raise")
    finally:
        evaluator.release.set()
        await store.close()


def assert_documented_imports(installed_root: Path, source_root: Path) -> None:
    """The documented public API must be importable from the installed wheel."""

    origin = Path(experia.__file__).resolve()
    assert origin.is_relative_to(installed_root), origin
    assert not origin.is_relative_to(source_root), origin
    assert set(experia.__all__) == EXPECTED_PUBLIC_API, set(experia.__all__)
    for name in EXPECTED_PUBLIC_API:
        assert getattr(experia, name) is not None
    # The concrete public symbols imported at module top must resolve, too.
    assert Learner is experia.Learner
    assert SQLiteStore is experia.SQLiteStore
    assert ConfigurationError is experia.ConfigurationError


async def main(installed_root: Path, source_root: Path) -> None:
    assert_documented_imports(installed_root, source_root)

    collector = EventCollector()
    package_logger = logging.getLogger("experia")
    previous_level = package_logger.level
    package_logger.setLevel(logging.DEBUG)
    package_logger.addHandler(collector)
    try:
        with tempfile.TemporaryDirectory() as raw_dir:
            workdir = Path(raw_dir)
            await phase_primary_pipeline(workdir, collector)
            await phase_downstream_failure_retention(workdir)
            await phase_cancel_shutdown_retention(workdir)
    finally:
        package_logger.removeHandler(collector)
        package_logger.setLevel(previous_level)

    print(SUCCESS_SENTINEL)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    arguments = parser.parse_args()
    asyncio.run(
        main(
            arguments.installed_root.resolve(),
            arguments.source_root.resolve(),
        )
    )
