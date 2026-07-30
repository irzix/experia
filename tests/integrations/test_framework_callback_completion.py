import ast
import asyncio
import logging
from collections import Counter
from pathlib import Path
from uuid import uuid4

import pytest

from experia.core.exceptions import EvaluationFailure, StorageError
from experia.core.learner import Learner
from experia.core.logging import (
    IntegrationEvent,
    IntegrationName,
    IntegrationOutcome,
    OperationType,
)
from experia.core.work import AsyncWorkManager
from experia.experience.evaluator import SimpleHeuristicEvaluator
from experia.integrations.langchain.callbacks import ExperiaCallbackHandler
from experia.memory.store import SQLiteStore

_CALLBACK_TIMEOUT_SECONDS = 2.0
_SUCCESS_TASK = "Complete the deployment"
_FAILURE_TASK = "Fail persistence for this deployment"


class _FlushSignallingManager(AsyncWorkManager):
    def __init__(self) -> None:
        super().__init__()
        self.flush_captured = asyncio.Event()

    def _capture_flush_generation(self, generation: int, cutoff: int) -> None:
        super()._capture_flush_generation(generation, cutoff)
        self.flush_captured.set()


class _ControlledFailureStore(SQLiteStore):
    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.all_saves_started = asyncio.Event()
        self.release_saves = asyncio.Event()
        self._started_tasks: set[str] = set()

    async def save_experience(self, experience) -> None:
        self._started_tasks.add(experience.task)
        if {_SUCCESS_TASK, _FAILURE_TASK} <= self._started_tasks:
            self.all_saves_started.set()

        await self.release_saves.wait()
        if experience.task == _FAILURE_TASK:
            raise StorageError(
                operation="save_experience",
                table="experiences",
                record_ids=experience.id,
            )
        await super().save_experience(experience)


async def _prepare_run(
    handler: ExperiaCallbackHandler,
    *,
    run_id,
    tool_run_id,
    task: str,
    tool_fails: bool,
) -> None:
    await handler.on_chain_start(
        serialized={"name": "AgentExecutor"},
        inputs={"input": task},
        run_id=run_id,
    )
    await handler.on_tool_start(
        serialized={"name": "Deploy"},
        input_str="deploy",
        run_id=tool_run_id,
        parent_run_id=run_id,
    )
    if tool_fails:
        await handler.on_tool_error(
            error=RuntimeError("deployment tool failed"),
            run_id=tool_run_id,
            parent_run_id=run_id,
        )
    else:
        await handler.on_tool_end(
            output="deployment completed",
            run_id=tool_run_id,
            parent_run_id=run_id,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("callback_mode", ["durable", "background"])
async def test_concurrent_completion_failure_and_orphan_are_isolated(
    tmp_path,
    caplog,
    callback_mode,
):
    store = _ControlledFailureStore(str(tmp_path / f"callbacks-{callback_mode}.db"))
    await store.initialize()
    learner = Learner(store=store, evaluator=SimpleHeuristicEvaluator())
    manager = _FlushSignallingManager()
    learner._work_manager = manager
    handler = ExperiaCallbackHandler(agent=learner, callback_mode=callback_mode)
    success_run_id, failure_run_id = uuid4(), uuid4()
    success_tool_id, failure_tool_id, orphan_tool_id = uuid4(), uuid4(), uuid4()

    await _prepare_run(
        handler,
        run_id=success_run_id,
        tool_run_id=success_tool_id,
        task=_SUCCESS_TASK,
        tool_fails=False,
    )
    await _prepare_run(
        handler,
        run_id=failure_run_id,
        tool_run_id=failure_tool_id,
        task=_FAILURE_TASK,
        tool_fails=True,
    )

    callback_results = asyncio.gather(
        handler.on_chain_end(
            outputs={"output": "deployment completed"},
            run_id=success_run_id,
        ),
        handler.on_chain_error(
            error=RuntimeError("deployment chain failed"),
            run_id=failure_run_id,
        ),
        handler.on_tool_end(
            output="orphaned output",
            run_id=orphan_tool_id,
            parent_run_id=success_run_id,
        ),
        return_exceptions=True,
    )

    try:
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="experia"):
            if callback_mode == "durable":
                await asyncio.wait_for(
                    store.all_saves_started.wait(),
                    timeout=_CALLBACK_TIMEOUT_SECONDS,
                )
                store.release_saves.set()
                results = await asyncio.wait_for(
                    callback_results,
                    timeout=_CALLBACK_TIMEOUT_SECONDS,
                )

                assert results[0] is None
                assert isinstance(results[1], StorageError)
                assert results[2] is None

                # Durable callback completion proves persistence before return.
                persisted = await store.get_recent_experiences(limit=10)
                assert [experience.task for experience in persisted] == [_SUCCESS_TASK]
                await asyncio.wait_for(
                    learner.flush(),
                    timeout=_CALLBACK_TIMEOUT_SECONDS,
                )
            else:
                results = await asyncio.wait_for(
                    callback_results,
                    timeout=_CALLBACK_TIMEOUT_SECONDS,
                )
                assert results == [None, None, None]
                await asyncio.wait_for(
                    store.all_saves_started.wait(),
                    timeout=_CALLBACK_TIMEOUT_SECONDS,
                )

                flush_task = asyncio.create_task(learner.flush())
                await asyncio.wait_for(
                    manager.flush_captured.wait(),
                    timeout=_CALLBACK_TIMEOUT_SECONDS,
                )
                store.release_saves.set()
                with pytest.raises(EvaluationFailure) as raised:
                    await asyncio.wait_for(
                        flush_task,
                        timeout=_CALLBACK_TIMEOUT_SECONDS,
                    )

                assert raised.value.operation == OperationType.RECORD.value
                assert len(raised.value.failures) == 1
                persisted = await store.get_recent_experiences(limit=10)
                assert [experience.task for experience in persisted] == [_SUCCESS_TASK]

        events = [
            record.experia_integration_event
            for record in caplog.records
            if isinstance(
                getattr(record, "experia_integration_event", None),
                IntegrationEvent,
            )
        ]
        assert Counter(events) == Counter(
            [
                IntegrationEvent(
                    schema_version=1,
                    integration=IntegrationName.LANGCHAIN,
                    outcome=IntegrationOutcome.RECORDED,
                    run_id=success_run_id,
                ),
                IntegrationEvent(
                    schema_version=1,
                    integration=IntegrationName.LANGCHAIN,
                    outcome=IntegrationOutcome.FAILED,
                    run_id=failure_run_id,
                ),
                IntegrationEvent(
                    schema_version=1,
                    integration=IntegrationName.LANGCHAIN,
                    outcome=IntegrationOutcome.ORPHAN,
                    run_id=orphan_tool_id,
                ),
            ]
        )
        assert handler.builder.active_runs == {}
    finally:
        store.release_saves.set()
        await store.close()


def test_framework_integration_tests_do_not_use_sleep_synchronization():
    violations = []
    integration_test_dir = Path(__file__).parent

    for test_path in sorted(integration_test_dir.glob("test_*.py")):
        tree = ast.parse(test_path.read_text(encoding="utf-8"), filename=str(test_path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            is_sleep_call = (
                isinstance(node.func, ast.Name) and node.func.id == "sleep"
            ) or (isinstance(node.func, ast.Attribute) and node.func.attr == "sleep")
            if is_sleep_call:
                violations.append(f"{test_path.name}:{node.lineno}")

    assert violations == [], (
        "Framework integration tests must synchronize with awaited callbacks, "
        f"events, or timeout-bounded flush(); sleep calls found at {violations}"
    )
