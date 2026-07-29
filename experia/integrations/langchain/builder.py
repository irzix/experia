"""Identifier-isolated LangChain run and tool state collection."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional
from uuid import UUID

from experia.core.learner import Learner
from experia.core.logging import (
    EVENT_SCHEMA_VERSION,
    IntegrationEvent,
    IntegrationName,
    IntegrationOutcome,
    emit_integration_event,
    logger,
)
from experia.integrations._candidates import (
    ExperienceCandidate,
    finalize_experience_candidate,
)
from experia.integrations._dispatch import (
    CallbackMode,
    dispatch_callback_record,
    validate_callback_mode,
)


class _RegistryMutation(str, Enum):
    """Result of a tool-state mutation."""

    APPLIED = "applied"
    DUPLICATE = "duplicate"
    ORPHAN = "orphan"


class _ToolRunOutcome(str, Enum):
    """Terminal state retained for a matched tool invocation."""

    SUCCESS = "success"
    FAILURE = "failure"


@dataclass(slots=True)
class ToolRunState:
    """State belonging to one tool identifier and its exact parent chain."""

    tool_run_id: UUID
    parent_run_id: UUID
    action: str
    result: str | None = None
    outcome: _ToolRunOutcome | None = None


@dataclass(slots=True)
class ChainRunState:
    """State belonging exclusively to one top-level chain identifier."""

    chain_run_id: UUID
    task: str
    context: dict[str, Any]
    tools: dict[UUID, ToolRunState] = field(default_factory=dict)
    tool_order: list[UUID] = field(default_factory=list)
    completed_tool_order: list[UUID] = field(default_factory=list)
    finalized: bool = False


class LangChainRunRegistry:
    """Lock-protected registry for isolated LangChain chain and tool runs."""

    def __init__(self) -> None:
        self.active_runs: dict[UUID, ChainRunState] = {}
        self._tool_parents: dict[UUID, UUID] = {}
        self._lock = asyncio.Lock()

    async def start_chain(self, run_id: UUID, *, task: str) -> bool:
        """Register a chain without replacing state from a duplicate start event."""

        async with self._lock:
            if run_id in self.active_runs:
                return False
            self.active_runs[run_id] = ChainRunState(
                chain_run_id=run_id,
                task=task,
                context={"chain_id": str(run_id)},
            )
            return True

    async def start_tool(
        self,
        tool_run_id: UUID,
        parent_run_id: UUID | None,
        *,
        action: str,
    ) -> _RegistryMutation:
        """Register a tool only when both its identifier and parent are consistent."""

        async with self._lock:
            if parent_run_id is None:
                return _RegistryMutation.ORPHAN
            chain = self.active_runs.get(parent_run_id)
            if chain is None or chain.finalized:
                return _RegistryMutation.ORPHAN

            registered_parent = self._tool_parents.get(tool_run_id)
            if registered_parent is not None:
                if registered_parent == parent_run_id:
                    return _RegistryMutation.DUPLICATE
                return _RegistryMutation.ORPHAN

            chain.tools[tool_run_id] = ToolRunState(
                tool_run_id=tool_run_id,
                parent_run_id=parent_run_id,
                action=action,
            )
            chain.tool_order.append(tool_run_id)
            self._tool_parents[tool_run_id] = parent_run_id
            return _RegistryMutation.APPLIED

    async def finish_tool(
        self,
        tool_run_id: UUID,
        parent_run_id: UUID | None,
        *,
        result: str,
        outcome: _ToolRunOutcome,
    ) -> _RegistryMutation:
        """Finalize only the tool matching both supplied identifiers."""

        async with self._lock:
            if parent_run_id is None:
                return _RegistryMutation.ORPHAN
            if self._tool_parents.get(tool_run_id) != parent_run_id:
                return _RegistryMutation.ORPHAN

            chain = self.active_runs.get(parent_run_id)
            if chain is None or chain.finalized:
                return _RegistryMutation.ORPHAN
            tool = chain.tools.get(tool_run_id)
            if tool is None or tool.parent_run_id != parent_run_id:
                return _RegistryMutation.ORPHAN
            if tool.outcome is not None:
                return _RegistryMutation.DUPLICATE

            tool.result = result
            tool.outcome = outcome
            chain.completed_tool_order.append(tool_run_id)
            return _RegistryMutation.APPLIED

    async def take_candidate(
        self,
        run_id: UUID,
        *,
        fallback_result: str,
        assert_accepting: Callable[[str], None],
    ) -> ExperienceCandidate | None:
        """Validate and atomically pop one run, if it has not been finalized."""

        async with self._lock:
            chain = self.active_runs.get(run_id)
            if chain is None or chain.finalized:
                return None

            result = fallback_result
            if chain.completed_tool_order:
                completed_tool = chain.tools[chain.completed_tool_order[-1]]
                if completed_tool.result is not None:
                    result = completed_tool.result

            candidate = finalize_experience_candidate(
                {
                    "task": chain.task,
                    "action": "\n".join(
                        chain.tools[tool_run_id].action
                        for tool_run_id in chain.tool_order
                    ),
                    "result": result,
                    "context": chain.context,
                },
                integration=IntegrationName.LANGCHAIN,
                run_id=run_id,
            )
            if candidate is not None:
                # Preserve an unconsumed run when the Learner is already shutting down.
                assert_accepting("record")

            chain.finalized = True
            self.active_runs.pop(run_id, None)
            for tool_run_id in chain.tool_order:
                if self._tool_parents.get(tool_run_id) == run_id:
                    self._tool_parents.pop(tool_run_id, None)
            return candidate


class ExperienceBuilder:
    """Compile identifier-matched LangChain events into cohesive experiences."""

    def __init__(
        self,
        agent: Learner,
        *,
        callback_mode: CallbackMode = "background",
    ):
        validated_mode = validate_callback_mode(
            callback_mode,
            feature="ExperienceBuilder",
        )
        self.agent = agent
        self.callback_mode = validated_mode
        self.registry = LangChainRunRegistry()
        # Retain the existing inspection attribute while the registry owns mutation.
        self.active_runs = self.registry.active_runs
        self._lock = self.registry._lock

    async def on_chain_start(
        self,
        run_id: UUID,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Capture the overall task or user prompt."""

        task = str(inputs)
        if isinstance(inputs, dict):
            for key in ("input", "query", "question", "messages"):
                if key in inputs:
                    task = str(inputs[key])
                    break

        if await self.registry.start_chain(run_id, task=task):
            logger.debug("ExperienceBuilder: Chain %s started.", run_id)

    async def on_tool_start(
        self,
        run_id: UUID,
        parent_run_id: Optional[UUID],
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Capture an action under its exact tool and parent-chain identifiers."""

        tool_name = serialized.get("name", "unknown_tool")
        action = f"Tool: {tool_name} | Input: {inputs}"
        mutation = await self.registry.start_tool(
            run_id,
            parent_run_id,
            action=action,
        )
        if mutation is _RegistryMutation.ORPHAN:
            self._emit_outcome(IntegrationOutcome.ORPHAN, run_id)
        elif mutation is _RegistryMutation.APPLIED:
            logger.debug(
                "ExperienceBuilder: Tool %s started for chain %s.",
                run_id,
                parent_run_id,
            )

    async def on_tool_end(
        self, run_id: UUID, parent_run_id: Optional[UUID], output: str, **kwargs: Any
    ) -> None:
        """Attach a successful result only to the matching tool invocation."""

        mutation = await self.registry.finish_tool(
            run_id,
            parent_run_id,
            result=str(output),
            outcome=_ToolRunOutcome.SUCCESS,
        )
        if mutation is _RegistryMutation.ORPHAN:
            self._emit_outcome(IntegrationOutcome.ORPHAN, run_id)
        elif mutation is _RegistryMutation.APPLIED:
            logger.debug(
                "ExperienceBuilder: Tool %s ended for chain %s.",
                run_id,
                parent_run_id,
            )

    async def on_tool_error(
        self,
        run_id: UUID,
        parent_run_id: Optional[UUID],
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        """Attach a failure result only to the matching tool invocation."""

        mutation = await self.registry.finish_tool(
            run_id,
            parent_run_id,
            result=f"ERROR: {error}",
            outcome=_ToolRunOutcome.FAILURE,
        )
        if mutation is _RegistryMutation.ORPHAN:
            self._emit_outcome(IntegrationOutcome.ORPHAN, run_id)
        elif mutation is _RegistryMutation.APPLIED:
            logger.debug(
                "ExperienceBuilder: Tool %s failed for chain %s.",
                run_id,
                parent_run_id,
            )

    async def on_chain_end(
        self, run_id: UUID, outputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Finalize one successful chain and dispatch its candidate."""

        await self._finalize_and_record(run_id, fallback_result=str(outputs))

    async def on_chain_error(
        self, run_id: UUID, error: BaseException, **kwargs: Any
    ) -> None:
        """Finalize one failed chain and dispatch its candidate."""

        await self._finalize_and_record(
            run_id,
            fallback_result=f"CHAIN ERROR: {error}",
        )

    async def _finalize_and_record(
        self,
        run_id: UUID,
        *,
        fallback_result: str,
    ) -> None:
        candidate = await self._take_record_candidate(
            run_id,
            fallback_result=fallback_result,
        )
        if candidate is None:
            return

        # Registry mutation and locking are complete before persistence begins.
        await dispatch_callback_record(
            self.agent,
            self.callback_mode,
            task=candidate["task"],
            action=candidate["action"],
            result=candidate["result"],
            context=candidate["context"],
            on_success=lambda: self._emit_outcome(IntegrationOutcome.RECORDED, run_id),
            on_failure=lambda: self._emit_outcome(IntegrationOutcome.FAILED, run_id),
        )
        logger.info(
            "ExperienceBuilder: Finalized experience for chain %s.",
            run_id,
        )

    async def _take_record_candidate(
        self,
        run_id: UUID,
        *,
        fallback_result: str,
    ) -> Optional[Dict[str, Any]]:
        """Atomically consume one candidate after checking Learner lifecycle state."""

        candidate = await self.registry.take_candidate(
            run_id,
            fallback_result=fallback_result,
            assert_accepting=self.agent._assert_accepting,
        )
        if candidate is None:
            return None
        return {
            "task": candidate.task,
            "action": candidate.action,
            "result": candidate.result,
            "context": candidate.context,
        }

    @staticmethod
    def _emit_outcome(outcome: IntegrationOutcome, run_id: UUID) -> None:
        emit_integration_event(
            IntegrationEvent(
                schema_version=EVENT_SCHEMA_VERSION,
                integration=IntegrationName.LANGCHAIN,
                outcome=outcome,
                run_id=run_id,
            )
        )


__all__ = [
    "ChainRunState",
    "ExperienceBuilder",
    "LangChainRunRegistry",
    "ToolRunState",
]
