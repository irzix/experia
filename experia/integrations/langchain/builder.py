import asyncio
from typing import Any, Dict, Optional
from uuid import UUID

from experia.core.learner import Learner
from experia.core.logging import logger


class ExperienceBuilder:
    """
    State machine that collects raw execution events (Chain Start, Tool Start, Tool End)
    and merges them into cohesive experiences to avoid spamming the cognitive engine.
    """

    def __init__(self, agent: Learner):
        self.agent = agent
        # Maps run_id to its current experience context
        self.active_runs: Dict[UUID, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def on_chain_start(
        self,
        run_id: UUID,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Capture the overall task or user prompt."""
        async with self._lock:
            # We try to extract the main input as the 'task'
            task_str = str(inputs)
            if isinstance(inputs, dict):
                # Common LangChain input keys
                for key in ["input", "query", "question", "messages"]:
                    if key in inputs:
                        task_str = str(inputs[key])
                        break

            self.active_runs[run_id] = {
                "task": task_str,
                "actions": [],
                "context": {"chain_id": str(run_id)},
            }
            logger.debug(
                f"ExperienceBuilder: Chain {run_id} started. Task: {task_str[:50]}..."
            )

    async def on_tool_start(
        self,
        run_id: UUID,
        parent_run_id: Optional[UUID],
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Capture the action taken by the agent."""
        async with self._lock:
            if parent_run_id in self.active_runs:
                tool_name = serialized.get("name", "unknown_tool")
                action_str = f"Tool: {tool_name} | Input: {inputs}"

                # We store the latest action. A single chain might call multiple tools.
                # In a more complex builder, we might record a sequence of actions.
                self.active_runs[parent_run_id]["actions"].append(action_str)
                logger.debug(
                    f"ExperienceBuilder: Tool {tool_name} started for chain {parent_run_id}"
                )

    async def on_tool_end(
        self, run_id: UUID, parent_run_id: Optional[UUID], output: str, **kwargs: Any
    ) -> None:
        """Capture the result of the action."""
        async with self._lock:
            if (
                parent_run_id in self.active_runs
                and self.active_runs[parent_run_id]["actions"]
            ):
                # Attach the result to the context so we know what happened
                self.active_runs[parent_run_id]["last_result"] = str(output)
                logger.debug(f"ExperienceBuilder: Tool ended for chain {parent_run_id}")

    async def on_tool_error(
        self,
        run_id: UUID,
        parent_run_id: Optional[UUID],
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        """Capture tool failures."""
        async with self._lock:
            if (
                parent_run_id in self.active_runs
                and self.active_runs[parent_run_id]["actions"]
            ):
                self.active_runs[parent_run_id]["last_result"] = f"ERROR: {str(error)}"
                logger.debug(f"ExperienceBuilder: Tool error for chain {parent_run_id}")

    async def on_chain_end(
        self, run_id: UUID, outputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Finalize the experience and send it to the Learner."""
        async with self._lock:
            if run_id in self.active_runs:
                run_data = self.active_runs.pop(run_id)

                # Only record if the agent actually took an action (used a tool)
                if not run_data["actions"]:
                    return

                # Summarize the actions taken
                action_summary = "\n".join(run_data["actions"])
                result_str = run_data.get("last_result", str(outputs))

                # Fire and forget the record to the learner
                asyncio.create_task(
                    self.agent.record(
                        task=run_data["task"],
                        action=action_summary,
                        result=result_str,
                        context=run_data["context"],
                    )
                )
                logger.info(
                    f"ExperienceBuilder: Compiled and recorded experience for chain {run_id}"
                )

    async def on_chain_error(
        self, run_id: UUID, error: BaseException, **kwargs: Any
    ) -> None:
        """Finalize the experience on chain failure."""
        async with self._lock:
            if run_id in self.active_runs:
                run_data = self.active_runs.pop(run_id)

                if not run_data["actions"]:
                    return

                action_summary = "\n".join(run_data["actions"])
                result_str = run_data.get("last_result", f"CHAIN ERROR: {str(error)}")

                asyncio.create_task(
                    self.agent.record(
                        task=run_data["task"],
                        action=action_summary,
                        result=result_str,
                        context=run_data["context"],
                    )
                )
                logger.info(
                    f"ExperienceBuilder: Compiled and recorded failed experience for chain {run_id}"
                )
