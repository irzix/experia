import asyncio
import os
import uuid

import pytest

from experia.core.learner import Learner
from experia.experience.evaluator import SimpleHeuristicEvaluator
from experia.integrations.langchain.callbacks import ExperiaCallbackHandler
from experia.memory.store import SQLiteStore


@pytest.mark.asyncio
async def test_experience_flow():
    """
    End-to-End test for the Experience State Machine (Builder -> Callback -> Learner -> Store).
    Simulates a LangChain chain execution with multiple tools.
    """
    db_path = "test_experience_flow.db"
    store = SQLiteStore(db_path)
    await store.initialize()

    try:
        evaluator = SimpleHeuristicEvaluator()
        agent = Learner(store=store, evaluator=evaluator)

        # We don't actually need the builder directly, ExperiaCallbackHandler instantiates it
        # but we can pass our agent to the callback handler.
        handler = ExperiaCallbackHandler(agent=agent)

        # Simulate a LangChain execution
        run_id = uuid.uuid4()
        tool1_run_id = uuid.uuid4()
        tool2_run_id = uuid.uuid4()

        # 1. Chain Starts (Task)
        await handler.on_chain_start(
            serialized={"name": "AgentExecutor"},
            inputs={"input": "Deploy the web app to production"},
            run_id=run_id,
            parent_run_id=None,
        )

        # 2. Tool 1 Starts & Ends (Action 1)
        await handler.on_tool_start(
            serialized={"name": "CheckEnv"},
            input_str="check env vars",
            inputs={"input": "check env vars"},
            run_id=tool1_run_id,
            parent_run_id=run_id,
        )
        await handler.on_tool_end(
            output="Env vars missing: DB_PASSWORD",
            run_id=tool1_run_id,
            parent_run_id=run_id,
        )

        # 3. Tool 2 Starts & Errors (Action 2)
        await handler.on_tool_start(
            serialized={"name": "DeployDocker"},
            input_str="docker-compose up -d",
            inputs={"input": "docker-compose up -d"},
            run_id=tool2_run_id,
            parent_run_id=run_id,
        )
        await handler.on_tool_error(
            error=Exception("Failed to bind port 80"),
            run_id=tool2_run_id,
            parent_run_id=run_id,
        )

        # 4. Chain Ends with Error
        await handler.on_chain_error(
            error=Exception("Deployment failed due to port conflict"),
            run_id=run_id,
            parent_run_id=None,
        )

        # Managed background persistence and evaluation share one flush boundary.
        await asyncio.wait_for(agent.flush(), timeout=1)

        # Verify the experience was properly compiled and stored
        recent = await store.get_recent_experiences(limit=5)

        assert len(recent) == 1
        exp = recent[0]

        assert exp.task == "Deploy the web app to production"
        assert "CheckEnv" in exp.action
        assert "DeployDocker" in exp.action
        assert "Failed to bind port 80" in exp.result

        # Ensure it was also evaluated (since Learner evaluates on record)
        # Using SimpleHeuristicEvaluator, it should have extracted a lesson if it failed
        # Wait, SimpleHeuristicEvaluator only creates lessons for failures, which this is!

        memories = await store.search_memories(query="DeployDocker", limit=5)
        # The heuristic evaluator looks for "failed" or "error" in the result and stores a lesson
        assert len(memories) > 0
        assert memories[0].type.name == "LESSON"

    finally:
        await store.close()
        for suffix in ("", "-wal", "-shm"):
            if os.path.exists(db_path + suffix):
                os.remove(db_path + suffix)
