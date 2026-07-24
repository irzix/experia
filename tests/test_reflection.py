import os
import uuid
from unittest.mock import AsyncMock, patch

import pytest

from experia.experience.models import ExperienceRecord
from experia.memory.models import MemoryType
from experia.memory.store import SQLiteStore
from experia.reflection.consolidation import ReflectionEngine


class MockChoice:
    def __init__(self, content):
        self.message = type("obj", (object,), {"content": content})


class MockResponse:
    def __init__(self, content):
        self.choices = [MockChoice(content)]


@pytest.mark.asyncio
async def test_reflection_engine():
    db_path = "test_reflection_engine.db"
    store = SQLiteStore(db_path)
    await store.initialize()

    try:
        engine = ReflectionEngine(store=store, model="test-model")

        # Add some experiences
        for _ in range(3):
            await store.save_experience(
                ExperienceRecord(
                    id=uuid.uuid4(),
                    task="deploy web app",
                    action="run docker compose",
                    result="success",
                )
            )

        with patch(
            "experia.reflection.consolidation.litellm.acompletion",
            new_callable=AsyncMock,
        ) as mock_acompletion:
            mock_acompletion.return_value = MockResponse(
                "Always use docker compose for web app deployments."
            )

            memory = await engine.reflect(batch_size=10)

            assert memory is not None
            assert memory.type == MemoryType.STRATEGY
            assert memory.importance == 1.0
            assert (
                memory.content == "Always use docker compose for web app deployments."
            )
    finally:
        await store.close()
        for suffix in ("", "-wal", "-shm"):
            if os.path.exists(db_path + suffix):
                os.remove(db_path + suffix)
