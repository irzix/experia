from unittest.mock import AsyncMock, patch

import pytest

from experia.experience.llm_evaluator import LLMEvaluator
from experia.experience.models import ExperienceRecord
from experia.improvement.rules import RuleGenerator
from experia.memory.models import MemoryType
from experia.memory.store import SQLiteStore


class MockChoice:
    def __init__(self, content):
        self.message = type("obj", (object,), {"content": content})


class MockResponse:
    def __init__(self, content):
        self.choices = [MockChoice(content)]


@pytest.mark.asyncio
async def test_llm_evaluator():
    evaluator = LLMEvaluator(model="test-model")
    experience = ExperienceRecord(
        task="run command", action="ls /foo", result="No such file or directory"
    )

    # Mock the acompletion call
    with patch(
        "experia.experience.llm_evaluator.litellm.acompletion", new_callable=AsyncMock
    ) as mock_acompletion:
        # Provide a mock JSON string that matches the schema
        mock_json = '{"lesson": "Check if directory exists before listing", "root_cause": "Typo in directory name", "confidence": 0.9}'
        mock_acompletion.return_value = MockResponse(mock_json)

        lesson = await evaluator.evaluate(experience)

        assert lesson is not None
        assert lesson.content == "Check if directory exists before listing"
        assert lesson.root_cause == "Typo in directory name"
        assert lesson.confidence == 0.9


@pytest.mark.asyncio
async def test_rule_generator():
    store = SQLiteStore(":memory:")
    await store.initialize()

    generator = RuleGenerator(store=store, model="test-model")

    import uuid

    from experia.experience.models import Lesson

    lesson = Lesson(
        experience_id=uuid.uuid4(),
        content="Always verify port availability using lsof -i :80 before starting the web server.",
        root_cause="Port conflict",
        confidence=1.0,
    )

    with patch(
        "experia.improvement.rules.litellm.acompletion", new_callable=AsyncMock
    ) as mock_acompletion:
        # Mock generating a valid rule
        mock_acompletion.return_value = MockResponse(
            "Always check port bindings before starting services."
        )

        memory = await generator.consolidate_lesson(lesson)

        assert memory is not None
        assert memory.type == MemoryType.RULE
        assert memory.importance == 1.0
        assert memory.content == "Always check port bindings before starting services."
