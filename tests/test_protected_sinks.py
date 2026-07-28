import logging
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from experia.core.exceptions import SanitizationError
from experia.core.learner import Learner
from experia.experience.models import ExperienceRecord, Lesson
from experia.security import DataProtectionLayer


class MockChoice:
    def __init__(self, content):
        self.message = SimpleNamespace(content=content)


class MockResponse:
    def __init__(self, content):
        self.choices = [MockChoice(content)]


class SinkStore:
    def __init__(self, experiences=None):
        self.experiences = list(experiences or [])
        self.saved_memories = []

    async def get_recent_experiences(self, limit=50):
        return self.experiences[:limit]

    async def save_memory(self, memory):
        self.saved_memories.append(memory)


class FailingOrUnserializableSanitizer:
    def __init__(self, target, mode):
        self.target = target
        self.mode = mode

    def sanitize(self, value, *, path):
        if path != self.target:
            if isinstance(value, bytearray):
                return value.decode()
            return value
        if self.mode == "raise":
            if isinstance(value, bytearray):
                value.extend(b"-mutated-copy")
            raise RuntimeError("private sanitizer failure")
        return object()


class PrefixSanitizer:
    def sanitize(self, value, *, path):
        if isinstance(value, str):
            return f"protected:{value}"
        return value


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["raise", "unserializable"])
@pytest.mark.parametrize(
    "target, expected_operation",
    [
        (("context", "credential"), "external_request"),
        (("model",), "log_metadata"),
    ],
)
async def test_llm_evaluator_protection_failure_sends_no_request_event_or_mutation(
    monkeypatch, caplog, mode, target, expected_operation
):
    from experia.experience import llm_evaluator as module

    request = AsyncMock()
    monkeypatch.setattr(module, "litellm", SimpleNamespace(acompletion=request))
    credential = bytearray(b"caller-secret")
    experience = ExperienceRecord(
        task="task",
        action="action",
        result="result",
        context={"credential": credential},
    )
    before = experience.model_copy(deep=True)
    evaluator = module.LLMEvaluator(
        model="test-model",
        data_protection=DataProtectionLayer(
            FailingOrUnserializableSanitizer(target, mode)
        ),
    )

    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="experia"):
        with pytest.raises(SanitizationError) as caught:
            await evaluator.evaluate(experience)

    assert caught.value.operation == expected_operation
    request.assert_not_awaited()
    assert caplog.records == []
    assert experience == before
    assert credential == bytearray(b"caller-secret")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["raise", "unserializable"])
@pytest.mark.parametrize(
    "target, expected_operation",
    [
        (("texts", 0), "external_request"),
        (("model",), "log_metadata"),
    ],
)
async def test_external_embedder_protection_failure_sends_no_request_event_or_mutation(
    monkeypatch, caplog, mode, target, expected_operation
):
    from experia.memory import embeddings as module

    request = AsyncMock()
    monkeypatch.setattr(module, "litellm", SimpleNamespace(aembedding=request))
    caller_text = bytearray(b"caller-secret")
    texts = [caller_text]
    before = deepcopy(texts)
    embedder = module.LiteLLMEmbedder(
        model="test-model",
        data_protection=DataProtectionLayer(
            FailingOrUnserializableSanitizer(target, mode)
        ),
    )

    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="experia"):
        with pytest.raises(SanitizationError) as caught:
            await embedder.embed(texts)

    assert caught.value.operation == expected_operation
    request.assert_not_awaited()
    assert caplog.records == []
    assert texts == before
    assert caller_text == bytearray(b"caller-secret")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["raise", "unserializable"])
@pytest.mark.parametrize(
    "target, expected_operation",
    [
        (("content",), "external_request"),
        (("model",), "log_metadata"),
    ],
)
async def test_rule_generator_protection_failure_sends_no_request_event_or_mutation(
    monkeypatch, caplog, mode, target, expected_operation
):
    from experia.improvement import rules as module

    request = AsyncMock()
    monkeypatch.setattr(module, "litellm", SimpleNamespace(acompletion=request))
    lesson = Lesson(
        experience_id=uuid4(),
        content="caller-secret lesson",
        root_cause="caller-secret cause",
    )
    before = lesson.model_copy(deep=True)
    store = SinkStore()
    generator = module.RuleGenerator(
        store=store,
        model="test-model",
        data_protection=DataProtectionLayer(
            FailingOrUnserializableSanitizer(target, mode)
        ),
    )

    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="experia"):
        with pytest.raises(SanitizationError) as caught:
            await generator.consolidate_lesson(lesson)

    assert caught.value.operation == expected_operation
    request.assert_not_awaited()
    assert caplog.records == []
    assert store.saved_memories == []
    assert lesson == before


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["raise", "unserializable"])
@pytest.mark.parametrize(
    "target, expected_operation",
    [
        (("experiences", 0, "result"), "external_request"),
        (("model",), "log_metadata"),
    ],
)
async def test_reflection_protection_failure_sends_no_request_event_or_mutation(
    monkeypatch, caplog, mode, target, expected_operation
):
    from experia.reflection import consolidation as module

    request = AsyncMock()
    monkeypatch.setattr(module, "litellm", SimpleNamespace(acompletion=request))
    experience = ExperienceRecord(
        task="caller-secret task",
        action="caller-secret action",
        result="caller-secret result",
    )
    before = experience.model_copy(deep=True)
    store = SinkStore([experience])
    engine = module.ReflectionEngine(
        store=store,
        model="test-model",
        data_protection=DataProtectionLayer(
            FailingOrUnserializableSanitizer(target, mode)
        ),
    )

    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="experia"):
        with pytest.raises(SanitizationError) as caught:
            await engine.reflect(batch_size=1)

    assert caught.value.operation == expected_operation
    request.assert_not_awaited()
    assert caplog.records == []
    assert store.saved_memories == []
    assert experience == before


@pytest.mark.asyncio
async def test_llm_evaluator_transmits_and_emits_only_protected_values(
    monkeypatch, caplog
):
    from experia.experience import llm_evaluator as module

    request = AsyncMock(
        return_value=MockResponse(
            '{"lesson":"safe lesson","root_cause":"safe cause","confidence":0.9}'
        )
    )
    monkeypatch.setattr(module, "litellm", SimpleNamespace(acompletion=request))
    evaluator = module.LLMEvaluator(
        model="test-model",
        data_protection=DataProtectionLayer(PrefixSanitizer()),
    )
    experience = ExperienceRecord(
        task="secret task",
        action="secret action",
        result="secret result",
        context={"token": "secret token"},
    )

    with caplog.at_level(logging.INFO, logger="experia"):
        await evaluator.evaluate(experience)

    user_prompt = request.await_args.kwargs["messages"][1]["content"]
    assert "protected:secret task" in user_prompt
    assert "protected:secret action" in user_prompt
    assert "protected:secret result" in user_prompt
    assert "protected:secret token" in user_prompt
    metadata = caplog.records[-1].experia_metadata
    assert metadata["operation"] == "protected:evaluation"
    assert metadata["model"] == "protected:test-model"


@pytest.mark.asyncio
async def test_external_embedder_transmits_and_emits_only_protected_values(
    monkeypatch, caplog
):
    from experia.memory import embeddings as module

    request = AsyncMock(return_value={"data": [{"embedding": [1.0, 2.0]}]})
    monkeypatch.setattr(module, "litellm", SimpleNamespace(aembedding=request))
    embedder = module.LiteLLMEmbedder(
        model="test-model",
        data_protection=DataProtectionLayer(PrefixSanitizer()),
    )

    with caplog.at_level(logging.DEBUG, logger="experia"):
        vector = await embedder.embed_one("secret text")

    assert vector == [1.0, 2.0]
    assert request.await_args.kwargs["input"] == ["protected:secret text"]
    metadata = caplog.records[-1].experia_metadata
    assert metadata["operation"] == "protected:embedding"
    assert metadata["model"] == "protected:test-model"


@pytest.mark.asyncio
async def test_rule_generator_transmits_and_emits_only_protected_values(
    monkeypatch, caplog
):
    from experia.improvement import rules as module

    request = AsyncMock(return_value=MockResponse("generated secret rule"))
    monkeypatch.setattr(module, "litellm", SimpleNamespace(acompletion=request))
    store = SinkStore()
    generator = module.RuleGenerator(
        store=store,
        model="test-model",
        data_protection=DataProtectionLayer(PrefixSanitizer()),
    )
    lesson = Lesson(
        experience_id=uuid4(),
        content="secret lesson",
        root_cause="secret cause",
    )

    with caplog.at_level(logging.INFO, logger="experia"):
        memory = await generator.consolidate_lesson(lesson)

    assert memory is not None
    user_prompt = request.await_args.kwargs["messages"][1]["content"]
    assert "protected:secret lesson" in user_prompt
    assert "protected:secret cause" in user_prompt
    metadata = caplog.records[-1].experia_metadata
    assert metadata["operation"] == "protected:rule_generation"
    assert "generated secret rule" not in caplog.records[-1].getMessage()


@pytest.mark.asyncio
async def test_reflection_transmits_and_emits_only_protected_values(
    monkeypatch, caplog
):
    from experia.reflection import consolidation as module

    request = AsyncMock(return_value=MockResponse("generated secret strategy"))
    monkeypatch.setattr(module, "litellm", SimpleNamespace(acompletion=request))
    store = SinkStore(
        [
            ExperienceRecord(
                task="secret task",
                action="secret action",
                result="secret result",
            )
        ]
    )
    engine = module.ReflectionEngine(
        store=store,
        model="test-model",
        data_protection=DataProtectionLayer(PrefixSanitizer()),
    )

    with caplog.at_level(logging.INFO, logger="experia"):
        memory = await engine.reflect(batch_size=1)

    assert memory is not None
    user_prompt = request.await_args.kwargs["messages"][1]["content"]
    assert "protected:secret task" in user_prompt
    assert "protected:secret action" in user_prompt
    assert "protected:secret result" in user_prompt
    metadata = caplog.records[-1].experia_metadata
    assert metadata["operation"] == "protected:reflection"
    assert "generated secret strategy" not in caplog.records[-1].getMessage()


class RecordingExternalEmbedder:
    def __init__(self):
        self.calls = []

    async def embed_one(self, text):
        self.calls.append(text)
        return [1.0]


class NullEvaluator:
    async def evaluate(self, experience):
        return None


class UnusedStore:
    pass


@pytest.mark.asyncio
async def test_learner_routes_arbitrary_external_embedder_through_boundary():
    embedder = RecordingExternalEmbedder()
    learner = Learner(
        store=UnusedStore(),
        evaluator=NullEvaluator(),
        embedder=embedder,
        data_protection=DataProtectionLayer(PrefixSanitizer()),
    )

    vector = await learner._embed_direct("secret text")

    assert vector == [1.0]
    assert embedder.calls == ["protected:secret text"]
