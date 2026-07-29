"""Focused contract tests for optional and planned features."""

import importlib
from copy import deepcopy
from uuid import uuid4

import pytest

from experia.core.dependencies import require_optional_dependency
from experia.core.exceptions import ConfigurationError, UnavailableFeatureError
from experia.experience.models import Lesson


class TrackingStore:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def get_recent_experiences(self, limit: int = 50):
        self.calls.append("get_recent_experiences")
        return []

    async def save_memory(self, memory) -> None:
        self.calls.append("save_memory")


@pytest.mark.parametrize(
    ("module_name", "class_name", "feature"),
    [
        ("experia.adapters.mem0", "Mem0Adapter", "mem0"),
        ("experia.adapters.postgres", "PostgresAdapter", "postgres"),
        ("experia.adapters.zep", "ZepAdapter", "zep"),
        ("experia.integrations.crewai", "CrewAIIntegration", "crewai"),
    ],
)
def test_planned_feature_paths_import_but_construction_is_explicitly_unavailable(
    module_name, class_name, feature
):
    module = importlib.import_module(module_name)
    placeholder = getattr(module, class_name)
    caller_config = {"nested": ["unchanged"]}
    before = deepcopy(caller_config)

    with pytest.raises(UnavailableFeatureError) as caught:
        placeholder(caller_config)

    assert caught.value.feature == feature
    assert caught.value.status == "planned"
    assert caller_config == before


def test_present_optional_dependency_path_is_a_noop():
    assert (
        require_optional_dependency(
            True,
            feature="available-feature",
            extra="experia[available]",
        )
        is None
    )


@pytest.mark.parametrize(
    ("module_name", "class_name", "dependency_name", "args", "feature", "extra"),
    [
        (
            "experia.experience.llm_evaluator",
            "LLMEvaluator",
            "litellm",
            ("test-model",),
            "LLMEvaluator",
            "experia[llm]",
        ),
        (
            "experia.memory.embeddings",
            "LiteLLMEmbedder",
            "litellm",
            ("test-model",),
            "LiteLLMEmbedder",
            "experia[llm]",
        ),
        (
            "experia.integrations.langchain.callbacks",
            "ExperiaCallbackHandler",
            "_LANGCHAIN_AVAILABLE",
            (object(),),
            "ExperiaCallbackHandler",
            "experia[langchain]",
        ),
        (
            "experia.integrations.langgraph.nodes",
            "ExperiaContextNode",
            "_LANGGRAPH_AVAILABLE",
            (object(), 9),
            "ExperiaContextNode",
            "experia[langgraph]",
        ),
        (
            "experia.integrations.langgraph.nodes",
            "ExperiaLearningNode",
            "_LANGGRAPH_AVAILABLE",
            (object(),),
            "ExperiaLearningNode",
            "experia[langgraph]",
        ),
    ],
)
def test_missing_dependency_constructor_fails_before_instance_mutation(
    monkeypatch,
    module_name,
    class_name,
    dependency_name,
    args,
    feature,
    extra,
):
    module = importlib.import_module(module_name)
    dependency_value = None if dependency_name == "litellm" else False
    monkeypatch.setattr(module, dependency_name, dependency_value)
    guarded_class = getattr(module, class_name)
    instance = object.__new__(guarded_class)

    with pytest.raises(ConfigurationError) as caught:
        guarded_class.__init__(instance, *args)

    assert caught.value.feature == feature
    assert caught.value.parameter == "dependency"
    assert caught.value.extra == extra
    assert extra in str(caught.value)
    assert vars(instance) == {}


def test_missing_langchain_retriever_dependency_fails_before_construction(monkeypatch):
    retrievers = importlib.import_module("experia.integrations.langchain.retrievers")
    monkeypatch.setattr(retrievers, "_LANGCHAIN_AVAILABLE", False)

    with pytest.raises(ConfigurationError) as caught:
        retrievers.ExperiaLearningRetriever(agent=object())

    assert caught.value.feature == "ExperiaLearningRetriever"
    assert caught.value.extra == "experia[langchain]"
    assert "experia[langchain]" in str(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("module_name", "class_name", "method_name", "feature"),
    [
        (
            "experia.improvement.rules",
            "RuleGenerator",
            "consolidate_lesson",
            "RuleGenerator",
        ),
        (
            "experia.reflection.consolidation",
            "ReflectionEngine",
            "reflect",
            "ReflectionEngine",
        ),
    ],
)
async def test_missing_llm_dependency_fails_before_store_or_object_mutation(
    monkeypatch, module_name, class_name, method_name, feature
):
    module = importlib.import_module(module_name)
    store = TrackingStore()
    guarded_object = getattr(module, class_name)(store=store, model="test-model")
    before = vars(guarded_object).copy()
    monkeypatch.setattr(module, "litellm", None)

    with pytest.raises(ConfigurationError) as caught:
        if method_name == "consolidate_lesson":
            lesson = Lesson(
                experience_id=uuid4(),
                content="A reusable lesson",
                root_cause="A known cause",
            )
            await guarded_object.consolidate_lesson(lesson)
        else:
            await guarded_object.reflect(batch_size=10)

    assert caught.value.feature == feature
    assert caught.value.extra == "experia[llm]"
    assert "experia[llm]" in str(caught.value)
    assert store.calls == []
    assert vars(guarded_object) == before


def test_available_llm_dependency_allows_construction(monkeypatch):
    evaluator_module = importlib.import_module("experia.experience.llm_evaluator")
    embedding_module = importlib.import_module("experia.memory.embeddings")
    available_dependency = object()
    monkeypatch.setattr(evaluator_module, "litellm", available_dependency)
    monkeypatch.setattr(embedding_module, "litellm", available_dependency)

    evaluator = evaluator_module.LLMEvaluator(model="present-model")
    embedder = embedding_module.LiteLLMEmbedder(model="present-model")

    assert evaluator.model == "present-model"
    assert embedder.model == "present-model"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("module_name", "class_name", "method_name"),
    [
        ("experia.experience.llm_evaluator", "LLMEvaluator", "evaluate"),
        ("experia.memory.embeddings", "LiteLLMEmbedder", "embed"),
    ],
)
async def test_dependency_loss_at_invocation_fails_before_processing_or_mutation(
    monkeypatch, module_name, class_name, method_name
):
    class TrackingProtection:
        def __init__(self):
            self.calls = []

        def protect_sink(self, fields, metadata):
            self.calls.append((fields, metadata))
            return fields, metadata

    module = importlib.import_module(module_name)
    protection = TrackingProtection()
    monkeypatch.setattr(module, "litellm", object())
    guarded_object = getattr(module, class_name)(
        "present-model",
        data_protection=protection,
    )
    before = vars(guarded_object).copy()
    monkeypatch.setattr(module, "litellm", None)

    if method_name == "evaluate":
        from experia.experience.models import ExperienceRecord

        argument = ExperienceRecord(task="task", action="action", result="result")
    else:
        argument = ["text that must not be processed"]

    with pytest.raises(ConfigurationError) as caught:
        await getattr(guarded_object, method_name)(argument)

    assert caught.value.extra == "experia[llm]"
    assert protection.calls == []
    assert vars(guarded_object) == before


@pytest.mark.parametrize(
    ("module_name", "class_name", "feature"),
    [
        ("experia.adapters", "Mem0MemoryAdapter", "mem0"),
        ("experia.adapters", "PostgresStore", "postgres"),
        ("experia.adapters", "PostgreSQLStore", "postgres"),
        ("experia.adapters", "ZepMemoryAdapter", "zep"),
        ("experia.integrations.crewai", "CrewAIAdapter", "crewai"),
    ],
)
def test_planned_feature_public_aliases_are_explicitly_unavailable(
    module_name, class_name, feature
):
    module = importlib.import_module(module_name)

    with pytest.raises(UnavailableFeatureError) as caught:
        getattr(module, class_name)()

    assert caught.value.feature == feature
    assert caught.value.status == "planned"


@pytest.mark.parametrize(
    ("module_name", "class_name", "availability_name", "args", "expected_state"),
    [
        (
            "experia.integrations.langchain.callbacks",
            "ExperiaCallbackHandler",
            "_LANGCHAIN_AVAILABLE",
            (object(),),
            "builder",
        ),
        (
            "experia.integrations.langgraph.nodes",
            "ExperiaContextNode",
            "_LANGGRAPH_AVAILABLE",
            (object(), 9),
            "limit",
        ),
        (
            "experia.integrations.langgraph.nodes",
            "ExperiaLearningNode",
            "_LANGGRAPH_AVAILABLE",
            (object(),),
            "callback_mode",
        ),
    ],
)
def test_present_framework_extra_allows_guarded_construction(
    monkeypatch,
    module_name,
    class_name,
    availability_name,
    args,
    expected_state,
):
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, availability_name, True)

    instance = getattr(module, class_name)(*args)

    assert hasattr(instance, expected_state)
