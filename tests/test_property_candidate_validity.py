"""Property coverage for framework candidate validation before persistence."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from enum import Enum
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from experia.core.learner import Learner
from experia.core.logging import (
    EVENT_SCHEMA_VERSION,
    IntegrationEvent,
    IntegrationName,
    IntegrationOutcome,
    logger,
)
from experia.experience.models import ExperienceRecord
from experia.integrations.langgraph.nodes import ExperiaLearningNode
from experia.integrations.langgraph.utils import default_messages_state_extractor
from experia.memory.store import SQLiteStore

_NON_BLANK_TEXT = st.builds(
    lambda prefix, value, suffix: f"{prefix}{value}{suffix}",
    prefix=st.sampled_from(("", " ", "\t")),
    value=st.text(
        alphabet=st.characters(
            blacklist_categories=("Cs", "Cc", "Zl", "Zp"),
            blacklist_characters=(" ",),
        ),
        min_size=1,
        max_size=24,
    ).filter(lambda value: bool(value.strip())),
    suffix=st.sampled_from(("", " ", "\t")),
)
_BLANK_TEXT = st.sampled_from(("", " ", "\t", "\n", "\r\n", "\u2003"))
_JSON_SCALAR = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2**31), max_value=2**31 - 1),
    st.floats(allow_nan=False, allow_infinity=False, width=32),
    _NON_BLANK_TEXT,
)
_CONTEXT = st.dictionaries(
    st.text(
        alphabet=st.characters(blacklist_categories=("Cs", "Cc")),
        min_size=1,
        max_size=12,
    ),
    _JSON_SCALAR,
    max_size=4,
)


class _ExtractorKind(str, Enum):
    DEFAULT = "default"
    CUSTOM = "custom"


class _InvalidShape(str, Enum):
    TASK = "task"
    ACTION = "action"
    RESULT = "result"
    CONTEXT = "context"
    MISSING = "missing"
    NON_MAPPING = "non_mapping"


class _NoLessonEvaluator:
    async def evaluate(self, _experience: ExperienceRecord) -> None:
        return None


class _IntegrationEventHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[IntegrationEvent] = []

    def emit(self, record: logging.LogRecord) -> None:
        event = getattr(record, "experia_integration_event", None)
        if isinstance(event, IntegrationEvent):
            self.events.append(event)


def _default_state(task: str, action: str, result: str) -> dict[str, Any]:
    tool_call = {
        "id": "generated-tool",
        "name": "generated_action",
        "args": {"value": action},
    }
    return {
        "messages": [
            {"type": "human", "content": task},
            {"type": "ai", "content": "", "tool_calls": [tool_call]},
            {
                "type": "tool",
                "content": result,
                "tool_call_id": "generated-tool",
            },
        ]
    }


def _custom_output(
    *,
    valid: bool,
    invalid_shape: _InvalidShape,
    task: str,
    action: str,
    result: str,
    blank: str,
    context: dict[str, Any],
) -> object:
    candidate: dict[str, Any] = {
        "task": task,
        "action": action,
        "result": result,
        "context": context,
    }
    if valid:
        return candidate
    if invalid_shape is _InvalidShape.NON_MAPPING:
        return [candidate]
    if invalid_shape is _InvalidShape.MISSING:
        candidate.pop("result")
    elif invalid_shape is _InvalidShape.CONTEXT:
        candidate["context"] = [context]
    else:
        candidate[invalid_shape.value] = blank
    return candidate


async def _exercise_candidate(
    *,
    extractor_kind: _ExtractorKind,
    valid: bool,
    invalid_shape: _InvalidShape,
    task: str,
    action: str,
    result: str,
    blank: str,
    context: dict[str, Any],
) -> None:
    store = SQLiteStore(":memory:")
    await store.initialize()
    learner = Learner(
        store=store,
        evaluator=_NoLessonEvaluator(),
        background_evaluation=False,
    )
    handler = _IntegrationEventHandler()
    previous_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    try:
        expected: Mapping[str, Any] | None
        if extractor_kind is _ExtractorKind.DEFAULT:
            default_task = (
                task if valid or invalid_shape is _InvalidShape.RESULT else blank
            )
            default_result = (
                result if valid or invalid_shape is not _InvalidShape.RESULT else blank
            )
            state = _default_state(default_task, action, default_result)
            expected = default_messages_state_extractor(state)
            node = ExperiaLearningNode(
                agent=learner,
                callback_mode="durable",
            )
        else:
            output = _custom_output(
                valid=valid,
                invalid_shape=invalid_shape,
                task=task,
                action=action,
                result=result,
                blank=blank,
                context=context,
            )
            expected = output if isinstance(output, Mapping) else None
            node = ExperiaLearningNode(
                agent=learner,
                extractor=lambda _state: output,
                callback_mode="durable",
            )
            state = {"messages": []}

        assert await node(state) == {}
        persisted = await store.get_recent_experiences()
        no_experience = [
            event
            for event in handler.events
            if event.outcome is IntegrationOutcome.NO_EXPERIENCE
        ]

        assert len(persisted) == int(valid)
        if valid:
            assert expected is not None
            record = persisted[0]
            assert record.task == expected["task"]
            assert record.action == expected["action"]
            assert record.result == expected["result"]
            expected_context = dict(expected.get("context", {}))
            expected_context["source"] = "langgraph"
            assert record.context == expected_context
            assert no_experience == []
        else:
            assert persisted == []
            assert no_experience == [
                IntegrationEvent(
                    schema_version=EVENT_SCHEMA_VERSION,
                    integration=IntegrationName.LANGGRAPH,
                    outcome=IntegrationOutcome.NO_EXPERIENCE,
                    run_id=None,
                )
            ]
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
        await learner.shutdown()
        await store.close()


# Feature: open-source-project-improvements, Property 21: Invalid candidates are never persisted
# **Validates: Requirements 6.5, 6.6**
@settings(max_examples=100, deadline=None)
@given(
    extractor_kind=st.sampled_from(tuple(_ExtractorKind)),
    valid=st.booleans(),
    invalid_shape=st.sampled_from(tuple(_InvalidShape)),
    task=_NON_BLANK_TEXT,
    action=_NON_BLANK_TEXT,
    result=_NON_BLANK_TEXT,
    blank=_BLANK_TEXT,
    context=_CONTEXT,
)
def test_only_valid_default_and_custom_candidates_are_persisted(
    extractor_kind: _ExtractorKind,
    valid: bool,
    invalid_shape: _InvalidShape,
    task: str,
    action: str,
    result: str,
    blank: str,
    context: dict[str, Any],
) -> None:
    asyncio.run(
        _exercise_candidate(
            extractor_kind=extractor_kind,
            valid=valid,
            invalid_shape=invalid_shape,
            task=task,
            action=action,
            result=result,
            blank=blank,
            context=context,
        )
    )
