"""Property tests for type-preserving storage serialization round trips."""

import struct
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from experia.experience.models import ExperienceRecord
from experia.memory.models import Memory, MemoryType
from experia.memory.store import SQLiteStore
from tests.quality_profiles import SERIALIZER_MIN_EXAMPLES

_SAFE_TEXT = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)),
    max_size=32,
)
_FINITE_FLOATS = st.floats(
    allow_nan=False,
    allow_infinity=False,
    width=64,
)
_JSON_SCALARS = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2**63), max_value=2**63 - 1),
    _FINITE_FLOATS,
    _SAFE_TEXT,
)
_FIXED_OFFSETS = st.integers(min_value=-(24 * 60 - 1), max_value=24 * 60 - 1).map(
    lambda minutes: timezone(timedelta(minutes=minutes))
)
_AWARE_DATETIMES = st.datetimes(
    min_value=datetime(2000, 1, 1),
    max_value=datetime(2035, 12, 31, 23, 59, 59, 999999),
    timezones=_FIXED_OFFSETS,
)


@st.composite
def _nested_json_mappings(draw: st.DrawFn) -> dict[str, Any]:
    """Generate every supported JSON scalar inside nested mappings and lists."""
    return {
        "none": None,
        "boolean": draw(st.booleans()),
        "integer": draw(st.integers(min_value=-(2**63), max_value=2**63 - 1)),
        "float": draw(_FINITE_FLOATS),
        "text": draw(_SAFE_TEXT),
        "nested": {
            "list": draw(st.lists(_JSON_SCALARS, min_size=0, max_size=6)),
            "object": {"leaf": draw(_JSON_SCALARS)},
        },
    }


@st.composite
def _experience_records(draw: st.DrawFn) -> ExperienceRecord:
    return ExperienceRecord(
        id=draw(st.uuids()),
        task=draw(_SAFE_TEXT),
        action=draw(_SAFE_TEXT),
        result=draw(_SAFE_TEXT),
        agent_role=draw(_SAFE_TEXT),
        context=draw(_nested_json_mappings()),
        created_at=draw(_AWARE_DATETIMES),
    )


@st.composite
def _memories(draw: st.DrawFn) -> Memory:
    reinforcement_count = draw(st.integers(min_value=0, max_value=10_000))
    return Memory(
        id=draw(st.uuids()),
        content=draw(_SAFE_TEXT),
        type=draw(st.sampled_from(tuple(MemoryType))),
        agent_role=draw(_SAFE_TEXT),
        confidence=draw(
            st.floats(
                min_value=0.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
                width=64,
            )
        ),
        importance=draw(
            st.floats(
                min_value=0.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
                width=64,
            )
        ),
        source=draw(st.one_of(st.none(), _SAFE_TEXT)),
        metadata=draw(_nested_json_mappings()),
        embedding=draw(st.lists(_FINITE_FLOATS, min_size=1, max_size=16)),
        reinforcement_count=reinforcement_count,
        success_count=draw(st.integers(min_value=0, max_value=reinforcement_count)),
        created_at=draw(_AWARE_DATETIMES),
        updated_at=draw(_AWARE_DATETIMES),
        expires_at=draw(_AWARE_DATETIMES),
    )


def _assert_json_value_preserved(actual: Any, expected: Any) -> None:
    assert type(actual) is type(expected)
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key, expected_value in expected.items():
            _assert_json_value_preserved(actual[key], expected_value)
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for actual_value, expected_value in zip(actual, expected):
            _assert_json_value_preserved(actual_value, expected_value)
    elif isinstance(expected, float):
        assert struct.pack("!d", actual) == struct.pack("!d", expected)
    else:
        assert actual == expected


def _assert_aware_instant_and_offset_preserved(
    actual: datetime, expected: datetime
) -> None:
    assert actual.tzinfo is not None
    assert actual.utcoffset() is not None
    assert actual.utcoffset() == expected.utcoffset()
    assert actual.astimezone(timezone.utc) == expected.astimezone(timezone.utc)


# Feature: open-source-project-improvements, Property 15: Storage serialization round trip is type-preserving
@pytest.mark.asyncio
@settings(max_examples=SERIALIZER_MIN_EXAMPLES, deadline=None)
@given(experience=_experience_records(), memory=_memories())
async def test_storage_serialization_round_trip_is_type_preserving(
    experience: ExperienceRecord, memory: Memory
) -> None:
    """**Validates: Requirements 3.9, 3.10, 3.11, 3.12, 7.4**"""
    store = SQLiteStore(":memory:")
    await store.initialize()
    try:
        await store.save_experience(experience)
        await store.save_memory(memory)

        loaded_experience = await store.get_experience(experience.id)
        loaded_memory = await store.get_memory(memory.id)

        assert loaded_experience == experience
        assert loaded_memory == memory
        assert isinstance(loaded_experience.id, UUID)
        assert isinstance(loaded_memory.id, UUID)
        assert type(loaded_memory.type) is MemoryType
        assert loaded_memory.type.value == memory.type.value

        _assert_json_value_preserved(loaded_experience.context, experience.context)
        _assert_json_value_preserved(loaded_memory.metadata, memory.metadata)

        assert loaded_memory.embedding is not None
        assert memory.embedding is not None
        assert len(loaded_memory.embedding) == len(memory.embedding)
        assert [struct.pack("!d", value) for value in loaded_memory.embedding] == [
            struct.pack("!d", value) for value in memory.embedding
        ]

        _assert_aware_instant_and_offset_preserved(
            loaded_experience.created_at, experience.created_at
        )
        _assert_aware_instant_and_offset_preserved(
            loaded_memory.created_at, memory.created_at
        )
        _assert_aware_instant_and_offset_preserved(
            loaded_memory.updated_at, memory.updated_at
        )
        assert loaded_memory.expires_at is not None
        assert memory.expires_at is not None
        _assert_aware_instant_and_offset_preserved(
            loaded_memory.expires_at, memory.expires_at
        )
    finally:
        await store.close()
