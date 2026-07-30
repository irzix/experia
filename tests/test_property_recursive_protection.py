"""Property tests for recursive protection at external and metadata sinks."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import BaseModel, ConfigDict

from experia.security import DataProtectionLayer

Path = tuple[str | int, ...]
_SAFE_TEXT = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)),
    max_size=24,
)
_SCALARS = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2**31), max_value=2**31 - 1),
    st.floats(allow_nan=False, allow_infinity=False, width=32),
    _SAFE_TEXT,
    st.binary(max_size=12),
    st.binary(max_size=12).map(bytearray),
)
_HASHABLE_SCALARS = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2**31), max_value=2**31 - 1),
    st.floats(allow_nan=False, allow_infinity=False, width=32),
    _SAFE_TEXT,
    st.binary(max_size=12),
)
_MAPPING_KEYS = st.one_of(
    st.integers(min_value=-10, max_value=10),
    st.text(
        alphabet=st.characters(blacklist_categories=("Cs",)),
        min_size=1,
        max_size=12,
    ),
)


class GeneratedModel(BaseModel):
    """Pydantic container used to exercise declared and extra fields."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    value: Any
    label: Any


_NESTED_VALUES = st.recursive(
    _SCALARS,
    lambda children: st.one_of(
        st.lists(children, max_size=3),
        st.lists(children, max_size=3).map(tuple),
        st.dictionaries(_MAPPING_KEYS, children, max_size=3),
        st.sets(_HASHABLE_SCALARS, max_size=3),
        st.sets(_HASHABLE_SCALARS, max_size=3).map(frozenset),
        st.builds(
            GeneratedModel,
            value=children,
            label=_SCALARS,
            extra_value=children,
        ),
    ),
    max_leaves=12,
)


@st.composite
def _supported_payloads(draw: st.DrawFn) -> Mapping[str, Any]:
    """Generate every supported container kind plus recursive nested values."""

    return OrderedDict(
        generated=draw(_NESTED_VALUES),
        sequences=[draw(_NESTED_VALUES), (draw(_SCALARS),)],
        sets={
            "mutable": draw(st.sets(_HASHABLE_SCALARS, max_size=3)),
            "immutable": draw(st.sets(_HASHABLE_SCALARS, max_size=3).map(frozenset)),
        },
        model=GeneratedModel(
            value=draw(_NESTED_VALUES),
            label=draw(_SCALARS),
            extra_value=draw(_NESTED_VALUES),
        ),
    )


@dataclass(frozen=True)
class ProtectedLeaf:
    """Unique sanitizer output proving which path produced a sink value."""

    path: Path


class RecordingSanitizer:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, Any]] = []

    def sanitize(self, value: Any, *, path: Path) -> ProtectedLeaf:
        self.calls.append((path, deepcopy(value)))
        return ProtectedLeaf(path)


def _path_component(key: object) -> str | int:
    if isinstance(key, str):
        return key
    if isinstance(key, int):
        return int(key)
    return str(key)


def _leaves(value: Any, path: Path) -> Iterator[tuple[Path, Any]]:
    if isinstance(value, BaseModel):
        for field_name in type(value).model_fields:
            yield from _leaves(
                getattr(value, field_name),
                path + (field_name,),
            )
        if value.model_extra:
            for field_name, field_value in value.model_extra.items():
                yield from _leaves(
                    field_value,
                    path + (_path_component(field_name),),
                )
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            yield from _leaves(nested, path + (_path_component(key),))
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for index, nested in enumerate(value):
            yield from _leaves(nested, path + (index,))
        return
    yield path, value


def _payload_leaves(payload: Mapping[str, Any]) -> list[tuple[Path, Any]]:
    return [
        leaf
        for key, value in payload.items()
        for leaf in _leaves(value, (_path_component(key),))
    ]


# Feature: open-source-project-improvements, Property 6: Recursive protection applies at every sink
@settings(max_examples=100, deadline=None)
@given(payload=_supported_payloads())
def test_recursive_protection_applies_at_every_sink(
    payload: Mapping[str, Any],
) -> None:
    """**Validates: Requirements 2.1, 2.2, 2.9**"""
    before = deepcopy(payload)
    expected_calls = _payload_leaves(payload)

    for sink_name in ("protect_external", "protect_metadata"):
        pass_through = getattr(DataProtectionLayer(), sink_name)(payload)
        assert pass_through == payload
        assert payload == before

        sanitizer = RecordingSanitizer()
        protected = getattr(DataProtectionLayer(sanitizer), sink_name)(payload)

        assert sanitizer.calls == expected_calls
        actual_replacements = [value for _, value in _payload_leaves(protected)]
        expected_replacements = [ProtectedLeaf(path) for path, _ in expected_calls]
        assert len(actual_replacements) == len(expected_replacements)
        assert set(actual_replacements) == set(expected_replacements)
        assert payload == before
