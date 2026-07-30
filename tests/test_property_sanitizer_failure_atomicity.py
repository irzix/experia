"""Property tests for sanitizer-failure atomicity."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from experia.core.exceptions import SanitizationError
from experia.security import DataProtectionLayer

_JSON_SCALARS = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2**31), max_value=2**31 - 1),
    st.floats(allow_nan=False, allow_infinity=False, width=32),
    st.text(max_size=20),
)
_JSON_VALUES = st.recursive(
    _JSON_SCALARS,
    lambda children: st.one_of(
        st.lists(children, max_size=3),
        st.dictionaries(st.text(max_size=10), children, max_size=3),
    ),
    max_leaves=8,
)
_JSON_OBJECTS = st.dictionaries(st.text(max_size=10), _JSON_VALUES, max_size=3)
_PATH_KEYS = st.text(min_size=1, max_size=10)


@dataclass(frozen=True)
class _NestedFailureCase:
    value: Any
    path: tuple[str | int, ...]


@st.composite
def _nested_failure_cases(draw: st.DrawFn) -> _NestedFailureCase:
    """Build a mutable target at a generated path through supported containers."""

    value: Any = bytearray(draw(st.binary(max_size=20)))
    path: tuple[str | int, ...] = ()

    for _ in range(draw(st.integers(min_value=1, max_value=5))):
        container = draw(st.sampled_from(("mapping", "list", "tuple")))
        if container == "mapping":
            key = draw(_PATH_KEYS)
            value = {key: value}
            path = (key, *path)
            continue

        prefix = draw(st.lists(_JSON_SCALARS, max_size=3))
        suffix = draw(st.lists(_JSON_SCALARS, max_size=3))
        index = len(prefix)
        members = [*prefix, value, *suffix]
        value = members if container == "list" else tuple(members)
        path = (index, *path)

    root = draw(_PATH_KEYS)
    return _NestedFailureCase(value={root: value}, path=(root, *path))


class _MutatingFailingSanitizer:
    def __init__(self, target_path: tuple[str | int, ...]) -> None:
        self.target_path = target_path
        self.visited_paths: list[tuple[str | int, ...]] = []

    def sanitize(self, value: Any, *, path: tuple[str | int, ...]) -> Any:
        self.visited_paths.append(path)
        if path == self.target_path:
            value.extend(b"-mutated-copy")
            raise RuntimeError("sanitizer failed after mutating its private copy")
        return value


# Feature: open-source-project-improvements, Property 7: Sanitizer failure is atomic and non-mutating
@pytest.mark.parametrize("failure_operation", ("external_request", "log_metadata"))
@settings(max_examples=100, deadline=None)
@given(
    failure_case=_nested_failure_cases(),
    safe_request=_JSON_OBJECTS,
    safe_metadata=_JSON_OBJECTS,
)
def test_sanitizer_failure_is_atomic_and_non_mutating(
    failure_operation: str,
    failure_case: _NestedFailureCase,
    safe_request: dict[str, Any],
    safe_metadata: dict[str, Any],
) -> None:
    """**Validates: Requirements 2.10, 2.11**"""

    request_fields = {"safe": safe_request}
    metadata = {"safe": safe_metadata}
    if failure_operation == "external_request":
        request_fields["generated"] = failure_case.value
    else:
        metadata["generated"] = failure_case.value

    target_path = ("generated", *failure_case.path)
    sanitizer = _MutatingFailingSanitizer(target_path)
    protection = DataProtectionLayer(sanitizer)
    request_before = deepcopy(request_fields)
    metadata_before = deepcopy(metadata)
    requests: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []

    def emit_to_sinks() -> None:
        protected_request, protected_metadata = protection.protect_sink(
            request_fields,
            metadata,
        )
        requests.append(protected_request)
        events.append(protected_metadata)

    with pytest.raises(SanitizationError) as raised:
        emit_to_sinks()

    assert raised.value.path == target_path
    assert raised.value.operation == failure_operation
    assert sanitizer.visited_paths.count(target_path) == 1
    assert requests == []
    assert events == []
    assert request_fields == request_before
    assert metadata == metadata_before
