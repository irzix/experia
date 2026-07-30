"""Property coverage for roadmap ownership/readiness schema uniqueness."""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from scripts.roadmap_manifest import (
    _KINDS as VALID_KINDS,
)
from scripts.roadmap_manifest import (
    _OWNERSHIP_KEYS as OWNERSHIP_KEYS,
)
from scripts.roadmap_manifest import (
    _READINESS_STATUSES as READINESS_STATUSES,
)
from scripts.roadmap_manifest import (
    RoadmapManifestError,
    validate_entry,
)

# Valid, non-placeholder values for each ownership selector. ``unassigned`` is the
# literal ``true`` while ``owner``/``team`` are public non-placeholder identities.
_OWNERSHIP_VALUES: dict[str, Any] = {
    "owner": "a-maintainer",
    "team": "core-team",
    "unassigned": True,
}

_KIND = st.sampled_from(sorted(VALID_KINDS))
_READINESS = st.sampled_from(sorted(READINESS_STATUSES))
# 0/1/many ownership selectors drawn as a unique subset of owner/team/unassigned.
_OWNERSHIP_SUBSET = st.lists(
    st.sampled_from(sorted(OWNERSHIP_KEYS)),
    unique=True,
    max_size=len(OWNERSHIP_KEYS),
)


@st.composite
def _roadmap_entries(draw: st.DrawFn) -> dict[str, Any]:
    """Build entries with 0/1/many ownership selectors and readiness values."""
    entry: dict[str, Any] = {
        "id": "x",
        "title": "Planned Item",
        "kind": draw(_KIND),
    }
    for key in draw(_OWNERSHIP_SUBSET):
        entry[key] = _OWNERSHIP_VALUES[key]

    readiness_kind = draw(st.sampled_from(("absent", "single", "many")))
    if readiness_kind == "single":
        entry["readiness"] = draw(_READINESS)
    elif readiness_kind == "many":
        # A collection of readiness statuses is "many", never exactly one.
        entry["readiness"] = draw(
            st.lists(_READINESS, min_size=2, max_size=len(READINESS_STATUSES))
        )
    return entry


def _has_exactly_one_ownership_selector(entry: dict[str, Any]) -> bool:
    return len([key for key in OWNERSHIP_KEYS if key in entry]) == 1


def _has_exactly_one_readiness_value(entry: dict[str, Any]) -> bool:
    readiness = entry.get("readiness")
    return isinstance(readiness, str) and readiness in READINESS_STATUSES


# Feature: open-source-project-improvements, Property 23: Planned ownership is unique
@settings(max_examples=100, deadline=None)
@given(entry=_roadmap_entries())
def test_entry_is_valid_iff_one_ownership_selector_and_one_readiness(
    entry: dict[str, Any],
) -> None:
    """**Validates: Requirements 9.5**"""
    expected_accept = _has_exactly_one_ownership_selector(
        entry
    ) and _has_exactly_one_readiness_value(entry)

    if expected_accept:
        assert validate_entry(entry) == entry["id"]
    else:
        with pytest.raises(RoadmapManifestError):
            validate_entry(entry)
