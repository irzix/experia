"""Enforce Quality Gate property-suite profiles and migration coverage."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from hypothesis import settings

from experia.memory.migrations import SUPPORTED_SCHEMA_VERSIONS
from tests import test_property_concurrent_feedback as concurrency_property
from tests import test_property_migration_preservation as migration_property
from tests import test_property_serialization_failure_atomicity as failure_property
from tests import test_property_storage_round_trip as round_trip_property
from tests.quality_profiles import (
    CI_PROFILE,
    CONCURRENCY_CI_MAX_OPERATIONS,
    CONCURRENCY_EXTENDED_MAX_OPERATIONS,
    CONCURRENCY_EXTENDED_PROFILE,
    CONCURRENCY_EXTENDED_PROFILE_ALIASES,
    CONCURRENCY_MIN_EXAMPLES,
    CONCURRENCY_MIN_OPERATIONS,
    CONCURRENCY_PROFILE,
    HYPOTHESIS_PROFILE_MIN_EXAMPLES,
    LOCAL_PROFILE,
    SERIALIZER_MIN_EXAMPLES,
    concurrency_max_operations,
)


def _effective_settings(property_test: Callable[..., Any]) -> settings:
    configured = getattr(property_test, "_hypothesis_internal_use_settings", None)
    assert configured is not None, f"{property_test.__name__} is not a Hypothesis test"
    return configured


@pytest.mark.parametrize(
    ("profile_name", "required_examples"),
    (
        (LOCAL_PROFILE, 100),
        (CI_PROFILE, 100),
        (CONCURRENCY_PROFILE, 50),
        (CONCURRENCY_EXTENDED_PROFILE, 100),
        ("concurrency-extended", 100),
    ),
)
def test_documented_hypothesis_profiles_preserve_quality_gate_minimums(
    profile_name: str,
    required_examples: int,
) -> None:
    declared_minimum = HYPOTHESIS_PROFILE_MIN_EXAMPLES[profile_name]
    profile = settings.get_profile(profile_name)

    assert declared_minimum >= required_examples
    assert profile.max_examples >= declared_minimum
    assert profile.deadline is None
    assert profile.derandomize is True
    assert profile.database is None


@pytest.mark.parametrize(
    "property_test",
    (
        failure_property.test_serialization_failure_has_no_partial_effect,
        round_trip_property.test_storage_serialization_round_trip_is_type_preserving,
    ),
    ids=("serialization-failure-atomicity", "storage-round-trip"),
)
def test_serializer_properties_run_at_least_100_generated_examples(
    property_test: Callable[..., Any],
) -> None:
    assert SERIALIZER_MIN_EXAMPLES >= 100
    assert _effective_settings(property_test).max_examples >= SERIALIZER_MIN_EXAMPLES


def test_concurrency_property_enforces_case_and_operation_bounds() -> None:
    property_test = (
        concurrency_property.test_concurrent_feedback_preserves_counts_and_confidence
    )

    assert CONCURRENCY_MIN_EXAMPLES >= 50
    assert _effective_settings(property_test).max_examples >= CONCURRENCY_MIN_EXAMPLES
    assert CONCURRENCY_MIN_OPERATIONS == 2
    assert CONCURRENCY_CI_MAX_OPERATIONS == 32
    assert CONCURRENCY_EXTENDED_MAX_OPERATIONS == 100
    assert concurrency_property._MAX_OPERATIONS == concurrency_max_operations(
        settings.get_current_profile_name()
    )


@pytest.mark.parametrize(
    ("profile_name", "expected_max_operations"),
    (
        (LOCAL_PROFILE, CONCURRENCY_CI_MAX_OPERATIONS),
        (CI_PROFILE, CONCURRENCY_CI_MAX_OPERATIONS),
        (CONCURRENCY_PROFILE, CONCURRENCY_CI_MAX_OPERATIONS),
        *(
            (profile_name, CONCURRENCY_EXTENDED_MAX_OPERATIONS)
            for profile_name in sorted(CONCURRENCY_EXTENDED_PROFILE_ALIASES)
        ),
    ),
)
def test_concurrency_operation_range_is_profile_controlled(
    profile_name: str,
    expected_max_operations: int,
) -> None:
    assert concurrency_max_operations(profile_name) == expected_max_operations


def test_migration_property_exercises_every_supported_fixture() -> None:
    property_test = migration_property.test_migration_is_preserving_and_idempotent
    parametrize_marks = [
        mark
        for mark in property_test.pytestmark
        if mark.name == "parametrize" and mark.args[0] == "fixture"
    ]

    assert len(parametrize_marks) == 1
    fixture_cases = tuple(parametrize_marks[0].args[1])
    assert tuple(case.schema_version for case in fixture_cases) == (
        SUPPORTED_SCHEMA_VERSIONS
    )
    assert len({case.schema_version for case in fixture_cases}) == len(fixture_cases)
    assert all(isinstance(case.script_path, Path) for case in fixture_cases)
    assert all(case.script_path.is_file() for case in fixture_cases)
