import inspect
import warnings
from copy import deepcopy

import pytest

from experia.core.deprecation import deprecated, warn_deprecated
from scripts.api_compatibility import (
    SnapshotCompatibilityError,
    assert_snapshots_compatible,
    compare_snapshots,
    deprecation_window_elapsed,
)

_NOT_DEPRECATED = {
    "is_deprecated": False,
    "message": None,
    "replacement": None,
    "since": None,
}


def _parameter(
    name,
    *,
    kind="positional_or_keyword",
    annotation="str",
    required=True,
    default=None,
):
    return {
        "name": name,
        "kind": kind,
        "annotation": annotation,
        "required": required,
        "default": default or {"kind": "required"},
    }


def _export(path="experia.Legacy", *, parameters=None, members=None, deprecation=None):
    return {
        "path": path,
        "target": path,
        "kind": "function",
        "async": False,
        "signature": {
            "parameters": parameters or [_parameter("value")],
            "return": "str",
        },
        "members": members or [],
        "deprecation": deprecation or deepcopy(_NOT_DEPRECATED),
    }


def _snapshot(version, exports):
    return {
        "schema_version": 1,
        "package": "experia",
        "package_version": version,
        "major_version": int(version.split(".", 1)[0]),
        "exports": exports,
    }


def _active_deprecation(*, since="0.6.0"):
    replacement = "experia.Replacement"
    return {
        "is_deprecated": True,
        "message": f"Legacy API is deprecated; use {replacement} instead.",
        "replacement": replacement,
        "since": since,
    }


def test_semantic_comparison_permits_reordering_and_additive_changes():
    baseline = _snapshot(
        "0.7.0",
        [
            _export(
                parameters=[
                    _parameter("value"),
                    _parameter(
                        "limit",
                        annotation="int",
                        required=False,
                        default={"kind": "value", "value": 10},
                    ),
                ],
                members=[
                    {
                        "name": "run",
                        "kind": "method",
                        "async": False,
                        "signature": {"parameters": [], "return": "str"},
                        "deprecation": deepcopy(_NOT_DEPRECATED),
                    }
                ],
            )
        ],
    )
    candidate_export = deepcopy(baseline["exports"][0])
    candidate_export["signature"]["parameters"].append(
        _parameter(
            "timeout",
            kind="keyword_only",
            annotation="float | None",
            required=False,
            default={"kind": "value", "value": None},
        )
    )
    candidate_export["members"].insert(
        0,
        {
            "name": "new_method",
            "kind": "method",
            "async": False,
            "signature": {"parameters": [], "return": "None"},
            "deprecation": deepcopy(_NOT_DEPRECATED),
        },
    )
    candidate = _snapshot(
        "0.8.0",
        [_export("experia.Added"), candidate_export],
    )

    report = compare_snapshots(baseline, candidate)

    assert report.compatible
    assert report.issues == ()
    assert_snapshots_compatible(baseline, candidate)


@pytest.mark.parametrize(
    ("mutate", "expected_code"),
    [
        (lambda export: export["signature"]["parameters"].clear(), "parameter_removed"),
        (
            lambda export: export["signature"]["parameters"].append(
                _parameter("required_addition")
            ),
            "required_parameter_added",
        ),
        (
            lambda export: export["signature"]["parameters"][0].update(
                annotation="str"
            ),
            "parameter_type_narrowed",
        ),
        (
            lambda export: export["signature"]["parameters"][0].update(
                required=True,
                default={"kind": "required"},
            ),
            "parameter_became_required",
        ),
    ],
)
def test_same_major_rejects_narrowed_signatures(mutate, expected_code):
    baseline_export = _export(
        parameters=[
            _parameter(
                "value",
                annotation="int | str",
                required=False,
                default={"kind": "value", "value": "all"},
            )
        ]
    )
    baseline = _snapshot("0.7.0", [baseline_export])
    candidate = deepcopy(baseline)
    candidate["package_version"] = "0.8.0"
    mutate(candidate["exports"][0])

    report = compare_snapshots(baseline, candidate)

    assert not report.compatible
    assert expected_code in {issue.code for issue in report.issues}
    with pytest.raises(SnapshotCompatibilityError, match=expected_code):
        assert_snapshots_compatible(baseline, candidate)


def test_same_major_rejects_export_and_member_removals():
    baseline_export = _export(
        members=[
            {
                "name": "run",
                "kind": "method",
                "async": False,
                "signature": {"parameters": [], "return": "str"},
                "deprecation": deepcopy(_NOT_DEPRECATED),
            }
        ]
    )
    baseline = _snapshot("0.7.0", [baseline_export, _export("experia.Other")])
    candidate_export = deepcopy(baseline_export)
    candidate_export["members"] = []
    candidate = _snapshot("0.8.0", [candidate_export])

    report = compare_snapshots(baseline, candidate)

    assert {issue.code for issue in report.issues} == {
        "export_removed",
        "member_removed",
    }


def test_greater_major_can_break_non_deprecated_api():
    baseline = _snapshot("0.7.0", [_export()])
    candidate = _snapshot("1.0.0", [])

    assert compare_snapshots(baseline, candidate).compatible


def test_deprecated_api_cannot_be_removed_before_two_minor_releases():
    baseline = _snapshot(
        "0.7.0",
        [_export(deprecation=_active_deprecation(since="0.6.0"))],
    )
    candidate = _snapshot("1.0.0", [])

    report = compare_snapshots(baseline, candidate)

    assert not report.compatible
    assert [issue.code for issue in report.issues] == ["deprecation_window_incomplete"]
    assert deprecation_window_elapsed("0.6.0", "0.7.9") is False
    assert deprecation_window_elapsed("0.6.0", "0.8.0") is True


def test_deprecated_api_may_be_removed_in_next_major_after_window():
    baseline = _snapshot(
        "0.8.0",
        [_export(deprecation=_active_deprecation(since="0.6.0"))],
    )
    candidate = _snapshot("1.0.0", [])

    assert compare_snapshots(baseline, candidate).compatible


def test_deprecation_metadata_requires_replacement_bearing_message():
    deprecation = _active_deprecation()
    deprecation["message"] = "Legacy API is deprecated."
    candidate = _snapshot("0.7.0", [_export(deprecation=deprecation)])

    report = compare_snapshots(_snapshot("0.7.0", []), candidate)

    assert [issue.code for issue in report.issues] == [
        "deprecation_message_missing_replacement"
    ]


def test_deprecated_sync_callable_warns_at_caller_and_remains_callable():
    @deprecated(since="0.6.0", replacement="experia.new_api")
    def legacy(value: int) -> int:
        return value + 1

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        call_line = inspect.currentframe().f_lineno + 1
        assert legacy(2) == 3

    assert len(caught) == 1
    assert caught[0].category is DeprecationWarning
    assert "experia.new_api" in str(caught[0].message)
    assert caught[0].filename == __file__
    assert caught[0].lineno == call_line
    assert legacy.__deprecated_since__ == "0.6.0"
    assert legacy.__deprecated_replacement__ == "experia.new_api"


@pytest.mark.asyncio
async def test_deprecated_async_callable_preserves_async_and_warning_behavior():
    @deprecated(
        since="0.6.0",
        replacement="experia.new_async_api",
        message="The legacy coroutine is deprecated.",
    )
    async def legacy(value: int) -> int:
        return value * 2

    assert inspect.iscoroutinefunction(legacy)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        call_line = inspect.currentframe().f_lineno + 1
        assert await legacy(3) == 6

    assert len(caught) == 1
    assert "experia.new_async_api" in str(caught[0].message)
    assert caught[0].filename == __file__
    assert caught[0].lineno == call_line


def test_warn_deprecated_uses_replacement_and_stacklevel_two():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        call_line = inspect.currentframe().f_lineno + 1
        warn_deprecated(
            "experia.legacy",
            since="0.6.0",
            replacement="experia.replacement",
        )

    assert len(caught) == 1
    assert "experia.replacement" in str(caught[0].message)
    assert caught[0].filename == __file__
    assert caught[0].lineno == call_line


def test_deprecated_class_remains_a_class_with_its_constructor_signature():
    @deprecated(since="0.6.0", replacement="experia.Replacement")
    class Legacy:
        def __init__(self, value: int) -> None:
            self.value = value

    assert inspect.isclass(Legacy)
    assert str(inspect.signature(Legacy)) == "(value: int) -> None"
    with pytest.warns(DeprecationWarning, match="experia.Replacement"):
        instance = Legacy(4)
    assert instance.value == 4


def test_deprecation_policy_rejects_invalid_release_or_replacement():
    with pytest.raises(ValueError, match="MAJOR.MINOR.PATCH"):
        deprecated(since="0.6", replacement="experia.new_api")
    with pytest.raises(ValueError, match="replacement"):
        deprecated(since="0.6.0", replacement="")


@pytest.mark.parametrize("current_version", ["0.6.0", "0.7.0"])
def test_deprecated_api_remains_callable_through_each_protected_minor_release(
    current_version,
):
    calls = []

    @deprecated(
        since="0.6.0",
        replacement="experia.current_api",
        message="Legacy endpoint is deprecated.",
    )
    def legacy(value):
        calls.append(value)
        return f"handled:{value}"

    assert not deprecation_window_elapsed("0.6.0", current_version)
    with pytest.warns(DeprecationWarning) as caught:
        assert legacy(current_version) == f"handled:{current_version}"

    assert calls == [current_version]
    assert str(caught[0].message) == (
        "Legacy endpoint is deprecated. Use experia.current_api instead."
    )
