import json
from pathlib import Path

from scripts.generate_api_snapshot import (
    ExportSpec,
    build_snapshot,
    canonical_json,
    supported_export_specs,
)

BASELINE_PATH = Path(__file__).resolve().parents[1] / "api-snapshot.json"


def _exports_by_path(snapshot):
    return {export["path"]: export for export in snapshot["exports"]}


def _parameters_by_name(signature):
    return {parameter["name"]: parameter for parameter in signature["parameters"]}


def test_baseline_is_the_canonical_snapshot_for_the_current_package():
    generated = build_snapshot()
    baseline_text = BASELINE_PATH.read_text(encoding="utf-8")

    assert baseline_text == canonical_json(generated)
    assert json.loads(baseline_text) == generated
    assert [export["path"] for export in generated["exports"]] == sorted(
        export["path"] for export in generated["exports"]
    )


def test_learner_evaluator_remains_required_in_current_major_baseline():
    exports = _exports_by_path(build_snapshot())
    learner = exports["experia.Learner"]
    parameters = _parameters_by_name(learner["signature"])

    assert parameters["store"]["default"] == {"kind": "required"}
    assert parameters["evaluator"]["required"] is True
    assert parameters["evaluator"]["default"] == {"kind": "required"}
    assert parameters["rule_generator"]["required"] is False
    assert parameters["rule_generator"]["default"] == {
        "kind": "value",
        "value": None,
    }


def test_task_1_1_errors_are_captured_through_all_supported_namespaces():
    paths = set(_exports_by_path(build_snapshot()))
    error_names = {
        "ConfigurationError",
        "EvaluationError",
        "EvaluationFailure",
        "ExperiaError",
        "FailureDetail",
        "LifecycleError",
        "SanitizationError",
        "StorageError",
        "UnavailableFeatureError",
    }

    for name in error_names:
        assert f"experia.{name}" in paths
        assert f"experia.core.{name}" in paths
        assert f"experia.core.exceptions.{name}" in paths


def test_snapshot_records_async_defaults_enums_and_deprecation_metadata():
    exports = _exports_by_path(build_snapshot())
    learner_members = {
        member["name"]: member for member in exports["experia.Learner"]["members"]
    }
    sqlite_parameters = _parameters_by_name(exports["experia.SQLiteStore"]["signature"])
    memory_type = exports["experia.MemoryType"]

    assert learner_members["record"]["async"] is True
    assert learner_members["record"]["deprecation"] == {
        "is_deprecated": False,
        "message": None,
        "replacement": None,
        "since": None,
    }
    assert sqlite_parameters["db_path"]["default"] == {
        "kind": "value",
        "value": "experia.db",
    }
    assert memory_type["kind"] == "enum"
    assert memory_type["values"] == [
        {"name": "FACT", "value": "fact"},
        {"name": "PREFERENCE", "value": "preference"},
        {"name": "LESSON", "value": "lesson"},
        {"name": "RULE", "value": "rule"},
        {"name": "STRATEGY", "value": "strategy"},
        {"name": "EXPERIENCE", "value": "experience"},
    ]


def test_canonical_json_is_independent_of_export_spec_input_order():
    specs = supported_export_specs()[:4]
    reversed_specs = tuple(reversed(specs))

    first = build_snapshot(specs, package_version="0.7.0")
    second = build_snapshot(reversed_specs, package_version="0.7.0")

    assert canonical_json(first) == canonical_json(second)


def test_a_minimal_alias_snapshot_records_the_canonical_target():
    snapshot = build_snapshot(
        (ExportSpec("experia.Learner", ("record",)),),
        package_version="0.7.0",
    )
    export = snapshot["exports"][0]

    assert export["path"] == "experia.Learner"
    assert export["target"] == "experia.core.learner.Learner"
    assert export["members"][0]["name"] == "record"
