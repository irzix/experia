"""Focused tests for reusable API, extras, and installed-example gates."""

from copy import deepcopy
from pathlib import Path

import pytest

from scripts.api_gate import (
    GateFailure,
    _install_requirement,
    check_snapshot,
    load_baseline,
    load_example_manifest,
    require_external_workspace,
    run_source_gate,
)
from scripts.generate_api_snapshot import build_snapshot

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "api-snapshot.json"
API_REFERENCE = ROOT / "API_REFERENCE.md"
MANIFEST = ROOT / "examples" / "installed-examples.json"


def test_source_gate_preserves_baseline_and_validates_documented_extras():
    before = BASELINE.read_bytes()

    messages = run_source_gate(
        baseline_path=BASELINE,
        api_reference=API_REFERENCE,
        manifest_path=MANIFEST,
        project_root=ROOT,
    )

    assert BASELINE.read_bytes() == before
    assert [message.split(":", 1)[0] for message in messages] == [
        "PASS api-baseline",
        "PASS api-reference",
        "PASS example-manifest",
        "PASS documented-extras",
    ]


def test_manifest_covers_base_and_every_documented_optional_extra():
    manifest = load_example_manifest(MANIFEST, project_root=ROOT)

    assert manifest.extras == ("langchain", "langgraph", "llm")
    assert {example.extras for example in manifest.examples} == {
        (),
        ("langchain",),
        ("langgraph",),
        ("llm",),
    }
    assert all(example.path.is_file() for example in manifest.examples)


def test_same_version_addition_fails_without_rewriting_the_baseline():
    baseline = load_baseline(BASELINE)
    candidate = deepcopy(build_snapshot())
    added = deepcopy(candidate["exports"][0])
    added["path"] = "experia.UnreviewedAddition"
    candidate["exports"].append(added)

    with pytest.raises(GateFailure, match="same package version") as caught:
        check_snapshot(baseline, candidate, installed=False)

    assert "experia.UnreviewedAddition" in str(caught.value)


def test_incompatible_signature_failure_names_the_path_and_reason():
    baseline = load_baseline(BASELINE)
    candidate = deepcopy(build_snapshot())
    learner = next(
        item for item in candidate["exports"] if item["path"] == "experia.Learner"
    )
    learner["signature"]["parameters"] = [
        parameter
        for parameter in learner["signature"]["parameters"]
        if parameter["name"] != "evaluator"
    ]

    with pytest.raises(GateFailure, match="parameter_removed") as caught:
        check_snapshot(baseline, candidate, installed=True)

    assert "experia.Learner(evaluator)" in str(caught.value)


def test_wheel_requirement_selects_only_the_example_declared_extras(tmp_path):
    wheel = tmp_path / "experia-0.7.0-py3-none-any.whl"
    wheel.touch()

    assert _install_requirement("experia", wheel, ()) == (
        f"experia @ {wheel.resolve().as_uri()}"
    )
    assert _install_requirement("experia", wheel, ("langchain",)) == (
        f"experia[langchain] @ {wheel.resolve().as_uri()}"
    )


def test_installed_gate_requires_workspace_outside_source_tree(tmp_path):
    assert require_external_workspace(tmp_path, ROOT) == tmp_path.resolve()

    with pytest.raises(GateFailure, match="execution-isolation") as caught:
        require_external_workspace(ROOT / "dist" / "installed-gate", ROOT)

    assert "workspace must be outside source tree" in str(caught.value)
    assert str(ROOT) in str(caught.value)
