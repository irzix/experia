"""Credential-free policy tests for the reproducible, provenance-attested release.

Validates Requirements 7.7 and 8.2-8.5, 8.9: the publish workflow must require a
passing reusable Quality Gate, reuse the tested wheel/sdist bytes without
rebuilding, verify count and SHA-256 identity, run the release identity and
artifact-metadata validators as publish-blocking gates, attach source/tag/workflow
provenance before publication, and upload exactly those tested bytes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPOSITORY_ROOT / ".github" / "workflows" / "publish.yml"
REUSABLE_QUALITY_GATE = "./.github/workflows/ci.yml"


def _load_workflow() -> Mapping[str, Any]:
    # BaseLoader keeps GitHub's `on` key and every scalar as text while still
    # parsing YAML structure; it cannot construct arbitrary Python objects.
    parsed = yaml.load(
        WORKFLOW_PATH.read_text(encoding="utf-8"), Loader=yaml.BaseLoader
    )
    assert isinstance(parsed, Mapping)
    return parsed


def _steps(job: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    steps = job.get("steps")
    assert isinstance(steps, Sequence) and not isinstance(steps, (str, bytes))
    return steps


def _step(job: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    matches = [step for step in _steps(job) if step.get("name") == name]
    assert len(matches) == 1, (
        f"expected one publish step named {name!r}, found {len(matches)}"
    )
    return matches[0]


def _step_index(job: Mapping[str, Any], name: str) -> int:
    for index, step in enumerate(_steps(job)):
        if step.get("name") == name:
            return index
    raise AssertionError(f"missing publish step named {name!r}")


def _run_text(job: Mapping[str, Any]) -> str:
    return "\n".join(str(step["run"]) for step in _steps(job) if "run" in step)


def test_publish_triggers_only_on_version_tags() -> None:
    workflow = _load_workflow()
    assert workflow["on"]["push"]["tags"] == ["v*.*.*"]
    # Least privilege at the workflow root.
    assert workflow["permissions"] == {"contents": "read"}


def test_publish_requires_the_reusable_quality_gate() -> None:
    jobs = _load_workflow()["jobs"]
    quality_gate = jobs["quality-gate"]
    publish = jobs["publish"]

    assert quality_gate["uses"] == REUSABLE_QUALITY_GATE
    assert publish["needs"] == "quality-gate"

    # An explicit publish-blocking check on the reusable gate's status output.
    guard = _step(publish, "Require a passing reusable Quality Gate")
    script = str(guard["run"])
    assert '"${QUALITY_GATE_STATUS}" != "passed"' in script
    assert "exit 1" in script
    assert publish["env"]["QUALITY_GATE_STATUS"] == (
        "${{ needs.quality-gate.outputs.quality-gate-status }}"
    )
    assert publish["env"]["TESTED_ARTIFACT_NAME"] == (
        "${{ needs.quality-gate.outputs.artifact-name }}"
    )


def test_publish_reuses_tested_bytes_without_rebuilding() -> None:
    publish = _load_workflow()["jobs"]["publish"]
    download = _step(
        publish, "Download the tested wheel and sdist from the Quality Gate"
    )

    assert download["uses"] == "actions/download-artifact@v4"
    assert download["with"]["name"] == (
        "${{ needs.quality-gate.outputs.artifact-name }}"
    )
    assert download["with"]["path"] == "dist/quality-gate"

    run_text = _run_text(publish)
    forbidden_rebuild_commands = (
        "python -m build",
        "python scripts/artifact_gate.py",
        "pip wheel",
        "hatch build",
    )
    assert all(command not in run_text for command in forbidden_rebuild_commands)


def test_publish_verifies_count_and_sha256_identity() -> None:
    publish = _load_workflow()["jobs"]["publish"]
    verify = _step(
        publish, "Verify tested artifact count and SHA-256 identity without rebuilding"
    )
    script = str(verify["run"])
    assert "artifact-manifest.json" in script
    assert "inspect_artifacts" in script
    assert "hashlib.sha256" in script
    assert "FAIL artifact-count" in script
    assert "FAIL artifact-identity" in script


def test_publish_runs_release_identity_and_metadata_gates() -> None:
    publish = _load_workflow()["jobs"]["publish"]

    identity = _step(
        publish, "Validate release tag, project version, and changelog identity"
    )
    identity_script = str(identity["run"])
    assert "scripts/release_identity.py" in identity_script
    assert "--release-date" in identity_script
    assert "FAIL release-identity" in identity_script

    metadata = _step(
        publish, "Inspect release metadata, README, license, and wheel contents"
    )
    metadata_script = str(metadata["run"])
    assert "scripts/release_artifacts.py" in metadata_script
    assert "FAIL release-artifact-metadata" in metadata_script

    # Both gates must run before publication.
    publish_index = _step_index(publish, "Publish the attested tested bytes to PyPI")
    assert (
        _step_index(
            publish, "Validate release tag, project version, and changelog identity"
        )
        < publish_index
    )
    assert (
        _step_index(
            publish, "Inspect release metadata, README, license, and wheel contents"
        )
        < publish_index
    )


def test_publish_attaches_provenance_before_publishing_exact_bytes() -> None:
    publish = _load_workflow()["jobs"]["publish"]

    assert publish["permissions"]["id-token"] == "write"
    assert publish["permissions"]["attestations"] == "write"

    attest = _step(
        publish, "Attest source, tag, and workflow provenance for the published bytes"
    )
    assert attest["uses"].startswith("actions/attest-build-provenance@")
    subjects = str(attest["with"]["subject-path"])
    assert "dist/publish/*.whl" in subjects
    assert "dist/publish/*.tar.gz" in subjects

    stage = _step(publish, "Stage the exact tested bytes for publication")
    stage_script = str(stage["run"])
    assert "dist/publish" in stage_script
    assert "FAIL publish-bytes" in stage_script

    publish_step = _step(publish, "Publish the attested tested bytes to PyPI")
    assert publish_step["uses"].startswith("pypa/gh-action-pypi-publish@")
    assert publish_step["with"]["packages-dir"] == "dist/publish"

    # Provenance is attached before the exact tested bytes are published.
    attest_index = _step_index(
        publish, "Attest source, tag, and workflow provenance for the published bytes"
    )
    provenance_index = _step_index(
        publish, "Record reproducible release provenance inputs"
    )
    publish_index = _step_index(publish, "Publish the attested tested bytes to PyPI")
    assert provenance_index < attest_index < publish_index


def test_provenance_inputs_identify_source_tag_and_run() -> None:
    publish = _load_workflow()["jobs"]["publish"]
    provenance = _step(publish, "Record reproducible release provenance inputs")
    script = str(provenance["run"])
    assert "source_revision" in script
    assert "release_tag" in script
    assert "workflow_run" in script

    env = publish["env"]
    assert env["SOURCE_REVISION"] == "${{ github.sha }}"
    assert env["RELEASE_TAG"] == "${{ github.ref_name }}"
    assert env["WORKFLOW_RUN"] == "${{ github.run_id }}"


def test_publish_failures_have_named_observed_diagnostics() -> None:
    publish = _load_workflow()["jobs"]["publish"]
    normalized = _run_text(publish).replace("\n", " ")
    diagnostic_names = (
        "quality-gate-dependency",
        "release-identity",
        "artifact-count",
        "artifact-identity",
        "quality-gate-smoke",
        "release-artifact-metadata",
        "publish-bytes",
        "provenance-inputs",
    )
    for diagnostic_name in diagnostic_names:
        marker = f"FAIL {diagnostic_name}"
        position = normalized.find(marker)
        assert position >= 0, f"missing named failure diagnostic {marker!r}"
        assert "observed" in normalized[position : position + 500]


def test_publish_job_steps_are_not_bypassable() -> None:
    publish = _load_workflow()["jobs"]["publish"]
    assert publish.get("continue-on-error") != "true"
    for step in _steps(publish):
        assert step.get("continue-on-error") != "true"
