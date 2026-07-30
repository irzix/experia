"""Credential-free policy tests for the required CI Quality Gate.

Validates Requirements 7.1-7.7 and 9.6.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPOSITORY_ROOT / ".github" / "workflows" / "ci.yml"
CONTRIBUTING_PATH = REPOSITORY_ROOT / "CONTRIBUTING.md"
SUPPORTED_PYTHONS = ("3.10", "3.11", "3.12")
QUALITY_COMMANDS = (
    "python -m ruff check .",
    "python -m ruff format --check .",
    "python -m pytest",
    "python scripts/coverage_gate.py .coverage.granular.json",
    "python scripts/artifact_gate.py --output-dir dist/quality-gate",
)
SETUP_COMMANDS = (
    "python -m pip install --upgrade pip",
    'python -m pip install -e ".[dev]"',
)
# Substrings that would indicate a documented command depends on a credential
# or secret, which Requirement 9.6 forbids for the fresh-checkout flow.
CREDENTIAL_MARKERS = (
    "token",
    "password",
    "passwd",
    "api_key",
    "api-key",
    "apikey",
    "secret",
    "credential",
    "bearer",
    "@github.com",
)


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
        f"expected one CI step named {name!r}, found {len(matches)}"
    )
    return matches[0]


def _run_text(job: Mapping[str, Any]) -> str:
    return "\n".join(str(step["run"]) for step in _steps(job) if "run" in step)


def _documented_quality_commands() -> tuple[str, ...]:
    document = CONTRIBUTING_PATH.read_text(encoding="utf-8")
    start = "<!-- BEGIN QUALITY GATE COMMANDS -->"
    end = "<!-- END QUALITY GATE COMMANDS -->"
    assert document.count(start) == document.count(end) == 1
    section = document.split(start, 1)[1].split(end, 1)[0].strip()
    assert section.startswith("```bash\n") and section.endswith("\n```")
    return tuple(
        line.strip()
        for line in section[len("```bash\n") : -len("\n```")].splitlines()
        if line.strip()
    )


def test_matrix_has_exact_supported_python_coverage_and_required_commands() -> None:
    workflow = _load_workflow()
    matrix_job = workflow["jobs"]["matrix"]

    assert workflow["on"]["pull_request"]["branches"] == ["main"]
    assert tuple(matrix_job["strategy"]["matrix"]["python-version"]) == (
        SUPPORTED_PYTHONS
    )
    assert matrix_job["strategy"]["fail-fast"] == "false"

    run_text = _run_text(matrix_job)
    for command in QUALITY_COMMANDS:
        assert command in run_text

    required_steps = (
        "Lint (ruff check)",
        "Format (ruff format --check)",
        "Full tests and coverage collection",
        "Granular coverage (85% per target)",
        "Build wheel and sdist once for Python ${{ matrix.python-version }}",
    )
    for step_name in required_steps:
        step = _step(matrix_job, step_name)
        assert "if" not in step
        assert step.get("continue-on-error") != "true"


def test_contributor_quality_commands_have_exact_ci_parity() -> None:
    matrix_job = _load_workflow()["jobs"]["matrix"]
    documented_commands = _documented_quality_commands()

    assert documented_commands == QUALITY_COMMANDS
    run_text = _run_text(matrix_job)
    assert all(command in run_text for command in documented_commands)


def test_aggregate_gate_requires_every_non_bypassable_job() -> None:
    workflow = _load_workflow()
    jobs = workflow["jobs"]
    artifact_job = jobs["artifact-gate"]
    aggregate_job = jobs["quality-gate"]

    assert artifact_job["needs"] == "matrix"
    assert aggregate_job["name"] == "Quality Gate"
    assert aggregate_job["if"] == "${{ always() }}"
    assert tuple(aggregate_job["needs"]) == ("matrix", "artifact-gate")
    for required_job in (jobs["matrix"], artifact_job, aggregate_job):
        assert required_job.get("continue-on-error") != "true"
        assert all(
            step.get("continue-on-error") != "true" for step in _steps(required_job)
        )

    status_step = _step(aggregate_job, "Require every observed Quality Gate check")
    assert status_step["env"] == {
        "MATRIX_RESULT": "${{ needs.matrix.result }}",
        "ARTIFACT_RESULT": "${{ needs.artifact-gate.result }}",
    }
    status_script = str(status_step["run"])
    assert '"${MATRIX_RESULT}" != "success"' in status_script
    assert '"${ARTIFACT_RESULT}" != "success"' in status_script
    assert "status=failed" in status_script
    assert "exit 1" in status_script

    outputs = workflow["on"]["workflow_call"]["outputs"]
    assert outputs["quality-gate-status"]["value"] == (
        "${{ jobs.quality-gate.outputs.status }}"
    )
    assert outputs["artifact-name"]["value"] == (
        "${{ jobs.quality-gate.outputs.artifact-name }}"
    )


def test_downstream_gate_reuses_uploaded_tested_artifacts_without_rebuild() -> None:
    jobs = _load_workflow()["jobs"]
    matrix_job = jobs["matrix"]
    artifact_job = jobs["artifact-gate"]
    upload = _step(matrix_job, "Upload tested Python 3.12 artifacts")
    download = _step(artifact_job, "Download tested artifacts")

    assert upload["if"] == "${{ matrix.python-version == '3.12' }}"
    assert upload["uses"] == "actions/upload-artifact@v4"
    assert download["uses"] == "actions/download-artifact@v4"
    assert upload["with"]["name"] == download["with"]["name"]
    assert upload["with"]["path"] == "dist/quality-gate/*"
    assert download["with"]["path"] == "dist/quality-gate"

    downstream_commands = _run_text(artifact_job)
    forbidden_rebuild_commands = (
        "python -m build",
        "python scripts/artifact_gate.py",
        "pip wheel",
        "hatch build",
    )
    assert all(
        command not in downstream_commands for command in forbidden_rebuild_commands
    )

    for step_name in (
        "Inspect tested artifact identity",
        "Smoke-test the downloaded wheel in a clean environment",
        "Execute documented quickstart against the downloaded wheel",
        "Run installed API, extras, and example gates on the downloaded wheel",
    ):
        assert "dist/quality-gate" in str(_step(artifact_job, step_name)["run"])


def test_required_failures_have_named_observed_diagnostics() -> None:
    workflow = _load_workflow()
    normalized = " ".join(_run_text(job) for job in workflow["jobs"].values()).replace(
        "\n", " "
    )
    diagnostic_names = (
        "lint",
        "format",
        "full-tests",
        "granular-coverage",
        "build-artifacts[",
        "artifact-manifest",
        "artifact-identity",
        "wheel-smoke",
        "documentation-source",
        "documentation-wheel",
        "artifact-api-gate",
        "quality-gate",
    )

    for diagnostic_name in diagnostic_names:
        marker = f"FAIL {diagnostic_name}"
        position = normalized.find(marker)
        assert position >= 0, f"missing named failure diagnostic {marker!r}"
        assert "observed" in normalized[position : position + 500]


def _delimited_bash_commands(start: str, end: str) -> tuple[str, ...]:
    document = CONTRIBUTING_PATH.read_text(encoding="utf-8")
    assert document.count(start) == document.count(end) == 1
    section = document.split(start, 1)[1].split(end, 1)[0].strip()
    assert section.startswith("```bash\n") and section.endswith("\n```")
    return tuple(
        line.strip()
        for line in section[len("```bash\n") : -len("\n```")].splitlines()
        if line.strip()
    )


def _documented_setup_commands() -> tuple[str, ...]:
    return _delimited_bash_commands(
        "<!-- BEGIN SETUP COMMANDS -->", "<!-- END SETUP COMMANDS -->"
    )


def _contributor_bash_blocks() -> tuple[str, ...]:
    document = CONTRIBUTING_PATH.read_text(encoding="utf-8")
    blocks: list[str] = []
    remaining = document
    fence = "```bash\n"
    while fence in remaining:
        _, remaining = remaining.split(fence, 1)
        body, remaining = remaining.split("```", 1)
        blocks.append(body)
    return tuple(blocks)


def test_contributor_setup_commands_have_exact_ci_parity() -> None:
    matrix_job = _load_workflow()["jobs"]["matrix"]
    documented_setup = _documented_setup_commands()

    # The documented fresh-checkout setup is exactly what CI runs, so a newcomer
    # following CONTRIBUTING.md reproduces the environment with no hidden steps.
    assert documented_setup == SETUP_COMMANDS

    setup_step = _step(matrix_job, "Set up the documented fresh-checkout environment")
    assert "if" not in setup_step
    assert setup_step.get("continue-on-error") != "true"
    setup_lines = tuple(
        line.strip() for line in str(setup_step["run"]).splitlines() if line.strip()
    )
    assert setup_lines == documented_setup

    run_text = _run_text(matrix_job)
    assert all(command in run_text for command in documented_setup)


def test_documented_contributor_commands_are_credential_free() -> None:
    documented = "\n".join(_contributor_bash_blocks()).lower()
    assert documented, "expected at least one documented bash command block"

    for marker in CREDENTIAL_MARKERS:
        assert marker not in documented, (
            f"documented contributor command references credential marker {marker!r}"
        )

    # Every documented setup and quality command must be reproducible from a
    # fresh checkout across each supported Python version with no credentials.
    for command in SETUP_COMMANDS + QUALITY_COMMANDS:
        assert command in documented or command.lower() in documented
