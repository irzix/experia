"""Credential-free release pipeline dry run driving the real validators.

This exercises the publishing path end to end without ever publishing. It
reuses the exact validators the ``publish.yml`` workflow invokes
(``release_identity``, ``artifact_gate.inspect_artifacts``, and
``release_artifacts``) and mirrors the workflow's inline glue (quality-gate
dependency, artifact count/SHA-256 identity, exact-byte staging, and
provenance-input generation). Every stage stops the pipeline before the
publication step, so no bytes are uploaded, no credentials are read, and no
network access is required.

Validates Requirements 7.7 and 8.1-8.9: a release must fail closed on tag,
project-version, or changelog identity mismatch, artifact count or byte-identity
changes, untruthful metadata, a missing/failed Quality Gate, or incomplete
provenance inputs, and must otherwise produce the exact tested bytes plus
reproducible provenance inputs without publishing.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from scripts.artifact_gate import (
    ArtifactGateError,
    canonical_json,
    inspect_artifacts,
)
from scripts.release_artifacts import ReleaseArtifactError, write_inspection_manifest
from scripts.release_identity import (
    ReleaseIdentity,
    ReleaseIdentityError,
    validate_release_identity,
)

# The synthetic wheel/sdist builders already produce artifacts that satisfy the
# full metadata/README/license/wheel-content inspection; reuse them verbatim so
# the dry run drives the same inspection code the release pipeline runs.
from tests.test_release_artifacts import _artifacts, _metadata

RELEASE_VERSION = "0.7.0"
RELEASE_DATE = "2026-07-24"
DEFAULT_CHANGELOG = f"# Changelog\n\n## [{RELEASE_VERSION}] - {RELEASE_DATE}\n"


class DryRunBlocked(RuntimeError):
    """A precise, publication-blocking dry-run failure with a named stage."""

    def __init__(self, stage: str, detail: str) -> None:
        self.stage = stage
        self.detail = detail
        super().__init__(f"FAIL {stage}: {detail}")


class _Publisher:
    """A spy standing in for the PyPI upload; a dry run must never call it."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *args: object, **kwargs: object) -> None:  # pragma: no cover
        self.calls += 1


@dataclass(frozen=True)
class DryRunResult:
    """Evidence proving the pipeline reached publication readiness only."""

    identity: ReleaseIdentity
    artifact_identity: list[dict[str, Any]]
    inspection: dict[str, Any]
    staged: tuple[str, ...]
    provenance: dict[str, str]
    ready_to_publish: bool
    published: bool


def _require_quality_gate(status: str, artifact_name: str) -> None:
    """Mirror publish.yml's reusable Quality Gate dependency guard."""
    if status != "passed":
        raise DryRunBlocked(
            "quality-gate-dependency", f"quality-gate-status={status or 'missing'}"
        )
    if not artifact_name:
        raise DryRunBlocked("quality-gate-dependency", "tested-artifact-name=missing")


def _validate_identity(
    tag: str, release_date: str, pyproject_path: Path, changelog_path: Path
) -> ReleaseIdentity:
    """Run the real tag/version/changelog validator as a publish-blocking gate."""
    try:
        return validate_release_identity(
            tag,
            release_date,
            pyproject_path=pyproject_path,
            changelog_path=changelog_path,
        )
    except ReleaseIdentityError as error:
        raise DryRunBlocked("release-identity", str(error)) from error


def _verify_artifact_identity(
    directory: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Mirror publish.yml's count and SHA-256 identity verification."""
    manifest_path = directory / "artifact-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DryRunBlocked("artifact-manifest", f"unreadable manifest: {error}")

    try:
        observed = [record.as_dict() for record in inspect_artifacts(directory)]
    except ArtifactGateError as error:
        raise DryRunBlocked("artifact-count", str(error)) from error

    expected = manifest.get("artifacts")
    smoke_status = manifest.get("smoke", {}).get("status")
    if observed != expected:
        raise DryRunBlocked(
            "artifact-identity", f"observed={observed!r}, manifest={expected!r}"
        )
    if smoke_status != "passed":
        raise DryRunBlocked(
            "quality-gate-smoke", f"matrix_smoke_status={smoke_status!r}"
        )

    for record in expected:
        artifact = directory / record["name"]
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        size = artifact.stat().st_size
        if digest != record["sha256"] or size != record["size_bytes"]:
            raise DryRunBlocked(
                "artifact-identity",
                f"name={record['name']!r}, sha256={digest!r}, size={size!r}",
            )
    return manifest, expected


def _inspect_metadata(
    directory: Path, manifest_path: Path, policy_path: Path
) -> dict[str, Any]:
    """Run the real metadata/README/license/wheel-content inspection."""
    try:
        return write_inspection_manifest(
            directory, manifest_path, policy_path=policy_path
        )
    except ReleaseArtifactError as error:
        raise DryRunBlocked("release-artifact-metadata", str(error)) from error


def _stage_bytes(
    source: Path, target: Path, manifest: dict[str, Any]
) -> tuple[str, ...]:
    """Mirror publish.yml staging: copy only the two tested bytes, re-verify."""
    target.mkdir(parents=True, exist_ok=True)
    staged: list[str] = []
    for record in manifest["artifacts"]:
        destination = target / record["name"]
        shutil.copy2(source / record["name"], destination)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if digest != record["sha256"]:
            raise DryRunBlocked(
                "publish-bytes", f"name={record['name']!r}, sha256={digest!r}"
            )
        staged.append(destination.name)

    present = sorted(path.name for path in target.iterdir())
    if len(staged) != 2 or present != sorted(staged):
        raise DryRunBlocked("publish-bytes", f"staged={present!r}")
    return tuple(staged)


def _record_provenance(env: dict[str, str]) -> dict[str, str]:
    """Mirror publish.yml's reproducible provenance-input generation."""
    record = {
        "source_revision": env.get("SOURCE_REVISION", ""),
        "release_tag": env.get("RELEASE_TAG", ""),
        "workflow_run": env.get("WORKFLOW_RUN", ""),
        "repository": env.get("GITHUB_REPOSITORY", ""),
        "workflow_ref": env.get("GITHUB_WORKFLOW_REF", ""),
    }
    required = ("source_revision", "release_tag", "workflow_run")
    missing = sorted(name for name in required if not record[name])
    if missing:
        raise DryRunBlocked("provenance-inputs", f"missing={missing!r}")
    return record


def run_release_pipeline(
    *,
    tag: str,
    release_date: str,
    pyproject_path: Path,
    changelog_path: Path,
    artifact_directory: Path,
    policy_path: Path,
    quality_gate_status: str,
    tested_artifact_name: str,
    provenance_env: dict[str, str],
    publish_directory: Path,
    inspection_manifest_path: Path,
    publisher: _Publisher,
    dry_run: bool = True,
) -> DryRunResult:
    """Run every non-publishing release stage in order and stop before upload."""
    _require_quality_gate(quality_gate_status, tested_artifact_name)
    identity = _validate_identity(tag, release_date, pyproject_path, changelog_path)
    manifest, artifact_identity = _verify_artifact_identity(artifact_directory)
    inspection = _inspect_metadata(
        artifact_directory, inspection_manifest_path, policy_path
    )
    staged = _stage_bytes(artifact_directory, publish_directory, manifest)
    provenance = _record_provenance(provenance_env)

    published = False
    if not dry_run:  # Publication is intentionally unreachable in the dry run.
        publisher(publish_directory)
        published = True
    return DryRunResult(
        identity=identity,
        artifact_identity=artifact_identity,
        inspection=inspection,
        staged=staged,
        provenance=provenance,
        ready_to_publish=True,
        published=published,
    )


@dataclass
class _Fixture:
    """A prepared, all-passing release the tests can selectively corrupt."""

    tag: str
    release_date: str
    pyproject_path: Path
    changelog_path: Path
    artifact_directory: Path
    policy_path: Path
    quality_gate_status: str
    tested_artifact_name: str
    provenance_env: dict[str, str]
    publish_directory: Path
    inspection_manifest_path: Path
    publisher: _Publisher
    wheel: Path
    sdist: Path

    def run(self) -> DryRunResult:
        return run_release_pipeline(
            tag=self.tag,
            release_date=self.release_date,
            pyproject_path=self.pyproject_path,
            changelog_path=self.changelog_path,
            artifact_directory=self.artifact_directory,
            policy_path=self.policy_path,
            quality_gate_status=self.quality_gate_status,
            tested_artifact_name=self.tested_artifact_name,
            provenance_env=self.provenance_env,
            publish_directory=self.publish_directory,
            inspection_manifest_path=self.inspection_manifest_path,
            publisher=self.publisher,
        )

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        (self.artifact_directory / "artifact-manifest.json").write_text(
            canonical_json(manifest), encoding="utf-8"
        )

    def read_manifest(self) -> dict[str, Any]:
        return json.loads(
            (self.artifact_directory / "artifact-manifest.json").read_text(
                encoding="utf-8"
            )
        )


def _write_manifest(directory: Path) -> dict[str, Any]:
    """Record the tested wheel/sdist identity exactly like the Quality Gate."""
    records = inspect_artifacts(directory)
    manifest = {
        "schema_version": 1,
        "artifacts": [record.as_dict() for record in records],
        "smoke": {"status": "passed", "package": "experia"},
    }
    (directory / "artifact-manifest.json").write_text(
        canonical_json(manifest), encoding="utf-8"
    )
    return manifest


def _prepare_release(
    tmp_path: Path,
    *,
    project_version: str = RELEASE_VERSION,
    changelog: str | None = None,
    metadata: bytes | None = None,
    **artifact_kwargs: Any,
) -> _Fixture:
    artifact_directory = tmp_path / "dist" / "quality-gate"
    artifact_directory.mkdir(parents=True)
    wheel, sdist, policy = _artifacts(
        artifact_directory, metadata=metadata, **artifact_kwargs
    )
    _write_manifest(artifact_directory)

    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        f'[project]\nname = "experia"\nversion = "{project_version}"\n',
        encoding="utf-8",
    )
    changelog_path = tmp_path / "CHANGELOG.md"
    changelog_path.write_text(
        DEFAULT_CHANGELOG if changelog is None else changelog, encoding="utf-8"
    )

    return _Fixture(
        tag=f"v{RELEASE_VERSION}",
        release_date=RELEASE_DATE,
        pyproject_path=pyproject,
        changelog_path=changelog_path,
        artifact_directory=artifact_directory,
        policy_path=policy,
        quality_gate_status="passed",
        tested_artifact_name="tested-distributions",
        provenance_env={
            "SOURCE_REVISION": "a" * 40,
            "RELEASE_TAG": f"v{RELEASE_VERSION}",
            "WORKFLOW_RUN": "123456789",
            "GITHUB_REPOSITORY": "irzix/experia",
            "GITHUB_WORKFLOW_REF": (
                "irzix/experia/.github/workflows/publish.yml"
                f"@refs/tags/v{RELEASE_VERSION}"
            ),
        },
        publish_directory=tmp_path / "dist" / "publish",
        inspection_manifest_path=artifact_directory
        / "release-artifact-inspection.json",
        publisher=_Publisher(),
        wheel=wheel,
        sdist=sdist,
    )


def test_dry_run_reaches_publication_readiness_without_publishing(tmp_path: Path):
    fixture = _prepare_release(tmp_path)

    result = fixture.run()

    # The pipeline completed every validation but never published anything.
    assert result.ready_to_publish is True
    assert result.published is False
    assert fixture.publisher.calls == 0

    # Real validators produced consistent identity and metadata evidence.
    assert result.identity.version == RELEASE_VERSION
    assert result.identity.release_date == RELEASE_DATE
    assert result.inspection["package"]["version"] == RELEASE_VERSION
    assert result.inspection["artifact_identity"]["counts_by_kind"] == {
        "wheel": 1,
        "sdist": 1,
    }

    # Exactly the two tested bytes were staged locally (never uploaded).
    assert result.staged == (fixture.wheel.name, fixture.sdist.name)
    staged_on_disk = sorted(path.name for path in fixture.publish_directory.iterdir())
    assert staged_on_disk == sorted(result.staged)
    assert any(name.endswith(".whl") for name in result.staged)
    assert any(name.endswith(".tar.gz") for name in result.staged)

    # Provenance inputs were generated and identify the source, tag, and run.
    assert result.provenance["source_revision"] == "a" * 40
    assert result.provenance["release_tag"] == f"v{RELEASE_VERSION}"
    assert result.provenance["workflow_run"] == "123456789"


@pytest.mark.parametrize(
    ("status", "artifact_name"),
    [
        ("failed", "tested-distributions"),
        ("", "tested-distributions"),
        ("passed", ""),
    ],
)
def test_dry_run_blocks_when_quality_gate_is_missing_or_failed(
    tmp_path: Path, status: str, artifact_name: str
):
    fixture = _prepare_release(tmp_path)
    fixture.quality_gate_status = status
    fixture.tested_artifact_name = artifact_name

    with pytest.raises(DryRunBlocked) as caught:
        fixture.run()

    assert caught.value.stage == "quality-gate-dependency"
    assert fixture.publisher.calls == 0
    # It stopped before staging any bytes for publication.
    assert not fixture.publish_directory.exists()


def test_dry_run_blocks_on_non_exact_release_tag(tmp_path: Path):
    fixture = _prepare_release(tmp_path)
    fixture.tag = "0.7"  # not MAJOR.MINOR.PATCH

    with pytest.raises(DryRunBlocked) as caught:
        fixture.run()

    assert caught.value.stage == "release-identity"
    assert "tag-syntax" in caught.value.detail
    assert fixture.publisher.calls == 0


def test_dry_run_blocks_on_tag_project_version_mismatch(tmp_path: Path):
    fixture = _prepare_release(tmp_path, project_version="0.7.1")
    fixture.tag = f"v{RELEASE_VERSION}"  # tag says 0.7.0, project says 0.7.1

    with pytest.raises(DryRunBlocked) as caught:
        fixture.run()

    assert caught.value.stage == "release-identity"
    assert "tag-project-version-identity" in caught.value.detail
    assert fixture.publisher.calls == 0


def test_dry_run_blocks_on_changelog_date_mismatch(tmp_path: Path):
    fixture = _prepare_release(
        tmp_path,
        changelog=f"# Changelog\n\n## [{RELEASE_VERSION}] - 2026-07-25\n",
    )

    with pytest.raises(DryRunBlocked) as caught:
        fixture.run()

    assert caught.value.stage == "release-identity"
    assert "changelog-date-identity" in caught.value.detail
    assert fixture.publisher.calls == 0


def test_dry_run_blocks_when_artifact_count_changes(tmp_path: Path):
    fixture = _prepare_release(tmp_path)
    # A second wheel appears after the tested manifest was recorded.
    (fixture.artifact_directory / "experia-0.7.0-2-py3-none-any.whl").write_bytes(
        b"unexpected extra wheel"
    )

    with pytest.raises(DryRunBlocked) as caught:
        fixture.run()

    assert caught.value.stage == "artifact-count"
    assert fixture.publisher.calls == 0
    assert not fixture.publish_directory.exists()


def test_dry_run_blocks_when_artifact_bytes_change_after_testing(tmp_path: Path):
    fixture = _prepare_release(tmp_path)
    # Mutate the wheel bytes so its SHA-256 no longer matches the tested manifest.
    with zipfile.ZipFile(fixture.wheel, "a") as archive:
        archive.comment = b"byte identity changed after the Quality Gate"

    with pytest.raises(DryRunBlocked) as caught:
        fixture.run()

    assert caught.value.stage == "artifact-identity"
    assert fixture.publisher.calls == 0
    assert not fixture.publish_directory.exists()


def test_dry_run_blocks_when_manifest_smoke_status_is_not_passed(tmp_path: Path):
    fixture = _prepare_release(tmp_path)
    manifest = fixture.read_manifest()
    manifest["smoke"]["status"] = "failed"
    fixture.write_manifest(manifest)

    with pytest.raises(DryRunBlocked) as caught:
        fixture.run()

    assert caught.value.stage == "quality-gate-smoke"
    assert fixture.publisher.calls == 0


def test_dry_run_blocks_on_untruthful_release_metadata(tmp_path: Path):
    # Non-MIT license expression must be caught by the metadata inspection.
    fixture = _prepare_release(
        tmp_path, metadata=_metadata(license_expression="Apache-2.0")
    )

    with pytest.raises(DryRunBlocked) as caught:
        fixture.run()

    assert caught.value.stage == "release-artifact-metadata"
    assert "license-expression" in caught.value.detail
    assert fixture.publisher.calls == 0


def test_dry_run_blocks_when_provenance_inputs_are_incomplete(tmp_path: Path):
    fixture = _prepare_release(tmp_path)
    fixture.provenance_env = dict(fixture.provenance_env)
    fixture.provenance_env["WORKFLOW_RUN"] = ""  # incomplete provenance

    with pytest.raises(DryRunBlocked) as caught:
        fixture.run()

    assert caught.value.stage == "provenance-inputs"
    assert "workflow_run" in caught.value.detail
    assert fixture.publisher.calls == 0
