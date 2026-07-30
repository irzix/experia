"""Focused tests for reusable build/install/smoke artifact tooling."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.artifact_gate import (
    ArtifactGateError,
    ArtifactRecord,
    SmokeResult,
    build_manifest,
    inspect_artifacts,
    validate_smoke_result,
)


def test_artifact_inspection_emits_exact_names_hashes_and_sizes(tmp_path: Path):
    wheel = tmp_path / "experia-0.7.0-py3-none-any.whl"
    sdist = tmp_path / "experia-0.7.0.tar.gz"
    wheel.write_bytes(b"wheel bytes")
    sdist.write_bytes(b"sdist bytes")

    records = inspect_artifacts(tmp_path)

    assert records == (
        ArtifactRecord(
            kind="wheel",
            name=wheel.name,
            sha256=hashlib.sha256(b"wheel bytes").hexdigest(),
            size_bytes=len(b"wheel bytes"),
        ),
        ArtifactRecord(
            kind="sdist",
            name=sdist.name,
            sha256=hashlib.sha256(b"sdist bytes").hexdigest(),
            size_bytes=len(b"sdist bytes"),
        ),
    )


@pytest.mark.parametrize(
    "artifact_names",
    [
        ("experia-0.7.0-py3-none-any.whl",),
        (
            "experia-0.7.0-py3-none-any.whl",
            "experia-0.7.0-py3-none-linux.whl",
            "experia-0.7.0.tar.gz",
        ),
    ],
)
def test_artifact_inspection_rejects_missing_or_duplicate_kinds(
    tmp_path: Path, artifact_names: tuple[str, ...]
):
    for artifact_name in artifact_names:
        (tmp_path / artifact_name).write_bytes(b"artifact")

    with pytest.raises(
        ArtifactGateError, match="Expected exactly one wheel and one sdist"
    ):
        inspect_artifacts(tmp_path)


def test_smoke_evidence_requires_import_and_working_directory_outside_repository(
    tmp_path: Path,
):
    repository = tmp_path / "repository"
    repository.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    payload = {
        "status": "passed",
        "package": "experia",
        "package_version": "0.7.0",
        "python_version": "3.10.0",
        "import_origin": str(outside / "site-packages" / "experia" / "__init__.py"),
        "working_directory": str(outside),
        "checks": [
            "clean-virtual-environment",
            "import-outside-source-tree",
            "sqlite-memory-round-trip",
            "idempotent-close",
        ],
    }

    result = validate_smoke_result(payload, repository)
    manifest = build_manifest(
        (
            ArtifactRecord("wheel", "experia.whl", "a" * 64, 10),
            ArtifactRecord("sdist", "experia.tar.gz", "b" * 64, 20),
        ),
        result,
    )

    assert isinstance(result, SmokeResult)
    assert manifest["artifacts"][0] == {
        "kind": "wheel",
        "name": "experia.whl",
        "sha256": "a" * 64,
        "size_bytes": 10,
    }
    assert manifest["smoke"]["import_origin_outside_repository"] is True
    assert manifest["smoke"]["working_directory_outside_repository"] is True

    payload["import_origin"] = str(repository / "experia" / "__init__.py")
    with pytest.raises(ArtifactGateError, match="inside the source repository"):
        validate_smoke_result(payload, repository)
