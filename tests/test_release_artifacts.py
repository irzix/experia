"""Focused tests for credential-free release artifact inspection."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import tarfile
import zipfile
from pathlib import Path
from typing import Callable

import pytest

from scripts.release_artifacts import (
    ReleaseArtifactError,
    inspect_release_artifacts,
    write_inspection_manifest,
)

README = "# Experia\n\nA typed learning package.\n"
LICENSE = "MIT License\n\nPermission is hereby granted.\n"
PYTHON_CLASSIFIERS = (
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
)


def _metadata(
    *,
    license_expression: str = "MIT",
    urls: dict[str, str] | None = None,
    classifiers: tuple[str, ...] = PYTHON_CLASSIFIERS,
    readme: str = README,
) -> bytes:
    project_urls = urls or {
        "Homepage": "https://experia.example/project",
        "Repository": "https://github.com/irzix/experia",
        "Issues": "https://github.com/irzix/experia/issues",
    }
    headers = [
        "Metadata-Version: 2.4",
        "Name: experia",
        "Version: 0.7.0",
        "Summary: Experience learning for AI agents",
        "Requires-Python: >=3.10",
        f"License-Expression: {license_expression}",
        "License-File: LICENSE",
        "Description-Content-Type: text/markdown; charset=UTF-8",
    ]
    headers.extend(
        f"Project-URL: {label}, {url}" for label, url in project_urls.items()
    )
    headers.extend(f"Classifier: {classifier}" for classifier in classifiers)
    return ("\n".join(headers) + "\n\n" + readme).encode("utf-8")


def _record_bytes(files: dict[str, bytes], record_path: str) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    for path in sorted(files):
        digest = base64.urlsafe_b64encode(hashlib.sha256(files[path]).digest()).rstrip(
            b"="
        )
        writer.writerow([path, f"sha256={digest.decode('ascii')}", len(files[path])])
    writer.writerow([record_path, "", ""])
    return output.getvalue().encode("utf-8")


def _write_wheel(
    directory: Path,
    metadata: bytes,
    *,
    marker: bytes = b"",
    extra_files: dict[str, bytes] | None = None,
    tamper_record_path: str | None = None,
) -> Path:
    wheel = directory / "experia-0.7.0-py3-none-any.whl"
    dist_info = "experia-0.7.0.dist-info"
    record_path = f"{dist_info}/RECORD"
    files = {
        "experia/__init__.py": b'__version__ = "0.7.0"\n',
        "experia/core.py": b"def typed(value: str) -> str:\n    return value\n",
        "experia/py.typed": marker,
        f"{dist_info}/METADATA": metadata,
        f"{dist_info}/WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        f"{dist_info}/licenses/LICENSE": LICENSE.encode(),
    }
    files.update(extra_files or {})
    record = _record_bytes(files, record_path)
    if tamper_record_path is not None:
        record = record.replace(b"sha256=", b"sha256=broken", 1)
    files[record_path] = record
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, content in files.items():
            archive.writestr(path, content)
    return wheel


def _write_sdist(
    directory: Path,
    metadata: bytes,
    *,
    marker: bytes = b"",
    readme: str = README,
) -> Path:
    sdist = directory / "experia-0.7.0.tar.gz"
    root = "experia-0.7.0"
    files = {
        f"{root}/PKG-INFO": metadata,
        f"{root}/README.md": readme.encode(),
        f"{root}/LICENSE": LICENSE.encode(),
        f"{root}/experia/__init__.py": b'__version__ = "0.7.0"\n',
        f"{root}/experia/core.py": b"def typed(value: str) -> str:\n    return value\n",
        f"{root}/experia/py.typed": marker,
    }
    with tarfile.open(sdist, "w:gz") as archive:
        for path, content in files.items():
            info = tarfile.TarInfo(path)
            info.size = len(content)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(content))
    return sdist


def _write_policy(directory: Path) -> Path:
    source = Path(__file__).resolve().parents[1] / "release-artifact-policy.json"
    policy = directory / "release-artifact-policy.json"
    policy.write_bytes(source.read_bytes())
    return policy


def _artifacts(
    directory: Path,
    *,
    metadata: bytes | None = None,
    wheel_marker: bytes = b"",
    sdist_marker: bytes = b"",
    wheel_extras: dict[str, bytes] | None = None,
    tamper_record_path: str | None = None,
    sdist_readme: str = README,
) -> tuple[Path, Path, Path]:
    metadata = metadata or _metadata()
    wheel = _write_wheel(
        directory,
        metadata,
        marker=wheel_marker,
        extra_files=wheel_extras,
        tamper_record_path=tamper_record_path,
    )
    sdist = _write_sdist(
        directory,
        metadata,
        marker=sdist_marker,
        readme=sdist_readme,
    )
    return wheel, sdist, _write_policy(directory)


def test_inspection_records_exact_identity_and_complete_local_evidence(tmp_path: Path):
    wheel, sdist, policy = _artifacts(tmp_path)
    manifest_path = tmp_path / "inspection.json"

    result = write_inspection_manifest(tmp_path, manifest_path, policy_path=policy)

    identity = result["artifact_identity"]
    assert identity["artifact_count"] == 2
    assert identity["counts_by_kind"] == {"wheel": 1, "sdist": 1}
    assert identity["artifacts"] == [
        {
            "kind": "wheel",
            "name": wheel.name,
            "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
            "size_bytes": wheel.stat().st_size,
        },
        {
            "kind": "sdist",
            "name": sdist.name,
            "sha256": hashlib.sha256(sdist.read_bytes()).hexdigest(),
            "size_bytes": sdist.stat().st_size,
        },
    ]
    assert result["package"]["license_expression"] == "MIT"
    assert result["package"]["project_url_validation"] == {
        "mode": "offline-https-shape-and-repository-policy",
        "network_access": False,
    }
    assert result["package"]["python_classifiers"] == list(PYTHON_CLASSIFIERS)
    assert result["readme"]["rendered"] is True
    assert result["readme"]["renderer"] == "offline-safe-markdown-preview-v1"
    assert result["license"]["bytes_match"] is True
    assert result["wheel"]["typing"]["mode"] == "inline"
    assert result["wheel"]["typing"]["bytes_match"] is True
    assert result["wheel"]["record"]["all_file_hashes_verified"] is True
    assert result["wheel"]["contents"]["python_modules_match_sdist"] is True
    assert (
        "tests/" in result["wheel"]["contents"]["verified_exclusions"]["path_prefixes"]
    )
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == result


def test_inspection_hashes_change_when_artifact_bytes_change(tmp_path: Path):
    wheel, _, policy = _artifacts(tmp_path)
    first = inspect_release_artifacts(tmp_path, policy_path=policy)
    first_hash = first["artifact_identity"]["artifacts"][0]["sha256"]

    with zipfile.ZipFile(wheel, "a") as archive:
        archive.comment = b"byte identity changed"
    second = inspect_release_artifacts(tmp_path, policy_path=policy)
    second_hash = second["artifact_identity"]["artifacts"][0]["sha256"]

    assert first_hash != second_hash


def test_inspection_rejects_duplicate_artifact_kind_with_observed_counts(
    tmp_path: Path,
):
    _, _, policy = _artifacts(tmp_path)
    (tmp_path / "experia-0.7.0-2-py3-none-any.whl").write_bytes(b"duplicate")

    with pytest.raises(ReleaseArtifactError) as captured:
        inspect_release_artifacts(tmp_path, policy_path=policy)

    assert captured.value.check == "artifact-count"
    assert captured.value.observed["wheel_count"] == 2
    assert captured.value.observed["sdist_count"] == 1


@pytest.mark.parametrize(
    ("metadata_factory", "expected_check"),
    [
        (lambda: _metadata(license_expression="Apache-2.0"), "license-expression"),
        (
            lambda: _metadata(
                urls={
                    "Homepage": "https://experia.example/project",
                    "Repository": "https://github.com/irzix/experia",
                    "Issues": "http://github.com/irzix/experia/issues",
                }
            ),
            "project-url-shape",
        ),
        (
            lambda: _metadata(classifiers=PYTHON_CLASSIFIERS[:-1]),
            "python-classifiers",
        ),
        (
            lambda: _metadata(readme="# Experia\n\n```python\nunclosed\n"),
            "rendered-readme",
        ),
    ],
)
def test_inspection_blocks_invalid_release_metadata(
    tmp_path: Path,
    metadata_factory: Callable[[], bytes],
    expected_check: str,
):
    metadata = metadata_factory()
    _, _, policy = _artifacts(
        tmp_path,
        metadata=metadata,
        sdist_readme=(
            "# Experia\n\n```python\nunclosed\n"
            if expected_check == "rendered-readme"
            else README
        ),
    )

    with pytest.raises(ReleaseArtifactError) as captured:
        inspect_release_artifacts(tmp_path, policy_path=policy)

    assert captured.value.check == expected_check


@pytest.mark.parametrize(
    ("artifact_arguments", "expected_check"),
    [
        ({"wheel_marker": b"partial\n"}, "typing-marker-bytes"),
        (
            {"wheel_extras": {"tests/test_leak.py": b"assert True\n"}},
            "wheel-allowed-roots",
        ),
        ({"tamper_record_path": "experia/__init__.py"}, "wheel-record-entry"),
    ],
)
def test_inspection_blocks_untruthful_or_unexpected_wheel_contents(
    tmp_path: Path,
    artifact_arguments: dict[str, object],
    expected_check: str,
):
    _, _, policy = _artifacts(tmp_path, **artifact_arguments)

    with pytest.raises(ReleaseArtifactError) as captured:
        inspect_release_artifacts(tmp_path, policy_path=policy)

    assert captured.value.check == expected_check
