"""Inspect release artifacts and emit credential-free byte-identity evidence.

The inspector never performs network requests. Project URL checks prove presence,
HTTPS shape, and the declared Repository/Issues relationship locally. Wheel
exclusions are defined in ``release-artifact-policy.json`` and are included in
inspection evidence so later release jobs can enforce the same policy.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import io
import ipaddress
import json
import re
import stat
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from email import policy as email_policy
from email.message import Message
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

try:
    from scripts.artifact_gate import (
        ArtifactGateError,
        ArtifactRecord,
        canonical_json,
        inspect_artifacts,
    )
except ModuleNotFoundError:  # Support direct execution from the scripts directory.
    from artifact_gate import (  # type: ignore[no-redef]
        ArtifactGateError,
        ArtifactRecord,
        canonical_json,
        inspect_artifacts,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = PROJECT_ROOT / "release-artifact-policy.json"
DEFAULT_ARTIFACT_DIRECTORY = PROJECT_ROOT / "dist" / "quality-gate"
DEFAULT_MANIFEST_NAME = "release-artifact-inspection.json"
MANIFEST_SCHEMA_VERSION = 1
_RENDERER_NAME = "offline-safe-markdown-preview-v1"
_VERSION_CLASSIFIER_PREFIX = "Programming Language :: Python :: "
_ARCHIVE_MEMBER_LIMIT = 64 * 1024 * 1024
_ARCHIVE_TOTAL_LIMIT = 256 * 1024 * 1024
_SEMANTIC_VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")


class ReleaseArtifactError(RuntimeError):
    """A precise, publication-blocking artifact inspection failure."""

    def __init__(self, check: str, observed: Any, detail: str):
        self.check = check
        self.observed = observed
        self.detail = detail
        super().__init__(f"{check} failed: {detail}; observed={observed!r}.")


@dataclass(frozen=True)
class InspectionPolicy:
    """Validated local policy for package metadata and archive contents."""

    path: Path
    package: str
    readme_path: str
    license_path: str
    license_expression: str
    requires_python: str
    required_project_url_labels: tuple[str, ...]
    supported_python_versions: tuple[str, ...]
    typing_marker: str
    typing_mode: str
    allowed_package_roots: tuple[str, ...]
    required_wheel_paths: tuple[str, ...]
    excluded_path_prefixes: tuple[str, ...]
    excluded_path_suffixes: tuple[str, ...]
    excluded_basenames: tuple[str, ...]


@dataclass(frozen=True)
class MetadataSnapshot:
    """Release metadata fields that must agree across wheel and sdist."""

    metadata_version: str
    name: str
    version: str
    summary: str
    requires_python: str
    description_content_type: str
    description: str
    license_expression: str
    license_files: tuple[str, ...]
    project_urls: Mapping[str, str]
    classifiers: tuple[str, ...]

    def comparable(self) -> dict[str, Any]:
        return {
            "metadata_version": self.metadata_version,
            "name": self.name,
            "version": self.version,
            "summary": self.summary,
            "requires_python": self.requires_python,
            "description_content_type": self.description_content_type,
            "description": _canonical_text(self.description),
            "license_expression": self.license_expression,
            "license_files": list(self.license_files),
            "project_urls": dict(self.project_urls),
            "classifiers": list(self.classifiers),
        }


@dataclass(frozen=True)
class ArchiveSnapshot:
    """Regular-file bytes from one safe distribution archive."""

    files: Mapping[str, bytes]

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(sorted(self.files))


def load_policy(path: Path) -> InspectionPolicy:
    """Load and strictly validate the checked-in artifact policy."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReleaseArtifactError("artifact-policy", str(path), str(error)) from error
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ReleaseArtifactError(
            "artifact-policy-schema",
            raw.get("schema_version") if isinstance(raw, dict) else type(raw).__name__,
            "expected schema_version 1",
        )
    wheel = raw.get("wheel")
    if not isinstance(wheel, dict):
        raise ReleaseArtifactError("artifact-policy-wheel", wheel, "expected an object")

    def required_string(container: Mapping[str, Any], key: str) -> str:
        value = container.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ReleaseArtifactError(
                "artifact-policy-field", {key: value}, "expected a non-blank string"
            )
        return value

    def string_tuple(container: Mapping[str, Any], key: str) -> tuple[str, ...]:
        value = container.get(key)
        if (
            not isinstance(value, list)
            or not value
            or any(not isinstance(item, str) or not item for item in value)
            or len(set(value)) != len(value)
        ):
            raise ReleaseArtifactError(
                "artifact-policy-field",
                {key: value},
                "expected a non-empty list of unique strings",
            )
        return tuple(value)

    typing_mode = required_string(raw, "typing_mode")
    if typing_mode not in {"inline", "partial"}:
        raise ReleaseArtifactError(
            "artifact-policy-field",
            {"typing_mode": typing_mode},
            "expected 'inline' or 'partial'",
        )

    return InspectionPolicy(
        path=path.resolve(),
        package=required_string(raw, "package"),
        readme_path=required_string(raw, "readme_path"),
        license_path=required_string(raw, "license_path"),
        license_expression=required_string(raw, "license_expression"),
        requires_python=required_string(raw, "requires_python"),
        required_project_url_labels=string_tuple(raw, "required_project_url_labels"),
        supported_python_versions=string_tuple(raw, "supported_python_versions"),
        typing_marker=required_string(raw, "typing_marker"),
        typing_mode=typing_mode,
        allowed_package_roots=string_tuple(wheel, "allowed_package_roots"),
        required_wheel_paths=string_tuple(wheel, "required_paths"),
        excluded_path_prefixes=string_tuple(wheel, "excluded_path_prefixes"),
        excluded_path_suffixes=string_tuple(wheel, "excluded_path_suffixes"),
        excluded_basenames=string_tuple(wheel, "excluded_basenames"),
    )


def inspect_release_artifacts(
    artifact_directory: Path,
    *,
    policy_path: Path = DEFAULT_POLICY,
) -> dict[str, Any]:
    """Inspect exactly one wheel/sdist and return deterministic release evidence."""
    artifact_directory = artifact_directory.resolve()
    artifact_policy = load_policy(policy_path)
    try:
        records = inspect_artifacts(artifact_directory)
    except ArtifactGateError as error:
        observed = _artifact_name_counts(artifact_directory)
        raise ReleaseArtifactError(
            "artifact-count", observed, "expected exactly one wheel and one sdist"
        ) from error

    wheel_record = _record_of_kind(records, "wheel")
    sdist_record = _record_of_kind(records, "sdist")
    wheel_path = artifact_directory / wheel_record.name
    sdist_path = artifact_directory / sdist_record.name
    wheel = _read_zip_archive(wheel_path, "wheel-archive")
    sdist = _read_sdist(sdist_path)

    dist_info = _wheel_dist_info(wheel)
    wheel_metadata = _parse_metadata(
        wheel.files[f"{dist_info}/METADATA"], "wheel-metadata"
    )
    sdist_root = _sdist_root(sdist)
    sdist_metadata = _parse_metadata(
        sdist.files[f"{sdist_root}/PKG-INFO"], "sdist-metadata"
    )
    _validate_metadata_parity(wheel_metadata, sdist_metadata)
    _validate_metadata_policy(wheel_metadata, artifact_policy)
    _validate_artifact_names(
        wheel_record.name,
        sdist_record.name,
        sdist_root,
        dist_info,
        wheel_metadata,
    )

    readme_evidence = _validate_readme(
        wheel_metadata,
        sdist,
        sdist_root,
        artifact_policy,
    )
    license_evidence = _validate_license(
        wheel_metadata,
        wheel,
        sdist,
        dist_info,
        sdist_root,
        artifact_policy,
    )
    exclusion_evidence = _validate_wheel_contents(
        wheel, sdist, dist_info, sdist_root, artifact_policy
    )
    typing_evidence = _validate_typing_marker(wheel, sdist, sdist_root, artifact_policy)
    record_evidence = _validate_wheel_record(wheel, dist_info)

    identity_artifacts = [record.as_dict() for record in records]
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_identity": {
            "algorithm": "sha256",
            "artifact_count": len(records),
            "counts_by_kind": {"wheel": 1, "sdist": 1},
            "artifacts": identity_artifacts,
        },
        "package": {
            "name": wheel_metadata.name,
            "version": wheel_metadata.version,
            "metadata_version": wheel_metadata.metadata_version,
            "summary": wheel_metadata.summary,
            "requires_python": wheel_metadata.requires_python,
            "license_expression": wheel_metadata.license_expression,
            "project_urls": dict(wheel_metadata.project_urls),
            "project_url_validation": {
                "mode": "offline-https-shape-and-repository-policy",
                "network_access": False,
            },
            "python_classifiers": [
                classifier
                for classifier in wheel_metadata.classifiers
                if classifier.startswith(_VERSION_CLASSIFIER_PREFIX)
            ],
        },
        "readme": readme_evidence,
        "license": license_evidence,
        "wheel": {
            "file_count": len(wheel.files),
            "dist_info": dist_info,
            "contents": exclusion_evidence,
            "record": record_evidence,
            "typing": typing_evidence,
        },
        "sdist": {"file_count": len(sdist.files), "root": sdist_root},
        "policy": {
            "path": artifact_policy.path.name,
            "sha256": _sha256(artifact_policy.path.read_bytes()),
            "schema_version": 1,
        },
    }


def write_inspection_manifest(
    artifact_directory: Path,
    manifest_path: Path,
    *,
    policy_path: Path = DEFAULT_POLICY,
) -> dict[str, Any]:
    """Inspect artifacts and atomically persist canonical JSON evidence."""
    result = inspect_release_artifacts(artifact_directory, policy_path=policy_path)
    manifest_path = manifest_path.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary.write_text(canonical_json(result), encoding="utf-8")
    temporary.replace(manifest_path)
    return result


def _record_of_kind(records: Sequence[ArtifactRecord], kind: str) -> ArtifactRecord:
    return next(record for record in records if record.kind == kind)


def _artifact_name_counts(directory: Path) -> dict[str, Any]:
    if not directory.is_dir():
        return {"directory": str(directory), "exists": False}
    wheel_names = sorted(path.name for path in directory.glob("*.whl"))
    sdist_names = sorted(
        path.name
        for path in directory.iterdir()
        if path.name.endswith((".tar.gz", ".zip"))
    )
    return {
        "wheel_count": len(wheel_names),
        "sdist_count": len(sdist_names),
        "wheels": wheel_names,
        "sdists": sdist_names,
    }


def _read_zip_archive(path: Path, check: str) -> ArchiveSnapshot:
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            raw_names = [info.filename for info in infos]
            if len(raw_names) != len(set(raw_names)):
                duplicates = sorted(
                    name for name in set(raw_names) if raw_names.count(name) > 1
                )
                raise ReleaseArtifactError(check, duplicates, "duplicate archive paths")
            files: dict[str, bytes] = {}
            total = 0
            for info in infos:
                safe_name = _safe_archive_path(info.filename, check)
                mode = (info.external_attr >> 16) & 0o170000
                if mode == stat.S_IFLNK:
                    raise ReleaseArtifactError(
                        check, safe_name, "symbolic links are not allowed"
                    )
                if info.is_dir():
                    continue
                if info.flag_bits & 0x1:
                    raise ReleaseArtifactError(
                        check, safe_name, "encrypted members are not allowed"
                    )
                total = _bounded_archive_size(check, safe_name, info.file_size, total)
                files[safe_name] = archive.read(info)
            return ArchiveSnapshot(files)
    except ReleaseArtifactError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        raise ReleaseArtifactError(check, path.name, str(error)) from error


def _read_sdist(path: Path) -> ArchiveSnapshot:
    if path.name.endswith(".zip"):
        return _read_zip_archive(path, "sdist-archive")
    try:
        with tarfile.open(path, mode="r:*") as archive:
            files: dict[str, bytes] = {}
            total = 0
            for member in archive.getmembers():
                safe_name = _safe_archive_path(member.name, "sdist-archive")
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ReleaseArtifactError(
                        "sdist-archive",
                        safe_name,
                        "links and special members are not allowed",
                    )
                if safe_name in files:
                    raise ReleaseArtifactError(
                        "sdist-archive", safe_name, "duplicate archive path"
                    )
                total = _bounded_archive_size(
                    "sdist-archive", safe_name, member.size, total
                )
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ReleaseArtifactError(
                        "sdist-archive", safe_name, "regular file is unreadable"
                    )
                files[safe_name] = extracted.read()
            return ArchiveSnapshot(files)
    except ReleaseArtifactError:
        raise
    except (OSError, tarfile.TarError) as error:
        raise ReleaseArtifactError("sdist-archive", path.name, str(error)) from error


def _safe_archive_path(name: str, check: str) -> str:
    if not name or "\\" in name:
        raise ReleaseArtifactError(check, name, "archive path is empty or non-POSIX")
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or ".." in path.parts
        or any(part == "" for part in path.parts)
    ):
        raise ReleaseArtifactError(check, name, "archive path is unsafe")
    return path.as_posix().rstrip("/")


def _bounded_archive_size(
    check: str, path: str, member_size: int, current_total: int
) -> int:
    if member_size > _ARCHIVE_MEMBER_LIMIT:
        raise ReleaseArtifactError(
            check,
            {"path": path, "size_bytes": member_size},
            f"member exceeds {_ARCHIVE_MEMBER_LIMIT} bytes",
        )
    total = current_total + member_size
    if total > _ARCHIVE_TOTAL_LIMIT:
        raise ReleaseArtifactError(
            check, total, f"archive exceeds {_ARCHIVE_TOTAL_LIMIT} uncompressed bytes"
        )
    return total


def _raw_metadata_body(data: bytes) -> bytes:
    """Return the raw bytes after the RFC 822 header/body separator."""
    positions = [
        (data.find(separator), len(separator))
        for separator in (b"\r\n\r\n", b"\n\n")
        if data.find(separator) != -1
    ]
    if not positions:
        return b""
    start, separator_length = min(positions)
    return data[start + separator_length :]


def _metadata_description(data: bytes, message: Message, check: str) -> str:
    """Decode the description body as UTF-8 per the core-metadata file encoding.

    ``email`` lossily replaces non-ASCII body bytes with U+FFFD when the message
    carries no charset header (as core metadata does not), which would corrupt a
    README that uses non-ASCII characters. Metadata 2.x files are UTF-8, so the
    raw body is decoded directly; a legacy folded ``Description`` header is used
    only when no body is present.
    """
    body = _raw_metadata_body(data)
    if body.strip():
        try:
            return body.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ReleaseArtifactError(
                check, "non-utf-8", "description body is not UTF-8"
            ) from error
    header_value = message.get("Description")
    if isinstance(header_value, str) and header_value.strip():
        return header_value
    raise ReleaseArtifactError(check, "", "README description is empty")


def _parse_metadata(data: bytes, check: str) -> MetadataSnapshot:
    try:
        message = BytesParser(policy=email_policy.default).parsebytes(data)
    except (UnicodeError, ValueError) as error:
        raise ReleaseArtifactError(check, "unparseable", str(error)) from error
    if message.is_multipart():
        raise ReleaseArtifactError(check, "multipart", "package metadata must be flat")

    metadata_version = _one_header(message, "Metadata-Version", check)
    name = _one_header(message, "Name", check)
    version = _one_header(message, "Version", check)
    summary = _one_header(message, "Summary", check)
    requires_python = _one_header(message, "Requires-Python", check)
    content_type = _one_header(message, "Description-Content-Type", check)
    license_expression = _one_header(message, "License-Expression", check)
    payload = _metadata_description(data, message, check)

    project_urls: dict[str, str] = {}
    for value in message.get_all("Project-URL", []):
        label, separator, url = value.partition(",")
        label, url = label.strip(), url.strip()
        if not separator or not label or not url or label in project_urls:
            raise ReleaseArtifactError(
                check, value, "Project-URL must contain one unique 'Label, URL' pair"
            )
        project_urls[label] = url

    return MetadataSnapshot(
        metadata_version=metadata_version,
        name=name,
        version=version,
        summary=summary,
        requires_python=requires_python,
        description_content_type=content_type,
        description=payload,
        license_expression=license_expression,
        license_files=tuple(message.get_all("License-File", [])),
        project_urls=project_urls,
        classifiers=tuple(message.get_all("Classifier", [])),
    )


def _one_header(message: Message, name: str, check: str) -> str:
    values = message.get_all(name, [])
    if len(values) != 1 or not values[0].strip():
        raise ReleaseArtifactError(
            check,
            {name: values},
            f"expected exactly one non-blank {name} header",
        )
    return values[0].strip()


def _validate_metadata_parity(wheel: MetadataSnapshot, sdist: MetadataSnapshot) -> None:
    wheel_value = wheel.comparable()
    sdist_value = sdist.comparable()
    if wheel_value != sdist_value:
        differing = sorted(
            key for key in wheel_value if wheel_value[key] != sdist_value[key]
        )
        raise ReleaseArtifactError(
            "metadata-parity", differing, "wheel and sdist metadata differ"
        )


def _validate_metadata_policy(
    metadata: MetadataSnapshot, artifact_policy: InspectionPolicy
) -> None:
    if metadata.name != artifact_policy.package:
        raise ReleaseArtifactError(
            "package-name", metadata.name, f"expected {artifact_policy.package!r}"
        )
    if not _SEMANTIC_VERSION.fullmatch(metadata.version):
        raise ReleaseArtifactError(
            "package-version", metadata.version, "expected MAJOR.MINOR.PATCH"
        )
    if not metadata.summary.strip():
        raise ReleaseArtifactError("package-summary", metadata.summary, "is blank")
    if metadata.requires_python != artifact_policy.requires_python:
        raise ReleaseArtifactError(
            "requires-python",
            metadata.requires_python,
            f"expected {artifact_policy.requires_python!r}",
        )
    if metadata.license_expression != artifact_policy.license_expression:
        raise ReleaseArtifactError(
            "license-expression",
            metadata.license_expression,
            f"expected SPDX expression {artifact_policy.license_expression!r}",
        )
    if artifact_policy.license_path not in metadata.license_files:
        raise ReleaseArtifactError(
            "license-metadata",
            list(metadata.license_files),
            f"expected License-File {artifact_policy.license_path!r}",
        )
    _validate_project_urls(metadata.project_urls, artifact_policy)
    _validate_python_classifiers(metadata.classifiers, artifact_policy)


def _validate_project_urls(
    project_urls: Mapping[str, str], artifact_policy: InspectionPolicy
) -> None:
    expected = set(artifact_policy.required_project_url_labels)
    observed = set(project_urls)
    if observed != expected:
        raise ReleaseArtifactError(
            "project-url-labels",
            sorted(observed),
            f"expected exactly {sorted(expected)!r}",
        )
    parsed = {
        label: _validate_https_url(label, url) for label, url in project_urls.items()
    }
    repository = parsed["Repository"]
    issues = parsed["Issues"]
    repository_path = repository.path.rstrip("/")
    if repository_path.endswith(".git"):
        repository_path = repository_path[:-4]
    expected_issues_path = f"{repository_path}/issues"
    if (
        issues.hostname != repository.hostname
        or issues.path.rstrip("/") != expected_issues_path
    ):
        raise ReleaseArtifactError(
            "project-url-policy",
            project_urls["Issues"],
            f"expected Issues URL on Repository host at {expected_issues_path!r}",
        )


def _validate_https_url(label: str, url: str):
    if any(character.isspace() for character in url):
        raise ReleaseArtifactError(
            "project-url-shape", {label: url}, "contains whitespace"
        )
    parsed = urlsplit(url)
    hostname = parsed.hostname
    if (
        parsed.scheme != "https"
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ReleaseArtifactError(
            "project-url-shape",
            {label: url},
            "expected credential-free HTTPS URL without a fragment",
        )
    if hostname == "localhost" or "." not in hostname:
        raise ReleaseArtifactError(
            "project-url-shape", {label: url}, "host is not publicly shaped"
        )
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        raise ReleaseArtifactError(
            "project-url-shape", {label: url}, "literal IP hosts are not allowed"
        )
    return parsed


def _validate_python_classifiers(
    classifiers: Sequence[str], artifact_policy: InspectionPolicy
) -> None:
    observed_versions = sorted(
        classifier.removeprefix(_VERSION_CLASSIFIER_PREFIX)
        for classifier in classifiers
        if classifier.startswith(_VERSION_CLASSIFIER_PREFIX)
        and re.fullmatch(
            r"[0-9]+\.[0-9]+",
            classifier.removeprefix(_VERSION_CLASSIFIER_PREFIX),
        )
    )
    expected_versions = sorted(artifact_policy.supported_python_versions)
    if observed_versions != expected_versions:
        raise ReleaseArtifactError(
            "python-classifiers",
            observed_versions,
            f"expected exactly {expected_versions!r}",
        )


def _validate_artifact_names(
    wheel_name: str,
    sdist_name: str,
    sdist_root: str,
    dist_info: str,
    metadata: MetadataSnapshot,
) -> None:
    normalized_name = re.sub(r"[-_.]+", "_", metadata.name)
    expected_dist_info = f"{normalized_name}-{metadata.version}.dist-info"
    expected_sdist_root = f"{metadata.name}-{metadata.version}"
    if dist_info != expected_dist_info:
        raise ReleaseArtifactError(
            "wheel-dist-info", dist_info, f"expected {expected_dist_info!r}"
        )
    if not wheel_name.startswith(f"{normalized_name}-{metadata.version}-"):
        raise ReleaseArtifactError(
            "wheel-filename",
            wheel_name,
            f"expected name/version prefix {normalized_name}-{metadata.version}-",
        )
    expected_sdist_names = {
        f"{expected_sdist_root}.tar.gz",
        f"{expected_sdist_root}.zip",
    }
    if sdist_name not in expected_sdist_names or sdist_root != expected_sdist_root:
        raise ReleaseArtifactError(
            "sdist-identity",
            {"filename": sdist_name, "root": sdist_root},
            f"expected {sorted(expected_sdist_names)!r} with root {expected_sdist_root!r}",
        )


def _wheel_dist_info(wheel: ArchiveSnapshot) -> str:
    metadata_paths = [
        path for path in wheel.files if path.endswith(".dist-info/METADATA")
    ]
    if len(metadata_paths) != 1:
        raise ReleaseArtifactError(
            "wheel-metadata-paths",
            metadata_paths,
            "expected exactly one .dist-info/METADATA",
        )
    dist_info = metadata_paths[0].removesuffix("/METADATA")
    for required in ("WHEEL", "RECORD"):
        path = f"{dist_info}/{required}"
        if path not in wheel.files:
            raise ReleaseArtifactError(
                "wheel-required-file", path, "required wheel file is missing"
            )
    return dist_info


def _sdist_root(sdist: ArchiveSnapshot) -> str:
    roots = {PurePosixPath(path).parts[0] for path in sdist.files}
    if len(roots) != 1:
        raise ReleaseArtifactError(
            "sdist-root", sorted(roots), "expected exactly one top-level directory"
        )
    root = next(iter(roots))
    if f"{root}/PKG-INFO" not in sdist.files:
        raise ReleaseArtifactError(
            "sdist-metadata-path", root, "top-level PKG-INFO is missing"
        )
    return root


def _validate_readme(
    metadata: MetadataSnapshot,
    sdist: ArchiveSnapshot,
    root: str,
    artifact_policy: InspectionPolicy,
) -> dict[str, Any]:
    media_type, _, parameters = metadata.description_content_type.partition(";")
    if media_type.strip().lower() != "text/markdown":
        raise ReleaseArtifactError(
            "readme-content-type",
            metadata.description_content_type,
            "expected text/markdown",
        )
    if parameters and "charset=utf-8" not in parameters.lower().replace(" ", ""):
        raise ReleaseArtifactError(
            "readme-content-type",
            metadata.description_content_type,
            "only an optional UTF-8 charset parameter is supported",
        )
    readme_path = f"{root}/{artifact_policy.readme_path}"
    readme_bytes = _required_file(sdist, readme_path, "sdist-readme")
    try:
        readme = readme_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ReleaseArtifactError(
            "sdist-readme", readme_path, "is not UTF-8"
        ) from error
    if _canonical_text(readme) != _canonical_text(metadata.description):
        raise ReleaseArtifactError(
            "rendered-readme-source",
            {
                "sdist_sha256": _sha256(_canonical_text(readme).encode()),
                "metadata_sha256": _sha256(
                    _canonical_text(metadata.description).encode()
                ),
            },
            "metadata description does not match packaged README",
        )
    rendered = _render_markdown_preview(readme)
    return {
        "path": artifact_policy.readme_path,
        "content_type": metadata.description_content_type,
        "source_sha256": _sha256(readme_bytes),
        "rendered_sha256": _sha256(rendered.encode("utf-8")),
        "renderer": _RENDERER_NAME,
        "rendered": True,
    }


def _render_markdown_preview(markdown: str) -> str:
    if "\x00" in markdown or re.search(r"<\s*(script|iframe|object)\b", markdown, re.I):
        raise ReleaseArtifactError(
            "rendered-readme", "unsafe-content", "contains unsafe HTML or NUL data"
        )
    fence: str | None = None
    rendered_lines: list[str] = []
    for line_number, line in enumerate(markdown.splitlines(), start=1):
        match = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if match:
            marker = match.group(1)[0]
            if fence is None:
                fence = marker
                rendered_lines.append("<pre><code>")
            elif fence == marker:
                fence = None
                rendered_lines.append("</code></pre>")
            else:
                rendered_lines.append(html.escape(line))
            continue
        escaped = html.escape(line)
        if fence is not None:
            rendered_lines.append(escaped)
        elif heading := re.match(r"^(#{1,6})\s+(.+)$", line):
            level = len(heading.group(1))
            rendered_lines.append(
                f"<h{level}>{html.escape(heading.group(2))}</h{level}>"
            )
        elif line.strip():
            rendered_lines.append(f"<p>{escaped}</p>")
    if fence is not None:
        raise ReleaseArtifactError(
            "rendered-readme", "unclosed-code-fence", "Markdown code fence is unclosed"
        )
    rendered = "\n".join(rendered_lines)
    if not rendered.strip():
        raise ReleaseArtifactError(
            "rendered-readme", "empty", "Markdown rendered to empty output"
        )
    return rendered


def _validate_license(
    metadata: MetadataSnapshot,
    wheel: ArchiveSnapshot,
    sdist: ArchiveSnapshot,
    dist_info: str,
    root: str,
    artifact_policy: InspectionPolicy,
) -> dict[str, Any]:
    source_path = f"{root}/{artifact_policy.license_path}"
    wheel_path = f"{dist_info}/licenses/{artifact_policy.license_path}"
    source = _required_file(sdist, source_path, "sdist-license")
    installed = _required_file(wheel, wheel_path, "wheel-license")
    if source != installed:
        raise ReleaseArtifactError(
            "license-bytes",
            {"sdist_sha256": _sha256(source), "wheel_sha256": _sha256(installed)},
            "wheel and sdist licenses differ",
        )
    if not source.startswith(b"MIT License"):
        raise ReleaseArtifactError(
            "license-content",
            source[:40].decode("utf-8", errors="replace"),
            "expected MIT license text",
        )
    return {
        "expression": metadata.license_expression,
        "source_path": artifact_policy.license_path,
        "wheel_path": wheel_path,
        "sha256": _sha256(source),
        "bytes_match": True,
    }


def _validate_wheel_contents(
    wheel: ArchiveSnapshot,
    sdist: ArchiveSnapshot,
    dist_info: str,
    root: str,
    artifact_policy: InspectionPolicy,
) -> dict[str, Any]:
    missing = sorted(set(artifact_policy.required_wheel_paths) - set(wheel.files))
    if missing:
        raise ReleaseArtifactError(
            "wheel-required-paths", missing, "required package files are missing"
        )
    allowed_roots = set(artifact_policy.allowed_package_roots) | {dist_info}
    unexpected_roots = sorted(
        {
            PurePosixPath(path).parts[0]
            for path in wheel.files
            if PurePosixPath(path).parts[0] not in allowed_roots
        }
    )
    if unexpected_roots:
        raise ReleaseArtifactError(
            "wheel-allowed-roots",
            unexpected_roots,
            f"expected only {sorted(allowed_roots)!r}",
        )

    violations: list[str] = []
    for path in wheel.files:
        basename = PurePosixPath(path).name
        if path.startswith(artifact_policy.excluded_path_prefixes):
            violations.append(path)
        elif path.endswith(artifact_policy.excluded_path_suffixes):
            violations.append(path)
        elif basename in artifact_policy.excluded_basenames:
            violations.append(path)
        elif "__pycache__" in PurePosixPath(path).parts:
            violations.append(path)
    if violations:
        raise ReleaseArtifactError(
            "wheel-exclusions", sorted(violations), "documented exclusions are present"
        )

    package_prefixes = tuple(
        f"{name}/" for name in artifact_policy.allowed_package_roots
    )
    wheel_modules = {
        path
        for path in wheel.files
        if path.startswith(package_prefixes) and path.endswith(".py")
    }
    source_modules = {
        path.removeprefix(f"{root}/")
        for path in sdist.files
        if path.removeprefix(f"{root}/").startswith(package_prefixes)
        and path.endswith(".py")
    }
    if wheel_modules != source_modules:
        raise ReleaseArtifactError(
            "wheel-python-modules",
            {
                "missing": sorted(source_modules - wheel_modules),
                "unexpected": sorted(wheel_modules - source_modules),
            },
            "wheel Python modules do not match the sdist package modules",
        )
    for module_path in wheel_modules:
        try:
            wheel.files[module_path].decode("utf-8")
        except UnicodeDecodeError as error:
            raise ReleaseArtifactError(
                "wheel-python-module", module_path, "source is not UTF-8"
            ) from error

    return {
        "allowed_roots": sorted(allowed_roots),
        "required_paths": list(artifact_policy.required_wheel_paths),
        "python_module_count": len(wheel_modules),
        "python_modules_match_sdist": True,
        "verified_exclusions": {
            "path_prefixes": list(artifact_policy.excluded_path_prefixes),
            "path_suffixes": list(artifact_policy.excluded_path_suffixes),
            "basenames": list(artifact_policy.excluded_basenames),
            "implicit": ["__pycache__/"],
        },
    }


def _validate_typing_marker(
    wheel: ArchiveSnapshot,
    sdist: ArchiveSnapshot,
    root: str,
    artifact_policy: InspectionPolicy,
) -> dict[str, Any]:
    wheel_marker = _required_file(
        wheel, artifact_policy.typing_marker, "wheel-typing-marker"
    )
    source_path = f"{root}/{artifact_policy.typing_marker}"
    source_marker = _required_file(sdist, source_path, "sdist-typing-marker")
    if wheel_marker != source_marker:
        raise ReleaseArtifactError(
            "typing-marker-bytes",
            {
                "wheel_sha256": _sha256(wheel_marker),
                "sdist_sha256": _sha256(source_marker),
            },
            "wheel and sdist py.typed markers differ",
        )
    try:
        marker_text = wheel_marker.decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise ReleaseArtifactError(
            "typing-marker-content", "non-UTF-8", "py.typed must be UTF-8"
        ) from error
    observed_mode = "partial" if marker_text == "partial" else "inline"
    if marker_text not in {"", "partial"}:
        raise ReleaseArtifactError(
            "typing-marker-content",
            marker_text,
            "expected an empty inline marker or the word 'partial'",
        )
    if observed_mode != artifact_policy.typing_mode:
        raise ReleaseArtifactError(
            "typing-marker-mode",
            observed_mode,
            f"policy declares {artifact_policy.typing_mode!r}",
        )
    return {
        "path": artifact_policy.typing_marker,
        "mode": observed_mode,
        "sha256": _sha256(wheel_marker),
        "present_in_wheel": True,
        "present_in_sdist": True,
        "bytes_match": True,
    }


def _validate_wheel_record(wheel: ArchiveSnapshot, dist_info: str) -> dict[str, Any]:
    record_path = f"{dist_info}/RECORD"
    try:
        rows = list(csv.reader(io.StringIO(wheel.files[record_path].decode("utf-8"))))
    except (UnicodeDecodeError, csv.Error) as error:
        raise ReleaseArtifactError("wheel-record", record_path, str(error)) from error
    entries: dict[str, tuple[str, str]] = {}
    for row in rows:
        if len(row) != 3:
            raise ReleaseArtifactError("wheel-record", row, "expected three columns")
        path, digest, size = row
        if path in entries:
            raise ReleaseArtifactError("wheel-record", path, "duplicate RECORD path")
        entries[path] = (digest, size)
    if set(entries) != set(wheel.files):
        raise ReleaseArtifactError(
            "wheel-record-paths",
            {
                "missing": sorted(set(wheel.files) - set(entries)),
                "unexpected": sorted(set(entries) - set(wheel.files)),
            },
            "RECORD paths do not match wheel contents",
        )
    for path, content in wheel.files.items():
        digest, size = entries[path]
        if path == record_path:
            if digest or size:
                raise ReleaseArtifactError(
                    "wheel-record-self",
                    entries[path],
                    "RECORD self-entry must be empty",
                )
            continue
        expected_digest = "sha256=" + base64.urlsafe_b64encode(
            hashlib.sha256(content).digest()
        ).rstrip(b"=").decode("ascii")
        if digest != expected_digest or size != str(len(content)):
            raise ReleaseArtifactError(
                "wheel-record-entry",
                {"path": path, "digest": digest, "size": size},
                f"expected digest={expected_digest!r}, size={len(content)!r}",
            )
    return {
        "path": record_path,
        "entry_count": len(entries),
        "all_paths_listed": True,
        "all_file_hashes_verified": True,
    }


def _required_file(archive: ArchiveSnapshot, path: str, check: str) -> bytes:
    try:
        return archive.files[path]
    except KeyError as error:
        raise ReleaseArtifactError(check, path, "required file is missing") from error


def _canonical_text(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n") + "\n"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _resolve(path: Path, root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIRECTORY,
        help=f"directory containing exactly one wheel/sdist (default: {DEFAULT_ARTIFACT_DIRECTORY})",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=DEFAULT_POLICY,
        help=f"local inspection policy (default: {DEFAULT_POLICY})",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help=f"output path (default: ARTIFACT_DIR/{DEFAULT_MANIFEST_NAME})",
    )
    arguments = parser.parse_args(argv)
    artifact_directory = _resolve(arguments.artifact_dir, PROJECT_ROOT)
    policy_path = _resolve(arguments.policy, PROJECT_ROOT)
    manifest_path = (
        _resolve(arguments.manifest, PROJECT_ROOT)
        if arguments.manifest is not None
        else artifact_directory / DEFAULT_MANIFEST_NAME
    )
    try:
        result = write_inspection_manifest(
            artifact_directory, manifest_path, policy_path=policy_path
        )
    except ReleaseArtifactError as error:
        print(f"release artifact inspection failed: {error}", file=sys.stderr)
        return 1
    print(canonical_json(result), end="")
    print(f"manifest: {manifest_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
