"""Machine-readable benchmark manifest generation and publication validation.

A benchmark manifest is the provenance record published alongside any benchmark
result. It records exactly what ran and where, so a reader can decide whether
the numbers are trustworthy and reproducible:

  * source commit and source-tree cleanliness,
  * package, Python, dependency, and operating-system versions,
  * the dataset identity, explicit seed, evaluator, and embedder,
  * the exact execution command, and
  * an offline / non-offline network classification that documents every
    required service and credential category.

The module is deliberately credential-free and never touches the network. Git
provenance is read through a short, sandboxed subprocess with a timeout;
everything else comes from the standard library.

``collect_benchmark_manifest`` is lenient: it records whatever the environment
reports, including gaps (for example an unknown commit outside a checkout).
``validate_benchmark_manifest`` is the strict publication gate: it rejects a
manifest that is incomplete (a missing, blank, or placeholder field) or
mismatched (a tampered identity, a dataset identity that does not match the
published result, or a network classification that contradicts its documented
services).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA = "experia.benchmark-manifest.v1"

# Runtime dependencies whose versions belong in every provenance record.
_RUNTIME_DEPENDENCIES: tuple[str, ...] = ("aiosqlite", "pydantic")

# Values that never count as a real, published provenance field. ``none`` and
# ``not-applicable`` are intentionally excluded: they are legitimate, documented
# evaluator/embedder classifications (for example the offline learning
# benchmark uses no embedder).
_PLACEHOLDER_VALUES = frozenset(
    {
        "",
        "unknown",
        "unset",
        "tbd",
        "todo",
        "changeme",
        "placeholder",
        "not-installed",
    }
)

# Fields excluded from the reproducible identity hash. ``manifest_id`` is the
# hash itself; ``generated_at`` is an informational wall-clock timestamp that
# must not perturb the identity of an otherwise identical manifest.
_IDENTITY_EXCLUDED = ("manifest_id", "generated_at")

_REQUIRED_FIELDS = (
    "benchmark",
    "command",
    "dataset",
    "dependencies",
    "embedder",
    "evaluator",
    "manifest_id",
    "network",
    "operating_system",
    "package_version",
    "python",
    "schema",
    "seed",
    "source",
)

_REQUIRED_OS_FIELDS = ("machine", "platform", "release", "system", "version")


class ManifestValidationError(ValueError):
    """Raised when a benchmark manifest is not fit for publication."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _identity(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def manifest_identity(manifest: Mapping[str, Any]) -> str:
    """Return the reproducible identity hash of a manifest's contents."""
    payload = {
        key: value for key, value in manifest.items() if key not in _IDENTITY_EXCLUDED
    }
    return _identity(payload)


def serialize_manifest(manifest: Mapping[str, Any]) -> str:
    """Stable, key-sorted serialization of a benchmark manifest."""
    return (
        json.dumps(
            manifest,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _git_output(repository_root: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def offline_network(*, notes: str = "") -> dict[str, Any]:
    """A network classification for a fully offline, credential-free benchmark."""
    return {
        "credential_categories": [],
        "notes": notes,
        "offline": True,
        "required_services": [],
    }


def service_network(
    *,
    required_services: Sequence[str],
    credential_categories: Sequence[str] = (),
    notes: str = "",
) -> dict[str, Any]:
    """A non-offline classification that documents its services and credentials.

    Requirement 11.8 obliges any benchmark needing network access, external
    services, or credentials to declare them explicitly, so an empty
    classification is rejected at construction time.
    """
    services = [str(service) for service in required_services]
    categories = [str(category) for category in credential_categories]
    if not services and not categories:
        raise ValueError(
            "a non-offline benchmark must document at least one required "
            "service or credential category"
        )
    return {
        "credential_categories": categories,
        "notes": notes,
        "offline": False,
        "required_services": services,
    }


def collect_benchmark_manifest(
    *,
    benchmark: str,
    command: Sequence[str],
    dataset_identity: Mapping[str, Any],
    seed: int,
    evaluator: str,
    embedder: str,
    network: Mapping[str, Any] | None = None,
    dependencies: Sequence[str] = _RUNTIME_DEPENDENCIES,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Assemble a benchmark manifest from the current environment and inputs.

    The dataset identity must carry a stable ``dataset_id``; the network
    classification defaults to fully offline. Collection is lenient by design:
    missing environment provenance (for example an unknown commit) is recorded
    verbatim so the publication validator can reject it later.
    """
    if (
        not isinstance(dataset_identity, Mapping)
        or "dataset_id" not in dataset_identity
    ):
        raise ValueError("dataset_identity must be a mapping containing 'dataset_id'")

    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[1]
    )
    commit = _git_output(root, "rev-parse", "HEAD")
    status = _git_output(root, "status", "--porcelain")
    clean = None if status is None else status == ""

    payload = {
        "benchmark": benchmark,
        "command": [str(part) for part in command],
        "dataset": dict(dataset_identity),
        "dependencies": {name: _package_version(name) for name in sorted(dependencies)},
        "embedder": embedder,
        "evaluator": evaluator,
        "network": dict(network) if network is not None else offline_network(),
        "operating_system": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "release": platform.release(),
            "system": platform.system(),
            "version": platform.version(),
        },
        "package_version": _package_version("experia"),
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "schema": MANIFEST_SCHEMA,
        "seed": seed,
        "source": {
            "clean": clean,
            "commit": commit or "unknown",
        },
    }
    return {
        "manifest_id": manifest_identity(payload),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **payload,
    }


def _is_blank(value: Any) -> bool:
    return (
        not isinstance(value, str)
        or not value.strip()
        or value.strip().lower() in _PLACEHOLDER_VALUES
    )


def _validate_network_classification(network: Any) -> None:
    if not isinstance(network, Mapping):
        raise ManifestValidationError("manifest network classification is missing")
    offline = network.get("offline")
    if not isinstance(offline, bool):
        raise ManifestValidationError(
            "manifest network classification must record 'offline' as a boolean"
        )
    services = network.get("required_services")
    credentials = network.get("credential_categories")
    if not isinstance(services, list) or any(_is_blank(item) for item in services):
        raise ManifestValidationError(
            "manifest required services must be a list of documented service names"
        )
    if not isinstance(credentials, list) or any(
        _is_blank(item) for item in credentials
    ):
        raise ManifestValidationError(
            "manifest credential categories must be a list of documented categories"
        )
    if offline and (services or credentials):
        raise ManifestValidationError(
            "an offline benchmark must not require services or credentials"
        )
    if not offline and not (services or credentials):
        raise ManifestValidationError(
            "a non-offline benchmark must document its required services or "
            "credential categories"
        )


def validate_benchmark_manifest(
    manifest: Any,
    *,
    expected_dataset_id: str | None = None,
    expected_command: Sequence[str] | None = None,
    require_clean_source: bool = True,
) -> None:
    """Reject an incomplete or mismatched manifest before publication.

    Raises :class:`ManifestValidationError` describing the first failed check.
    Passing ``expected_dataset_id`` or ``expected_command`` additionally rejects
    a manifest whose dataset identity or command does not match the result being
    published.
    """
    if not isinstance(manifest, Mapping):
        raise ManifestValidationError("manifest must be a mapping")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ManifestValidationError("manifest schema is not supported")

    missing = [field for field in _REQUIRED_FIELDS if field not in manifest]
    if missing:
        raise ManifestValidationError(
            "manifest is missing required fields: " + ", ".join(sorted(missing))
        )

    for field in ("benchmark", "evaluator", "embedder", "package_version"):
        if _is_blank(manifest[field]):
            raise ManifestValidationError(
                f"manifest field {field!r} is blank or a placeholder"
            )

    command = manifest["command"]
    if (
        not isinstance(command, Sequence)
        or isinstance(command, (str, bytes))
        or not command
    ):
        raise ManifestValidationError(
            "manifest command must be a non-empty argument list"
        )
    if any(_is_blank(part) for part in command):
        raise ManifestValidationError("manifest command contains a blank argument")

    seed = manifest["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ManifestValidationError("manifest seed must be an integer")

    dataset = manifest["dataset"]
    if not isinstance(dataset, Mapping) or _is_blank(dataset.get("dataset_id")):
        raise ManifestValidationError("manifest dataset identity is missing")

    source = manifest["source"]
    if not isinstance(source, Mapping):
        raise ManifestValidationError("manifest source provenance is missing")
    if _is_blank(source.get("commit")):
        raise ManifestValidationError("manifest source commit is missing or unknown")
    clean = source.get("clean")
    if not isinstance(clean, bool):
        raise ManifestValidationError(
            "manifest source cleanliness must be recorded as a boolean"
        )
    if require_clean_source and not clean:
        raise ManifestValidationError(
            "manifest records an unclean source tree; refusing publication"
        )

    python = manifest["python"]
    if (
        not isinstance(python, Mapping)
        or _is_blank(python.get("implementation"))
        or _is_blank(python.get("version"))
    ):
        raise ManifestValidationError("manifest Python identity is incomplete")

    operating_system = manifest["operating_system"]
    if not isinstance(operating_system, Mapping):
        raise ManifestValidationError("manifest operating-system identity is missing")
    for field in _REQUIRED_OS_FIELDS:
        if _is_blank(operating_system.get(field)):
            raise ManifestValidationError(
                f"manifest operating-system field {field!r} is incomplete"
            )

    dependencies = manifest["dependencies"]
    if not isinstance(dependencies, Mapping) or not dependencies:
        raise ManifestValidationError("manifest dependency versions are missing")
    for name, version in dependencies.items():
        if _is_blank(name) or _is_blank(version):
            raise ManifestValidationError(
                f"manifest dependency version for {name!r} is missing"
            )

    _validate_network_classification(manifest["network"])

    if manifest_identity(manifest) != manifest["manifest_id"]:
        raise ManifestValidationError("manifest identity does not match its contents")

    if (
        expected_dataset_id is not None
        and dataset.get("dataset_id") != expected_dataset_id
    ):
        raise ManifestValidationError(
            "manifest dataset identity does not match the published result"
        )
    if expected_command is not None and list(command) != list(expected_command):
        raise ManifestValidationError(
            "manifest command does not match the published result"
        )
