"""Focused tests for benchmark manifest generation and the publication gate.

These cover the guarantees task 16.2 adds on top of the deterministic offline
learning benchmark (task 16.1):

  * generation records every provenance field required for publication — source
    commit and cleanliness, package/Python/dependency/OS versions, dataset
    identity, seed, evaluator, embedder, command, and network classification;
  * the offline learning benchmark is classified as offline with no services or
    credentials;
  * ``validate_benchmark_manifest`` rejects incomplete or mismatched
    environment/dataset manifests before publication.

Everything here is fully offline (no LLM, no network, no credentials).

Validates: Requirements 5.9, 11.6, 11.8
"""

from __future__ import annotations

from copy import deepcopy

import pytest

from benchmark.learning_benchmark import (
    EMBEDDER_IDENTITY,
    EVALUATOR_IDENTITY,
    LearningScenario,
    build_manifest,
    run_benchmark,
)
from benchmark.manifest import (
    MANIFEST_SCHEMA,
    ManifestValidationError,
    collect_benchmark_manifest,
    manifest_identity,
    offline_network,
    serialize_manifest,
    service_network,
    validate_benchmark_manifest,
)

COMMAND = ("python", "-m", "benchmark.learning_benchmark", "--manifest", "m.json")


def _signed(manifest: dict) -> dict:
    """Return a manifest with a correct identity hash and a fixed timestamp."""
    manifest = deepcopy(manifest)
    manifest.pop("manifest_id", None)
    manifest["manifest_id"] = manifest_identity(manifest)
    manifest["generated_at"] = "2025-01-01T00:00:00+00:00"
    return manifest


def _complete_manifest(**overrides) -> dict:
    """A structurally complete, clean, offline manifest for gate tests."""
    manifest = {
        "benchmark": "learning",
        "command": list(COMMAND),
        "dataset": {"dataset_id": "abc123", "kind": "learning-scenario"},
        "dependencies": {"aiosqlite": "0.20.0", "pydantic": "2.9.2"},
        "embedder": "none",
        "evaluator": "SimpleHeuristicEvaluator",
        "network": offline_network(),
        "operating_system": {
            "machine": "arm64",
            "platform": "macOS-14",
            "processor": "arm",
            "release": "23.0.0",
            "system": "Darwin",
            "version": "Darwin Kernel 23.0.0",
        },
        "package_version": "0.7.0",
        "python": {"implementation": "CPython", "version": "3.11.9"},
        "schema": MANIFEST_SCHEMA,
        "seed": 1729,
        "source": {"clean": True, "commit": "a" * 40},
    }
    manifest.update(overrides)
    return _signed(manifest)


# --------------------------------------------------------------------------- #
# Network classification helpers (Requirement 11.8)
# --------------------------------------------------------------------------- #
def test_offline_network_has_no_services_or_credentials() -> None:
    network = offline_network()
    assert network["offline"] is True
    assert network["required_services"] == []
    assert network["credential_categories"] == []


def test_service_network_documents_services_and_credentials() -> None:
    network = service_network(
        required_services=["openai-api"],
        credential_categories=["api-key"],
    )
    assert network["offline"] is False
    assert network["required_services"] == ["openai-api"]
    assert network["credential_categories"] == ["api-key"]


def test_service_network_rejects_empty_documentation() -> None:
    with pytest.raises(ValueError, match="must document at least one"):
        service_network(required_services=[], credential_categories=[])


# --------------------------------------------------------------------------- #
# Generation records every provenance field (Requirements 11.6, 11.8)
# --------------------------------------------------------------------------- #
def test_collect_records_all_required_provenance_fields() -> None:
    manifest = collect_benchmark_manifest(
        benchmark="learning",
        command=COMMAND,
        dataset_identity={"dataset_id": "abc123", "kind": "learning-scenario"},
        seed=1729,
        evaluator=EVALUATOR_IDENTITY,
        embedder=EMBEDDER_IDENTITY,
    )

    # Commit + source-tree cleanliness are recorded (values depend on git state).
    assert "commit" in manifest["source"]
    assert "clean" in manifest["source"]
    # Package/Python/dependency/OS versions.
    assert isinstance(manifest["package_version"], str)
    assert set(manifest["python"]) == {"implementation", "version"}
    assert set(manifest["dependencies"]) == {"aiosqlite", "pydantic"}
    assert set(manifest["operating_system"]) >= {
        "machine",
        "platform",
        "release",
        "system",
        "version",
    }
    # Dataset identity, seed, evaluator, embedder, command.
    assert manifest["dataset"]["dataset_id"] == "abc123"
    assert manifest["seed"] == 1729
    assert manifest["evaluator"] == EVALUATOR_IDENTITY
    assert manifest["embedder"] == EMBEDDER_IDENTITY
    assert manifest["command"] == list(COMMAND)
    # Offline/non-offline classification and self-consistent identity.
    assert manifest["network"]["offline"] is True
    assert manifest["manifest_id"] == manifest_identity(manifest)


def test_collect_requires_dataset_id() -> None:
    with pytest.raises(ValueError, match="dataset_id"):
        collect_benchmark_manifest(
            benchmark="learning",
            command=COMMAND,
            dataset_identity={"kind": "learning-scenario"},
            seed=1,
            evaluator="e",
            embedder="none",
        )


async def test_learning_benchmark_manifest_is_offline_and_complete() -> None:
    report = await run_benchmark(LearningScenario(seed=2024, rounds=2))
    manifest = build_manifest(report, command=COMMAND)

    assert manifest["benchmark"] == "learning"
    assert manifest["evaluator"] == "SimpleHeuristicEvaluator"
    assert manifest["embedder"] == "none"
    assert manifest["seed"] == report["seed"]
    assert manifest["dataset"]["dataset_id"] == report["controlled_inputs_id"]
    assert manifest["dataset"]["outcomes_id"] == report["outcomes_id"]
    assert manifest["network"] == offline_network()


async def test_manifest_identity_is_reproducible_for_identical_reports() -> None:
    # Identity excludes the wall-clock timestamp, so two builds from identical
    # reports in the same environment share one manifest identity.
    report = await run_benchmark(LearningScenario(seed=42, rounds=3))
    first = build_manifest(report, command=COMMAND)
    second = build_manifest(report, command=COMMAND)

    assert first["manifest_id"] == second["manifest_id"]
    assert first["manifest_id"] == manifest_identity(first)


# --------------------------------------------------------------------------- #
# Publication gate accepts a complete, clean, offline manifest
# --------------------------------------------------------------------------- #
def test_validate_accepts_complete_offline_manifest() -> None:
    validate_benchmark_manifest(_complete_manifest(), expected_dataset_id="abc123")


def test_validate_accepts_documented_non_offline_manifest() -> None:
    manifest = _complete_manifest(
        network=service_network(
            required_services=["evaluator-api"],
            credential_categories=["api-key"],
        )
    )
    validate_benchmark_manifest(manifest)


def test_serialize_manifest_round_trips_and_is_key_sorted() -> None:
    import json

    manifest = _complete_manifest()
    serialized = serialize_manifest(manifest)
    assert json.loads(serialized) == manifest
    assert serialized == json.dumps(manifest, indent=2, sort_keys=True) + "\n"


# --------------------------------------------------------------------------- #
# Publication gate rejects incomplete manifests (Requirements 5.9, 11.6)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "field",
    ["dataset", "dependencies", "network", "operating_system", "python", "source"],
)
def test_validate_rejects_missing_field(field) -> None:
    manifest = _complete_manifest()
    manifest.pop(field)
    manifest = _signed(manifest)
    with pytest.raises(ManifestValidationError, match="missing required fields"):
        validate_benchmark_manifest(manifest)


def test_validate_rejects_unknown_commit() -> None:
    manifest = _complete_manifest(source={"clean": True, "commit": "unknown"})
    with pytest.raises(ManifestValidationError, match="source commit"):
        validate_benchmark_manifest(manifest)


def test_validate_rejects_unclean_source_for_publication() -> None:
    manifest = _complete_manifest(source={"clean": False, "commit": "a" * 40})
    with pytest.raises(ManifestValidationError, match="unclean source tree"):
        validate_benchmark_manifest(manifest)
    # The same manifest is acceptable when the clean-source gate is relaxed.
    validate_benchmark_manifest(manifest, require_clean_source=False)


def test_validate_rejects_non_boolean_cleanliness() -> None:
    manifest = _complete_manifest(source={"clean": "yes", "commit": "a" * 40})
    with pytest.raises(ManifestValidationError, match="cleanliness"):
        validate_benchmark_manifest(manifest)


def test_validate_rejects_missing_dependency_version() -> None:
    manifest = _complete_manifest(
        dependencies={"aiosqlite": "0.20.0", "pydantic": "not-installed"}
    )
    with pytest.raises(ManifestValidationError, match="dependency version"):
        validate_benchmark_manifest(manifest)


def test_validate_rejects_incomplete_operating_system() -> None:
    manifest = _complete_manifest(
        operating_system={
            "machine": "arm64",
            "platform": "",
            "release": "23.0.0",
            "system": "Darwin",
            "version": "Darwin Kernel 23.0.0",
        }
    )
    with pytest.raises(ManifestValidationError, match="operating-system field"):
        validate_benchmark_manifest(manifest)


def test_validate_rejects_blank_seed_and_command() -> None:
    with pytest.raises(ManifestValidationError, match="seed must be an integer"):
        validate_benchmark_manifest(_complete_manifest(seed=True))
    with pytest.raises(ManifestValidationError, match="non-empty argument list"):
        validate_benchmark_manifest(_complete_manifest(command=[]))


# --------------------------------------------------------------------------- #
# Publication gate rejects mismatched manifests (Requirements 5.9, 11.8)
# --------------------------------------------------------------------------- #
def test_validate_rejects_tampered_identity() -> None:
    manifest = _complete_manifest()
    # Alter a field after signing so the recorded identity no longer matches.
    manifest["seed"] = manifest["seed"] + 1
    with pytest.raises(ManifestValidationError, match="identity does not match"):
        validate_benchmark_manifest(manifest)


def test_validate_rejects_mismatched_dataset_identity() -> None:
    manifest = _complete_manifest()
    with pytest.raises(
        ManifestValidationError, match="dataset identity does not match"
    ):
        validate_benchmark_manifest(manifest, expected_dataset_id="different")


def test_validate_rejects_mismatched_command() -> None:
    manifest = _complete_manifest()
    with pytest.raises(ManifestValidationError, match="command does not match"):
        validate_benchmark_manifest(manifest, expected_command=["python", "other"])


def test_validate_rejects_offline_manifest_that_requires_services() -> None:
    manifest = _complete_manifest(
        network={
            "credential_categories": [],
            "notes": "",
            "offline": True,
            "required_services": ["some-service"],
        }
    )
    with pytest.raises(ManifestValidationError, match="offline benchmark must not"):
        validate_benchmark_manifest(manifest)


def test_validate_rejects_non_offline_without_documentation() -> None:
    manifest = _complete_manifest(
        network={
            "credential_categories": [],
            "notes": "",
            "offline": False,
            "required_services": [],
        }
    )
    with pytest.raises(ManifestValidationError, match="non-offline benchmark must"):
        validate_benchmark_manifest(manifest)


def test_validate_rejects_unsupported_schema() -> None:
    manifest = _complete_manifest(schema="experia.something-else.v9")
    with pytest.raises(ManifestValidationError, match="schema is not supported"):
        validate_benchmark_manifest(manifest)
