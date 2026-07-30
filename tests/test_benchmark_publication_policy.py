"""Publication-policy tests spanning the reproducible benchmark suite (task 16.5).

Tasks 16.1–16.3 each proved one piece in isolation: the deterministic learning
runner, the provenance manifest and its publication gate, and the enforceable
offline harness. This module verifies the pieces hold together as a single
*publication policy* — the end-to-end contract a maintainer relies on when
publishing benchmark numbers:

  * **Provenance / environment fields** — a published learning result carries
    every field required for reproducibility (commit, source cleanliness,
    package/Python/dependency/OS versions, dataset identity, seed, evaluator,
    embedder, command, network classification), and a published retrieval result
    carries the equivalent environment identity. (Requirements 11.6)
  * **Clean-state enforcement** — every benchmark variant starts from clean
    persisted state, and the publication gate refuses a result recorded against
    an unclean source tree. (Requirements 11.6, 11.7)
  * **Offline network denial** — the offline benchmark completes with external
    network access denied, its manifest is classified offline, and denying the
    network does not change the deterministic result. (Requirements 11.8, 11.9)
  * **Identical query / dataset settings** — a before/after retrieval comparison
    is emitted only when the dataset identity, query/warmup/limit settings,
    fixed query inputs, and reference environment all match. (Requirement 5.9)
  * **Before/after retrieval metrics** — that comparison publishes p50 latency,
    p95 latency, peak additional memory, result count, and recall@10 for both
    variants and their deltas. (Requirement 5.9)

Everything here is fully offline (no LLM, no network, no credentials).

Validates: Requirements 5.9, 11.6, 11.7, 11.8, 11.9
"""

from __future__ import annotations

import socket
from copy import deepcopy

import pytest

from benchmark.learning_benchmark import (
    EMBEDDER_IDENTITY,
    EVALUATOR_IDENTITY,
    LearningScenario,
    build_manifest,
    run_benchmark,
    run_offline_benchmark,
)
from benchmark.manifest import (
    ManifestValidationError,
    manifest_identity,
    offline_network,
    validate_benchmark_manifest,
)
from benchmark.offline import NetworkAccessDeniedError, deny_network
from benchmark.retrieval_benchmark import (
    COMPARISON_SCHEMA,
    REPORT_SCHEMA,
    compare_reports,
    run_retrieval_benchmark,
)
from benchmark.retrieval_dataset import (
    ReferenceDatasetConfig,
    generate_reference_dataset,
)

LEARNING_COMMAND = (
    "python",
    "-m",
    "benchmark.learning_benchmark",
    "--manifest",
    "m.json",
)
RETRIEVAL_COMMAND = ("python", "-m", "benchmark.retrieval_benchmark", "run")

# The five before/after metrics a retrieval-change publication must report.
REQUIRED_RETRIEVAL_METRICS = frozenset(
    {
        "p50_latency_ms",
        "p95_latency_ms",
        "peak_additional_memory_bytes",
        "result_count_total",
        "recall_at_10",
    }
)


def _resign(manifest: dict) -> dict:
    """Return a copy whose recorded identity matches its (mutated) contents."""
    manifest = deepcopy(manifest)
    manifest.pop("manifest_id", None)
    manifest["manifest_id"] = manifest_identity(manifest)
    return manifest


# --------------------------------------------------------------------------- #
# Provenance / environment fields (Requirement 11.6)
# --------------------------------------------------------------------------- #
async def test_published_learning_result_records_every_provenance_field() -> None:
    report = await run_offline_benchmark(LearningScenario(seed=2024, rounds=2))
    manifest = build_manifest(report, command=LEARNING_COMMAND)

    # Source commit and cleanliness are recorded (values depend on git state).
    assert isinstance(manifest["source"]["commit"], str)
    assert isinstance(manifest["source"]["clean"], bool)
    # Package / Python / dependency / OS versions.
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
    # Dataset identity, seed, evaluator, embedder, command bound to the report.
    assert manifest["dataset"]["dataset_id"] == report["controlled_inputs_id"]
    assert manifest["dataset"]["outcomes_id"] == report["outcomes_id"]
    assert manifest["seed"] == report["seed"]
    assert manifest["evaluator"] == EVALUATOR_IDENTITY
    assert manifest["embedder"] == EMBEDDER_IDENTITY
    assert manifest["command"] == list(LEARNING_COMMAND)
    # Offline/non-offline classification, self-consistent identity.
    assert manifest["network"] == offline_network()
    assert manifest["manifest_id"] == manifest_identity(manifest)


async def test_published_retrieval_result_records_environment_identity(
    tmp_path,
) -> None:
    artifacts = await generate_reference_dataset(
        tmp_path / "artifacts",
        config=ReferenceDatasetConfig(
            memory_count=120,
            query_count=6,
            seed=4321,
            embedding_dimension=8,
            batch_size=17,
        ),
    )
    report = await run_retrieval_benchmark(
        database_path=artifacts.database_path,
        queries_path=artifacts.queries_path,
        manifest_path=artifacts.manifest_path,
        variant="before",
        warmup_query_count=2,
        measured_query_count=6,
        command=RETRIEVAL_COMMAND,
    )

    environment = report["environment_identity"]
    # Source commit and cleanliness.
    assert (
        isinstance(environment["source"]["commit"], str)
        and environment["source"]["commit"]
    )
    assert "clean" in environment["source"]
    # Package / Python / dependency / OS / sqlite versions and the exact command.
    assert isinstance(environment["package_version"], str)
    assert set(environment["python"]) == {"implementation", "version"}
    assert set(environment["dependencies"]) == {"aiosqlite", "pydantic"}
    assert set(environment["operating_system"]) >= {
        "machine",
        "platform",
        "system",
        "version",
    }
    assert isinstance(environment["sqlite"], str)
    assert environment["command"] == list(RETRIEVAL_COMMAND)
    # Dataset identity and offline classification travel with the result.
    assert report["dataset_identity"] == artifacts.manifest["identity"]
    assert environment["network"]["offline"] is True


# --------------------------------------------------------------------------- #
# Clean-state enforcement (Requirements 11.6, 11.7)
# --------------------------------------------------------------------------- #
async def test_every_published_variant_starts_from_clean_persisted_state() -> None:
    report = await run_offline_benchmark(LearningScenario(seed=7, rounds=3))
    assert report["variants"], "the benchmark must produce comparison variants"
    assert all(
        variant["clean_start"] is True for variant in report["variants"].values()
    )


async def test_publication_gate_rejects_result_from_unclean_source_tree() -> None:
    report = await run_offline_benchmark(LearningScenario(seed=13, rounds=2))
    manifest = build_manifest(report, command=LEARNING_COMMAND)

    # Normalise the git-dependent source to a known-clean, known-commit state so
    # the policy is exercised deterministically regardless of the working tree.
    clean = _resign({**manifest, "source": {"clean": True, "commit": "a" * 40}})
    validate_benchmark_manifest(
        clean, expected_dataset_id=report["controlled_inputs_id"]
    )

    unclean = _resign({**clean, "source": {"clean": False, "commit": "a" * 40}})
    with pytest.raises(ManifestValidationError, match="unclean source tree"):
        validate_benchmark_manifest(unclean)
    # The same result is publishable only when the clean-state gate is relaxed.
    validate_benchmark_manifest(unclean, require_clean_source=False)


# --------------------------------------------------------------------------- #
# Offline network denial (Requirements 11.8, 11.9)
# --------------------------------------------------------------------------- #
async def test_offline_benchmark_completes_under_network_denial() -> None:
    scenario = LearningScenario(seed=2024, rounds=3)
    guarded = await run_offline_benchmark(scenario)
    unguarded = await run_benchmark(scenario)

    # Denying the network cannot change the deterministic published result.
    assert guarded == unguarded
    assert guarded["variants"]["experia"]["totals"]["successes"] > 0
    manifest = build_manifest(guarded, command=LEARNING_COMMAND)
    assert manifest["network"] == offline_network()
    assert manifest["network"]["required_services"] == []
    assert manifest["network"]["credential_categories"] == []


def test_network_denial_blocks_accidental_external_access() -> None:
    with deny_network():
        with pytest.raises(NetworkAccessDeniedError):
            socket.create_connection(("example.com", 80), timeout=0.01)


# --------------------------------------------------------------------------- #
# Before/after retrieval metrics under identical settings (Requirement 5.9)
# --------------------------------------------------------------------------- #
async def test_retrieval_change_publishes_all_before_after_metrics(tmp_path) -> None:
    artifacts = await generate_reference_dataset(
        tmp_path / "artifacts",
        config=ReferenceDatasetConfig(
            memory_count=120,
            query_count=6,
            seed=4321,
            embedding_dimension=8,
            batch_size=17,
        ),
    )
    shared = dict(
        database_path=artifacts.database_path,
        queries_path=artifacts.queries_path,
        manifest_path=artifacts.manifest_path,
        warmup_query_count=2,
        measured_query_count=6,
        command=RETRIEVAL_COMMAND,
    )
    # Identical queries, dataset contents, and environment for both variants.
    before = await run_retrieval_benchmark(variant="before", **shared)
    after = await run_retrieval_benchmark(variant="after", **shared)

    comparison = compare_reports(before, after)

    assert comparison["schema"] == COMPARISON_SCHEMA
    # All five publication metrics are reported for both variants and the deltas.
    assert set(comparison["before"]["metrics"]) == REQUIRED_RETRIEVAL_METRICS
    assert set(comparison["after"]["metrics"]) == REQUIRED_RETRIEVAL_METRICS
    assert set(comparison["deltas"]) == REQUIRED_RETRIEVAL_METRICS
    for delta in comparison["deltas"].values():
        assert set(delta) == {"absolute", "percent"}
    # The comparison records the identical controlled inputs it enforced.
    controlled = comparison["controlled_inputs"]
    assert controlled["dataset_id"] == artifacts.manifest["identity"]["dataset_id"]
    assert controlled["measured_query_count"] == 6
    assert controlled["query_limit"] == 10
    assert controlled["warmup_query_count"] == 2
    assert (
        controlled["environment_comparison_id"]
        == before["environment_identity"]["comparison_id"]
    )


def _retrieval_report(variant: str, **overrides) -> dict:
    """A complete synthetic retrieval report carrying all five metrics."""
    report = {
        "benchmark": {
            "measured_query_count": 2,
            "query_limit": 10,
            "variant": variant,
            "warmup_query_count": 1,
        },
        "dataset_identity": {"dataset_id": "dataset-1"},
        "environment_identity": {
            "comparison_id": "environment-1",
            "environment_id": f"run-{variant}",
            "source": {"clean": True, "commit": f"{variant}-commit"},
        },
        "metrics": {
            "p50_latency_ms": 2.0,
            "p95_latency_ms": 3.0,
            "peak_additional_memory_bytes": 100,
            "recall_at_10": 0.9,
            "result_count_total": 20,
        },
        "query_outcomes": [
            {"oracle_ids": [str(index) for index in range(10)], "query_id": "q0"},
            {"oracle_ids": [str(index) for index in range(10)], "query_id": "q1"},
        ],
        "report_id": f"{variant}-report",
        "schema": REPORT_SCHEMA,
    }
    for key, value in overrides.items():
        report[key] = value
    return report


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda report: report["dataset_identity"].__setitem__(
                "dataset_id", "dataset-2"
            ),
            "dataset identities differ",
        ),
        (
            lambda report: report["benchmark"].__setitem__("warmup_query_count", 5),
            "query or warmup settings differ",
        ),
        (
            lambda report: report["benchmark"].__setitem__("measured_query_count", 3),
            "query or warmup settings differ",
        ),
        (
            lambda report: report["benchmark"].__setitem__("query_limit", 25),
            "query or warmup settings differ",
        ),
        (
            lambda report: report["environment_identity"].__setitem__(
                "comparison_id", "environment-2"
            ),
            "reference environment settings differ",
        ),
        (
            lambda report: report["query_outcomes"][0].__setitem__("query_id", "qX"),
            "fixed query inputs differ",
        ),
        (
            lambda report: report["query_outcomes"][1].__setitem__(
                "oracle_ids", [str(index) for index in range(1, 11)]
            ),
            "fixed query inputs differ",
        ),
    ],
)
def test_comparison_refused_when_controlled_inputs_differ(mutate, match) -> None:
    before = _retrieval_report("before")
    after = _retrieval_report("after")
    mutate(after)
    with pytest.raises(ValueError, match=match):
        compare_reports(before, after)


def test_comparison_accepts_matching_controlled_inputs() -> None:
    # The same synthetic pair, differing only in outcome metrics, compares
    # cleanly and reports every required metric delta.
    before = _retrieval_report("before")
    after = _retrieval_report(
        "after",
        metrics={
            "p50_latency_ms": 1.5,
            "p95_latency_ms": 2.5,
            "peak_additional_memory_bytes": 90,
            "recall_at_10": 1.0,
            "result_count_total": 20,
        },
    )

    comparison = compare_reports(before, after)

    assert set(comparison["deltas"]) == REQUIRED_RETRIEVAL_METRICS
    assert comparison["deltas"]["p95_latency_ms"]["absolute"] == pytest.approx(-0.5)
    assert comparison["deltas"]["recall_at_10"]["absolute"] == pytest.approx(0.1)
    assert comparison["deltas"]["result_count_total"]["absolute"] == 0
