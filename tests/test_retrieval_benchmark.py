"""Reduced-size validation for the 100k/1k retrieval benchmark tooling."""

from __future__ import annotations

import json
import sqlite3
from copy import deepcopy

import pytest

from benchmark.retrieval_benchmark import (
    COMPARISON_SCHEMA,
    ENVIRONMENT_SCHEMA,
    REPORT_SCHEMA,
    compare_reports,
    run_retrieval_benchmark,
)
from benchmark.retrieval_dataset import (
    DEFAULT_MEMORY_COUNT,
    DEFAULT_QUERY_COUNT,
    ReferenceDatasetConfig,
    generate_reference_dataset,
    load_reference_queries,
)


def test_reference_defaults_and_reduced_config_validation() -> None:
    config = ReferenceDatasetConfig()

    assert config.memory_count == DEFAULT_MEMORY_COUNT == 100_000
    assert config.query_count == DEFAULT_QUERY_COUNT == 1_000
    assert config.role_count == 100
    with pytest.raises(ValueError, match="at least 10 memories"):
        ReferenceDatasetConfig(memory_count=39, query_count=4)
    with pytest.raises(ValueError, match="integer"):
        ReferenceDatasetConfig(seed=True)


@pytest.mark.asyncio
async def test_reduced_dataset_generation_is_logically_and_byte_deterministic(
    tmp_path,
) -> None:
    config = ReferenceDatasetConfig(
        memory_count=40,
        query_count=4,
        seed=1234,
        embedding_dimension=4,
        batch_size=13,
    )

    first = await generate_reference_dataset(tmp_path / "first", config=config)
    second = await generate_reference_dataset(tmp_path / "second", config=config)

    assert first.manifest["identity"] == second.manifest["identity"]
    assert (
        first.manifest["artifacts"]["database"]["sha256"]
        == second.manifest["artifacts"]["database"]["sha256"]
    )
    assert first.queries_path.read_bytes() == second.queries_path.read_bytes()
    assert sum(first.manifest["distribution"]["memory_types"].values()) == 40
    assert first.manifest["distribution"]["agent_roles"]["count"] == 4

    queries = load_reference_queries(
        first.queries_path,
        expected_dataset_id=first.manifest["identity"]["dataset_id"],
    )
    assert len(queries) == 4
    assert all(len(query.oracle_ids) == 10 for query in queries)

    connection = sqlite3.connect(first.database_path)
    try:
        memory_count = connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        band_count = connection.execute(
            "SELECT COUNT(*) FROM memory_vector_bands"
        ).fetchone()[0]
        status = connection.execute(
            "SELECT status FROM memory_vector_index_state WHERE singleton = 1"
        ).fetchone()[0]
    finally:
        connection.close()
    assert memory_count == 40
    assert band_count == 40 * 8
    assert status == "ready"


@pytest.mark.asyncio
async def test_reduced_benchmark_smoke_emits_complete_machine_readable_report(
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
    output_path = tmp_path / "retrieval-before.json"

    report = await run_retrieval_benchmark(
        database_path=artifacts.database_path,
        queries_path=artifacts.queries_path,
        manifest_path=artifacts.manifest_path,
        variant="before",
        output_path=output_path,
        warmup_query_count=2,
        measured_query_count=6,
        command=("python", "-m", "benchmark.retrieval_benchmark", "run"),
    )

    assert report["schema"] == REPORT_SCHEMA
    assert report["dataset_identity"] == artifacts.manifest["identity"]
    assert report["benchmark"]["acceptance_thresholds_enforced"] is False
    assert report["benchmark"]["warmup_query_count"] == 2
    assert report["benchmark"]["measured_query_count"] == 6
    assert report["metrics"]["p50_latency_ms"] <= report["metrics"]["p95_latency_ms"]
    assert report["metrics"]["peak_additional_memory_bytes"] >= 0
    assert report["metrics"]["result_count_total"] == 60
    assert report["metrics"]["recall_at_10"] == 1.0
    assert report["metrics"]["recall_at_10_min"] == 1.0
    assert len(report["query_outcomes"]) == 6
    assert report["environment_identity"]["schema"] == ENVIRONMENT_SCHEMA
    assert report["environment_identity"]["source"]["commit"]
    assert "clean" in report["environment_identity"]["source"]
    assert report["environment_identity"]["network"]["offline"] is True
    assert json.loads(output_path.read_text(encoding="utf-8")) == report


def test_before_after_comparison_requires_identical_controlled_inputs() -> None:
    base_report = {
        "benchmark": {
            "measured_query_count": 1,
            "query_limit": 10,
            "variant": "before",
            "warmup_query_count": 1,
        },
        "dataset_identity": {"dataset_id": "dataset-1"},
        "environment_identity": {
            "comparison_id": "environment-1",
            "environment_id": "run-before",
            "source": {"clean": True, "commit": "before-commit"},
        },
        "metrics": {
            "p50_latency_ms": 2.0,
            "p95_latency_ms": 3.0,
            "peak_additional_memory_bytes": 100,
            "recall_at_10": 0.9,
            "result_count_total": 10,
        },
        "query_outcomes": [
            {"oracle_ids": [str(index) for index in range(10)], "query_id": "q0"}
        ],
        "report_id": "before-report",
        "schema": REPORT_SCHEMA,
    }
    after_report = deepcopy(base_report)
    after_report["benchmark"]["variant"] = "after"
    after_report["environment_identity"]["environment_id"] = "run-after"
    after_report["environment_identity"]["source"]["commit"] = "after-commit"
    after_report["metrics"]["p95_latency_ms"] = 2.5
    after_report["metrics"]["recall_at_10"] = 1.0
    after_report["report_id"] = "after-report"

    comparison = compare_reports(base_report, after_report)

    assert comparison["schema"] == COMPARISON_SCHEMA
    assert comparison["before"]["metrics"]["p95_latency_ms"] == 3.0
    assert comparison["after"]["metrics"]["p95_latency_ms"] == 2.5
    assert comparison["deltas"]["p95_latency_ms"]["absolute"] == -0.5
    assert comparison["deltas"]["recall_at_10"]["absolute"] == pytest.approx(0.1)
    assert comparison["controlled_inputs"]["dataset_id"] == "dataset-1"

    mismatched = deepcopy(after_report)
    mismatched["dataset_identity"]["dataset_id"] = "dataset-2"
    with pytest.raises(ValueError, match="dataset identities differ"):
        compare_reports(base_report, mismatched)
