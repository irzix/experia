"""Offline retrieval benchmark runner and before/after report comparator.

Run the reference workload after generating artifacts::

    python -m benchmark.retrieval_dataset
    python -m benchmark.retrieval_benchmark run --variant before

After changing retrieval code, run the same generated artifacts as ``after`` and
compare the reports::

    python -m benchmark.retrieval_benchmark compare \
        --before benchmark/artifacts/retrieval-before.json \
        --after benchmark/artifacts/retrieval-after.json

The runner performs a fixed warmup, measures all 1,000 queries by default with
``time.perf_counter_ns()``, samples peak additional resident memory, and reports
exact-oracle recall@10, p50/p95 latency, and result counts. It records but does
not enforce the environment-dependent acceptance thresholds from task 9.10.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from benchmark.retrieval_dataset import (
    DEFAULT_DATABASE_NAME,
    DEFAULT_MANIFEST_NAME,
    DEFAULT_QUERY_NAME,
    ReferenceQuery,
    load_dataset_manifest,
    load_reference_queries,
    sha256_file,
)
from experia.memory.store import SQLiteStore

REPORT_SCHEMA = "experia.retrieval-benchmark-report.v1"
COMPARISON_SCHEMA = "experia.retrieval-benchmark-comparison.v1"
ENVIRONMENT_SCHEMA = "experia.retrieval-benchmark-environment.v1"
DEFAULT_WARMUP_QUERY_COUNT = 50
DEFAULT_REPORT_NAME = "retrieval-report.json"
_RUNTIME_DEPENDENCIES = ("aiosqlite", "pydantic")


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


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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


def collect_environment_identity(
    *,
    command: Sequence[str],
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Collect publication provenance and a comparison-safe environment key."""
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[1]
    )
    commit = _git_output(root, "rev-parse", "HEAD")
    status = _git_output(root, "status", "--porcelain")
    clean = None if status is None else status == ""
    dependencies = {
        name: _package_version(name) for name in sorted(_RUNTIME_DEPENDENCIES)
    }
    operating_system = {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "system": platform.system(),
        "version": platform.version(),
    }
    python = {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
    }
    comparison_payload = {
        "dependencies": dependencies,
        "operating_system": operating_system,
        "python": python,
        "sqlite": sqlite3.sqlite_version,
    }
    payload = {
        "command": list(command),
        "comparison_id": _identity(comparison_payload),
        "dependencies": dependencies,
        "embedder": "deterministic-reference-v1",
        "evaluator": "not-applicable",
        "network": {
            "credential_categories": [],
            "offline": True,
            "required_services": [],
        },
        "operating_system": operating_system,
        "package_version": _package_version("experia"),
        "python": python,
        "schema": ENVIRONMENT_SCHEMA,
        "source": {
            "clean": clean,
            "commit": commit or "unknown",
        },
        "sqlite": sqlite3.sqlite_version,
    }
    return {"environment_id": _identity(payload), **payload}


def _resource_rss_bytes() -> tuple[int, str] | None:
    try:
        import resource
    except ImportError:
        return None
    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    multiplier = 1 if sys.platform == "darwin" else 1024
    return int(maximum * multiplier), "resource_maxrss"


def _current_rss_bytes() -> tuple[int, str]:
    statm = Path("/proc/self/statm")
    if statm.exists():
        try:
            resident_pages = int(statm.read_text(encoding="ascii").split()[1])
            return resident_pages * os.sysconf("SC_PAGE_SIZE"), "proc_statm"
        except (OSError, ValueError, IndexError):
            pass
    resource_value = _resource_rss_bytes()
    if resource_value is not None:
        return resource_value
    # A portable fallback still captures Python-owned peak allocations. It is
    # used only where the operating system exposes neither current nor max RSS.
    import tracemalloc

    if not tracemalloc.is_tracing():
        tracemalloc.start()
    current, _ = tracemalloc.get_traced_memory()
    return current, "tracemalloc_current"


class PeakMemoryTracker:
    """Sample process memory without adding a runtime dependency."""

    def __init__(self, *, interval_seconds: float = 0.005) -> None:
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._baseline = 0
        self._peak = 0
        self.method = "unknown"

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("PeakMemoryTracker has already started")
        self._baseline, self.method = _current_rss_bytes()
        self._peak = self._baseline
        self._thread = threading.Thread(
            target=self._sample_until_stopped,
            name="experia-benchmark-memory-sampler",
            daemon=True,
        )
        self._thread.start()

    def sample(self) -> None:
        current, method = _current_rss_bytes()
        if method != self.method:
            raise RuntimeError(
                "Process memory sampling method changed during benchmark"
            )
        self._peak = max(self._peak, current)

    def stop(self) -> int:
        if self._thread is None:
            raise RuntimeError("PeakMemoryTracker has not started")
        self.sample()
        self._stop.set()
        self._thread.join(timeout=1)
        if self._thread.is_alive():
            raise RuntimeError("Peak memory sampler did not stop")
        return max(0, self._peak - self._baseline)

    def _sample_until_stopped(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            self.sample()


def _nearest_rank(values: Sequence[int], percentile: float) -> int:
    if not values:
        raise ValueError("Cannot calculate a percentile without observations")
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


async def _execute_query(store: SQLiteStore, query: ReferenceQuery) -> list[Any]:
    # ReferenceQuery fixes started_at as part of the query identity. The public
    # compatibility wrapper captures wall-clock time, so benchmark the engine
    # directly while retaining the store's lifecycle lease.
    async with store._operation():
        return await store._retrieval_engine.search(query.to_retrieval_query())


def _validate_artifacts(
    *,
    database_path: Path,
    queries_path: Path,
    manifest: dict[str, Any],
) -> None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Dataset artifact identities are missing")
    for name, path in (("database", database_path), ("queries", queries_path)):
        expected = artifacts.get(name)
        if not isinstance(expected, dict) or not isinstance(
            expected.get("sha256"), str
        ):
            raise ValueError(f"Dataset {name} identity is missing")
        if sha256_file(path) != expected["sha256"]:
            raise ValueError(f"Dataset {name} SHA-256 does not match manifest")


async def run_retrieval_benchmark(
    *,
    database_path: Path,
    queries_path: Path,
    manifest_path: Path,
    variant: str,
    output_path: Path | None = None,
    warmup_query_count: int = DEFAULT_WARMUP_QUERY_COUNT,
    measured_query_count: int | None = None,
    command: Sequence[str] = (),
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Run a fixed query prefix and emit a machine-readable benchmark report."""
    if not isinstance(variant, str) or not variant.strip():
        raise ValueError("variant must be non-empty text")
    if (
        isinstance(warmup_query_count, bool)
        or not isinstance(warmup_query_count, int)
        or warmup_query_count < 0
    ):
        raise ValueError("warmup_query_count must be a non-negative integer")

    database_path = Path(database_path).resolve()
    queries_path = Path(queries_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    manifest = load_dataset_manifest(manifest_path)
    _validate_artifacts(
        database_path=database_path,
        queries_path=queries_path,
        manifest=manifest,
    )
    dataset_identity = manifest["identity"]
    queries = load_reference_queries(
        queries_path,
        expected_dataset_id=dataset_identity["dataset_id"],
    )
    if len(queries) != dataset_identity["query_count"]:
        raise ValueError("Manifest query count does not match query artifact")
    if measured_query_count is None:
        measured_query_count = len(queries)
    if (
        isinstance(measured_query_count, bool)
        or not isinstance(measured_query_count, int)
        or not 1 <= measured_query_count <= len(queries)
    ):
        raise ValueError("measured_query_count must select a non-empty query prefix")
    measured_queries = queries[:measured_query_count]

    environment = collect_environment_identity(
        command=command,
        repository_root=repository_root,
    )
    store = SQLiteStore(str(database_path))
    tracker = PeakMemoryTracker()
    tracker_started = False
    try:
        await store.initialize()
        count_cursor = await store._require_conn().execute(
            "SELECT COUNT(*) FROM memories"
        )
        stored_memory_count = int((await count_cursor.fetchone())[0])
        if stored_memory_count != dataset_identity["memory_count"]:
            raise ValueError("Database memory count does not match manifest")
        if not await store._vector_index.is_ready():
            raise ValueError("Reference database vector index is not ready")

        for warmup_index in range(warmup_query_count):
            await _execute_query(store, queries[warmup_index % len(queries)])

        latencies_ns: list[int] = []
        outcomes: list[dict[str, Any]] = []
        recalls: list[float] = []
        result_counts: list[int] = []
        tracker.start()
        tracker_started = True
        for query in measured_queries:
            started_ns = time.perf_counter_ns()
            results = await _execute_query(store, query)
            elapsed_ns = time.perf_counter_ns() - started_ns
            tracker.sample()
            result_ids = tuple(str(memory.id) for memory in results)
            oracle_set = set(query.oracle_ids)
            recall = len(oracle_set.intersection(result_ids[:10])) / 10
            latencies_ns.append(elapsed_ns)
            recalls.append(recall)
            result_counts.append(len(result_ids))
            outcomes.append(
                {
                    "latency_ns": elapsed_ns,
                    "oracle_ids": list(query.oracle_ids),
                    "query_id": query.query_id,
                    "recall_at_10": recall,
                    "result_count": len(result_ids),
                    "result_ids": list(result_ids),
                }
            )
        peak_additional_memory_bytes = tracker.stop()
        tracker_started = False
    finally:
        if tracker_started:
            tracker.stop()
        await store.close()

    p50_ns = _nearest_rank(latencies_ns, 0.50)
    p95_ns = _nearest_rank(latencies_ns, 0.95)
    metrics = {
        "p50_latency_ms": p50_ns / 1_000_000,
        "p95_latency_ms": p95_ns / 1_000_000,
        "peak_additional_memory_bytes": peak_additional_memory_bytes,
        "peak_additional_memory_mib": peak_additional_memory_bytes / (1024 * 1024),
        "recall_at_10": sum(recalls) / len(recalls),
        "recall_at_10_min": min(recalls),
        "result_count_max": max(result_counts),
        "result_count_min": min(result_counts),
        "result_count_total": sum(result_counts),
    }
    report_payload = {
        "benchmark": {
            "acceptance_thresholds_enforced": False,
            "environment_dependent": True,
            "measured_query_count": measured_query_count,
            "memory_sampling_method": tracker.method,
            "query_limit": 10,
            "timer": "time.perf_counter_ns",
            "variant": variant.strip(),
            "warmup_query_count": warmup_query_count,
        },
        "dataset_identity": dataset_identity,
        "environment_identity": environment,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "query_outcomes": outcomes,
        "schema": REPORT_SCHEMA,
    }
    report = {"report_id": _identity(report_payload), **report_payload}
    if output_path is not None:
        _write_json(Path(output_path), report)
    return report


def _load_report(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Benchmark report could not be read") from exc
    if not isinstance(value, dict) or value.get("schema") != REPORT_SCHEMA:
        raise ValueError("Unsupported benchmark report schema")
    return value


def _metric_delta(
    before: float | int, after: float | int
) -> dict[str, float | int | None]:
    return {
        "absolute": after - before,
        "percent": None if before == 0 else ((after - before) / before) * 100,
    }


def compare_reports(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Compare reports only when dataset, queries, and environment match."""
    if before.get("schema") != REPORT_SCHEMA or after.get("schema") != REPORT_SCHEMA:
        raise ValueError("Both inputs must be retrieval benchmark reports")
    before_dataset = before["dataset_identity"]
    after_dataset = after["dataset_identity"]
    if before_dataset["dataset_id"] != after_dataset["dataset_id"]:
        raise ValueError("Before/after dataset identities differ")

    controlled_fields = (
        "measured_query_count",
        "query_limit",
        "warmup_query_count",
    )
    if any(
        before["benchmark"][field] != after["benchmark"][field]
        for field in controlled_fields
    ):
        raise ValueError("Before/after query or warmup settings differ")
    if (
        before["environment_identity"]["comparison_id"]
        != after["environment_identity"]["comparison_id"]
    ):
        raise ValueError("Before/after reference environment settings differ")

    before_queries = before["query_outcomes"]
    after_queries = after["query_outcomes"]
    before_inputs = [
        (outcome["query_id"], outcome["oracle_ids"]) for outcome in before_queries
    ]
    after_inputs = [
        (outcome["query_id"], outcome["oracle_ids"]) for outcome in after_queries
    ]
    if before_inputs != after_inputs:
        raise ValueError("Before/after fixed query inputs differ")

    metric_names = (
        "p50_latency_ms",
        "p95_latency_ms",
        "peak_additional_memory_bytes",
        "result_count_total",
        "recall_at_10",
    )
    before_metrics = {name: before["metrics"][name] for name in metric_names}
    after_metrics = {name: after["metrics"][name] for name in metric_names}
    comparison_payload = {
        "before": {
            "environment_id": before["environment_identity"]["environment_id"],
            "metrics": before_metrics,
            "report_id": before["report_id"],
            "source": before["environment_identity"]["source"],
            "variant": before["benchmark"]["variant"],
        },
        "controlled_inputs": {
            "dataset_id": before_dataset["dataset_id"],
            "environment_comparison_id": before["environment_identity"][
                "comparison_id"
            ],
            **{field: before["benchmark"][field] for field in controlled_fields},
            "query_ids": [outcome["query_id"] for outcome in before_queries],
        },
        "deltas": {
            name: _metric_delta(before_metrics[name], after_metrics[name])
            for name in metric_names
        },
        "after": {
            "environment_id": after["environment_identity"]["environment_id"],
            "metrics": after_metrics,
            "report_id": after["report_id"],
            "source": after["environment_identity"]["source"],
            "variant": after["benchmark"]["variant"],
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema": COMPARISON_SCHEMA,
    }
    comparison = {
        "comparison_id": _identity(comparison_payload),
        **comparison_payload,
    }
    if output_path is not None:
        _write_json(Path(output_path), comparison)
    return comparison


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or compare deterministic Experia retrieval benchmarks."
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)

    run = subparsers.add_parser("run", help="Run one benchmark variant.")
    run.add_argument(
        "--artifact-directory",
        type=Path,
        default=Path("benchmark/artifacts"),
    )
    run.add_argument("--database", type=Path)
    run.add_argument("--queries", type=Path)
    run.add_argument("--dataset-manifest", type=Path)
    run.add_argument("--variant", required=True)
    run.add_argument("--output", type=Path)
    run.add_argument(
        "--warmup-queries",
        type=int,
        default=DEFAULT_WARMUP_QUERY_COUNT,
    )
    run.add_argument(
        "--measured-queries",
        type=int,
        help=(
            "Measure a fixed query prefix. Omit for all 1,000 reference queries; "
            "use a reduced value only for smoke validation."
        ),
    )

    compare = subparsers.add_parser(
        "compare",
        help="Emit a controlled before/after comparison.",
    )
    compare.add_argument("--before", type=Path, required=True)
    compare.add_argument("--after", type=Path, required=True)
    compare.add_argument("--output", type=Path)
    return parser


async def _main_async(arguments: argparse.Namespace) -> int:
    if arguments.operation == "run":
        directory = arguments.artifact_directory
        database = arguments.database or directory / DEFAULT_DATABASE_NAME
        queries = arguments.queries or directory / DEFAULT_QUERY_NAME
        manifest = arguments.dataset_manifest or directory / DEFAULT_MANIFEST_NAME
        output = arguments.output or directory / DEFAULT_REPORT_NAME
        report = await run_retrieval_benchmark(
            database_path=database,
            queries_path=queries,
            manifest_path=manifest,
            variant=arguments.variant,
            output_path=output,
            warmup_query_count=arguments.warmup_queries,
            measured_query_count=arguments.measured_queries,
            command=(
                sys.executable,
                "-m",
                "benchmark.retrieval_benchmark",
                *sys.argv[1:],
            ),
        )
        print(
            json.dumps(
                {
                    "metrics": report["metrics"],
                    "output": str(output),
                    "report_id": report["report_id"],
                },
                sort_keys=True,
            )
        )
        return 0

    before = _load_report(arguments.before)
    after = _load_report(arguments.after)
    output = arguments.output or arguments.after.with_name("retrieval-comparison.json")
    comparison = compare_reports(before, after, output_path=output)
    print(
        json.dumps(
            {
                "comparison_id": comparison["comparison_id"],
                "deltas": comparison["deltas"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    arguments = _parser().parse_args()
    return asyncio.run(_main_async(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
