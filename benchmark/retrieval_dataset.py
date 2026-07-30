"""Deterministic reference dataset generation for retrieval benchmarks.

The reference defaults are 100,000 memories and 1,000 semantic queries. Memory
``i`` belongs to query cluster ``i % query_count`` and one of at most 100
role-isolated partitions. Every memory has a deterministic UUID, timestamp,
type, confidence, importance, metadata, and finite embedding. The manifest
records the exact type/role distribution and generation formulas.

Each query artifact carries an exact top-10 oracle. The oracle is computed by
scoring every eligible memory in the query's role partition with the documented
retrieval score and total tie order; it is not derived from index candidates.
Reduced sizes are supported for smoke tests, while command defaults retain the
published 100,000/1,000 reference scale.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import heapq
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from experia.memory.models import Memory, MemoryType
from experia.memory.retrieval import RetrievalQuery
from experia.memory.store import SQLiteStore

DATASET_SCHEMA = "experia.reference-retrieval-dataset.v1"
QUERY_SCHEMA = "experia.reference-retrieval-queries.v1"
GENERATOR_VERSION = 1
DEFAULT_MEMORY_COUNT = 100_000
DEFAULT_QUERY_COUNT = 1_000
DEFAULT_SEED = 20_250_301
DEFAULT_EMBEDDING_DIMENSION = 16
DEFAULT_BATCH_SIZE = 500
DEFAULT_DATABASE_NAME = "reference-retrieval-v1.db"
DEFAULT_QUERY_NAME = "reference-retrieval-v1.queries.json"
DEFAULT_MANIFEST_NAME = "reference-retrieval-v1.dataset.json"
REFERENCE_CREATED_AT = datetime(2025, 1, 1, tzinfo=timezone.utc)
REFERENCE_QUERY_STARTED_AT = datetime(2030, 1, 1, tzinfo=timezone.utc)
_MEMORY_TYPES = (
    MemoryType.FACT,
    MemoryType.PREFERENCE,
    MemoryType.LESSON,
    MemoryType.RULE,
    MemoryType.STRATEGY,
    MemoryType.EXPERIENCE,
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _hash_record(digest: Any, value: Any) -> None:
    digest.update(_canonical_json(value).encode("utf-8"))
    digest.update(b"\n")


@dataclass(frozen=True)
class ReferenceDatasetConfig:
    """Logical and operational settings for one dataset generation."""

    memory_count: int = DEFAULT_MEMORY_COUNT
    query_count: int = DEFAULT_QUERY_COUNT
    seed: int = DEFAULT_SEED
    embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION
    batch_size: int = DEFAULT_BATCH_SIZE

    def __post_init__(self) -> None:
        for name in (
            "memory_count",
            "query_count",
            "seed",
            "embedding_dimension",
            "batch_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.query_count <= 0:
            raise ValueError("query_count must be positive")
        if self.memory_count < self.query_count * 10:
            raise ValueError("memory_count must provide at least 10 memories per query")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.embedding_dimension < 2:
            raise ValueError("embedding_dimension must be at least 2")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    @property
    def role_count(self) -> int:
        return min(100, self.query_count)

    def identity_settings(self) -> dict[str, int]:
        """Return only logical settings; batch size cannot alter identity."""
        return {
            "embedding_dimension": self.embedding_dimension,
            "generator_version": GENERATOR_VERSION,
            "memory_count": self.memory_count,
            "query_count": self.query_count,
            "role_count": self.role_count,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class ReferenceQuery:
    """A fixed semantic query and its independently computed exact oracle."""

    query_id: str
    agent_role: str
    embedding: tuple[float, ...]
    oracle_ids: tuple[str, ...]
    started_at: datetime = REFERENCE_QUERY_STARTED_AT
    limit: int = 10

    def to_retrieval_query(self) -> RetrievalQuery:
        return RetrievalQuery(
            limit=self.limit,
            agent_role=self.agent_role,
            query_embedding=self.embedding,
            include_expired=False,
            started_at=self.started_at,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_role": self.agent_role,
            "embedding": list(self.embedding),
            "limit": self.limit,
            "oracle_ids": list(self.oracle_ids),
            "query_id": self.query_id,
            "started_at": self.started_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ReferenceQuery:
        try:
            query_id = value["query_id"]
            agent_role = value["agent_role"]
            embedding = value["embedding"]
            oracle_ids = value["oracle_ids"]
            limit = value["limit"]
            started_at = datetime.fromisoformat(value["started_at"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Malformed reference query") from exc
        if not isinstance(query_id, str) or not query_id:
            raise ValueError("Reference query_id must be non-empty text")
        if not isinstance(agent_role, str) or not agent_role:
            raise ValueError("Reference agent_role must be non-empty text")
        if limit != 10:
            raise ValueError("Reference query limit must be 10")
        if not isinstance(embedding, list) or not embedding:
            raise ValueError("Reference query embedding must be non-empty")
        if any(
            isinstance(component, bool)
            or not isinstance(component, (int, float))
            or not math.isfinite(component)
            for component in embedding
        ):
            raise ValueError("Reference query embedding must contain finite numbers")
        if not isinstance(oracle_ids, list) or len(oracle_ids) != 10:
            raise ValueError("Reference query must contain exactly 10 oracle IDs")
        if any(not isinstance(memory_id, str) for memory_id in oracle_ids):
            raise ValueError("Reference oracle IDs must be text")
        if started_at.tzinfo is None or started_at.utcoffset() is None:
            raise ValueError("Reference query start must be timezone-aware")
        return cls(
            query_id=query_id,
            agent_role=agent_role,
            embedding=tuple(float(component) for component in embedding),
            oracle_ids=tuple(oracle_ids),
            started_at=started_at,
            limit=limit,
        )


@dataclass(frozen=True)
class ReferenceDatasetArtifacts:
    """Paths and manifest emitted by one successful generation."""

    database_path: Path
    queries_path: Path
    manifest_path: Path
    manifest: dict[str, Any]


def default_artifact_paths(output_directory: Path) -> tuple[Path, Path, Path]:
    return (
        output_directory / DEFAULT_DATABASE_NAME,
        output_directory / DEFAULT_QUERY_NAME,
        output_directory / DEFAULT_MANIFEST_NAME,
    )


def _normalized(values: list[float]) -> tuple[float, ...]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        raise ValueError("Deterministic vector material produced a zero vector")
    return tuple(round(value / norm, 12) for value in values)


def _hash_vector(label: str, dimension: int) -> tuple[float, ...]:
    material = hashlib.shake_256(label.encode("utf-8")).digest(dimension * 2)
    values = []
    for offset in range(0, len(material), 2):
        unsigned = int.from_bytes(material[offset : offset + 2], "big")
        values.append((unsigned - 32_767.5) / 32_767.5)
    return _normalized(values)


def _query_vector(
    config: ReferenceDatasetConfig, query_index: int
) -> tuple[float, ...]:
    return _hash_vector(
        f"experia-reference-query-v{GENERATOR_VERSION}:{config.seed}:{query_index}",
        config.embedding_dimension,
    )


def _memory_vector(
    config: ReferenceDatasetConfig,
    *,
    query_index: int,
    ordinal: int,
    base: tuple[float, ...],
) -> tuple[float, ...]:
    if ordinal == 0:
        return base
    noise = _hash_vector(
        "experia-reference-memory-noise-"
        f"v{GENERATOR_VERSION}:{config.seed}:{query_index}:{ordinal}",
        config.embedding_dimension,
    )
    scale = min(0.25, ordinal * 0.002)
    return _normalized(
        [
            component + scale * perturbation
            for component, perturbation in zip(base, noise)
        ]
    )


def _role_for_query(config: ReferenceDatasetConfig, query_index: int) -> str:
    return f"role-{query_index % config.role_count:03d}"


def _cosine_similarity(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for left, right in zip(a, b):
        dot += left * right
        norm_a += left * left
        norm_b += right * right
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def _oracle_key(
    *,
    query_embedding: tuple[float, ...],
    memory_embedding: tuple[float, ...],
    importance: float,
    confidence: float,
    created_at: datetime,
    memory_id: UUID,
) -> tuple[float, float, float, float, int]:
    similarity = _cosine_similarity(query_embedding, memory_embedding)
    score = 0.75 * similarity + 0.25 * importance
    return (
        score,
        importance,
        confidence,
        created_at.timestamp(),
        -memory_id.int,
    )


def _memory_payload(memory: Memory) -> dict[str, Any]:
    return {
        "agent_role": memory.agent_role,
        "confidence": memory.confidence,
        "content": memory.content,
        "created_at": memory.created_at.isoformat(),
        "embedding": memory.embedding,
        "expires_at": None,
        "id": str(memory.id),
        "importance": memory.importance,
        "metadata": memory.metadata,
        "reinforcement_count": memory.reinforcement_count,
        "source": memory.source,
        "success_count": memory.success_count,
        "type": memory.type.value,
        "updated_at": memory.updated_at.isoformat(),
    }


def _build_memory(
    config: ReferenceDatasetConfig,
    *,
    memory_index: int,
    query_vectors: tuple[tuple[float, ...], ...],
) -> Memory:
    query_index = memory_index % config.query_count
    ordinal = memory_index // config.query_count
    created_at = REFERENCE_CREATED_AT + timedelta(seconds=memory_index)
    return Memory(
        id=UUID(int=memory_index + 1),
        content=(
            f"reference memory {memory_index:06d}; cluster {query_index:04d}; "
            f"ordinal {ordinal:03d}"
        ),
        type=_MEMORY_TYPES[memory_index % len(_MEMORY_TYPES)],
        agent_role=_role_for_query(config, query_index),
        confidence=round(1.0 - (ordinal % 20) * 0.02, 6),
        importance=round(1.0 - min(ordinal, 99) / 100, 6),
        source=f"reference-retrieval-dataset-v{GENERATOR_VERSION}",
        metadata={
            "cluster": query_index,
            "ordinal": ordinal,
            "seed": config.seed,
        },
        embedding=list(
            _memory_vector(
                config,
                query_index=query_index,
                ordinal=ordinal,
                base=query_vectors[query_index],
            )
        ),
        reinforcement_count=ordinal % 7,
        success_count=ordinal % 5,
        created_at=created_at,
        updated_at=created_at,
        expires_at=None,
    )


def _cleanup_database(path: Path) -> None:
    for candidate in (
        path,
        Path(f"{path}-journal"),
        Path(f"{path}-wal"),
        Path(f"{path}-shm"),
    ):
        candidate.unlink(missing_ok=True)


async def generate_reference_dataset(
    output_directory: Path,
    *,
    config: ReferenceDatasetConfig = ReferenceDatasetConfig(),
    replace: bool = False,
) -> ReferenceDatasetArtifacts:
    """Generate a SQLite dataset, fixed query artifact, and identity manifest."""
    output_directory = Path(output_directory).resolve()
    database_path, queries_path, manifest_path = default_artifact_paths(
        output_directory
    )
    existing = [
        path for path in (database_path, queries_path, manifest_path) if path.exists()
    ]
    if existing and not replace:
        raise FileExistsError(f"Reference artifacts already exist: {existing[0]}")
    output_directory.mkdir(parents=True, exist_ok=True)
    if replace:
        _cleanup_database(database_path)
        queries_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)

    query_vectors = tuple(
        _query_vector(config, query_index) for query_index in range(config.query_count)
    )
    query_roles = tuple(
        _role_for_query(config, query_index)
        for query_index in range(config.query_count)
    )
    queries_by_role: dict[str, list[int]] = defaultdict(list)
    for query_index, role in enumerate(query_roles):
        queries_by_role[role].append(query_index)

    oracle_heaps: list[list[tuple[tuple[float, float, float, float, int], str]]] = [
        [] for _ in range(config.query_count)
    ]
    memory_digest = hashlib.sha256()
    type_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()

    store = SQLiteStore(str(database_path))
    try:
        await store.initialize()
        fixed_audit_time = REFERENCE_CREATED_AT.isoformat()
        async with store._transactions.write(
            operation="benchmark_dataset",
            table="schema_migrations",
        ) as connection:
            await connection.execute(
                "UPDATE schema_migrations SET applied_at = ?",
                (fixed_audit_time,),
            )

        for batch_start in range(0, config.memory_count, config.batch_size):
            memory_rows: list[tuple[object, ...]] = []
            band_rows: list[tuple[object, ...]] = []
            batch_end = min(batch_start + config.batch_size, config.memory_count)
            for memory_index in range(batch_start, batch_end):
                memory = _build_memory(
                    config,
                    memory_index=memory_index,
                    query_vectors=query_vectors,
                )
                encoded = store._serializer.encode_memory(memory)
                memory_rows.append((*encoded.values(), encoded.embedding_dimension))
                band_rows.extend(
                    store._vector_index.band_rows(
                        encoded.id,
                        memory.embedding or (),
                    )
                )
                payload = _memory_payload(memory)
                _hash_record(memory_digest, payload)
                type_counts[memory.type.value] += 1
                role_counts[memory.agent_role] += 1

                memory_embedding = tuple(memory.embedding or ())
                for query_index in queries_by_role[memory.agent_role]:
                    key = _oracle_key(
                        query_embedding=query_vectors[query_index],
                        memory_embedding=memory_embedding,
                        importance=memory.importance,
                        confidence=memory.confidence,
                        created_at=memory.created_at,
                        memory_id=memory.id,
                    )
                    heap = oracle_heaps[query_index]
                    ranked = (key, str(memory.id))
                    if len(heap) < 10:
                        heapq.heappush(heap, ranked)
                    elif key > heap[0][0]:
                        heapq.heapreplace(heap, ranked)

            async with store._transactions.write(
                operation="benchmark_dataset",
                table="memories,memory_vector_bands",
            ) as connection:
                await connection.executemany(
                    """
                    INSERT INTO memories
                    (id, content, type, agent_role, confidence, importance, source,
                     metadata, embedding, reinforcement_count, success_count,
                     created_at, updated_at, expires_at, embedding_dimension)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    memory_rows,
                )
                await connection.executemany(
                    "INSERT INTO memory_vector_bands "
                    "(memory_id, dimension, band, bucket, index_version) "
                    "VALUES (?, ?, ?, ?, ?)",
                    band_rows,
                )

        if not await store._vector_index.is_ready():
            raise RuntimeError("Generated vector index is not ready")
        await store._require_conn().execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except BaseException:
        await store.close()
        _cleanup_database(database_path)
        raise
    else:
        await store.close()

    query_digest = hashlib.sha256()
    queries: list[ReferenceQuery] = []
    for query_index in range(config.query_count):
        oracle_ids = tuple(
            memory_id
            for _, memory_id in sorted(
                oracle_heaps[query_index],
                reverse=True,
            )
        )
        if len(oracle_ids) != 10:
            raise RuntimeError(
                f"Exact oracle for query {query_index} contains {len(oracle_ids)} rows"
            )
        query = ReferenceQuery(
            query_id=f"query-{query_index:04d}",
            agent_role=query_roles[query_index],
            embedding=query_vectors[query_index],
            oracle_ids=oracle_ids,
        )
        queries.append(query)
        _hash_record(query_digest, query.to_dict())

    identity_payload = {
        **config.identity_settings(),
        "logical_memories_sha256": memory_digest.hexdigest(),
        "logical_queries_sha256": query_digest.hexdigest(),
    }
    dataset_id = hashlib.sha256(
        _canonical_json(identity_payload).encode("utf-8")
    ).hexdigest()
    query_document = {
        "dataset_id": dataset_id,
        "queries": [query.to_dict() for query in queries],
        "query_count": config.query_count,
        "schema": QUERY_SCHEMA,
    }
    _write_json(queries_path, query_document)

    database_sha256 = sha256_file(database_path)
    queries_sha256 = sha256_file(queries_path)
    manifest = {
        "artifacts": {
            "database": {
                "file": database_path.name,
                "sha256": database_sha256,
                "size_bytes": database_path.stat().st_size,
            },
            "queries": {
                "file": queries_path.name,
                "sha256": queries_sha256,
                "size_bytes": queries_path.stat().st_size,
            },
        },
        "distribution": {
            "agent_roles": {
                "count": config.role_count,
                "counts": dict(sorted(role_counts.items())),
                "formula": "role-{(query_index modulo role_count):03d}",
            },
            "assignment": "memory_index modulo query_count",
            "created_at_start": REFERENCE_CREATED_AT.isoformat(),
            "embedding": {
                "dimension": config.embedding_dimension,
                "generator": "normalized SHAKE-256 cluster vector plus bounded deterministic noise",
            },
            "expiry": "all memories are non-expiring",
            "memory_id": "UUID integer value memory_index + 1",
            "memory_types": dict(sorted(type_counts.items())),
            "oracle": (
                "exact score over every same-role eligible memory using score DESC, "
                "importance DESC, confidence DESC, creation DESC, ID ASC"
            ),
        },
        "identity": {
            "dataset_id": dataset_id,
            **identity_payload,
        },
        "query_contract": {
            "fixed_order": "query_id ascending",
            "include_expired": False,
            "limit": 10,
            "started_at": REFERENCE_QUERY_STARTED_AT.isoformat(),
        },
        "reference_defaults": {
            "memory_count": DEFAULT_MEMORY_COUNT,
            "query_count": DEFAULT_QUERY_COUNT,
        },
        "schema": DATASET_SCHEMA,
    }
    _write_json(manifest_path, manifest)
    return ReferenceDatasetArtifacts(
        database_path=database_path,
        queries_path=queries_path,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def load_dataset_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Dataset manifest could not be read") from exc
    if not isinstance(value, dict) or value.get("schema") != DATASET_SCHEMA:
        raise ValueError("Unsupported dataset manifest schema")
    identity = value.get("identity")
    if not isinstance(identity, dict) or not isinstance(
        identity.get("dataset_id"), str
    ):
        raise ValueError("Dataset manifest identity is missing")
    return value


def load_reference_queries(
    path: Path,
    *,
    expected_dataset_id: str,
) -> tuple[ReferenceQuery, ...]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Reference queries could not be read") from exc
    if not isinstance(value, dict) or value.get("schema") != QUERY_SCHEMA:
        raise ValueError("Unsupported reference query schema")
    if value.get("dataset_id") != expected_dataset_id:
        raise ValueError("Reference query dataset identity does not match manifest")
    raw_queries = value.get("queries")
    if not isinstance(raw_queries, list):
        raise ValueError("Reference query list is missing")
    queries = tuple(ReferenceQuery.from_dict(query) for query in raw_queries)
    if value.get("query_count") != len(queries):
        raise ValueError("Reference query count does not match payload")
    if tuple(query.query_id for query in queries) != tuple(
        sorted(query.query_id for query in queries)
    ):
        raise ValueError("Reference queries must be ordered by query_id")
    return queries


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the deterministic Experia retrieval reference dataset.",
        epilog=(
            "Defaults generate the release reference scale (100,000 memories and "
            "1,000 fixed queries). Use reduced counts only for local smoke validation."
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("benchmark/artifacts"),
    )
    parser.add_argument("--memory-count", type=int, default=DEFAULT_MEMORY_COUNT)
    parser.add_argument("--query-count", type=int, default=DEFAULT_QUERY_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--embedding-dimension",
        type=int,
        default=DEFAULT_EMBEDDING_DIMENSION,
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing generated artifacts in the output directory.",
    )
    return parser


async def _main_async(arguments: argparse.Namespace) -> int:
    artifacts = await generate_reference_dataset(
        arguments.output_directory,
        config=ReferenceDatasetConfig(
            memory_count=arguments.memory_count,
            query_count=arguments.query_count,
            seed=arguments.seed,
            embedding_dimension=arguments.embedding_dimension,
            batch_size=arguments.batch_size,
        ),
        replace=arguments.replace,
    )
    print(
        json.dumps(
            {
                "database": str(artifacts.database_path),
                "dataset_id": artifacts.manifest["identity"]["dataset_id"],
                "manifest": str(artifacts.manifest_path),
                "queries": str(artifacts.queries_path),
            },
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    return asyncio.run(_main_async(_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
