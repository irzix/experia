"""Deterministic, rebuildable SQLite vector candidate index."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache

import aiosqlite

from experia.core.exceptions import StorageError
from experia.memory.serialization import EncodedMemory
from experia.memory.transactions import SQLiteTransactionManager

CURRENT_VECTOR_INDEX_VERSION = 1
VECTOR_INDEX_BAND_COUNT = 8
VECTOR_INDEX_BITS_PER_BAND = 8
DEFAULT_REBUILD_BATCH_SIZE = 256
INDEX_STATUS_READY = "ready"
INDEX_STATUS_REBUILDING = "rebuilding"


@dataclass(frozen=True)
class VectorIndexStatus:
    """Persisted availability and rebuild cursor for one index version."""

    index_version: int
    status: str
    last_memory_id: str | None

    @property
    def ready(self) -> bool:
        return (
            self.index_version == CURRENT_VECTOR_INDEX_VERSION
            and self.status == INDEX_STATUS_READY
        )


@dataclass(frozen=True)
class VectorIndexRebuildResult:
    """Summary of work committed by one resumable rebuild invocation."""

    index_version: int
    processed_memories: int
    complete: bool
    last_memory_id: str | None


@lru_cache(maxsize=32)
def _projection_signs(
    index_version: int,
    dimension: int,
    band_count: int,
    bits_per_band: int,
) -> tuple[tuple[int, ...], ...]:
    """Return stable Rademacher projection vectors for one dimension."""
    projection_count = band_count * bits_per_band
    seed = b"experia-lsh" + struct.pack(
        ">IIII", index_version, dimension, band_count, bits_per_band
    )
    material = hashlib.shake_256(seed).digest(projection_count * dimension)
    return tuple(
        tuple(
            1 if material[projection * dimension + component] & 1 else -1
            for component in range(dimension)
        )
        for projection in range(projection_count)
    )


def _stored_embedding(value: object, *, memory_id: str) -> tuple[float, ...]:
    try:
        if not isinstance(value, str):
            raise TypeError("stored embedding is not text")
        decoded = json.loads(value)
        if not isinstance(decoded, list) or any(
            isinstance(component, bool)
            or not isinstance(component, (int, float))
            or not math.isfinite(component)
            for component in decoded
        ):
            raise TypeError("stored embedding is not a finite numeric vector")
        return tuple(float(component) for component in decoded)
    except Exception as exc:
        raise StorageError(
            "Stored embedding cannot be indexed.",
            operation="rebuild_index",
            table="memories",
            record_ids=(memory_id,),
            field="embedding",
        ) from exc


class VectorCandidateIndex:
    """Maintain deterministic LSH bands as rebuildable derived data.

    Source writes call :meth:`maintain_memory` with their existing transaction.
    Rebuilds commit bounded batches and persist a cursor after each batch, so a
    cancellation or failure can resume without changing source records. A
    candidate result of ``None`` means callers must use the exact streaming
    fallback because the index is absent or rebuilding.
    """

    def __init__(
        self,
        connection: Callable[[], aiosqlite.Connection],
        transactions: SQLiteTransactionManager,
        *,
        index_version: int = CURRENT_VECTOR_INDEX_VERSION,
        band_count: int = VECTOR_INDEX_BAND_COUNT,
        bits_per_band: int = VECTOR_INDEX_BITS_PER_BAND,
    ) -> None:
        for parameter, value in (
            ("index_version", index_version),
            ("band_count", band_count),
            ("bits_per_band", bits_per_band),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{parameter} must be a positive integer")
        self._connection = connection
        self._transactions = transactions
        self.index_version = index_version
        self.band_count = band_count
        self.bits_per_band = bits_per_band

    def band_rows(
        self,
        memory_id: str,
        embedding: Sequence[float],
    ) -> tuple[tuple[object, ...], ...]:
        """Build deterministic, versioned SQLite rows for one embedding."""
        vector = self._validated_vector(embedding)
        dimension = len(vector)
        # Large-magnitude finite embeddings can overflow the projection sum
        # below. Scaling the whole vector by one positive power of two divides
        # every projection by the same positive factor, so the sign of every
        # projection - and therefore every bucket - is unchanged, while the
        # running sum stays finite. Power-of-two scaling is exact, so buckets
        # for normal-range vectors are byte-identical.
        peak = max((abs(component) for component in vector), default=0.0)
        if peak > 1.0:
            _, exponent = math.frexp(peak)
            vector = tuple(math.ldexp(component, -exponent) for component in vector)
        projections = _projection_signs(
            self.index_version,
            dimension,
            self.band_count,
            self.bits_per_band,
        )
        rows: list[tuple[object, ...]] = []
        for band in range(self.band_count):
            bucket = 0
            projection_offset = band * self.bits_per_band
            for bit in range(self.bits_per_band):
                coefficients = projections[projection_offset + bit]
                projection = math.fsum(
                    component * coefficient
                    for component, coefficient in zip(vector, coefficients)
                )
                if projection >= 0.0:
                    bucket |= 1 << bit
            rows.append((memory_id, dimension, band, bucket, self.index_version))
        return tuple(rows)

    async def maintain_encoded_memory(
        self,
        conn: aiosqlite.Connection,
        memory: EncodedMemory,
    ) -> None:
        """Replace one memory's bands inside the caller's source transaction."""
        embedding = (
            None
            if memory.embedding is None
            else _stored_embedding(memory.embedding, memory_id=memory.id)
        )
        await self.maintain_memory(conn, memory.id, embedding)

    async def maintain_memory(
        self,
        conn: aiosqlite.Connection,
        memory_id: str,
        embedding: Sequence[float] | None,
    ) -> None:
        """Replace one memory's derived rows without opening a transaction."""
        state = await self._read_status(conn)
        if state is None:
            raise StorageError(
                "Vector index state is unavailable.",
                operation="maintain_index",
                table="memory_vector_index_state",
                record_ids=(memory_id,),
            )
        if state.index_version != self.index_version:
            await conn.execute(
                "UPDATE memory_vector_index_state "
                "SET index_version = ?, status = ?, last_memory_id = NULL "
                "WHERE singleton = 1",
                (self.index_version, INDEX_STATUS_REBUILDING),
            )

        await conn.execute(
            "DELETE FROM memory_vector_bands WHERE memory_id = ?",
            (memory_id,),
        )
        if embedding is None:
            return
        await conn.executemany(
            "INSERT INTO memory_vector_bands "
            "(memory_id, dimension, band, bucket, index_version) "
            "VALUES (?, ?, ?, ?, ?)",
            self.band_rows(memory_id, embedding),
        )

    async def status(self) -> VectorIndexStatus | None:
        """Return ``None`` when the derived index schema is unavailable."""
        try:
            return await self._read_status(self._connection())
        except aiosqlite.OperationalError as exc:
            if "no such table" in str(exc).casefold():
                return None
            raise StorageError(
                "Vector index status could not be read.",
                operation="index_status",
                table="memory_vector_index_state",
            ) from exc

    async def is_ready(self) -> bool:
        """Report whether current-version candidate data is safe to query."""
        status = await self.status()
        return status is not None and status.ready

    async def candidate_ids(
        self,
        query_embedding: Sequence[float],
        *,
        budget: int,
        memory_type: str | None = None,
        agent_role: str | None = None,
        started_at: datetime | None = None,
        include_expired: bool = True,
    ) -> tuple[str, ...] | None:
        """Return bounded, eligible, same-dimension LSH candidates.

        ``None`` means the derived index is absent or rebuilding and callers
        must use their exact streaming fallback. Candidate order is stable by
        matched-band count and memory identifier.
        """
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
            raise ValueError("budget must be a non-negative integer")
        if not isinstance(include_expired, bool):
            raise ValueError("include_expired must be a boolean")
        if not include_expired and (
            started_at is None
            or started_at.tzinfo is None
            or started_at.utcoffset() is None
        ):
            raise ValueError(
                "an aware started_at is required when expired rows are excluded"
            )

        state = await self.status()
        if state is None or not state.ready:
            return None

        query_rows = self.band_rows("", query_embedding)
        try:
            conn = self._connection()
            values_sql = ", ".join("(?, ?)" for _ in query_rows)
            parameters: list[object] = []
            for _, _, band, bucket, _ in query_rows:
                parameters.extend((band, bucket))

            query_dimension = query_rows[0][1]
            match_clauses = [
                "indexed.dimension = ?",
                "source.embedding_dimension = ?",
                "indexed.index_version = ?",
            ]
            parameters.extend((query_dimension, query_dimension, self.index_version))
            if memory_type is not None:
                match_clauses.append("source.type = ?")
                parameters.append(memory_type)
            if agent_role:
                match_clauses.append(
                    "(source.agent_role = ? OR "
                    "(source.agent_role = ? AND source.type = ?))"
                )
                parameters.extend((agent_role, "global", "strategy"))
            if not include_expired:
                match_clauses.append(
                    "(source.expires_at IS NULL OR "
                    "julianday(source.expires_at) > julianday(?))"
                )
                parameters.append(started_at.isoformat())

            parameters.extend(
                (
                    budget,
                    self.index_version,
                    INDEX_STATUS_READY,
                )
            )
            cursor = await conn.execute(
                f"""
                WITH query_bands(band, bucket) AS (VALUES {values_sql}),
                matches AS (
                    SELECT indexed.memory_id, COUNT(*) AS matched_bands
                    FROM memory_vector_bands AS indexed
                    JOIN query_bands
                      ON query_bands.band = indexed.band
                     AND query_bands.bucket = indexed.bucket
                    JOIN memories AS source ON source.id = indexed.memory_id
                    WHERE {" AND ".join(match_clauses)}
                    GROUP BY indexed.memory_id
                    ORDER BY matched_bands DESC, indexed.memory_id ASC
                    LIMIT ?
                )
                SELECT state.index_version, state.status, matches.memory_id,
                       matches.matched_bands
                FROM memory_vector_index_state AS state
                LEFT JOIN matches
                  ON state.index_version = ? AND state.status = ?
                WHERE state.singleton = 1
                ORDER BY matches.matched_bands DESC, matches.memory_id ASC
                """,
                tuple(parameters),
            )
            rows = await cursor.fetchall()
            if not rows:
                return None
            if (
                int(rows[0][0]) != self.index_version
                or str(rows[0][1]) != INDEX_STATUS_READY
            ):
                return None
            return tuple(str(row[2]) for row in rows if row[2] is not None)
        except aiosqlite.OperationalError as exc:
            if "no such table" in str(exc).casefold():
                return None
            raise StorageError(
                "Vector candidates could not be read.",
                operation="index_candidates",
                table="memory_vector_bands",
            ) from exc
        except StorageError:
            raise
        except Exception as exc:
            raise StorageError(
                "Vector candidates could not be read.",
                operation="index_candidates",
                table="memory_vector_bands",
            ) from exc

    async def rebuild(
        self,
        *,
        batch_size: int = DEFAULT_REBUILD_BATCH_SIZE,
        force: bool = False,
    ) -> VectorIndexRebuildResult:
        """Build current-version rows in resumable committed batches."""
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(force, bool):
            raise ValueError("force must be a boolean")

        processed = 0
        async with self._transactions.write(
            operation="rebuild_index",
            table="memory_vector_bands,memory_vector_index_state",
        ) as conn:
            state = await self._read_status(conn)
            if state is None:
                raise StorageError(
                    "Vector index state is unavailable.",
                    operation="rebuild_index",
                    table="memory_vector_index_state",
                )
            if force or state.index_version != self.index_version:
                await conn.execute("DELETE FROM memory_vector_bands")
                await conn.execute(
                    "UPDATE memory_vector_index_state "
                    "SET index_version = ?, status = ?, last_memory_id = NULL "
                    "WHERE singleton = 1",
                    (self.index_version, INDEX_STATUS_REBUILDING),
                )
            elif state.ready:
                return VectorIndexRebuildResult(
                    index_version=self.index_version,
                    processed_memories=0,
                    complete=True,
                    last_memory_id=None,
                )

        while True:
            batch: list[tuple[object, object]] = []
            complete = False
            last_memory_id: str | None = None
            async with self._transactions.write(
                operation="rebuild_index",
                table="memory_vector_bands,memory_vector_index_state",
            ) as conn:
                state = await self._read_status(conn)
                if state is None:
                    raise StorageError(
                        "Vector index state is unavailable.",
                        operation="rebuild_index",
                        table="memory_vector_index_state",
                    )
                cursor = await conn.execute(
                    "SELECT id, embedding FROM memories "
                    "WHERE embedding IS NOT NULL "
                    "AND (? IS NULL OR id > ?) ORDER BY id LIMIT ?",
                    (state.last_memory_id, state.last_memory_id, batch_size),
                )
                batch = list(await cursor.fetchall())
                if batch:
                    for raw_memory_id, stored_embedding in batch:
                        memory_id = str(raw_memory_id)
                        await self.maintain_memory(
                            conn,
                            memory_id,
                            _stored_embedding(stored_embedding, memory_id=memory_id),
                        )
                    last_memory_id = str(batch[-1][0])
                    await conn.execute(
                        "UPDATE memory_vector_index_state "
                        "SET status = ?, last_memory_id = ? WHERE singleton = 1",
                        (INDEX_STATUS_REBUILDING, last_memory_id),
                    )
                else:
                    await conn.execute(
                        "DELETE FROM memory_vector_bands "
                        "WHERE index_version != ? OR memory_id IN "
                        "(SELECT id FROM memories WHERE embedding IS NULL)",
                        (self.index_version,),
                    )
                    missing_cursor = await conn.execute(
                        """
                        SELECT memories.id
                        FROM memories
                        LEFT JOIN memory_vector_bands AS indexed
                          ON indexed.memory_id = memories.id
                         AND indexed.dimension = memories.embedding_dimension
                         AND indexed.index_version = ?
                         AND indexed.band >= 0 AND indexed.band < ?
                        WHERE memories.embedding IS NOT NULL
                        GROUP BY memories.id
                        HAVING COUNT(indexed.band) != ?
                        ORDER BY memories.id
                        LIMIT 1
                        """,
                        (
                            self.index_version,
                            self.band_count,
                            self.band_count,
                        ),
                    )
                    missing = await missing_cursor.fetchone()
                    if missing is None:
                        await conn.execute(
                            "UPDATE memory_vector_index_state "
                            "SET status = ?, last_memory_id = NULL WHERE singleton = 1",
                            (INDEX_STATUS_READY,),
                        )
                        complete = True
                    else:
                        await conn.execute(
                            "UPDATE memory_vector_index_state "
                            "SET status = ?, last_memory_id = NULL WHERE singleton = 1",
                            (INDEX_STATUS_REBUILDING,),
                        )

            processed += len(batch)
            if complete:
                return VectorIndexRebuildResult(
                    index_version=self.index_version,
                    processed_memories=processed,
                    complete=True,
                    last_memory_id=None,
                )

    @staticmethod
    async def _read_status(
        conn: aiosqlite.Connection,
    ) -> VectorIndexStatus | None:
        cursor = await conn.execute(
            "SELECT index_version, status, last_memory_id "
            "FROM memory_vector_index_state WHERE singleton = 1"
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return VectorIndexStatus(
            index_version=int(row[0]),
            status=str(row[1]),
            last_memory_id=None if row[2] is None else str(row[2]),
        )

    @staticmethod
    def _validated_vector(embedding: Sequence[float]) -> tuple[float, ...]:
        try:
            vector = tuple(embedding)
        except TypeError as exc:
            raise ValueError("embedding must be a finite numeric sequence") from exc
        if any(
            isinstance(component, bool)
            or not isinstance(component, (int, float))
            or not math.isfinite(component)
            for component in vector
        ):
            raise ValueError("embedding must be a finite numeric sequence")
        return tuple(float(component) for component in vector)


__all__ = [
    "CURRENT_VECTOR_INDEX_VERSION",
    "DEFAULT_REBUILD_BATCH_SIZE",
    "INDEX_STATUS_READY",
    "INDEX_STATUS_REBUILDING",
    "VECTOR_INDEX_BAND_COUNT",
    "VECTOR_INDEX_BITS_PER_BAND",
    "VectorCandidateIndex",
    "VectorIndexRebuildResult",
    "VectorIndexStatus",
]
