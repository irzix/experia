"""Validated, bounded retrieval with indexed candidates and exact reranking."""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone

import aiosqlite

from experia.core.exceptions import ConfigurationError, StorageError
from experia.core.logging import (
    RetrievalDiagnostic,
    RetrievalDiagnosticCode,
    logger,
)
from experia.memory.embeddings import cosine_similarity
from experia.memory.models import Memory, MemoryType
from experia.memory.serialization import StorageSerializer
from experia.memory.vector_index import VectorCandidateIndex

MAX_RETRIEVAL_LIMIT = 100_000
# Ready-index semantic queries exact-rerank no more than this many source rows.
# The fixed cap keeps candidate memory independent of total stored-memory count.
DEFAULT_INDEX_CANDIDATE_BUDGET = 2_048
_STREAM_BATCH_SIZE = 256
_CANDIDATE_ID_BATCH_SIZE = 500
_MEMORY_COLUMNS = (
    "id, content, type, agent_role, confidence, importance, source, metadata, "
    "embedding, reinforcement_count, success_count, created_at, updated_at, "
    "expires_at"
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class RetrievalQuery:
    """One validated retrieval request with a stable UTC start time."""

    text: str = ""
    limit: int = 10
    memory_type: MemoryType | None = None
    agent_role: str | None = None
    query_embedding: tuple[float, ...] | None = None
    include_expired: bool = False
    started_at: datetime = field(default_factory=lambda: _utc_now())

    def __post_init__(self) -> None:
        if (
            not isinstance(self.limit, int)
            or isinstance(self.limit, bool)
            or not 0 <= self.limit <= MAX_RETRIEVAL_LIMIT
        ):
            raise ConfigurationError(
                "Retrieval limit must be a non-boolean integer from 0 through 100000.",
                feature="retrieval",
                parameter="limit",
            )

        if self.started_at.tzinfo is None or self.started_at.utcoffset() is None:
            raise ConfigurationError(
                "Retrieval start time must be timezone-aware.",
                feature="retrieval",
                parameter="started_at",
            )

        object.__setattr__(self, "started_at", self.started_at.astimezone(timezone.utc))
        if self.query_embedding is not None:
            object.__setattr__(self, "query_embedding", tuple(self.query_embedding))


def memory_is_eligible(memory: Memory, query: RetrievalQuery) -> bool:
    """Apply exact role and expiry boundaries against one query snapshot."""
    if query.memory_type is not None and memory.type != query.memory_type:
        return False

    if query.agent_role and not (
        memory.agent_role == query.agent_role
        or (memory.agent_role == "global" and memory.type == MemoryType.STRATEGY)
    ):
        return False

    return (
        query.include_expired
        or memory.expires_at is None
        or memory.expires_at > query.started_at
    )


def _memory_rank_key(
    memory: Memory, query: RetrievalQuery
) -> tuple[float, float, float, float, int]:
    """Return a key where larger values are better in the documented order."""
    if query.query_embedding:
        similarity = (
            cosine_similarity(list(query.query_embedding), memory.embedding)
            if memory.embedding
            else 0.0
        )
        score = 0.75 * similarity + 0.25 * memory.importance
    else:
        score = 0.0
    if math.isnan(score):
        score = -math.inf

    return (
        score,
        memory.importance,
        memory.confidence,
        memory.created_at.timestamp(),
        -memory.id.int,
    )


@dataclass(order=True)
class _RankedMemory:
    key: tuple[float, float, float, float, int]
    memory: Memory = field(compare=False)


class _BoundedMemoryRanker:
    """Retain only the exact best ``limit`` candidates seen so far."""

    def __init__(self, query: RetrievalQuery):
        self._query = query
        self._heap: list[_RankedMemory] = []

    def __len__(self) -> int:
        return len(self._heap)

    def add(self, memory: Memory) -> None:
        if self._query.limit == 0:
            return

        ranked = _RankedMemory(_memory_rank_key(memory, self._query), memory)
        if len(self._heap) < self._query.limit:
            heapq.heappush(self._heap, ranked)
        elif ranked.key > self._heap[0].key:
            heapq.heapreplace(self._heap, ranked)

    def results(self) -> list[Memory]:
        """Return retained memories from best to worst deterministically."""
        return [
            ranked.memory
            for ranked in sorted(self._heap, key=lambda item: item.key, reverse=True)
        ]


def rank_memories(memories: Sequence[Memory], query: RetrievalQuery) -> list[Memory]:
    """Return at most ``limit`` memories in the documented total order."""
    ranker = _BoundedMemoryRanker(query)
    for memory in memories:
        ranker.add(memory)
    return ranker.results()


class RetrievalEngine:
    """Run filtered retrieval behind the existing store search methods.

    A ready semantic index contributes at most ``candidate_budget`` source rows.
    Those approximate candidates are always scored and ordered exactly. If the
    derived index is absent or rebuilding, the engine emits one safe diagnostic
    and uses the bounded exact streaming path instead.
    """

    def __init__(
        self,
        connection: Callable[[], aiosqlite.Connection],
        serializer: StorageSerializer,
        candidate_index: VectorCandidateIndex,
        *,
        candidate_budget: int = DEFAULT_INDEX_CANDIDATE_BUDGET,
    ) -> None:
        if (
            isinstance(candidate_budget, bool)
            or not isinstance(candidate_budget, int)
            or candidate_budget <= 0
        ):
            raise ValueError("candidate_budget must be a positive integer")
        self._connection = connection
        self._serializer = serializer
        self._candidate_index = candidate_index
        self._candidate_budget = candidate_budget

    async def search(self, query: RetrievalQuery) -> list[Memory]:
        """Search via a ready index or the bounded exact streaming fallback."""
        if query.limit == 0:
            return []

        try:
            if not query.query_embedding:
                return await self._stream_exact(query)

            candidate_ids = await self._candidate_index.candidate_ids(
                query.query_embedding,
                budget=self._candidate_budget,
                memory_type=(
                    query.memory_type.value if query.memory_type is not None else None
                ),
                agent_role=query.agent_role,
                started_at=query.started_at,
                include_expired=query.include_expired,
            )
            if candidate_ids is None:
                self._emit_diagnostic(
                    RetrievalDiagnostic(
                        code=RetrievalDiagnosticCode.INDEX_FALLBACK,
                        memory_id=None,
                        stored_dimension=None,
                        query_dimension=None,
                    )
                )
                return await self._stream_exact(query)

            return await self._rank_indexed_candidates(query, candidate_ids)
        except StorageError:
            raise
        except Exception as exc:
            raise StorageError(
                "Failed to search memories.",
                operation="search",
                table="memories",
            ) from exc

    async def _rank_indexed_candidates(
        self,
        query: RetrievalQuery,
        candidate_ids: Sequence[str],
    ) -> list[Memory]:
        """Load a bounded candidate set and apply the exact total rank order."""
        ranker = _BoundedMemoryRanker(query)
        query_dimension = len(query.query_embedding or ())
        diagnosed_ids: set[object] = set()
        unique_ids = tuple(dict.fromkeys(candidate_ids))[: self._candidate_budget]

        for offset in range(0, len(unique_ids), _CANDIDATE_ID_BATCH_SIZE):
            batch = unique_ids[offset : offset + _CANDIDATE_ID_BATCH_SIZE]
            placeholders = ", ".join("?" for _ in batch)
            cursor = await self._connection().execute(
                f"SELECT {_MEMORY_COLUMNS} FROM memories WHERE id IN ({placeholders})",
                batch,
            )
            try:
                while rows := await cursor.fetchmany(_STREAM_BATCH_SIZE):
                    for row in rows:
                        memory = self._serializer.decode_memory(row)
                        if not memory_is_eligible(memory, query):
                            continue
                        if self._is_dimension_mismatch(
                            memory,
                            query_dimension,
                            diagnosed_ids,
                        ):
                            continue
                        ranker.add(memory)
            finally:
                await cursor.close()

        remaining_budget = self._candidate_budget - len(unique_ids)
        if remaining_budget:
            await self._add_unembedded_candidates(
                query,
                ranker,
                budget=remaining_budget,
            )

        await self._emit_indexed_mismatch_diagnostics(
            query,
            query_dimension=query_dimension,
            diagnosed_ids=diagnosed_ids,
        )
        return ranker.results()

    async def _add_unembedded_candidates(
        self,
        query: RetrievalQuery,
        ranker: _BoundedMemoryRanker,
        *,
        budget: int,
    ) -> None:
        """Preserve semantic-search behavior for bounded unembedded memories."""
        clauses, params = self._source_filters(query)
        clauses.append("embedding IS NULL")
        cursor = await self._connection().execute(
            f"SELECT {_MEMORY_COLUMNS} FROM memories WHERE "
            f"{' AND '.join(clauses)} "
            "ORDER BY importance DESC, confidence DESC, created_at DESC, id ASC",
            tuple(params),
        )
        accepted = 0
        try:
            while accepted < budget and (
                rows := await cursor.fetchmany(_STREAM_BATCH_SIZE)
            ):
                for row in rows:
                    memory = self._serializer.decode_memory(row)
                    if not memory_is_eligible(memory, query):
                        continue
                    ranker.add(memory)
                    accepted += 1
                    if accepted == budget:
                        break
        finally:
            await cursor.close()

    async def _emit_indexed_mismatch_diagnostics(
        self,
        query: RetrievalQuery,
        *,
        query_dimension: int,
        diagnosed_ids: set[object],
    ) -> None:
        """Diagnose eligible incompatible source vectors without loading payloads."""
        clauses, params = self._source_filters(query)
        clauses.extend(
            (
                "embedding IS NOT NULL",
                "(embedding_dimension IS NULL OR embedding_dimension != ?)",
            )
        )
        params.append(query_dimension)
        cursor = await self._connection().execute(
            f"SELECT {_MEMORY_COLUMNS} FROM memories WHERE {' AND '.join(clauses)}",
            tuple(params),
        )
        try:
            while rows := await cursor.fetchmany(_STREAM_BATCH_SIZE):
                for row in rows:
                    memory = self._serializer.decode_memory(row)
                    if memory_is_eligible(memory, query):
                        self._is_dimension_mismatch(
                            memory,
                            query_dimension,
                            diagnosed_ids,
                        )
        finally:
            await cursor.close()

    async def _stream_exact(self, query: RetrievalQuery) -> list[Memory]:
        """Stream every eligible row while retaining only the exact top K."""
        clauses, params = self._source_filters(
            query,
            include_text=not bool(query.query_embedding),
        )
        cursor = await self._connection().execute(
            f"SELECT {_MEMORY_COLUMNS} FROM memories WHERE {' AND '.join(clauses)}",
            tuple(params),
        )
        ranker = _BoundedMemoryRanker(query)
        query_dimension = len(query.query_embedding) if query.query_embedding else None
        diagnosed_ids: set[object] = set()
        try:
            while rows := await cursor.fetchmany(_STREAM_BATCH_SIZE):
                for row in rows:
                    memory = self._serializer.decode_memory(row)
                    if not memory_is_eligible(memory, query):
                        continue
                    if query_dimension is not None and self._is_dimension_mismatch(
                        memory,
                        query_dimension,
                        diagnosed_ids,
                    ):
                        continue
                    ranker.add(memory)
            return ranker.results()
        finally:
            await cursor.close()

    @staticmethod
    def _source_filters(
        query: RetrievalQuery,
        *,
        include_text: bool = False,
    ) -> tuple[list[str], list[object]]:
        clauses = ["1=1"]
        params: list[object] = []
        if query.memory_type is not None:
            clauses.append("type = ?")
            params.append(query.memory_type.value)
        if query.agent_role:
            clauses.append("(agent_role = ? OR (agent_role = ? AND type = ?))")
            params.extend(
                (
                    query.agent_role,
                    "global",
                    MemoryType.STRATEGY.value,
                )
            )
        if include_text and query.text:
            clauses.append("content LIKE ?")
            params.append(f"%{query.text}%")
        return clauses, params

    def _is_dimension_mismatch(
        self,
        memory: Memory,
        query_dimension: int,
        diagnosed_ids: set[object],
    ) -> bool:
        if memory.embedding is None or len(memory.embedding) == query_dimension:
            return False
        if memory.id not in diagnosed_ids:
            diagnosed_ids.add(memory.id)
            self._emit_diagnostic(
                RetrievalDiagnostic(
                    code=RetrievalDiagnosticCode.DIMENSION_MISMATCH,
                    memory_id=memory.id,
                    stored_dimension=len(memory.embedding),
                    query_dimension=query_dimension,
                )
            )
        return True

    @staticmethod
    def _emit_diagnostic(diagnostic: RetrievalDiagnostic) -> None:
        message = (
            "Memory omitted due to incompatible embedding dimension."
            if diagnostic.code is RetrievalDiagnosticCode.DIMENSION_MISMATCH
            else "Vector index unavailable; using exact streaming fallback."
        )
        logger.info(
            message,
            extra={"experia_retrieval_diagnostic": diagnostic},
        )


__all__ = [
    "DEFAULT_INDEX_CANDIDATE_BUDGET",
    "MAX_RETRIEVAL_LIMIT",
    "RetrievalEngine",
    "RetrievalQuery",
    "memory_is_eligible",
    "rank_memories",
]
