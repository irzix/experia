"""Property coverage for bounded, isolated, deterministic retrieval."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from experia.memory.models import Memory, MemoryType
from experia.memory.retrieval import RetrievalQuery
from experia.memory.store import SQLiteStore
from experia.memory.vector_index import INDEX_STATUS_REBUILDING

_QUERY_STARTED_AT = datetime(2030, 1, 1, 12, 0, tzinfo=timezone.utc)
_REQUESTED_ROLES = ("researcher", "writer")
_OTHER_ROLE = {"researcher": "writer", "writer": "researcher"}
_RANK_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
_VECTORS = (None, (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0))
_QUERY_VECTORS = (None, (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0))


@dataclass(frozen=True)
class _RetrievalCase:
    memories: tuple[Memory, ...]
    permutation: tuple[int, ...]
    requested_role: str
    limit: int
    include_expired: bool
    query_embedding: tuple[float, ...] | None


@st.composite
def _retrieval_cases(draw: st.DrawFn) -> _RetrievalCase:
    requested_role = draw(st.sampled_from(_REQUESTED_ROLES))
    count = draw(st.integers(min_value=0, max_value=12))
    memories = []
    for index in range(count):
        expires_offset = draw(st.one_of(st.none(), st.sampled_from((-1, 0, 1))))
        created_offset = draw(st.integers(min_value=-2, max_value=2))
        embedding = draw(st.sampled_from(_VECTORS))
        memories.append(
            Memory(
                id=UUID(int=index + 1),
                content=f"generated memory {index}",
                type=draw(st.sampled_from(tuple(MemoryType))),
                agent_role=draw(
                    st.sampled_from(
                        (requested_role, "global", _OTHER_ROLE[requested_role])
                    )
                ),
                confidence=draw(st.sampled_from(_RANK_VALUES)),
                importance=draw(st.sampled_from(_RANK_VALUES)),
                created_at=_QUERY_STARTED_AT + timedelta(seconds=created_offset),
                updated_at=_QUERY_STARTED_AT + timedelta(seconds=created_offset),
                expires_at=(
                    None
                    if expires_offset is None
                    else _QUERY_STARTED_AT + timedelta(seconds=expires_offset)
                ),
                embedding=None if embedding is None else list(embedding),
            )
        )

    return _RetrievalCase(
        memories=tuple(memories),
        permutation=draw(st.permutations(tuple(range(count)))),
        requested_role=requested_role,
        limit=draw(st.integers(min_value=0, max_value=count + 2)),
        include_expired=draw(st.booleans()),
        query_embedding=draw(st.sampled_from(_QUERY_VECTORS)),
    )


def _cosine_similarity(
    query_embedding: tuple[float, ...], memory_embedding: list[float] | None
) -> float:
    if not memory_embedding:
        return 0.0
    dot_product = sum(
        query_value * memory_value
        for query_value, memory_value in zip(query_embedding, memory_embedding)
    )
    query_norm = math.sqrt(sum(value * value for value in query_embedding))
    memory_norm = math.sqrt(sum(value * value for value in memory_embedding))
    if query_norm == 0.0 or memory_norm == 0.0:
        return 0.0
    return dot_product / (query_norm * memory_norm)


def _exact_oracle(case: _RetrievalCase) -> tuple[UUID, ...]:
    eligible = []
    for memory in case.memories:
        role_matches = memory.agent_role == case.requested_role or (
            memory.agent_role == "global" and memory.type is MemoryType.STRATEGY
        )
        expiry_matches = (
            case.include_expired
            or memory.expires_at is None
            or memory.expires_at > _QUERY_STARTED_AT
        )
        if role_matches and expiry_matches:
            eligible.append(memory)

    def rank_key(memory: Memory) -> tuple[float, float, float, float, int]:
        score = 0.0
        if case.query_embedding is not None:
            score = (
                0.75 * _cosine_similarity(case.query_embedding, memory.embedding)
                + 0.25 * memory.importance
            )
        return (
            -score,
            -memory.importance,
            -memory.confidence,
            -memory.created_at.timestamp(),
            memory.id.int,
        )

    return tuple(memory.id for memory in sorted(eligible, key=rank_key)[: case.limit])


async def _retrieve_in_order(
    case: _RetrievalCase, insertion_order: tuple[int, ...]
) -> tuple[UUID, ...]:
    store = SQLiteStore(":memory:")
    await store.initialize()
    try:
        for index in insertion_order:
            await store.save_memory(case.memories[index])

        if case.query_embedding is not None:
            async with store._transactions.write(
                operation="property_test_setup",
                table="memory_vector_index_state",
            ) as connection:
                await connection.execute(
                    "UPDATE memory_vector_index_state SET status = ? "
                    "WHERE singleton = 1",
                    (INDEX_STATUS_REBUILDING,),
                )

        query = RetrievalQuery(
            limit=case.limit,
            agent_role=case.requested_role,
            query_embedding=case.query_embedding,
            include_expired=case.include_expired,
            started_at=_QUERY_STARTED_AT,
        )
        results = await store._retrieval_engine.search(query)
        return tuple(memory.id for memory in results)
    finally:
        await store.close()


# Feature: open-source-project-improvements, Property 17: Retrieval is bounded, role-isolated, expiry-safe, and deterministic
@pytest.mark.asyncio
@settings(max_examples=100, deadline=None)
@given(case=_retrieval_cases())
async def test_retrieval_is_bounded_isolated_expiry_safe_and_deterministic(
    case: _RetrievalCase,
) -> None:
    """**Validates: Requirements 5.3, 5.4, 5.5, 5.7**"""
    canonical_order = tuple(range(len(case.memories)))
    expected_ids = _exact_oracle(case)

    canonical_ids = await _retrieve_in_order(case, canonical_order)
    permuted_ids = await _retrieve_in_order(case, case.permutation)

    assert canonical_ids == expected_ids
    assert permuted_ids == expected_ids
    assert canonical_ids == permuted_ids
    assert len(canonical_ids) <= case.limit

    memories_by_id = {memory.id: memory for memory in case.memories}
    for memory_id in canonical_ids:
        memory = memories_by_id[memory_id]
        assert memory.agent_role == case.requested_role or (
            memory.agent_role == "global" and memory.type is MemoryType.STRATEGY
        )
        if not case.include_expired:
            assert memory.expires_at is None or memory.expires_at > _QUERY_STARTED_AT
