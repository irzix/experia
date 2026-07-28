from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

from experia.core.exceptions import ConfigurationError
from experia.memory.models import Memory, MemoryType
from experia.memory.retrieval import MAX_RETRIEVAL_LIMIT, RetrievalQuery
from experia.memory.store import SQLiteStore


@pytest.fixture
async def retrieval_store(tmp_path):
    store = SQLiteStore(str(tmp_path / "retrieval.db"))
    await store.initialize()
    yield store
    await store.close()


@pytest.mark.parametrize(
    "invalid_limit",
    [True, False, -1, MAX_RETRIEVAL_LIMIT + 1, 1.0, "10", None],
)
def test_retrieval_query_rejects_invalid_limits(invalid_limit):
    with pytest.raises(ConfigurationError) as raised:
        RetrievalQuery(limit=invalid_limit)

    assert raised.value.feature == "retrieval"
    assert raised.value.parameter == "limit"


def test_retrieval_query_accepts_documented_limit_boundaries():
    assert RetrievalQuery(limit=0).limit == 0
    assert RetrievalQuery(limit=MAX_RETRIEVAL_LIMIT).limit == MAX_RETRIEVAL_LIMIT


@pytest.mark.asyncio
async def test_limit_validation_and_zero_return_happen_before_store_io():
    uninitialized = SQLiteStore(":memory:")

    with pytest.raises(ConfigurationError):
        await uninitialized.search_memories(limit=True)
    assert await uninitialized.search_memories(limit=0) == []


@pytest.mark.asyncio
async def test_role_filter_shares_only_global_strategies(retrieval_store):
    memories = [
        Memory(
            id=UUID(int=1),
            content="requested role fact",
            type=MemoryType.FACT,
            agent_role="researcher",
        ),
        Memory(
            id=UUID(int=2),
            content="requested role strategy",
            type=MemoryType.STRATEGY,
            agent_role="researcher",
        ),
        Memory(
            id=UUID(int=3),
            content="other role strategy",
            type=MemoryType.STRATEGY,
            agent_role="writer",
        ),
        Memory(
            id=UUID(int=4),
            content="global strategy",
            type=MemoryType.STRATEGY,
            agent_role="global",
        ),
        Memory(
            id=UUID(int=5),
            content="global fact",
            type=MemoryType.FACT,
            agent_role="global",
        ),
    ]
    for memory in memories:
        await retrieval_store.save_memory(memory)

    results = await retrieval_store.search_memories(agent_role="researcher", limit=10)

    assert {memory.content for memory in results} == {
        "requested role fact",
        "requested role strategy",
        "global strategy",
    }


@pytest.mark.asyncio
async def test_expiry_uses_one_utc_query_start_and_excludes_equality(
    retrieval_store, monkeypatch
):
    query_start = datetime(2030, 1, 1, 12, 0, tzinfo=timezone.utc)
    clock_calls = 0

    def fixed_utc_now():
        nonlocal clock_calls
        clock_calls += 1
        return query_start

    monkeypatch.setattr("experia.memory.retrieval._utc_now", fixed_utc_now)
    memories = [
        Memory(
            id=UUID(int=11),
            content="expired before start",
            type=MemoryType.FACT,
            expires_at=query_start - timedelta(microseconds=1),
        ),
        Memory(
            id=UUID(int=12),
            content="expires at start",
            type=MemoryType.FACT,
            expires_at=query_start,
        ),
        Memory(
            id=UUID(int=13),
            content="expires after start",
            type=MemoryType.FACT,
            expires_at=query_start + timedelta(microseconds=1),
        ),
    ]
    for memory in memories:
        await retrieval_store.save_memory(memory)

    results = await retrieval_store.search_memories(limit=10)

    assert clock_calls == 1
    assert [memory.content for memory in results] == ["expires after start"]
    assert (
        len(await retrieval_store.search_memories(limit=10, include_expired=True)) == 3
    )


@pytest.mark.asyncio
async def test_ranking_uses_complete_deterministic_total_order(retrieval_store):
    old = datetime(2025, 1, 1, tzinfo=timezone.utc)
    new = old + timedelta(days=1)
    memories = [
        Memory(
            id=UUID(int=25),
            content="highest importance",
            type=MemoryType.FACT,
            importance=0.9,
            confidence=0.1,
            created_at=old,
        ),
        Memory(
            id=UUID(int=24),
            content="highest confidence",
            type=MemoryType.FACT,
            importance=0.8,
            confidence=0.9,
            created_at=old,
        ),
        Memory(
            id=UUID(int=23),
            content="newest",
            type=MemoryType.FACT,
            importance=0.8,
            confidence=0.8,
            created_at=new,
        ),
        Memory(
            id=UUID(int=22),
            content="larger id",
            type=MemoryType.FACT,
            importance=0.8,
            confidence=0.8,
            created_at=old,
        ),
        Memory(
            id=UUID(int=21),
            content="smaller id",
            type=MemoryType.FACT,
            importance=0.8,
            confidence=0.8,
            created_at=old,
        ),
    ]
    for memory in reversed(memories):
        await retrieval_store.save_memory(memory)

    results = await retrieval_store.search_memories(limit=10)

    assert [memory.content for memory in results] == [
        "highest importance",
        "highest confidence",
        "newest",
        "smaller id",
        "larger id",
    ]


@pytest.mark.asyncio
async def test_semantic_score_precedes_deterministic_tie_breakers(
    retrieval_store, monkeypatch
):
    created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    memories = [
        Memory(
            id=UUID(int=32),
            content="lower semantic score",
            type=MemoryType.FACT,
            importance=1.0,
            confidence=1.0,
            created_at=created_at,
            embedding=[0.0, 1.0],
        ),
        Memory(
            id=UUID(int=31),
            content="higher semantic score",
            type=MemoryType.FACT,
            importance=0.1,
            confidence=0.1,
            created_at=created_at,
            embedding=[1.0, 0.0],
        ),
    ]
    for memory in memories:
        await retrieval_store.save_memory(memory)

    async def candidate_ids(query_embedding, **kwargs):
        return tuple(str(memory.id) for memory in memories)

    monkeypatch.setattr(
        retrieval_store._vector_index,
        "candidate_ids",
        candidate_ids,
    )
    results = await retrieval_store.search_memories(query_embedding=[1.0, 0.0], limit=2)

    assert [memory.content for memory in results] == [
        "higher semantic score",
        "lower semantic score",
    ]


@pytest.mark.asyncio
async def test_exact_fallback_streams_with_a_bounded_heap_and_is_deterministic(
    retrieval_store, monkeypatch, caplog
):
    import logging
    import math

    import aiosqlite

    from experia.core.logging import RetrievalDiagnosticCode
    from experia.memory import retrieval as retrieval_module
    from experia.memory.vector_index import INDEX_STATUS_REBUILDING

    created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    memories = []
    for index in range(300):
        memory = Memory(
            id=UUID(int=1_000 + index),
            content=f"candidate {index}",
            type=MemoryType.FACT,
            importance=(index % 11) / 10,
            confidence=(index % 7) / 6,
            created_at=created_at + timedelta(seconds=index % 5),
            embedding=[float(index % 5), 1.0],
        )
        memories.append(memory)
        await retrieval_store.save_memory(memory)

    async with retrieval_store._transactions.write(
        operation="test_setup",
        table="memory_vector_index_state",
    ) as connection:
        await connection.execute(
            "UPDATE memory_vector_index_state SET status = ? WHERE singleton = 1",
            (INDEX_STATUS_REBUILDING,),
        )
    caplog.set_level(logging.INFO, logger="experia")

    query_embedding = [1.0, 0.0]
    limit = 7

    def oracle_key(memory):
        vector = memory.embedding
        assert vector is not None
        norm = math.sqrt(sum(component * component for component in vector))
        similarity = vector[0] / norm if norm else 0.0
        score = 0.75 * similarity + 0.25 * memory.importance
        return (
            -score,
            -memory.importance,
            -memory.confidence,
            -memory.created_at.timestamp(),
            str(memory.id),
        )

    expected_ids = [memory.id for memory in sorted(memories, key=oracle_key)[:limit]]
    fetchmany_calls = []
    maximum_heap_size = 0
    original_fetchmany = aiosqlite.Cursor.fetchmany
    original_add = retrieval_module._BoundedMemoryRanker.add

    async def track_fetchmany(cursor, size=None):
        rows = await original_fetchmany(cursor, size)
        fetchmany_calls.append((size, len(rows)))
        return rows

    async def forbid_fetchall(cursor):
        raise AssertionError("streaming fallback must not call fetchall()")

    def track_add(ranker, memory):
        nonlocal maximum_heap_size
        original_add(ranker, memory)
        maximum_heap_size = max(maximum_heap_size, len(ranker))

    monkeypatch.setattr(aiosqlite.Cursor, "fetchmany", track_fetchmany)
    monkeypatch.setattr(aiosqlite.Cursor, "fetchall", forbid_fetchall)
    monkeypatch.setattr(retrieval_module._BoundedMemoryRanker, "add", track_add)

    first = await retrieval_store.search_memories(
        query_embedding=query_embedding,
        limit=limit,
    )
    second = await retrieval_store.search_memories(
        query_embedding=query_embedding,
        limit=limit,
    )

    assert [memory.id for memory in first] == expected_ids
    assert [memory.id for memory in second] == expected_ids
    assert maximum_heap_size == limit
    assert len(fetchmany_calls) >= 6
    assert all(size == 256 for size, _ in fetchmany_calls)
    assert all(row_count <= 256 for _, row_count in fetchmany_calls)
    fallback_diagnostics = [
        record.experia_retrieval_diagnostic
        for record in caplog.records
        if hasattr(record, "experia_retrieval_diagnostic")
        and record.experia_retrieval_diagnostic.code
        is RetrievalDiagnosticCode.INDEX_FALLBACK
    ]
    assert len(fallback_diagnostics) == 2


@pytest.mark.asyncio
async def test_ready_index_candidates_use_documented_budget_and_exact_reranking(
    retrieval_store, monkeypatch, caplog
):
    import logging

    from experia.core.logging import RetrievalDiagnosticCode
    from experia.memory.retrieval import DEFAULT_INDEX_CANDIDATE_BUDGET

    memories = [
        Memory(
            id=UUID(int=3_001),
            content="weaker exact candidate",
            type=MemoryType.FACT,
            importance=1.0,
            embedding=[0.0, 1.0],
        ),
        Memory(
            id=UUID(int=3_002),
            content="stronger exact candidate",
            type=MemoryType.FACT,
            importance=0.1,
            embedding=[1.0, 0.0],
        ),
    ]
    for memory in memories:
        await retrieval_store.save_memory(memory)

    observed = {}

    async def candidate_ids(query_embedding, **kwargs):
        observed["embedding"] = tuple(query_embedding)
        observed.update(kwargs)
        return tuple(str(memory.id) for memory in memories)

    monkeypatch.setattr(
        retrieval_store._vector_index,
        "candidate_ids",
        candidate_ids,
    )
    caplog.set_level(logging.INFO, logger="experia")

    results = await retrieval_store.search_memories(
        query_embedding=[1.0, 0.0],
        limit=2,
    )

    assert [memory.content for memory in results] == [
        "stronger exact candidate",
        "weaker exact candidate",
    ]
    assert observed["embedding"] == (1.0, 0.0)
    assert observed["budget"] == DEFAULT_INDEX_CANDIDATE_BUDGET
    assert observed["memory_type"] is None
    assert observed["agent_role"] is None
    assert observed["include_expired"] is False
    assert not any(
        hasattr(record, "experia_retrieval_diagnostic")
        and record.experia_retrieval_diagnostic.code
        is RetrievalDiagnosticCode.INDEX_FALLBACK
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_public_search_methods_delegate_to_retrieval_engine(
    retrieval_store, monkeypatch
):
    expected = Memory(
        content="delegated candidate",
        type=MemoryType.LESSON,
        embedding=[1.0, 0.0],
    )
    queries = []

    async def search(query):
        queries.append(query)
        return [expected]

    monkeypatch.setattr(retrieval_store._retrieval_engine, "search", search)

    searched = await retrieval_store.search_memories(
        query="delegated",
        memory_type=MemoryType.LESSON,
        agent_role="researcher",
        limit=3,
        query_embedding=[1.0, 0.0],
    )
    similar = await retrieval_store.find_similar_memory(
        [1.0, 0.0],
        memory_type=MemoryType.LESSON,
        agent_role="researcher",
    )

    assert searched == [expected]
    assert similar == expected
    assert len(queries) == 2
    assert queries[0].text == "delegated"
    assert queries[0].limit == 3
    assert queries[1].limit == 1


@pytest.mark.asyncio
async def test_dimension_mismatch_is_omitted_diagnosed_and_does_not_mutate_storage(
    retrieval_store, caplog
):
    import logging

    from experia.core.logging import (
        RetrievalDiagnosticCode,
    )

    matching = Memory(
        id=UUID(int=2_001),
        content="matching vector",
        type=MemoryType.FACT,
        embedding=[1.0, 0.0],
    )
    incompatible = Memory(
        id=UUID(int=2_002),
        content="incompatible vector payload",
        type=MemoryType.FACT,
        embedding=[1.0, 0.0, 0.0],
    )
    without_embedding = Memory(
        id=UUID(int=2_003),
        content="unembedded candidate",
        type=MemoryType.FACT,
        embedding=None,
    )
    for memory in (matching, incompatible, without_embedding):
        await retrieval_store.save_memory(memory)

    connection = retrieval_store._require_conn()
    before_cursor = await connection.execute(
        "SELECT id, embedding, embedding_dimension FROM memories ORDER BY id"
    )
    before = await before_cursor.fetchall()
    caplog.set_level(logging.INFO, logger="experia")

    results = await retrieval_store.search_memories(
        query_embedding=[1.0, 0.0],
        limit=10,
    )

    after_cursor = await connection.execute(
        "SELECT id, embedding, embedding_dimension FROM memories ORDER BY id"
    )
    after = await after_cursor.fetchall()
    diagnostics = [
        record.experia_retrieval_diagnostic
        for record in caplog.records
        if hasattr(record, "experia_retrieval_diagnostic")
    ]

    assert {memory.id for memory in results} == {
        matching.id,
        without_embedding.id,
    }
    assert before == after
    assert len(diagnostics) == 1
    assert diagnostics[0].code is RetrievalDiagnosticCode.DIMENSION_MISMATCH
    assert diagnostics[0].memory_id == incompatible.id
    assert diagnostics[0].stored_dimension == 3
    assert diagnostics[0].query_dimension == 2
    assert "incompatible vector payload" not in caplog.text
