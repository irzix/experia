"""Focused transactional and resumable vector index coverage."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import pytest

from experia.core.exceptions import StorageError
from experia.memory.models import Memory, MemoryType
from experia.memory.store import SQLiteStore
from experia.memory.vector_index import (
    CURRENT_VECTOR_INDEX_VERSION,
    INDEX_STATUS_READY,
    INDEX_STATUS_REBUILDING,
    VECTOR_INDEX_BAND_COUNT,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "sqlite"


async def _source_snapshot(store: SQLiteStore) -> tuple[tuple[object, ...], ...]:
    cursor = await store._require_conn().execute(
        "SELECT id, content, embedding, embedding_dimension FROM memories ORDER BY id"
    )
    return tuple(await cursor.fetchall())


async def _band_snapshot(store: SQLiteStore) -> tuple[tuple[object, ...], ...]:
    cursor = await store._require_conn().execute(
        "SELECT memory_id, dimension, band, bucket, index_version "
        "FROM memory_vector_bands ORDER BY memory_id, band"
    )
    return tuple(await cursor.fetchall())


@pytest.mark.asyncio
async def test_memory_save_maintains_deterministic_versioned_bands_in_one_transaction(
    tmp_path,
) -> None:
    store = SQLiteStore(str(tmp_path / "transactional-index.db"))
    await store.initialize()
    memory = Memory(
        id=UUID("10000000-0000-4000-8000-000000000001"),
        content="transactionally indexed",
        type=MemoryType.FACT,
        embedding=[0.25, -0.5, 1.0],
    )
    statements: list[str] = []
    await store._require_conn().set_trace_callback(statements.append)
    try:
        await store.save_memory(memory)

        expected = store._vector_index.band_rows(str(memory.id), memory.embedding or [])
        assert await _band_snapshot(store) == expected
        assert all(row[4] == CURRENT_VECTOR_INDEX_VERSION for row in expected)
        assert len(expected) == VECTOR_INDEX_BAND_COUNT
        assert expected == store._vector_index.band_rows(
            str(memory.id), memory.embedding or []
        )

        normalized = [" ".join(statement.split()).upper() for statement in statements]
        begin_index = normalized.index("BEGIN IMMEDIATE")
        source_index = next(
            index
            for index, statement in enumerate(normalized)
            if statement.startswith("INSERT OR REPLACE INTO MEMORIES")
        )
        first_band_index = next(
            index
            for index, statement in enumerate(normalized)
            if statement.startswith("INSERT INTO MEMORY_VECTOR_BANDS")
        )
        commit_index = normalized.index("COMMIT")
        assert begin_index < source_index < first_band_index < commit_index
        assert normalized.count("BEGIN IMMEDIATE") == 1
        assert normalized.count("COMMIT") == 1

        status = await store._vector_index.status()
        assert status is not None
        assert status.status == INDEX_STATUS_READY
        assert status.ready
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_index_maintenance_failure_rolls_back_source_and_derived_rows(
    tmp_path, monkeypatch
) -> None:
    store = SQLiteStore(str(tmp_path / "index-rollback.db"))
    await store.initialize()
    memory = Memory(
        id=UUID("10000000-0000-4000-8000-000000000002"),
        content="original source value",
        type=MemoryType.LESSON,
        embedding=[1.0, 0.0],
    )
    try:
        await store.save_memory(memory)
        source_before = await _source_snapshot(store)
        bands_before = await _band_snapshot(store)

        async def fail_after_deleting_bands(conn, encoded_memory):
            await conn.execute(
                "DELETE FROM memory_vector_bands WHERE memory_id = ?",
                (encoded_memory.id,),
            )
            raise RuntimeError("injected index maintenance failure")

        monkeypatch.setattr(
            store._vector_index,
            "maintain_encoded_memory",
            fail_after_deleting_bands,
        )
        replacement = memory.model_copy(
            update={"content": "must roll back", "embedding": [0.0, 1.0]}
        )

        with pytest.raises(StorageError) as raised:
            await store.save_memory(replacement)

        assert raised.value.operation == "save"
        assert raised.value.table == "memories"
        assert raised.value.record_ids == (str(memory.id),)
        assert isinstance(raised.value.__cause__, RuntimeError)
        assert await _source_snapshot(store) == source_before
        assert await _band_snapshot(store) == bands_before
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_candidate_ids_apply_dimension_role_expiry_and_budget(
    tmp_path,
) -> None:
    store = SQLiteStore(str(tmp_path / "filtered-candidates.db"))
    await store.initialize()
    started_at = datetime(2030, 1, 1, tzinfo=timezone.utc)
    memories = [
        Memory(
            id=UUID(int=10),
            content="requested role",
            type=MemoryType.FACT,
            agent_role="researcher",
            embedding=[1.0, 0.0],
        ),
        Memory(
            id=UUID(int=11),
            content="global strategy",
            type=MemoryType.STRATEGY,
            agent_role="global",
            embedding=[1.0, 0.0],
        ),
        Memory(
            id=UUID(int=12),
            content="other role",
            type=MemoryType.FACT,
            agent_role="writer",
            embedding=[1.0, 0.0],
        ),
        Memory(
            id=UUID(int=13),
            content="global non-strategy",
            type=MemoryType.FACT,
            agent_role="global",
            embedding=[1.0, 0.0],
        ),
        Memory(
            id=UUID(int=14),
            content="expired",
            type=MemoryType.FACT,
            agent_role="researcher",
            expires_at=started_at - timedelta(seconds=1),
            embedding=[1.0, 0.0],
        ),
        Memory(
            id=UUID(int=15),
            content="wrong dimension",
            type=MemoryType.FACT,
            agent_role="researcher",
            embedding=[1.0, 0.0, 0.0],
        ),
    ]
    try:
        for memory in memories:
            await store.save_memory(memory)

        candidates = await store._vector_index.candidate_ids(
            [1.0, 0.0],
            budget=2,
            agent_role="researcher",
            started_at=started_at,
            include_expired=False,
        )

        assert candidates == (str(memories[0].id), str(memories[1].id))
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_interrupted_rebuild_resumes_from_committed_cursor_without_source_changes(
    tmp_path, monkeypatch
) -> None:
    store = SQLiteStore(str(tmp_path / "resumable-index.db"))
    await store.initialize()
    memories = [
        Memory(
            id=UUID(int=index),
            content=f"memory {index}",
            type=MemoryType.FACT,
            embedding=[float(index), 1.0] if index < 3 else [1.0, 0.0, 0.0],
        )
        for index in range(1, 4)
    ]
    try:
        for memory in memories:
            await store.save_memory(memory)
        source_before = await _source_snapshot(store)
        async with store._transactions.write(
            operation="test_setup",
            table="memory_vector_bands,memory_vector_index_state",
        ) as conn:
            await conn.execute("DELETE FROM memory_vector_bands")
            await conn.execute(
                "UPDATE memory_vector_index_state "
                "SET status = ?, last_memory_id = NULL WHERE singleton = 1",
                (INDEX_STATUS_REBUILDING,),
            )

        assert await store._vector_index.candidate_ids([1.0, 0.0], budget=10) is None
        fallback = await store.search_memories(
            query_embedding=[1.0, 0.0],
            limit=10,
        )
        assert {memory.id for memory in fallback} == {memories[0].id, memories[1].id}

        original_maintain = store._vector_index.maintain_memory
        calls = 0

        async def fail_second_batch(conn, memory_id, embedding):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected rebuild interruption")
            await original_maintain(conn, memory_id, embedding)

        monkeypatch.setattr(
            store._vector_index,
            "maintain_memory",
            fail_second_batch,
        )
        with pytest.raises(StorageError) as raised:
            await store.rebuild_vector_index(batch_size=1)
        assert raised.value.operation == "rebuild_index"
        assert isinstance(raised.value.__cause__, RuntimeError)

        status = await store._vector_index.status()
        assert status is not None
        assert status.status == INDEX_STATUS_REBUILDING
        assert status.last_memory_id == str(memories[0].id)
        assert len(await _band_snapshot(store)) == VECTOR_INDEX_BAND_COUNT
        assert await _source_snapshot(store) == source_before

        monkeypatch.setattr(
            store._vector_index,
            "maintain_memory",
            original_maintain,
        )
        result = await store.rebuild_vector_index(batch_size=1)

        assert result.complete
        assert result.processed_memories == 2
        assert result.last_memory_id is None
        status = await store._vector_index.status()
        assert status is not None and status.ready
        assert len(await _band_snapshot(store)) == 3 * VECTOR_INDEX_BAND_COUNT
        assert await _source_snapshot(store) == source_before

        candidates = await store._vector_index.candidate_ids(
            memories[0].embedding or [],
            budget=10,
        )
        assert candidates is not None
        assert str(memories[0].id) in candidates
        assert str(memories[2].id) not in candidates
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_version_two_database_uses_streaming_fallback_until_rebuild(
    tmp_path,
) -> None:
    database_path = tmp_path / "schema-v2.db"
    connection = sqlite3.connect(database_path)
    try:
        connection.executescript(
            (FIXTURE_ROOT / "schema-v2.sql").read_text(encoding="utf-8")
        )
    finally:
        connection.close()

    store = SQLiteStore(str(database_path))
    await store.initialize()
    try:
        status = await store._vector_index.status()
        assert status is not None
        assert status.status == INDEX_STATUS_REBUILDING
        assert not status.ready

        results = await store.search_memories(
            query_embedding=[0.125, -0.5, 1.25, 0.0],
            limit=10,
            include_expired=True,
        )
        assert [memory.id for memory in results] == [
            UUID("33333333-3333-4333-8333-333333333333")
        ]
        assert (
            await store._vector_index.candidate_ids(
                [0.125, -0.5, 1.25, 0.0],
                budget=10,
            )
            is None
        )

        source_before = await _source_snapshot(store)
        rebuilt = await store.rebuild_vector_index(batch_size=1)
        assert rebuilt.complete
        assert rebuilt.processed_memories == 2
        assert await _source_snapshot(store) == source_before
        assert await store._vector_index.is_ready()
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_streaming_retrieval_survives_absent_derived_index(tmp_path) -> None:
    store = SQLiteStore(str(tmp_path / "absent-index.db"))
    await store.initialize()
    memory = Memory(
        content="source remains searchable",
        type=MemoryType.FACT,
        embedding=[1.0, 0.0],
    )
    try:
        await store.save_memory(memory)
        async with store._transactions.write(
            operation="test_setup",
            table="memory_vector_bands",
        ) as conn:
            await conn.execute("DROP TABLE memory_vector_bands")

        assert await store._vector_index.candidate_ids([1.0, 0.0], budget=10) is None
        results = await store.search_memories(
            query_embedding=[1.0, 0.0],
            limit=10,
        )
        assert [result.id for result in results] == [memory.id]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_ready_index_and_fallback_preserve_exact_tie_order(tmp_path) -> None:
    store = SQLiteStore(str(tmp_path / "exact-tie-order.db"))
    await store.initialize()
    tied_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    insertion_ids = (UUID(int=105), UUID(int=101), UUID(int=104), UUID(int=102))
    memories = [
        Memory(
            id=memory_id,
            content=f"exact tie {memory_id.int}",
            type=MemoryType.FACT,
            confidence=0.75,
            importance=0.5,
            embedding=[1.0, 0.0],
            created_at=tied_at,
            updated_at=tied_at,
        )
        for memory_id in insertion_ids
    ]
    expected_ids = [UUID(int=101), UUID(int=102), UUID(int=104)]
    try:
        for memory in memories:
            await store.save_memory(memory)

        assert await store._vector_index.is_ready()
        indexed = await store.search_memories(
            query_embedding=[1.0, 0.0],
            limit=3,
        )

        async with store._transactions.write(
            operation="test_setup",
            table="memory_vector_index_state",
        ) as conn:
            await conn.execute(
                "UPDATE memory_vector_index_state SET status = ? WHERE singleton = 1",
                (INDEX_STATUS_REBUILDING,),
            )

        assert (
            await store._vector_index.candidate_ids(
                [1.0, 0.0],
                budget=10,
            )
            is None
        )
        fallback = await store.search_memories(
            query_embedding=[1.0, 0.0],
            limit=3,
        )

        assert [memory.id for memory in indexed] == expected_ids
        assert [memory.id for memory in fallback] == expected_ids
    finally:
        await store.close()
