"""Property coverage for incompatible semantic embedding dimensions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from experia.core.logging import RetrievalDiagnosticCode
from experia.memory.models import Memory, MemoryType
from experia.memory.store import SQLiteStore

_FIXED_INSTANT = datetime(2025, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class _DimensionCase:
    query_embedding: tuple[float, ...]
    memories: tuple[Memory, ...]
    matching_ids: frozenset[UUID]
    incompatible_dimensions: dict[UUID, int]


class _RecordCollector(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _unit_vector(dimension: int) -> list[float]:
    return [1.0, *([0.0] * (dimension - 1))]


@st.composite
def _mixed_dimension_cases(draw: st.DrawFn) -> _DimensionCase:
    query_dimension = draw(st.integers(min_value=1, max_value=8))
    incompatible_dimension = st.sampled_from(
        tuple(dimension for dimension in range(1, 9) if dimension != query_dimension)
    )
    incompatible_dimensions = draw(
        st.lists(incompatible_dimension, min_size=1, max_size=6)
    )
    matching_count = draw(st.integers(min_value=1, max_value=3))
    memory_ids = draw(
        st.lists(
            st.uuids(),
            min_size=matching_count + len(incompatible_dimensions),
            max_size=matching_count + len(incompatible_dimensions),
            unique=True,
        )
    )

    matching_ids = frozenset(memory_ids[:matching_count])
    dimensions = [query_dimension] * matching_count + incompatible_dimensions
    memories = tuple(
        Memory(
            id=memory_id,
            content=f"generated memory {index}",
            type=MemoryType.FACT,
            embedding=_unit_vector(dimension),
            created_at=_FIXED_INSTANT,
            updated_at=_FIXED_INSTANT,
        )
        for index, (memory_id, dimension) in enumerate(zip(memory_ids, dimensions))
    )
    return _DimensionCase(
        query_embedding=tuple(_unit_vector(query_dimension)),
        memories=memories,
        matching_ids=matching_ids,
        incompatible_dimensions={
            memory.id: len(memory.embedding or [])
            for memory in memories
            if memory.id not in matching_ids
        },
    )


async def _persisted_state_bytes(store: SQLiteStore) -> bytes:
    """Return a canonical byte representation of the complete SQLite state."""
    connection = store._require_conn()
    version_cursor = await connection.execute("PRAGMA user_version")
    version = await version_cursor.fetchone()
    dump = [f"PRAGMA user_version={int(version[0])};"]
    dump.extend([statement async for statement in connection.iterdump()])
    return "\n".join(dump).encode("utf-8")


# Feature: open-source-project-improvements, Property 18: Incompatible embeddings are omitted and diagnosed
@pytest.mark.asyncio
@settings(max_examples=100, deadline=None)
@given(case=_mixed_dimension_cases())
async def test_incompatible_embeddings_are_omitted_and_diagnosed(
    case: _DimensionCase,
) -> None:
    """**Validates: Requirements 5.6**"""
    store = SQLiteStore(":memory:")
    await store.initialize()
    try:
        for memory in case.memories:
            await store.save_memory(memory)
        assert await store._vector_index.is_ready()

        before = await _persisted_state_bytes(store)
        handler = _RecordCollector()
        experia_logger = logging.getLogger("experia")
        previous_level = experia_logger.level
        experia_logger.addHandler(handler)
        experia_logger.setLevel(logging.INFO)
        try:
            results = await store.search_memories(
                query_embedding=list(case.query_embedding),
                limit=len(case.memories),
            )
        finally:
            experia_logger.removeHandler(handler)
            experia_logger.setLevel(previous_level)

        after = await _persisted_state_bytes(store)
        diagnostics = [
            record.experia_retrieval_diagnostic
            for record in handler.records
            if hasattr(record, "experia_retrieval_diagnostic")
        ]

        assert {memory.id for memory in results} == case.matching_ids
        assert len(diagnostics) == len(case.incompatible_dimensions)
        assert all(
            diagnostic.code is RetrievalDiagnosticCode.DIMENSION_MISMATCH
            for diagnostic in diagnostics
        )
        assert {
            diagnostic.memory_id: diagnostic.stored_dimension
            for diagnostic in diagnostics
        } == case.incompatible_dimensions
        assert all(
            diagnostic.query_dimension == len(case.query_embedding)
            for diagnostic in diagnostics
        )
        assert before == after
    finally:
        await store.close()
