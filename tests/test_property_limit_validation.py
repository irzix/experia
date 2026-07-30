"""Property coverage for retrieval-limit validation before storage I/O."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from experia.core.exceptions import ConfigurationError
from experia.memory.retrieval import MAX_RETRIEVAL_LIMIT, RetrievalQuery
from experia.memory.store import SQLiteStore

_NESTED_NON_INTEGER_VALUES = st.recursive(
    st.one_of(
        st.none(),
        st.booleans(),
        st.floats(allow_nan=True, allow_infinity=True),
        st.text(max_size=24),
        st.binary(max_size=24),
    ),
    lambda children: st.one_of(
        st.lists(children, max_size=4),
        st.tuples(children, children),
        st.dictionaries(st.text(max_size=8), children, max_size=4),
    ),
    max_leaves=8,
)
_LIMIT_VALUES = st.one_of(
    st.sampled_from(
        (
            -1,
            0,
            1,
            MAX_RETRIEVAL_LIMIT,
            MAX_RETRIEVAL_LIMIT + 1,
            True,
            False,
        )
    ),
    st.integers(),
    _NESTED_NON_INTEGER_VALUES,
)


def _is_documented_limit(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= MAX_RETRIEVAL_LIMIT
    )


# Feature: open-source-project-improvements, Property 16: Limit validation happens before I/O
@pytest.mark.asyncio
@settings(max_examples=100, deadline=None)
@given(limit=_LIMIT_VALUES)
async def test_limit_acceptance_is_exact_and_invalid_values_precede_storage_io(
    limit: Any,
) -> None:
    """**Validates: Requirements 5.1, 5.2**"""
    store = SQLiteStore(":memory:")
    storage_leases = 0
    executed_queries: list[RetrievalQuery] = []

    @asynccontextmanager
    async def storage_access_spy() -> AsyncIterator[None]:
        nonlocal storage_leases
        storage_leases += 1
        yield

    async def query_spy(query: RetrievalQuery) -> list[Any]:
        executed_queries.append(query)
        return []

    store._operation = storage_access_spy
    store._search_memories = query_spy

    if _is_documented_limit(limit):
        assert await store.search_memories(limit=limit) == []
        expected_io_count = int(limit != 0)
        assert storage_leases == expected_io_count
        assert len(executed_queries) == expected_io_count
        if executed_queries:
            assert executed_queries[0].limit == limit
        return

    with pytest.raises(ConfigurationError) as raised:
        await store.search_memories(limit=limit)

    assert raised.value.feature == "retrieval"
    assert raised.value.parameter == "limit"
    assert storage_leases == 0
    assert executed_queries == []
