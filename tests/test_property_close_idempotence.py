"""Property tests for SQLite store close idempotence."""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import aiosqlite
from hypothesis import given
from hypothesis import strategies as st

from experia.memory.models import Memory, MemoryType
from experia.memory.store import SQLiteStore


async def _persisted_snapshot(db_path: Path) -> dict[str, list[tuple[Any, ...]]]:
    connection = await aiosqlite.connect(db_path)
    try:
        snapshot: dict[str, list[tuple[Any, ...]]] = {}
        for table in ("experiences", "lessons", "memories"):
            cursor = await connection.execute(f"SELECT * FROM {table} ORDER BY id")
            snapshot[table] = await cursor.fetchall()
        return snapshot
    finally:
        await connection.close()


async def _exercise_close_schedule(
    overlapping_callers: int, repeated_wave_sizes: list[int]
) -> None:
    with TemporaryDirectory() as directory:
        db_path = Path(directory) / "close-idempotence.db"
        store = SQLiteStore(str(db_path))
        await store.initialize()
        memory = Memory(
            content="Persisted state remains stable across close calls",
            type=MemoryType.LESSON,
            metadata={"property": 14, "nested": {"stable": True}},
            embedding=[0.125, 0.5, 0.875],
        )
        await store.save_memory(memory)
        expected_snapshot = await _persisted_snapshot(db_path)

        connection = store._require_conn()
        original_connection_close = connection.close
        original_coordinate_close = store._coordinate_close
        physical_close_started = asyncio.Event()
        release_physical_close = asyncio.Event()
        all_initial_callers_coordinated = asyncio.Event()
        physical_close_calls = 0
        coordinated_callers = 0

        async def controlled_connection_close() -> None:
            nonlocal physical_close_calls
            physical_close_calls += 1
            physical_close_started.set()
            await release_physical_close.wait()
            await original_connection_close()

        async def observed_coordinate_close():
            nonlocal coordinated_callers
            close_task = await original_coordinate_close()
            coordinated_callers += 1
            if coordinated_callers == overlapping_callers:
                all_initial_callers_coordinated.set()
            return close_task

        connection.close = controlled_connection_close
        store._coordinate_close = observed_coordinate_close

        initial_calls = [
            asyncio.create_task(store.close()) for _ in range(overlapping_callers)
        ]
        try:
            await physical_close_started.wait()
            await all_initial_callers_coordinated.wait()
            assert all(not call.done() for call in initial_calls)
        finally:
            release_physical_close.set()

        assert await asyncio.gather(*initial_calls) == [None] * overlapping_callers
        assert physical_close_calls == 1
        assert store._lifecycle_state == "closed"
        assert store._conn is None
        assert store._close_task is not None
        assert store._close_task.done()
        assert store._close_task.exception() is None
        assert await _persisted_snapshot(db_path) == expected_snapshot

        for wave_size in repeated_wave_sizes:
            calls = [asyncio.create_task(store.close()) for _ in range(wave_size)]
            assert await asyncio.gather(*calls) == [None] * wave_size
            assert physical_close_calls == 1
            assert store._lifecycle_state == "closed"
            assert store._conn is None
            assert await _persisted_snapshot(db_path) == expected_snapshot

        reopened = SQLiteStore(str(db_path))
        await reopened.initialize()
        try:
            assert await reopened.get_memory(memory.id) == memory
        finally:
            await reopened.close()


# Feature: open-source-project-improvements, Property 14: Close is idempotent
# **Validates: Requirements 3.8**
@given(
    overlapping_callers=st.integers(min_value=2, max_value=8),
    repeated_wave_sizes=st.lists(
        st.integers(min_value=1, max_value=8),
        min_size=1,
        max_size=5,
    ),
)
def test_close_is_idempotent(
    overlapping_callers: int, repeated_wave_sizes: list[int]
) -> None:
    asyncio.run(_exercise_close_schedule(overlapping_callers, repeated_wave_sizes))
