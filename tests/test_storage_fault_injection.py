"""Focused fault-injection tests for SQLite serialization and writes."""

from pathlib import Path
from unittest.mock import patch

import aiosqlite
import pytest

from experia.core.exceptions import StorageError
from experia.experience.models import ExperienceRecord
from experia.memory.models import Memory, MemoryType
from experia.memory.store import SQLiteStore

_TABLES = ("experiences", "lessons", "memories")


class InjectedRollbackFailure(RuntimeError):
    """Sentinel used to distinguish a failed rollback from the primary failure."""


@pytest.fixture
async def populated_store(tmp_path):
    db_path = tmp_path / "storage-faults.db"
    store = SQLiteStore(str(db_path))
    await store.initialize()
    experience = ExperienceRecord(
        task="Inspect the deployment",
        action="Read service logs",
        result="Found the failure",
        context={"attempt": 1},
    )
    memory = Memory(
        content="Inspect logs before retrying",
        type=MemoryType.LESSON,
        metadata={"source": "test"},
    )
    await store.save_experience(experience)
    await store.save_memory(memory)
    try:
        yield store, db_path, experience, memory
    finally:
        connection = store._conn
        if connection is not None:
            await connection.rollback()
        await store.close()


async def _persisted_snapshot(db_path: Path) -> tuple[tuple[str, tuple], ...]:
    """Read committed source rows through a connection independent of the store."""
    connection = await aiosqlite.connect(db_path)
    try:
        snapshot = []
        for table in _TABLES:
            cursor = await connection.execute(f"SELECT * FROM {table} ORDER BY id")
            snapshot.append((table, tuple(await cursor.fetchall())))
        return tuple(snapshot)
    finally:
        await connection.close()


# Validates: Requirements 3.13
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("table", "field", "malformed", "expected_field"),
    [
        pytest.param("experiences", "context", '{"missing":', "context", id="json"),
        pytest.param("memories", "type", "unknown", "type", id="enum"),
        pytest.param("experiences", "id", "not-a-uuid", "id", id="uuid"),
        pytest.param(
            "experiences",
            "created_at",
            "not-a-timestamp",
            "created_at",
            id="timestamp",
        ),
    ],
)
async def test_malformed_stored_rows_raise_contextual_error_without_mutation(
    populated_store,
    table: str,
    field: str,
    malformed: str,
    expected_field: str,
):
    store, db_path, experience, memory = populated_store
    connection = store._require_conn()
    record = experience if table == "experiences" else memory
    await connection.execute(
        f"UPDATE {table} SET {field} = ? WHERE id = ?",
        (malformed, str(record.id)),
    )
    await connection.commit()
    expected_record_id = malformed if field == "id" else str(record.id)
    before = await _persisted_snapshot(db_path)

    with pytest.raises(StorageError) as raised:
        if table == "memories":
            await store.get_memory(memory.id)
        elif field == "id":
            await store.get_recent_experiences()
        else:
            await store.get_experience(experience.id)

    error = raised.value
    assert error.operation == "decode"
    assert error.table == table
    assert error.record_ids == (expected_record_id,)
    assert error.field == expected_field
    assert await _persisted_snapshot(db_path) == before


# Validates: Requirements 3.5, 3.14
@pytest.mark.asyncio
@pytest.mark.parametrize("failure_point", ["execute", "commit", "rollback"])
async def test_write_failures_raise_contextual_error_without_persisting(
    populated_store,
    failure_point: str,
):
    store, db_path, _, _ = populated_store
    candidate = ExperienceRecord(
        task="Candidate task",
        action="Candidate action",
        result="Candidate result",
    )
    connection = store._require_conn()
    original_execute = connection.execute
    original_rollback = connection.rollback
    before = await _persisted_snapshot(db_path)
    primary_failure = StorageError(
        "Injected execute failure.",
        operation="save",
        table="experiences",
        record_ids=(candidate.id,),
    )

    if failure_point == "execute":

        async def fail_execute(sql, *args, **kwargs):
            if " ".join(sql.split()).upper().startswith("INSERT INTO EXPERIENCES"):
                raise aiosqlite.OperationalError("injected execute failure")
            return await original_execute(sql, *args, **kwargs)

        with patch.object(connection, "execute", new=fail_execute):
            with pytest.raises(StorageError) as raised:
                await store.save_experience(candidate)
        after = await _persisted_snapshot(db_path)
    elif failure_point == "commit":

        async def fail_commit() -> None:
            raise aiosqlite.OperationalError("injected commit failure")

        with patch.object(connection, "commit", new=fail_commit):
            with pytest.raises(StorageError) as raised:
                await store.save_experience(candidate)
        after = await _persisted_snapshot(db_path)
    else:

        async def execute_then_fail(sql, *args, **kwargs):
            result = await original_execute(sql, *args, **kwargs)
            if " ".join(sql.split()).upper().startswith("INSERT INTO EXPERIENCES"):
                raise primary_failure
            return result

        async def fail_rollback() -> None:
            raise InjectedRollbackFailure("injected rollback failure")

        with (
            patch.object(connection, "execute", new=execute_then_fail),
            patch.object(connection, "rollback", new=fail_rollback),
        ):
            with pytest.raises(StorageError) as raised:
                await store.save_experience(candidate)
            after = await _persisted_snapshot(db_path)
        await original_rollback()

    error = raised.value
    assert error.operation == "save"
    assert error.table == "experiences"
    assert error.record_ids == (str(candidate.id),)
    assert after == before
    if failure_point == "rollback":
        assert error is primary_failure
        assert isinstance(error.__cause__, InjectedRollbackFailure)
    else:
        assert isinstance(error.__cause__, aiosqlite.OperationalError)
