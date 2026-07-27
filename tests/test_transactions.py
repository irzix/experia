import asyncio

import aiosqlite
import pytest

from experia.core.exceptions import StorageError
from experia.memory.transactions import SQLiteTransactionManager


class CommitFailure(RuntimeError):
    pass


class RollbackFailure(RuntimeError):
    pass


class CommitFailingConnection:
    def __init__(self, connection: aiosqlite.Connection) -> None:
        self.connection = connection

    async def execute(self, *args, **kwargs):
        return await self.connection.execute(*args, **kwargs)

    async def commit(self) -> None:
        raise CommitFailure("injected commit failure")

    async def rollback(self) -> None:
        await self.connection.rollback()


class RollbackFailingConnection:
    def __init__(self, connection: aiosqlite.Connection) -> None:
        self.connection = connection

    async def execute(self, *args, **kwargs):
        return await self.connection.execute(*args, **kwargs)

    async def commit(self) -> None:
        await self.connection.commit()

    async def rollback(self) -> None:
        raise RollbackFailure("injected rollback failure")


@pytest.mark.asyncio
async def test_write_serializes_writers_and_uses_immediate_transactions(tmp_path):
    conn = await aiosqlite.connect(tmp_path / "serialized.db")
    try:
        await conn.execute("CREATE TABLE entries (value TEXT NOT NULL)")
        await conn.commit()
        statements: list[str] = []
        await conn.set_trace_callback(statements.append)
        manager = SQLiteTransactionManager(lambda: conn, asyncio.Lock())
        first_entered = asyncio.Event()
        release_first = asyncio.Event()
        second_attempting = asyncio.Event()
        second_entered = asyncio.Event()

        async def first_writer() -> None:
            async with manager.write(table="entries", record_ids=("first",)):
                await conn.execute("INSERT INTO entries VALUES ('first')")
                first_entered.set()
                await release_first.wait()

        async def second_writer() -> None:
            second_attempting.set()
            async with manager.write(table="entries", record_ids=("second",)):
                second_entered.set()
                await conn.execute("INSERT INTO entries VALUES ('second')")

        first = asyncio.create_task(first_writer())
        await first_entered.wait()
        second = asyncio.create_task(second_writer())
        await second_attempting.wait()
        try:
            assert not second_entered.is_set()
        finally:
            release_first.set()
        await asyncio.gather(first, second)

        cursor = await conn.execute("SELECT value FROM entries ORDER BY value")
        assert await cursor.fetchall() == [("first",), ("second",)]
        normalized = [statement.strip().upper() for statement in statements]
        assert normalized.count("BEGIN IMMEDIATE") == 2
        assert normalized.count("COMMIT") == 2
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_write_rolls_back_body_failure_with_context(tmp_path):
    conn = await aiosqlite.connect(tmp_path / "rollback.db")
    try:
        await conn.execute("CREATE TABLE entries (value TEXT NOT NULL)")
        await conn.commit()
        manager = SQLiteTransactionManager(lambda: conn, asyncio.Lock())

        with pytest.raises(StorageError) as raised:
            async with manager.write(
                operation="save",
                table="entries",
                record_ids=("record-1",),
            ):
                await conn.execute("INSERT INTO entries VALUES ('partial')")
                raise RuntimeError("injected statement failure")

        error = raised.value
        assert error.operation == "save"
        assert error.table == "entries"
        assert error.record_ids == ("record-1",)
        assert isinstance(error.__cause__, RuntimeError)
        cursor = await conn.execute("SELECT value FROM entries")
        assert await cursor.fetchall() == []
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_commit_failure_is_contextual_and_rolls_back(tmp_path):
    conn = await aiosqlite.connect(tmp_path / "commit.db")
    try:
        await conn.execute("CREATE TABLE entries (value TEXT NOT NULL)")
        await conn.commit()
        failing = CommitFailingConnection(conn)
        manager = SQLiteTransactionManager(lambda: failing, asyncio.Lock())

        with pytest.raises(StorageError) as raised:
            async with manager.write(
                operation="save",
                table="entries",
                record_ids=("record-2",),
            ) as transaction:
                await transaction.execute("INSERT INTO entries VALUES ('partial')")

        error = raised.value
        assert error.operation == "save"
        assert error.table == "entries"
        assert error.record_ids == ("record-2",)
        assert isinstance(error.__cause__, CommitFailure)
        cursor = await conn.execute("SELECT value FROM entries")
        assert await cursor.fetchall() == []
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_rollback_failure_keeps_original_contextual_error(tmp_path):
    conn = await aiosqlite.connect(tmp_path / "rollback_failure.db")
    try:
        await conn.execute("CREATE TABLE entries (value TEXT NOT NULL)")
        await conn.commit()
        failing = RollbackFailingConnection(conn)
        manager = SQLiteTransactionManager(lambda: failing, asyncio.Lock())
        original = StorageError(
            "Original write failure.",
            operation="save",
            table="entries",
            record_ids=("record-3",),
        )

        with pytest.raises(StorageError) as raised:
            async with manager.write(
                operation="different-operation",
                table="different-table",
                record_ids=("different-record",),
            ):
                await conn.execute("INSERT INTO entries VALUES ('partial')")
                raise original

        assert raised.value is original
        assert raised.value.operation == "save"
        assert raised.value.table == "entries"
        assert raised.value.record_ids == ("record-3",)
        assert isinstance(raised.value.__cause__, RollbackFailure)
    finally:
        await conn.rollback()
        await conn.close()
