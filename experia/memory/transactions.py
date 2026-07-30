"""Shared transaction boundary for SQLite writes."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager
from typing import AsyncIterator, NoReturn

import aiosqlite

from experia.core.exceptions import StorageError


class SQLiteTransactionManager:
    """Serialize and atomically commit writes on a caller-owned connection."""

    def __init__(
        self,
        connection: Callable[[], aiosqlite.Connection],
        write_lock: asyncio.Lock,
    ) -> None:
        self._connection = connection
        self._write_lock = write_lock

    @asynccontextmanager
    async def write(
        self,
        *,
        table: str,
        record_ids: Sequence[str] = (),
        operation: str = "write",
        migration: str | None = None,
    ) -> AsyncIterator[aiosqlite.Connection]:
        """Run one write under ``BEGIN IMMEDIATE`` and a shared writer lock."""
        identifiers = tuple(str(record_id) for record_id in record_ids)
        async with self._write_lock:
            conn = self._connection()
            try:
                await conn.execute("BEGIN IMMEDIATE")
                yield conn
                await conn.commit()
            except BaseException as original:
                rollback_failure = await self._rollback(conn)
                self._raise_failure(
                    original,
                    rollback_failure=rollback_failure,
                    operation=operation,
                    table=table,
                    record_ids=identifiers,
                    migration=migration,
                )

    @staticmethod
    async def _rollback(
        conn: aiosqlite.Connection,
    ) -> BaseException | None:
        try:
            await conn.rollback()
        except BaseException as rollback_failure:
            # The primary failure remains the transaction error. Clear the
            # implicit context so attaching rollback as its cause cannot form
            # a circular exception chain.
            rollback_failure.__context__ = None
            return rollback_failure
        return None

    @staticmethod
    def _raise_failure(
        original: BaseException,
        *,
        rollback_failure: BaseException | None,
        operation: str,
        table: str,
        record_ids: tuple[str, ...],
        migration: str | None,
    ) -> NoReturn:
        if isinstance(original, StorageError):
            if rollback_failure is not None:
                raise original from rollback_failure
            raise original

        if not isinstance(original, Exception):
            if rollback_failure is not None:
                raise original from rollback_failure
            raise original

        error = StorageError(
            "SQLite write transaction failed.",
            operation=operation,
            table=table,
            record_ids=record_ids,
            migration=migration,
        )
        if rollback_failure is not None:
            raise error from rollback_failure
        raise error from original


__all__ = ["SQLiteTransactionManager"]
