"""Deterministic migration lifecycle and fault-injection coverage."""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path
from unittest.mock import patch

import aiosqlite
import pytest

from experia.core.exceptions import StorageError
from experia.memory.migrations import (
    CURRENT_SCHEMA_VERSION,
    MIGRATIONS,
    Migration,
    SchemaMigrator,
)
from experia.memory.transactions import SQLiteTransactionManager

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "sqlite"
_SOURCE_TABLES = ("experiences", "lessons", "memories")


def _materialize_fixture(script_name: str, database_path: Path) -> None:
    connection = sqlite3.connect(database_path)
    try:
        connection.executescript(
            (FIXTURE_ROOT / script_name).read_text(encoding="utf-8")
        )
    finally:
        connection.close()


def _migrator(
    connection: aiosqlite.Connection,
    migrations: Sequence[Migration] = MIGRATIONS,
) -> SchemaMigrator:
    transactions = SQLiteTransactionManager(lambda: connection, asyncio.Lock())
    return SchemaMigrator(transactions, migrations)


async def _schema_version(connection: aiosqlite.Connection) -> int:
    cursor = await connection.execute("PRAGMA user_version")
    row = await cursor.fetchone()
    assert row is not None
    return int(row[0])


async def _database_snapshot(connection: aiosqlite.Connection) -> tuple[object, ...]:
    """Capture schema and rows so transactional DDL rollback is observable."""
    schema_cursor = await connection.execute(
        "SELECT type, name, tbl_name, sql FROM sqlite_master "
        "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
    )
    schema = tuple(await schema_cursor.fetchall())
    table_names = sorted(
        name for object_type, name, _, _ in schema if object_type == "table"
    )
    rows = []
    for table in table_names:
        cursor = await connection.execute(f'SELECT * FROM "{table}" ORDER BY 1')
        rows.append((table, tuple(await cursor.fetchall())))
    return await _schema_version(connection), schema, tuple(rows)


async def _source_columns(
    connection: aiosqlite.Connection,
) -> dict[str, tuple[str, ...]]:
    columns: dict[str, tuple[str, ...]] = {}
    for table in _SOURCE_TABLES:
        cursor = await connection.execute(f'PRAGMA table_info("{table}")')
        columns[table] = tuple(str(row[1]) for row in await cursor.fetchall())
    return columns


async def _source_snapshot(
    connection: aiosqlite.Connection,
    columns: Mapping[str, Sequence[str]],
) -> tuple[tuple[str, tuple[tuple[object, ...], ...]], ...]:
    snapshot = []
    for table in _SOURCE_TABLES:
        selected = ", ".join(f'"{column}"' for column in columns[table])
        cursor = await connection.execute(
            f'SELECT {selected} FROM "{table}" ORDER BY id'
        )
        snapshot.append((table, tuple(await cursor.fetchall())))
    return tuple(snapshot)


async def _assert_successful_retry(
    connection: aiosqlite.Connection,
    *,
    source_columns: Mapping[str, Sequence[str]],
    source_before: tuple[tuple[str, tuple[tuple[object, ...], ...]], ...],
) -> None:
    migrator = _migrator(connection)
    await migrator.migrate(connection)

    assert await _schema_version(connection) == CURRENT_SCHEMA_VERSION
    assert await _source_snapshot(connection, source_columns) == source_before
    cursor = await connection.execute(
        "SELECT version, name FROM schema_migrations ORDER BY version"
    )
    assert await cursor.fetchall() == [
        (1, "normalize_source_schema"),
        (2, "add_vector_index_schema"),
        (3, "add_vector_index_state"),
    ]

    completed = await _database_snapshot(connection)
    await migrator.migrate(connection)
    assert await _database_snapshot(connection) == completed


# Validates: Requirements 3.15, 7.6
@pytest.mark.asyncio
async def test_migration_step_failure_rolls_back_entire_chain_and_is_retry_safe(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "step-failure.db"
    _materialize_fixture("schema-v0.sql", database_path)
    connection = await aiosqlite.connect(database_path)

    async def fail_after_changes(transaction: aiosqlite.Connection) -> None:
        await transaction.execute(
            "CREATE TABLE injected_partial_migration (value TEXT NOT NULL)"
        )
        await transaction.execute("UPDATE memories SET content = 'must be rolled back'")
        raise RuntimeError("injected migration step failure")

    failing_migration = Migration(
        1,
        2,
        "injected_vector_index_failure",
        fail_after_changes,
    )
    migrator = _migrator(
        connection,
        (MIGRATIONS[0], failing_migration, MIGRATIONS[2]),
    )
    try:
        source_columns = await _source_columns(connection)
        source_before = await _source_snapshot(connection, source_columns)
        database_before = await _database_snapshot(connection)

        with pytest.raises(StorageError) as raised:
            await migrator.migrate(connection)

        error = raised.value
        assert error.operation == "migrate"
        assert error.table == "schema_migrations"
        assert error.record_ids == ()
        assert error.migration == "injected_vector_index_failure"
        assert isinstance(error.__cause__, RuntimeError)
        assert await _database_snapshot(connection) == database_before

        await _assert_successful_retry(
            connection,
            source_columns=source_columns,
            source_before=source_before,
        )
    finally:
        await connection.close()


# Validates: Requirements 3.15, 7.6
@pytest.mark.asyncio
async def test_migration_commit_failure_rolls_back_and_retry_is_idempotent(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "commit-failure.db"
    _materialize_fixture("schema-v1.sql", database_path)
    connection = await aiosqlite.connect(database_path)
    migrator = _migrator(connection)
    try:
        source_columns = await _source_columns(connection)
        source_before = await _source_snapshot(connection, source_columns)
        database_before = await _database_snapshot(connection)

        async def fail_commit() -> None:
            raise aiosqlite.OperationalError("injected migration commit failure")

        with patch.object(connection, "commit", new=fail_commit):
            with pytest.raises(StorageError) as raised:
                await migrator.migrate(connection)

        error = raised.value
        assert error.operation == "migrate"
        assert error.table == "schema_migrations"
        assert error.record_ids == ()
        assert error.migration == f"1->{CURRENT_SCHEMA_VERSION}"
        assert isinstance(error.__cause__, aiosqlite.OperationalError)
        assert await _database_snapshot(connection) == database_before

        await _assert_successful_retry(
            connection,
            source_columns=source_columns,
            source_before=source_before,
        )
    finally:
        await connection.close()


# Validates: Requirements 3.15, 7.6
@pytest.mark.asyncio
async def test_explicit_downgrade_is_rejected_without_starting_a_transaction(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "downgrade.db"
    _materialize_fixture("schema-v2.sql", database_path)
    connection = await aiosqlite.connect(database_path)
    statements: list[str] = []
    await connection.set_trace_callback(statements.append)
    try:
        database_before = await _database_snapshot(connection)

        with pytest.raises(StorageError) as raised:
            await _migrator(connection).migrate(connection, target_version=1)

        error = raised.value
        assert error.operation == "migrate"
        assert error.table == "schema_migrations"
        assert error.record_ids == ()
        assert error.migration == "2->1"
        assert await _database_snapshot(connection) == database_before
        normalized = {" ".join(statement.split()).upper() for statement in statements}
        assert "BEGIN IMMEDIATE" not in normalized
        assert "COMMIT" not in normalized
        assert "ROLLBACK" not in normalized
    finally:
        await connection.close()


# Validates: Requirements 3.15, 7.6
@pytest.mark.asyncio
async def test_vector_index_schema_rebuild_changes_no_source_record_values(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "derived-index.db"
    _materialize_fixture("schema-v1.sql", database_path)
    connection = await aiosqlite.connect(database_path)
    try:
        source_columns = await _source_columns(connection)
        source_before = await _source_snapshot(connection, source_columns)

        migrator = _migrator(connection)
        await migrator.migrate(connection)

        assert await _source_snapshot(connection, source_columns) == source_before
        cursor = await connection.execute(
            "SELECT id, embedding_dimension FROM memories ORDER BY id"
        )
        assert await cursor.fetchall() == [
            ("33333333-3333-4333-8333-333333333333", 4),
            ("44444444-4444-4444-8444-444444444444", 3),
        ]
        cursor = await connection.execute(
            "SELECT memory_id, dimension, band, bucket, index_version "
            "FROM memory_vector_bands ORDER BY memory_id, band"
        )
        assert await cursor.fetchall() == []

        source_after_first_rebuild = await _source_snapshot(connection, source_columns)
        await migrator.migrate(connection)
        assert (
            await _source_snapshot(connection, source_columns)
            == source_after_first_rebuild
        )
    finally:
        await connection.close()
