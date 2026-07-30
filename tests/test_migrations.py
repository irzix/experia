"""Focused unit coverage for versioned SQLite schema migration."""

import asyncio
from datetime import datetime, timezone
from uuid import UUID

import aiosqlite
import pytest

from experia.core.exceptions import StorageError
from experia.memory.migrations import CURRENT_SCHEMA_VERSION, SchemaMigrator
from experia.memory.models import Memory, MemoryType
from experia.memory.store import SQLiteStore
from experia.memory.transactions import SQLiteTransactionManager


def _migrator(
    connection: aiosqlite.Connection,
) -> SchemaMigrator:
    return SchemaMigrator(SQLiteTransactionManager(lambda: connection, asyncio.Lock()))


@pytest.mark.asyncio
async def test_fresh_migration_is_one_ordered_chain_and_target_is_noop(tmp_path):
    connection = await aiosqlite.connect(tmp_path / "fresh.db")
    statements: list[str] = []
    await connection.set_trace_callback(statements.append)
    migrator = _migrator(connection)
    try:
        await migrator.migrate(connection)

        normalized = [" ".join(statement.split()) for statement in statements]
        assert normalized.count("BEGIN IMMEDIATE") == 1
        assert normalized.count("COMMIT") == 1
        version_updates = [
            statement
            for statement in normalized
            if statement.upper().startswith("PRAGMA USER_VERSION =")
        ]
        assert version_updates == [f"PRAGMA user_version = {CURRENT_SCHEMA_VERSION}"]
        version_update_index = normalized.index(version_updates[0])
        assert version_update_index > next(
            index
            for index, statement in enumerate(normalized)
            if "CREATE TABLE IF NOT EXISTS memory_vector_bands" in statement
        )

        cursor = await connection.execute("PRAGMA user_version")
        assert await cursor.fetchone() == (CURRENT_SCHEMA_VERSION,)
        cursor = await connection.execute(
            "SELECT version, name FROM schema_migrations ORDER BY version"
        )
        assert await cursor.fetchall() == [
            (1, "normalize_source_schema"),
            (2, "add_vector_index_schema"),
            (3, "add_vector_index_state"),
        ]
        cursor = await connection.execute("PRAGMA table_info(memories)")
        assert "embedding_dimension" in {row[1] for row in await cursor.fetchall()}
        cursor = await connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name = 'memory_vector_bands'"
        )
        assert await cursor.fetchone() == ("memory_vector_bands",)

        statements.clear()
        await migrator.migrate(connection)
        normalized = [" ".join(statement.split()).upper() for statement in statements]
        assert "BEGIN IMMEDIATE" not in normalized
        assert "COMMIT" not in normalized
        assert not any(
            statement.startswith("PRAGMA USER_VERSION =") for statement in normalized
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_legacy_schema_is_detected_and_source_values_are_preserved(tmp_path):
    db_path = tmp_path / "legacy.db"
    connection = await aiosqlite.connect(db_path)
    memory_id = UUID("22222222-2222-2222-2222-222222222222")
    experience_id = UUID("11111111-1111-1111-1111-111111111111")
    created_at = datetime(2024, 1, 2, 3, 4, tzinfo=timezone.utc).isoformat()
    embedding = "[0.25,-0.5,1.0]"
    try:
        await connection.execute(
            """
            CREATE TABLE experiences (
                id TEXT PRIMARY KEY, task TEXT NOT NULL, action TEXT NOT NULL,
                result TEXT NOT NULL, context TEXT, created_at TEXT NOT NULL
            )
            """
        )
        await connection.execute(
            """
            CREATE TABLE lessons (
                id TEXT PRIMARY KEY, experience_id TEXT NOT NULL,
                content TEXT NOT NULL, confidence REAL NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        await connection.execute(
            """
            CREATE TABLE memories (
                id TEXT PRIMARY KEY, content TEXT NOT NULL, type TEXT NOT NULL,
                confidence REAL NOT NULL, importance REAL NOT NULL, source TEXT,
                metadata TEXT, embedding TEXT, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL, expires_at TEXT
            )
            """
        )
        await connection.execute(
            "INSERT INTO experiences VALUES (?, ?, ?, ?, ?, ?)",
            (
                str(experience_id),
                "legacy task",
                "legacy action",
                "legacy result",
                '{"offset":"preserved"}',
                created_at,
            ),
        )
        await connection.execute(
            "INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(memory_id),
                "legacy memory",
                "lesson",
                0.75,
                0.8,
                "legacy-source",
                '{"nested":{"value":1}}',
                embedding,
                created_at,
                created_at,
                None,
            ),
        )
        await connection.commit()
    finally:
        await connection.close()

    store = SQLiteStore(str(db_path))
    await store.initialize()
    try:
        connection = store._require_conn()
        cursor = await connection.execute(
            "SELECT content, type, confidence, importance, source, metadata, "
            "embedding, created_at, updated_at, expires_at, agent_role, "
            "reinforcement_count, success_count, embedding_dimension "
            "FROM memories WHERE id = ?",
            (str(memory_id),),
        )
        assert await cursor.fetchone() == (
            "legacy memory",
            "lesson",
            0.75,
            0.8,
            "legacy-source",
            '{"nested":{"value":1}}',
            embedding,
            created_at,
            created_at,
            None,
            "default",
            0,
            0,
            3,
        )
        cursor = await connection.execute(
            "SELECT task, action, result, context, created_at, agent_role "
            "FROM experiences WHERE id = ?",
            (str(experience_id),),
        )
        assert await cursor.fetchone() == (
            "legacy task",
            "legacy action",
            "legacy result",
            '{"offset":"preserved"}',
            created_at,
            "default",
        )

        loaded = await store.get_memory(memory_id)
        assert loaded == Memory(
            id=memory_id,
            content="legacy memory",
            type=MemoryType.LESSON,
            confidence=0.75,
            importance=0.8,
            source="legacy-source",
            metadata={"nested": {"value": 1}},
            embedding=[0.25, -0.5, 1.0],
            created_at=datetime.fromisoformat(created_at),
            updated_at=datetime.fromisoformat(created_at),
        )
        cursor = await connection.execute("SELECT * FROM memory_vector_bands")
        assert await cursor.fetchall() == []
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_new_memory_persists_embedding_dimension(tmp_path):
    store = SQLiteStore(str(tmp_path / "dimension.db"))
    await store.initialize()
    memory = Memory(
        content="dimension-aware",
        type=MemoryType.FACT,
        embedding=[1.0, 0.0, -1.0, 0.5],
    )
    try:
        await store.save_memory(memory)
        cursor = await store._require_conn().execute(
            "SELECT embedding, embedding_dimension FROM memories WHERE id = ?",
            (str(memory.id),),
        )
        assert await cursor.fetchone() == ("[1.0,0.0,-1.0,0.5]", 4)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_migrator_rejects_downgrade_without_schema_changes(tmp_path):
    connection = await aiosqlite.connect(tmp_path / "future.db")
    try:
        await connection.execute("CREATE TABLE sentinel (value TEXT NOT NULL)")
        await connection.execute("INSERT INTO sentinel VALUES ('preserved')")
        await connection.execute(f"PRAGMA user_version = {CURRENT_SCHEMA_VERSION + 1}")
        await connection.commit()
        migrator = _migrator(connection)

        with pytest.raises(StorageError) as raised:
            await migrator.migrate(connection)

        error = raised.value
        assert error.operation == "migrate"
        assert error.table == "schema_migrations"
        assert error.migration == (
            f"{CURRENT_SCHEMA_VERSION + 1}->{CURRENT_SCHEMA_VERSION}"
        )
        cursor = await connection.execute("PRAGMA user_version")
        assert await cursor.fetchone() == (CURRENT_SCHEMA_VERSION + 1,)
        cursor = await connection.execute("SELECT value FROM sentinel")
        assert await cursor.fetchall() == [("preserved",)]
    finally:
        await connection.close()
