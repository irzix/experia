"""Versioned, failure-atomic SQLite schema migrations."""

from __future__ import annotations

import json
import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone

import aiosqlite

from experia.core.exceptions import StorageError
from experia.memory.transactions import SQLiteTransactionManager

MigrationStep = Callable[[aiosqlite.Connection], Awaitable[None]]

CURRENT_SCHEMA_VERSION = 3
MIN_SUPPORTED_SCHEMA_VERSION = 0
SUPPORTED_SCHEMA_VERSIONS = tuple(
    range(MIN_SUPPORTED_SCHEMA_VERSION, CURRENT_SCHEMA_VERSION + 1)
)


@dataclass(frozen=True)
class Migration:
    """One ordered transition in the SQLite schema registry."""

    from_version: int
    to_version: int
    name: str
    apply: MigrationStep = field(repr=False, compare=False)


_BASE_COLUMNS = {
    "experiences": {"id", "task", "action", "result", "context", "created_at"},
    "lessons": {"id", "experience_id", "content", "confidence", "created_at"},
    "memories": {
        "id",
        "content",
        "type",
        "confidence",
        "importance",
        "source",
        "metadata",
        "created_at",
        "updated_at",
        "expires_at",
    },
}
_VERSION_ONE_COLUMNS = {
    "experiences": _BASE_COLUMNS["experiences"] | {"agent_role"},
    "lessons": _BASE_COLUMNS["lessons"] | {"agent_role", "root_cause"},
    "memories": _BASE_COLUMNS["memories"]
    | {"agent_role", "embedding", "reinforcement_count", "success_count"},
}


async def _table_columns(conn: aiosqlite.Connection, table: str) -> set[str]:
    cursor = await conn.execute(f"PRAGMA table_info({table})")
    return {str(row[1]) for row in await cursor.fetchall()}


async def _add_column_if_missing(
    conn: aiosqlite.Connection,
    *,
    table: str,
    column: str,
    definition: str,
) -> None:
    if column not in await _table_columns(conn, table):
        await conn.execute(f"ALTER TABLE {table} ADD COLUMN {definition}")


async def _ensure_registry_table(conn: aiosqlite.Connection) -> None:
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            applied_at TEXT NOT NULL
        )
        """
    )


async def _normalize_source_schema(conn: aiosqlite.Connection) -> None:
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS experiences (
            id TEXT PRIMARY KEY,
            task TEXT NOT NULL,
            action TEXT NOT NULL,
            result TEXT NOT NULL,
            agent_role TEXT NOT NULL DEFAULT 'default',
            context TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS lessons (
            id TEXT PRIMARY KEY,
            experience_id TEXT NOT NULL,
            content TEXT NOT NULL,
            agent_role TEXT NOT NULL DEFAULT 'default',
            root_cause TEXT,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (experience_id) REFERENCES experiences (id)
        )
        """
    )
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            type TEXT NOT NULL,
            agent_role TEXT NOT NULL DEFAULT 'default',
            confidence REAL NOT NULL,
            importance REAL NOT NULL,
            source TEXT,
            metadata TEXT,
            embedding TEXT,
            reinforcement_count INTEGER NOT NULL DEFAULT 0,
            success_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            expires_at TEXT
        )
        """
    )

    additions = (
        ("experiences", "agent_role", "agent_role TEXT DEFAULT 'default'"),
        ("lessons", "agent_role", "agent_role TEXT DEFAULT 'default'"),
        ("lessons", "root_cause", "root_cause TEXT"),
        ("memories", "agent_role", "agent_role TEXT DEFAULT 'default'"),
        ("memories", "embedding", "embedding TEXT"),
        (
            "memories",
            "reinforcement_count",
            "reinforcement_count INTEGER NOT NULL DEFAULT 0",
        ),
        (
            "memories",
            "success_count",
            "success_count INTEGER NOT NULL DEFAULT 0",
        ),
    )
    for table, column, definition in additions:
        await _add_column_if_missing(
            conn,
            table=table,
            column=column,
            definition=definition,
        )

    for ddl in (
        "CREATE INDEX IF NOT EXISTS idx_exp_created ON experiences(created_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_lessons_exp ON lessons(experience_id)",
        "CREATE INDEX IF NOT EXISTS idx_mem_role_type ON memories(agent_role, type)",
        "CREATE INDEX IF NOT EXISTS idx_mem_rank ON memories(importance DESC, confidence DESC)",
        "CREATE INDEX IF NOT EXISTS idx_mem_expires ON memories(expires_at)",
    ):
        await conn.execute(ddl)
    await _ensure_registry_table(conn)


async def _embedding_dimension(
    stored_embedding: object,
    *,
    memory_id: str,
) -> int:
    try:
        if not isinstance(stored_embedding, str):
            raise TypeError("stored embedding is not text")
        decoded = json.loads(stored_embedding)
        if not isinstance(decoded, list) or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in decoded
        ):
            raise TypeError("stored embedding is not a finite numeric vector")
        return len(decoded)
    except Exception as exc:
        raise StorageError(
            "Stored embedding cannot be migrated.",
            operation="migrate",
            table="memories",
            record_ids=(memory_id,),
            migration="add_vector_index_schema",
            field="embedding",
        ) from exc


async def _add_vector_index_schema(conn: aiosqlite.Connection) -> None:
    # Version-one databases can predate the registry while already having all
    # source columns. Re-running normalization restores any missing source
    # indexes without rewriting source rows.
    await _normalize_source_schema(conn)
    await _add_column_if_missing(
        conn,
        table="memories",
        column="embedding_dimension",
        definition=(
            "embedding_dimension INTEGER "
            "CHECK (embedding_dimension IS NULL OR embedding_dimension >= 0)"
        ),
    )

    cursor = await conn.execute(
        "SELECT id, embedding FROM memories WHERE embedding IS NOT NULL"
    )
    for memory_id, stored_embedding in await cursor.fetchall():
        dimension = await _embedding_dimension(
            stored_embedding,
            memory_id=str(memory_id),
        )
        await conn.execute(
            "UPDATE memories SET embedding_dimension = ? WHERE id = ?",
            (dimension, memory_id),
        )

    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_vector_bands (
            memory_id TEXT NOT NULL,
            dimension INTEGER NOT NULL,
            band INTEGER NOT NULL,
            bucket INTEGER NOT NULL,
            index_version INTEGER NOT NULL,
            PRIMARY KEY (memory_id, band),
            FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
        )
        """
    )
    await conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_vector_bucket
        ON memory_vector_bands(dimension, index_version, band, bucket)
        """
    )


async def _add_vector_index_state(conn: aiosqlite.Connection) -> None:
    """Add persisted rebuild readiness without populating derived band rows."""
    from experia.memory.vector_index import (
        CURRENT_VECTOR_INDEX_VERSION,
        INDEX_STATUS_READY,
        INDEX_STATUS_REBUILDING,
    )

    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_vector_index_state (
            singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
            index_version INTEGER NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('rebuilding', 'ready')),
            last_memory_id TEXT
        )
        """
    )
    cursor = await conn.execute(
        "SELECT 1 FROM memories WHERE embedding IS NOT NULL LIMIT 1"
    )
    has_embeddings = await cursor.fetchone() is not None
    await conn.execute(
        "INSERT OR IGNORE INTO memory_vector_index_state "
        "(singleton, index_version, status, last_memory_id) VALUES (1, ?, ?, NULL)",
        (
            CURRENT_VECTOR_INDEX_VERSION,
            INDEX_STATUS_REBUILDING if has_embeddings else INDEX_STATUS_READY,
        ),
    )


MIGRATIONS: tuple[Migration, ...] = (
    Migration(0, 1, "normalize_source_schema", _normalize_source_schema),
    Migration(1, 2, "add_vector_index_schema", _add_vector_index_schema),
    Migration(2, 3, "add_vector_index_state", _add_vector_index_state),
)


class SchemaMigrator:
    """Apply an ordered migration chain in one immediate transaction."""

    def __init__(
        self,
        transactions: SQLiteTransactionManager,
        migrations: Sequence[Migration] = MIGRATIONS,
    ) -> None:
        self._transactions = transactions
        self._migrations = tuple(migrations)
        self._validate_registry()

    async def migrate(
        self,
        conn: aiosqlite.Connection,
        *,
        target_version: int = CURRENT_SCHEMA_VERSION,
    ) -> None:
        """Migrate to ``target_version`` without implicit downgrades."""
        if isinstance(target_version, bool) or not isinstance(target_version, int):
            raise StorageError(
                "Schema target version is invalid.",
                operation="migrate",
                table="schema_migrations",
                migration=str(target_version),
            )

        initial_version = await self._read_version(conn)
        self._reject_downgrade(initial_version, target_version)
        if initial_version == target_version:
            return

        chain_context = f"{initial_version}->{target_version}"
        async with self._transactions.write(
            operation="migrate",
            table="schema_migrations",
            migration=chain_context,
        ) as transaction:
            pragma_version = await self._read_version(transaction)
            self._reject_downgrade(pragma_version, target_version)

            effective_version = pragma_version
            if pragma_version == 0:
                effective_version = await self._detect_legacy_version(transaction)
                if effective_version:
                    await _ensure_registry_table(transaction)
                    for migration in self._migrations:
                        if migration.to_version <= effective_version:
                            await self._record_migration(transaction, migration)

            self._reject_downgrade(effective_version, target_version)
            chain = self._migration_chain(effective_version, target_version)
            for migration in chain:
                try:
                    await migration.apply(transaction)
                    await self._record_migration(transaction, migration)
                except StorageError:
                    raise
                except Exception as exc:
                    raise StorageError(
                        "SQLite schema migration failed.",
                        operation="migrate",
                        table="schema_migrations",
                        migration=migration.name,
                    ) from exc

            # user_version is the source of truth and changes only after every
            # migration step and audit record in this chain has succeeded.
            await transaction.execute(f"PRAGMA user_version = {target_version}")

    def _validate_registry(self) -> None:
        expected = 0
        versions: set[int] = set()
        names: set[str] = set()
        for migration in self._migrations:
            if migration.from_version != expected:
                raise ValueError("Migration registry must be contiguous and ordered")
            if migration.to_version != migration.from_version + 1:
                raise ValueError("Each migration must advance exactly one version")
            if migration.to_version in versions or migration.name in names:
                raise ValueError("Migration versions and names must be unique")
            versions.add(migration.to_version)
            names.add(migration.name)
            expected = migration.to_version

    async def _detect_legacy_version(self, conn: aiosqlite.Connection) -> int:
        cursor = await conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )
        table_names = {str(row[0]) for row in await cursor.fetchall()}
        source_tables = table_names & _BASE_COLUMNS.keys()
        if not source_tables:
            return 0

        observed: dict[str, set[str]] = {}
        for table in source_tables:
            columns = await _table_columns(conn, table)
            observed[table] = columns
            missing = _BASE_COLUMNS[table] - columns
            if missing:
                raise StorageError(
                    "Legacy SQLite schema is not supported.",
                    operation="migrate",
                    table=table,
                    migration="legacy_detection",
                    field=sorted(missing)[0],
                )

        if source_tables == _VERSION_ONE_COLUMNS.keys() and all(
            _VERSION_ONE_COLUMNS[table] <= observed[table] for table in source_tables
        ):
            return 1
        return 0

    def _migration_chain(
        self, current_version: int, target_version: int
    ) -> tuple[Migration, ...]:
        chain: list[Migration] = []
        version = current_version
        while version < target_version:
            migration = next(
                (
                    candidate
                    for candidate in self._migrations
                    if candidate.from_version == version
                ),
                None,
            )
            if migration is None or migration.to_version > target_version:
                raise StorageError(
                    "No supported SQLite migration path exists.",
                    operation="migrate",
                    table="schema_migrations",
                    migration=f"{current_version}->{target_version}",
                )
            chain.append(migration)
            version = migration.to_version
        return tuple(chain)

    @staticmethod
    async def _read_version(conn: aiosqlite.Connection) -> int:
        cursor = await conn.execute("PRAGMA user_version")
        row = await cursor.fetchone()
        return int(row[0])

    @staticmethod
    def _reject_downgrade(current_version: int, target_version: int) -> None:
        if target_version < current_version:
            raise StorageError(
                "SQLite schema downgrades are not supported.",
                operation="migrate",
                table="schema_migrations",
                migration=f"{current_version}->{target_version}",
            )

    @staticmethod
    async def _record_migration(
        conn: aiosqlite.Connection,
        migration: Migration,
    ) -> None:
        cursor = await conn.execute(
            "SELECT name FROM schema_migrations WHERE version = ?",
            (migration.to_version,),
        )
        existing = await cursor.fetchone()
        if existing is not None:
            if existing[0] != migration.name:
                raise StorageError(
                    "Migration audit record conflicts with the registry.",
                    operation="migrate",
                    table="schema_migrations",
                    record_ids=(str(migration.to_version),),
                    migration=migration.name,
                )
            return
        await conn.execute(
            "INSERT INTO schema_migrations (version, name, applied_at) "
            "VALUES (?, ?, ?)",
            (
                migration.to_version,
                migration.name,
                datetime.now(timezone.utc).isoformat(),
            ),
        )


__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "MIN_SUPPORTED_SCHEMA_VERSION",
    "MIGRATIONS",
    "SUPPORTED_SCHEMA_VERSIONS",
    "Migration",
    "SchemaMigrator",
]
