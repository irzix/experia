"""Property coverage for preserving and idempotent SQLite migrations."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import UUID

import aiosqlite
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from experia.experience.models import ExperienceRecord, Lesson
from experia.memory.migrations import (
    CURRENT_SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    SchemaMigrator,
)
from experia.memory.models import Memory, MemoryType
from experia.memory.transactions import SQLiteTransactionManager
from tests.quality_profiles import STANDARD_PBT_MIN_EXAMPLES

_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "sqlite"
_MANIFEST_PATH = _FIXTURE_ROOT / "schema-support.json"
_TABLES_WITH_SOURCE_RECORDS = ("experiences", "lessons", "memories")
_TARGET_TABLES = (
    "experiences",
    "lessons",
    "memories",
    "schema_migrations",
    "memory_vector_bands",
    "memory_vector_index_state",
)


@dataclass(frozen=True)
class _FixtureCase:
    schema_version: int
    script_path: Path


@dataclass(frozen=True)
class _RecordBundle:
    experience: ExperienceRecord
    lesson: Lesson
    memories: tuple[Memory, ...]


@dataclass(frozen=True)
class _DatabaseSnapshot:
    schema_version: int
    structure: tuple[tuple[Any, ...], ...]
    records: tuple[tuple[str, tuple[str, ...], tuple[tuple[Any, ...], ...]], ...]


def _load_manifest() -> dict[str, Any]:
    return json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))


def _load_fixture_cases() -> tuple[_FixtureCase, ...]:
    manifest = _load_manifest()
    cases = tuple(
        _FixtureCase(
            schema_version=entry["schema_version"],
            script_path=_FIXTURE_ROOT / entry["path"],
        )
        for entry in manifest["fixtures"]
    )
    declared_versions = tuple(case.schema_version for case in cases)
    if declared_versions != SUPPORTED_SCHEMA_VERSIONS:
        raise AssertionError(
            "Migration property fixtures must cover every supported schema version"
        )
    return cases


_FIXTURE_CASES = _load_fixture_cases()
_FIXTURE_IDS = {
    UUID(_load_manifest()["representative_values"]["experience_id"]),
    UUID(_load_manifest()["representative_values"]["lesson_id"]),
    *(UUID(value) for value in _load_manifest()["representative_values"]["memory_ids"]),
}

_SAFE_TEXT = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)),
    max_size=48,
)
_JSON_KEYS = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)),
    max_size=16,
)
_FINITE_FLOATS = st.floats(
    min_value=-1_000_000.0,
    max_value=1_000_000.0,
    allow_nan=False,
    allow_infinity=False,
    width=64,
)
_UNIT_FLOATS = st.floats(
    min_value=0.0,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
    width=64,
)
_JSON_SCALARS = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2**63), max_value=2**63 - 1),
    _FINITE_FLOATS,
    _SAFE_TEXT,
)
_JSON_VALUES = st.recursive(
    _JSON_SCALARS,
    lambda children: st.one_of(
        st.lists(children, max_size=4),
        st.dictionaries(_JSON_KEYS, children, max_size=4),
    ),
    max_leaves=12,
)
_JSON_MAPPINGS = st.dictionaries(_JSON_KEYS, _JSON_VALUES, max_size=5)
_FIXED_OFFSETS = st.integers(min_value=-(24 * 60 - 1), max_value=24 * 60 - 1).map(
    lambda minutes: timezone(timedelta(minutes=minutes))
)
_AWARE_DATETIMES = st.datetimes(
    min_value=datetime(2000, 1, 1),
    max_value=datetime(2035, 12, 31, 23, 59, 59, 999999),
    timezones=_FIXED_OFFSETS,
)
_NON_FIXTURE_UUIDS = st.uuids(version=4).filter(lambda value: value not in _FIXTURE_IDS)


@st.composite
def _record_bundles(draw: st.DrawFn) -> _RecordBundle:
    ids = draw(
        st.lists(
            _NON_FIXTURE_UUIDS,
            min_size=4,
            max_size=4,
            unique=True,
        )
    )
    experience = ExperienceRecord(
        id=ids[0],
        task=draw(_SAFE_TEXT),
        action=draw(_SAFE_TEXT),
        result=draw(_SAFE_TEXT),
        agent_role=draw(_SAFE_TEXT),
        context=draw(st.one_of(st.none(), _JSON_MAPPINGS)),
        created_at=draw(_AWARE_DATETIMES),
    )
    lesson = Lesson(
        id=ids[1],
        experience_id=experience.id,
        content=draw(_SAFE_TEXT),
        agent_role=draw(_SAFE_TEXT),
        root_cause=draw(st.one_of(st.none(), _SAFE_TEXT)),
        confidence=draw(_UNIT_FLOATS),
        created_at=draw(_AWARE_DATETIMES),
    )

    reinforcement_count = draw(st.integers(min_value=0, max_value=10_000))
    embedded_memory = Memory(
        id=ids[2],
        content=draw(_SAFE_TEXT),
        type=draw(st.sampled_from(tuple(MemoryType))),
        agent_role=draw(_SAFE_TEXT),
        confidence=draw(_UNIT_FLOATS),
        importance=draw(_UNIT_FLOATS),
        source=draw(st.one_of(st.none(), _SAFE_TEXT)),
        metadata=draw(st.one_of(st.none(), _JSON_MAPPINGS)),
        embedding=draw(st.lists(_FINITE_FLOATS, max_size=8)),
        reinforcement_count=reinforcement_count,
        success_count=draw(st.integers(min_value=0, max_value=reinforcement_count)),
        created_at=draw(_AWARE_DATETIMES),
        updated_at=draw(_AWARE_DATETIMES),
        expires_at=draw(st.one_of(st.none(), _AWARE_DATETIMES)),
    )
    unembedded_memory = Memory(
        id=ids[3],
        content=draw(_SAFE_TEXT),
        type=draw(st.sampled_from(tuple(MemoryType))),
        agent_role=draw(_SAFE_TEXT),
        confidence=draw(_UNIT_FLOATS),
        importance=draw(_UNIT_FLOATS),
        source=draw(st.one_of(st.none(), _SAFE_TEXT)),
        metadata=draw(st.one_of(st.none(), _JSON_MAPPINGS)),
        embedding=None,
        reinforcement_count=0,
        success_count=0,
        created_at=draw(_AWARE_DATETIMES),
        updated_at=draw(_AWARE_DATETIMES),
        expires_at=draw(st.one_of(st.none(), _AWARE_DATETIMES)),
    )
    return _RecordBundle(
        experience=experience,
        lesson=lesson,
        memories=(embedded_memory, unembedded_memory),
    )


def _json_text(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _datetime_text(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _materialize_fixture_and_insert_records(
    fixture: _FixtureCase,
    database_path: Path,
    records: _RecordBundle,
) -> None:
    connection = sqlite3.connect(database_path)
    try:
        connection.executescript(fixture.script_path.read_text(encoding="utf-8"))

        experience = records.experience
        if fixture.schema_version == 0:
            connection.execute(
                "INSERT INTO experiences "
                "(id, task, action, result, context, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    str(experience.id),
                    experience.task,
                    experience.action,
                    experience.result,
                    _json_text(experience.context),
                    _datetime_text(experience.created_at),
                ),
            )
        else:
            connection.execute(
                "INSERT INTO experiences "
                "(id, task, action, result, agent_role, context, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    str(experience.id),
                    experience.task,
                    experience.action,
                    experience.result,
                    experience.agent_role,
                    _json_text(experience.context),
                    _datetime_text(experience.created_at),
                ),
            )

        lesson = records.lesson
        if fixture.schema_version == 0:
            connection.execute(
                "INSERT INTO lessons "
                "(id, experience_id, content, confidence, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    str(lesson.id),
                    str(lesson.experience_id),
                    lesson.content,
                    lesson.confidence,
                    _datetime_text(lesson.created_at),
                ),
            )
        else:
            connection.execute(
                "INSERT INTO lessons "
                "(id, experience_id, content, agent_role, root_cause, confidence, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    str(lesson.id),
                    str(lesson.experience_id),
                    lesson.content,
                    lesson.agent_role,
                    lesson.root_cause,
                    lesson.confidence,
                    _datetime_text(lesson.created_at),
                ),
            )

        for memory in records.memories:
            common_values = (
                str(memory.id),
                memory.content,
                memory.type.value,
                memory.confidence,
                memory.importance,
                memory.source,
                _json_text(memory.metadata),
                _json_text(memory.embedding),
                _datetime_text(memory.created_at),
                _datetime_text(memory.updated_at),
                _datetime_text(memory.expires_at),
            )
            if fixture.schema_version == 0:
                connection.execute(
                    "INSERT INTO memories "
                    "(id, content, type, confidence, importance, source, metadata, "
                    "embedding, created_at, updated_at, expires_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    common_values,
                )
                continue

            versioned_values = (
                str(memory.id),
                memory.content,
                memory.type.value,
                memory.agent_role,
                memory.confidence,
                memory.importance,
                memory.source,
                _json_text(memory.metadata),
                _json_text(memory.embedding),
                memory.reinforcement_count,
                memory.success_count,
                _datetime_text(memory.created_at),
                _datetime_text(memory.updated_at),
                _datetime_text(memory.expires_at),
            )
            if fixture.schema_version == 1:
                connection.execute(
                    "INSERT INTO memories "
                    "(id, content, type, agent_role, confidence, importance, source, "
                    "metadata, embedding, reinforcement_count, success_count, "
                    "created_at, updated_at, expires_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    versioned_values,
                )
            else:
                connection.execute(
                    "INSERT INTO memories "
                    "(id, content, type, agent_role, confidence, importance, source, "
                    "metadata, embedding, reinforcement_count, success_count, "
                    "created_at, updated_at, expires_at, embedding_dimension) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        *versioned_values,
                        None if memory.embedding is None else len(memory.embedding),
                    ),
                )
        connection.commit()
    finally:
        connection.close()


def _source_snapshot(
    database_path: Path,
) -> tuple[tuple[str, tuple[str, ...], tuple[tuple[Any, ...], ...]], ...]:
    connection = sqlite3.connect(database_path)
    try:
        snapshot = []
        for table in _TABLES_WITH_SOURCE_RECORDS:
            columns = tuple(
                row[1] for row in connection.execute(f"PRAGMA table_info({table})")
            )
            selected_columns = ", ".join(f'"{column}"' for column in columns)
            rows = tuple(
                connection.execute(
                    f'SELECT {selected_columns} FROM "{table}" ORDER BY id'
                ).fetchall()
            )
            snapshot.append((table, columns, rows))
        return tuple(snapshot)
    finally:
        connection.close()


async def _assert_source_values_preserved(
    connection: aiosqlite.Connection,
    source_snapshot: tuple[
        tuple[str, tuple[str, ...], tuple[tuple[Any, ...], ...]], ...
    ],
) -> None:
    for table, columns, expected_rows in source_snapshot:
        selected_columns = ", ".join(f'"{column}"' for column in columns)
        cursor = await connection.execute(
            f'SELECT {selected_columns} FROM "{table}" ORDER BY id'
        )
        assert tuple(await cursor.fetchall()) == expected_rows


async def _assert_generated_target_values(
    connection: aiosqlite.Connection,
    fixture_version: int,
    records: _RecordBundle,
) -> None:
    experience = records.experience
    cursor = await connection.execute(
        "SELECT id, task, action, result, agent_role, context, created_at "
        "FROM experiences WHERE id = ?",
        (str(experience.id),),
    )
    assert await cursor.fetchone() == (
        str(experience.id),
        experience.task,
        experience.action,
        experience.result,
        experience.agent_role if fixture_version >= 1 else "default",
        _json_text(experience.context),
        _datetime_text(experience.created_at),
    )

    lesson = records.lesson
    cursor = await connection.execute(
        "SELECT id, experience_id, content, agent_role, root_cause, confidence, "
        "created_at FROM lessons WHERE id = ?",
        (str(lesson.id),),
    )
    assert await cursor.fetchone() == (
        str(lesson.id),
        str(lesson.experience_id),
        lesson.content,
        lesson.agent_role if fixture_version >= 1 else "default",
        lesson.root_cause if fixture_version >= 1 else None,
        lesson.confidence,
        _datetime_text(lesson.created_at),
    )

    memory_ids = tuple(str(memory.id) for memory in records.memories)
    cursor = await connection.execute(
        "SELECT id, content, type, agent_role, confidence, importance, source, "
        "metadata, embedding, reinforcement_count, success_count, created_at, "
        "updated_at, expires_at, embedding_dimension FROM memories "
        "WHERE id IN (?, ?) ORDER BY id",
        memory_ids,
    )
    expected_memories = []
    for memory in records.memories:
        expected_memories.append(
            (
                str(memory.id),
                memory.content,
                memory.type.value,
                memory.agent_role if fixture_version >= 1 else "default",
                memory.confidence,
                memory.importance,
                memory.source,
                _json_text(memory.metadata),
                _json_text(memory.embedding),
                memory.reinforcement_count if fixture_version >= 1 else 0,
                memory.success_count if fixture_version >= 1 else 0,
                _datetime_text(memory.created_at),
                _datetime_text(memory.updated_at),
                _datetime_text(memory.expires_at),
                None if memory.embedding is None else len(memory.embedding),
            )
        )
    assert tuple(await cursor.fetchall()) == tuple(
        sorted(expected_memories, key=lambda row: row[0])
    )


async def _database_snapshot(
    connection: aiosqlite.Connection,
) -> _DatabaseSnapshot:
    version_cursor = await connection.execute("PRAGMA user_version")
    version_row = await version_cursor.fetchone()

    structure_cursor = await connection.execute(
        "SELECT type, name, tbl_name, sql FROM sqlite_master "
        "WHERE type IN ('table', 'index', 'trigger') "
        "AND name NOT LIKE 'sqlite_%' ORDER BY type, name"
    )
    structure = tuple(await structure_cursor.fetchall())

    records = []
    for table in _TARGET_TABLES:
        columns_cursor = await connection.execute(f"PRAGMA table_info({table})")
        column_info = await columns_cursor.fetchall()
        columns = tuple(str(row[1]) for row in column_info)
        primary_key_columns = tuple(
            str(row[1]) for row in sorted(column_info, key=lambda row: row[5]) if row[5]
        )
        selected_columns = ", ".join(f'"{column}"' for column in columns)
        order_columns = primary_key_columns or columns
        order_by = ", ".join(f'"{column}"' for column in order_columns)
        rows_cursor = await connection.execute(
            f'SELECT {selected_columns} FROM "{table}" ORDER BY {order_by}'
        )
        records.append((table, columns, tuple(await rows_cursor.fetchall())))

    return _DatabaseSnapshot(
        schema_version=int(version_row[0]),
        structure=structure,
        records=tuple(records),
    )


# Feature: open-source-project-improvements, Property 13: Migration is preserving and idempotent
@pytest.mark.parametrize(
    "fixture",
    _FIXTURE_CASES,
    ids=lambda fixture: f"schema-v{fixture.schema_version}",
)
@pytest.mark.asyncio
@settings(max_examples=STANDARD_PBT_MIN_EXAMPLES, deadline=None)
@given(records=_record_bundles(), repeat_count=st.integers(min_value=1, max_value=3))
async def test_migration_is_preserving_and_idempotent(
    fixture: _FixtureCase,
    records: _RecordBundle,
    repeat_count: int,
) -> None:
    """**Validates: Requirements 3.6, 3.7, 7.6**"""
    with TemporaryDirectory() as directory:
        database_path = Path(directory) / f"schema-v{fixture.schema_version}.db"
        _materialize_fixture_and_insert_records(fixture, database_path, records)
        before_migration = _source_snapshot(database_path)

        connection = await aiosqlite.connect(database_path)
        migrator = SchemaMigrator(
            SQLiteTransactionManager(lambda: connection, asyncio.Lock())
        )
        try:
            await migrator.migrate(connection)

            await _assert_source_values_preserved(connection, before_migration)
            await _assert_generated_target_values(
                connection,
                fixture.schema_version,
                records,
            )
            after_first_migration = await _database_snapshot(connection)
            assert after_first_migration.schema_version == CURRENT_SCHEMA_VERSION

            for _ in range(repeat_count):
                await migrator.migrate(connection)
                assert await _database_snapshot(connection) == after_first_migration
        finally:
            await connection.close()
