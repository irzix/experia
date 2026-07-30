"""Repository fixture coverage for every supported SQLite schema version."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest

from experia.experience.models import Lesson
from experia.memory.migrations import (
    CURRENT_SCHEMA_VERSION,
    MIN_SUPPORTED_SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
)
from experia.memory.models import MemoryType
from experia.memory.serialization import StorageSerializer
from experia.memory.store import SQLiteStore

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "sqlite"
MANIFEST_PATH = FIXTURE_ROOT / "schema-support.json"


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _materialize_fixture(script_name: str, database_path: Path) -> None:
    connection = sqlite3.connect(database_path)
    try:
        connection.executescript(
            (FIXTURE_ROOT / script_name).read_text(encoding="utf-8")
        )
    finally:
        connection.close()


def test_schema_support_manifest_covers_every_declared_version() -> None:
    manifest = _manifest()
    window = manifest["support_window"]
    fixtures = manifest["fixtures"]

    assert window["minimum_schema_version"] == MIN_SUPPORTED_SCHEMA_VERSION
    assert window["maximum_schema_version"] == CURRENT_SCHEMA_VERSION
    assert window["current_schema_version"] == CURRENT_SCHEMA_VERSION
    assert tuple(window["supported_schema_versions"]) == SUPPORTED_SCHEMA_VERSIONS
    assert tuple(entry["schema_version"] for entry in fixtures) == (
        SUPPORTED_SCHEMA_VERSIONS
    )

    declared_paths = {entry["path"] for entry in fixtures}
    repository_paths = {path.name for path in FIXTURE_ROOT.glob("schema-v*.sql")}
    assert declared_paths == repository_paths
    for entry in fixtures:
        fixture_bytes = (FIXTURE_ROOT / entry["path"]).read_bytes()
        assert hashlib.sha256(fixture_bytes).hexdigest() == entry["sha256"]


def test_each_fixture_contains_representative_typed_storage_values(tmp_path) -> None:
    manifest = _manifest()
    expected = manifest["representative_values"]

    for entry in manifest["fixtures"]:
        database_path = tmp_path / f"schema-v{entry['schema_version']}.db"
        _materialize_fixture(entry["path"], database_path)
        connection = sqlite3.connect(database_path)
        try:
            assert connection.execute("PRAGMA user_version").fetchone() == (
                entry["schema_version"],
            )

            experience_row = connection.execute(
                "SELECT id, context, created_at FROM experiences"
            ).fetchone()
            assert experience_row is not None
            assert UUID(experience_row[0]) == UUID(expected["experience_id"])
            assert json.loads(experience_row[1]) == expected["context"]
            assert datetime.fromisoformat(experience_row[2]).utcoffset() == timedelta(
                hours=5, minutes=30
            )

            lesson_row = connection.execute(
                "SELECT id, experience_id, created_at FROM lessons"
            ).fetchone()
            assert lesson_row is not None
            assert UUID(lesson_row[0]) == UUID(expected["lesson_id"])
            assert UUID(lesson_row[1]) == UUID(expected["experience_id"])
            assert datetime.fromisoformat(lesson_row[2]).utcoffset() == timedelta(
                hours=-4
            )

            memory_rows = connection.execute(
                "SELECT id, type, metadata, embedding, created_at, updated_at, "
                "expires_at FROM memories ORDER BY id"
            ).fetchall()
            assert [row[0] for row in memory_rows] == expected["memory_ids"]
            assert [MemoryType(row[1]).value for row in memory_rows] == expected[
                "memory_types"
            ]
            assert [json.loads(row[3]) for row in memory_rows] == expected["embeddings"]
            assert json.loads(memory_rows[0][2])["audit"]["verified"] is True
            assert json.loads(memory_rows[1][2])["constraints"]["retry"]["backoff"] == [
                0.25,
                0.5,
            ]
            assert datetime.fromisoformat(memory_rows[0][4]).utcoffset() == timedelta(
                hours=9
            )
            assert datetime.fromisoformat(memory_rows[0][6]).utcoffset() == timedelta(
                hours=-7
            )
            assert datetime.fromisoformat(memory_rows[1][4]).utcoffset() == timedelta(
                hours=14
            )
            assert datetime.fromisoformat(memory_rows[1][5]).utcoffset() == timedelta(
                hours=-4
            )
        finally:
            connection.close()


@pytest.mark.asyncio
async def test_each_supported_fixture_migrates_and_decodes_typed_values(
    tmp_path,
) -> None:
    manifest = _manifest()
    representative = manifest["representative_values"]
    serializer = StorageSerializer()

    for entry in manifest["fixtures"]:
        database_path = tmp_path / f"migrated-v{entry['schema_version']}.db"
        _materialize_fixture(entry["path"], database_path)
        store = SQLiteStore(str(database_path))
        await store.initialize()
        try:
            connection = store._require_conn()
            version_row = await (
                await connection.execute("PRAGMA user_version")
            ).fetchone()
            assert version_row == (CURRENT_SCHEMA_VERSION,)

            experience = await store.get_experience(
                UUID(representative["experience_id"])
            )
            assert experience is not None
            assert isinstance(experience.id, UUID)
            assert experience.context == representative["context"]
            assert (
                experience.agent_role
                == entry["expected_after_migration"]["experience_agent_role"]
            )
            assert experience.created_at.utcoffset() == timedelta(hours=5, minutes=30)

            lesson_cursor = await connection.execute(
                "SELECT id, experience_id, content, agent_role, root_cause, "
                "confidence, created_at FROM lessons"
            )
            lesson = serializer.decode_lesson(await lesson_cursor.fetchone())
            assert isinstance(lesson, Lesson)
            assert isinstance(lesson.id, UUID)
            assert (
                lesson.agent_role
                == entry["expected_after_migration"]["lesson_agent_role"]
            )
            assert (
                lesson.root_cause
                == entry["expected_after_migration"]["lesson_root_cause"]
            )
            assert lesson.created_at.utcoffset() == timedelta(hours=-4)

            memories = [
                await store.get_memory(UUID(memory_id))
                for memory_id in representative["memory_ids"]
            ]
            assert all(memory is not None for memory in memories)
            assert [memory.type for memory in memories if memory is not None] == [
                MemoryType(value) for value in representative["memory_types"]
            ]
            assert [memory.embedding for memory in memories if memory is not None] == (
                representative["embeddings"]
            )
            assert [memory.agent_role for memory in memories if memory is not None] == (
                entry["expected_after_migration"]["memory_agent_roles"]
            )
            assert [
                memory.reinforcement_count for memory in memories if memory is not None
            ] == entry["expected_after_migration"]["reinforcement_counts"]
            assert [
                memory.success_count for memory in memories if memory is not None
            ] == (entry["expected_after_migration"]["success_counts"])

            dimension_cursor = await connection.execute(
                "SELECT embedding_dimension FROM memories ORDER BY id"
            )
            assert [row[0] for row in await dimension_cursor.fetchall()] == [4, 3]
        finally:
            await store.close()
