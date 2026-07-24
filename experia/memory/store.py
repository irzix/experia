import json
from datetime import datetime
from typing import List, Optional
from uuid import UUID

import aiosqlite

from experia.core.exceptions import StorageError
from experia.experience.models import ExperienceRecord, Lesson
from experia.memory.models import Memory, MemoryType


class SQLiteStore:
    """
    An asynchronous SQLite-based local storage backend for Experia.
    Implements the MemoryStore Protocol.
    """

    def __init__(self, db_path: str = "experia.db"):
        self.db_path = db_path

    async def initialize(self) -> None:
        """Initializes the database schema if it doesn't exist."""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                # Experiences Table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS experiences (
                        id TEXT PRIMARY KEY,
                        task TEXT NOT NULL,
                        action TEXT NOT NULL,
                        result TEXT NOT NULL,
                        agent_role TEXT NOT NULL DEFAULT 'default',
                        context TEXT,
                        created_at TEXT NOT NULL
                    )
                """)

                # Lessons Table
                await conn.execute("""
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
                """)

                # Check for existing table and add root_cause/agent_role if missing (migration)
                cursor = await conn.execute("PRAGMA table_info(lessons)")
                columns = [row[1] for row in await cursor.fetchall()]
                if "root_cause" not in columns:
                    await conn.execute("ALTER TABLE lessons ADD COLUMN root_cause TEXT")
                if "agent_role" not in columns:
                    await conn.execute(
                        "ALTER TABLE lessons ADD COLUMN agent_role TEXT DEFAULT 'default'"
                    )

                cursor = await conn.execute("PRAGMA table_info(experiences)")
                exp_columns = [row[1] for row in await cursor.fetchall()]
                if "agent_role" not in exp_columns:
                    await conn.execute(
                        "ALTER TABLE experiences ADD COLUMN agent_role TEXT DEFAULT 'default'"
                    )

                # Memories Table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS memories (
                        id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        type TEXT NOT NULL,
                        agent_role TEXT NOT NULL DEFAULT 'default',
                        confidence REAL NOT NULL,
                        importance REAL NOT NULL,
                        source TEXT,
                        metadata TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        expires_at TEXT
                    )
                """)

                cursor = await conn.execute("PRAGMA table_info(memories)")
                mem_columns = [row[1] for row in await cursor.fetchall()]
                if "agent_role" not in mem_columns:
                    await conn.execute(
                        "ALTER TABLE memories ADD COLUMN agent_role TEXT DEFAULT 'default'"
                    )

                await conn.commit()
        except Exception as e:
            raise StorageError(f"Failed to initialize SQLite database: {e}")

    # --- Experience Methods ---

    async def save_experience(self, experience: ExperienceRecord) -> None:
        """Saves a raw experience record asynchronously."""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute(
                    """
                    INSERT INTO experiences (id, task, action, result, agent_role, context, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(experience.id),
                        experience.task,
                        experience.action,
                        experience.result,
                        experience.agent_role,
                        json.dumps(experience.context) if experience.context else None,
                        experience.created_at.isoformat(),
                    ),
                )
                await conn.commit()
        except Exception as e:
            raise StorageError(f"Failed to save experience: {e}")

    async def get_experience(self, experience_id: UUID) -> Optional[ExperienceRecord]:
        """Retrieves an experience by its ID asynchronously."""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                cursor = await conn.execute(
                    "SELECT * FROM experiences WHERE id = ?", (str(experience_id),)
                )
                row = await cursor.fetchone()

                if row:
                    return ExperienceRecord(
                        id=UUID(row[0]),
                        task=row[1],
                        action=row[2],
                        result=row[3],
                        context=json.loads(row[4]) if row[4] else {},
                        created_at=datetime.fromisoformat(row[5]),
                    )
            return None
        except Exception as e:
            raise StorageError(f"Failed to retrieve experience: {e}")

    async def get_recent_experiences(self, limit: int = 50) -> List[ExperienceRecord]:
        """Retrieves the most recent experiences asynchronously."""
        experiences = []
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                cursor = await conn.execute(
                    "SELECT * FROM experiences ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                )
                rows = await cursor.fetchall()

                for row in rows:
                    experiences.append(
                        ExperienceRecord(
                            id=UUID(row[0]),
                            task=row[1],
                            action=row[2],
                            result=row[3],
                            agent_role=row[4],
                            context=json.loads(row[5]) if row[5] else {},
                            created_at=datetime.fromisoformat(row[6]),
                        )
                    )
            return experiences
        except Exception as e:
            raise StorageError(f"Failed to retrieve recent experiences: {e}")

    # --- Lesson Methods ---

    async def save_lesson(self, lesson: Lesson) -> None:
        """Saves an extracted lesson asynchronously."""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute(
                    """
                    INSERT INTO lessons (id, experience_id, content, agent_role, root_cause, confidence, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(lesson.id),
                        str(lesson.experience_id),
                        lesson.content,
                        lesson.agent_role,
                        lesson.root_cause,
                        lesson.confidence,
                        lesson.created_at.isoformat(),
                    ),
                )
                await conn.commit()
        except Exception as e:
            raise StorageError(f"Failed to save lesson: {e}")

    # --- Memory Methods ---

    async def save_memory(self, memory: Memory) -> None:
        """Saves a memory object asynchronously."""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute(
                    """
                    INSERT OR REPLACE INTO memories 
                    (id, content, type, agent_role, confidence, importance, source, metadata, 
                    created_at, updated_at, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(memory.id),
                        memory.content,
                        memory.type.value,
                        memory.agent_role,
                        memory.confidence,
                        memory.importance,
                        memory.source,
                        json.dumps(memory.metadata) if memory.metadata else None,
                        memory.created_at.isoformat(),
                        memory.updated_at.isoformat(),
                        memory.expires_at.isoformat() if memory.expires_at else None,
                    ),
                )
                await conn.commit()
        except Exception as e:
            raise StorageError(f"Failed to save memory: {e}")

    async def search_memories(
        self,
        query: str = "",
        memory_type: Optional[MemoryType] = None,
        agent_role: Optional[str] = None,
        limit: int = 10,
    ) -> List[Memory]:
        """
        Asynchronous simple text-based search for memories.
        """
        memories = []
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                sql = "SELECT * FROM memories WHERE 1=1"
                params = []

                if query:
                    sql += " AND content LIKE ?"
                    params.append(f"%{query}%")

                if memory_type:
                    sql += " AND type = ?"
                    params.append(memory_type.value)

                if agent_role:
                    # Allow either the specific role or the 'global' shared role (e.g. STRATEGY)
                    sql += " AND (agent_role = ? OR type = ?)"
                    params.extend([agent_role, MemoryType.STRATEGY.value])

                sql += " ORDER BY importance DESC, confidence DESC LIMIT ?"
                params.append(limit)

                cursor = await conn.execute(sql, tuple(params))
                rows = await cursor.fetchall()

                for row in rows:
                    memories.append(
                        Memory(
                            id=UUID(row[0]),
                            content=row[1],
                            type=MemoryType(row[2]),
                            agent_role=row[3],
                            confidence=row[4],
                            importance=row[5],
                            source=row[6],
                            metadata=json.loads(row[7]) if row[7] else {},
                            created_at=datetime.fromisoformat(row[8]),
                            updated_at=datetime.fromisoformat(row[9]),
                            expires_at=datetime.fromisoformat(row[10])
                            if row[10]
                            else None,
                        )
                    )
            return memories
        except Exception as e:
            raise StorageError(f"Failed to search memories: {e}")
