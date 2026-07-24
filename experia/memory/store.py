import json
import sqlite3
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from experia.experience.models import ExperienceRecord, Lesson
from experia.memory.models import Memory, MemoryType


class SQLiteStore:
    """
    A simple SQLite-based local storage for Experia.
    Ideal for the default Local Mode without external infrastructure dependencies.
    """

    def __init__(self, db_path: str = "experia.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initializes the database schema if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Experiences Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiences (
                    id TEXT PRIMARY KEY,
                    task TEXT NOT NULL,
                    action TEXT NOT NULL,
                    result TEXT NOT NULL,
                    context TEXT,
                    created_at TEXT NOT NULL
                )
            """)

            # Lessons Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS lessons (
                    id TEXT PRIMARY KEY,
                    experience_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (experience_id) REFERENCES experiences (id)
                )
            """)

            # Memories Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    importance REAL NOT NULL,
                    source TEXT,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT
                )
            """)
            conn.commit()

    # --- Experience Methods ---

    def save_experience(self, experience: ExperienceRecord) -> None:
        """Saves a raw experience record."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO experiences (id, task, action, result, context, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    str(experience.id),
                    experience.task,
                    experience.action,
                    experience.result,
                    json.dumps(experience.context) if experience.context else None,
                    experience.created_at.isoformat(),
                ),
            )
            conn.commit()

    def get_experience(self, experience_id: UUID) -> Optional[ExperienceRecord]:
        """Retrieves an experience by its ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM experiences WHERE id = ?", (str(experience_id),)
            )
            row = cursor.fetchone()

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

    # --- Lesson Methods ---

    def save_lesson(self, lesson: Lesson) -> None:
        """Saves an extracted lesson."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO lessons (id, experience_id, content, confidence, created_at)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    str(lesson.id),
                    str(lesson.experience_id),
                    lesson.content,
                    lesson.confidence,
                    lesson.created_at.isoformat(),
                ),
            )
            conn.commit()

    # --- Memory Methods ---

    def save_memory(self, memory: Memory) -> None:
        """Saves a memory object."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO memories 
                (id, content, type, confidence, importance, source, metadata, 
                created_at, updated_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    str(memory.id),
                    memory.content,
                    memory.type.value,
                    memory.confidence,
                    memory.importance,
                    memory.source,
                    json.dumps(memory.metadata) if memory.metadata else None,
                    memory.created_at.isoformat(),
                    memory.updated_at.isoformat(),
                    memory.expires_at.isoformat() if memory.expires_at else None,
                ),
            )
            conn.commit()

    def search_memories(
        self,
        query: str = "",
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
    ) -> List[Memory]:
        """
        Simple text-based search for memories.
        In the future (v0.4), this will be backed by a Vector DB.
        """
        memories = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            sql = "SELECT * FROM memories WHERE 1=1"
            params = []

            if query:
                sql += " AND content LIKE ?"
                params.append(f"%{query}%")

            if memory_type:
                sql += " AND type = ?"
                params.append(memory_type.value)

            sql += " ORDER BY importance DESC, confidence DESC LIMIT ?"
            params.append(limit)

            cursor.execute(sql, tuple(params))
            rows = cursor.fetchall()

            for row in rows:
                memories.append(
                    Memory(
                        id=UUID(row[0]),
                        content=row[1],
                        type=MemoryType(row[2]),
                        confidence=row[3],
                        importance=row[4],
                        source=row[5],
                        metadata=json.loads(row[6]) if row[6] else {},
                        created_at=datetime.fromisoformat(row[7]),
                        updated_at=datetime.fromisoformat(row[8]),
                        expires_at=datetime.fromisoformat(row[9]) if row[9] else None,
                    )
                )
        return memories
