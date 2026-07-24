import asyncio
import json
from datetime import datetime, timezone
from typing import List, Optional
from uuid import UUID

import aiosqlite

from experia.core.exceptions import StorageError
from experia.core.logging import logger
from experia.experience.models import ExperienceRecord, Lesson
from experia.memory.embeddings import cosine_similarity
from experia.memory.models import Memory, MemoryType


class SQLiteStore:
    """
    An asynchronous SQLite-based local storage backend for Experia.
    Implements the MemoryStore Protocol.

    A single connection is opened in ``initialize()`` and reused for the
    lifetime of the store (WAL mode allows concurrent reads while a write
    lock serialises writers). Call ``close()`` when done.
    """

    def __init__(self, db_path: str = "experia.db"):
        self.db_path = db_path
        self._conn: Optional[aiosqlite.Connection] = None
        self._write_lock = asyncio.Lock()

    def _require_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise StorageError("Store not initialized. Call `await store.initialize()`.")
        return self._conn

    async def initialize(self) -> None:
        """Opens the connection and initialises the schema and indexes."""
        try:
            if self._conn is None:
                self._conn = await aiosqlite.connect(self.db_path)
                # WAL improves read/write concurrency but is unsupported on some
                # (e.g. networked) filesystems — enable it best-effort.
                try:
                    await self._conn.execute("PRAGMA journal_mode=WAL")
                except Exception as e:
                    logger.debug(f"WAL mode unavailable, using default journal: {e}")
                await self._conn.execute("PRAGMA foreign_keys=ON")

            conn = self._conn

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
                    embedding TEXT,
                    reinforcement_count INTEGER NOT NULL DEFAULT 0,
                    success_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT
                )
            """)

            await self._run_migrations(conn)
            await self._create_indexes(conn)
            await conn.commit()
        except Exception as e:
            raise StorageError(f"Failed to initialize SQLite database: {e}")

    async def _run_migrations(self, conn: aiosqlite.Connection) -> None:
        """Additive, idempotent column migrations for existing databases."""
        migrations = {
            "lessons": [
                ("root_cause", "ALTER TABLE lessons ADD COLUMN root_cause TEXT"),
                (
                    "agent_role",
                    "ALTER TABLE lessons ADD COLUMN agent_role TEXT DEFAULT 'default'",
                ),
            ],
            "experiences": [
                (
                    "agent_role",
                    "ALTER TABLE experiences ADD COLUMN agent_role TEXT DEFAULT 'default'",
                ),
            ],
            "memories": [
                (
                    "agent_role",
                    "ALTER TABLE memories ADD COLUMN agent_role TEXT DEFAULT 'default'",
                ),
                ("embedding", "ALTER TABLE memories ADD COLUMN embedding TEXT"),
                (
                    "reinforcement_count",
                    "ALTER TABLE memories ADD COLUMN reinforcement_count INTEGER NOT NULL DEFAULT 0",
                ),
                (
                    "success_count",
                    "ALTER TABLE memories ADD COLUMN success_count INTEGER NOT NULL DEFAULT 0",
                ),
            ],
        }
        for table, cols in migrations.items():
            cursor = await conn.execute(f"PRAGMA table_info({table})")
            existing = {row[1] for row in await cursor.fetchall()}
            for column, ddl in cols:
                if column not in existing:
                    await conn.execute(ddl)

    async def _create_indexes(self, conn: aiosqlite.Connection) -> None:
        for ddl in (
            "CREATE INDEX IF NOT EXISTS idx_exp_created ON experiences(created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_lessons_exp ON lessons(experience_id)",
            "CREATE INDEX IF NOT EXISTS idx_mem_role_type ON memories(agent_role, type)",
            "CREATE INDEX IF NOT EXISTS idx_mem_rank ON memories(importance DESC, confidence DESC)",
            "CREATE INDEX IF NOT EXISTS idx_mem_expires ON memories(expires_at)",
        ):
            await conn.execute(ddl)

    async def close(self) -> None:
        """Closes the underlying connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    # --- Experience Methods ---

    async def save_experience(self, experience: ExperienceRecord) -> None:
        conn = self._require_conn()
        try:
            async with self._write_lock:
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
        conn = self._require_conn()
        try:
            cursor = await conn.execute(
                "SELECT id, task, action, result, agent_role, context, created_at "
                "FROM experiences WHERE id = ?",
                (str(experience_id),),
            )
            row = await cursor.fetchone()
            return self._row_to_experience(row) if row else None
        except Exception as e:
            raise StorageError(f"Failed to retrieve experience: {e}")

    async def get_recent_experiences(self, limit: int = 50) -> List[ExperienceRecord]:
        conn = self._require_conn()
        try:
            cursor = await conn.execute(
                "SELECT id, task, action, result, agent_role, context, created_at "
                "FROM experiences ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
            rows = await cursor.fetchall()
            return [self._row_to_experience(row) for row in rows]
        except Exception as e:
            raise StorageError(f"Failed to retrieve recent experiences: {e}")

    @staticmethod
    def _row_to_experience(row) -> ExperienceRecord:
        return ExperienceRecord(
            id=UUID(row[0]),
            task=row[1],
            action=row[2],
            result=row[3],
            agent_role=row[4],
            context=json.loads(row[5]) if row[5] else {},
            created_at=datetime.fromisoformat(row[6]),
        )

    # --- Lesson Methods ---

    async def save_lesson(self, lesson: Lesson) -> None:
        conn = self._require_conn()
        try:
            async with self._write_lock:
                await self._insert_lesson(conn, lesson)
                await conn.commit()
        except Exception as e:
            raise StorageError(f"Failed to save lesson: {e}")

    async def save_lesson_and_memory(self, lesson: Lesson, memory: Memory) -> None:
        """Persist a lesson and its derived memory atomically in one transaction."""
        conn = self._require_conn()
        try:
            async with self._write_lock:
                await self._insert_lesson(conn, lesson)
                await self._upsert_memory(conn, memory)
                await conn.commit()
        except Exception as e:
            try:
                await conn.rollback()
            except Exception:
                pass
            raise StorageError(f"Failed to save lesson and memory: {e}")

    @staticmethod
    async def _insert_lesson(conn: aiosqlite.Connection, lesson: Lesson) -> None:
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

    # --- Memory Methods ---

    async def save_memory(self, memory: Memory) -> None:
        conn = self._require_conn()
        try:
            async with self._write_lock:
                await self._upsert_memory(conn, memory)
                await conn.commit()
        except Exception as e:
            raise StorageError(f"Failed to save memory: {e}")

    @staticmethod
    async def _upsert_memory(conn: aiosqlite.Connection, memory: Memory) -> None:
        await conn.execute(
            """
            INSERT OR REPLACE INTO memories
            (id, content, type, agent_role, confidence, importance, source, metadata,
             embedding, reinforcement_count, success_count, created_at, updated_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                json.dumps(memory.embedding) if memory.embedding else None,
                memory.reinforcement_count,
                memory.success_count,
                memory.created_at.isoformat(),
                memory.updated_at.isoformat(),
                memory.expires_at.isoformat() if memory.expires_at else None,
            ),
        )

    async def get_memory(self, memory_id: UUID) -> Optional[Memory]:
        conn = self._require_conn()
        try:
            cursor = await conn.execute(
                "SELECT * FROM memories WHERE id = ?", (str(memory_id),)
            )
            row = await cursor.fetchone()
            return self._row_to_memory(row) if row else None
        except Exception as e:
            raise StorageError(f"Failed to retrieve memory: {e}")

    async def search_memories(
        self,
        query: str = "",
        memory_type: Optional[MemoryType] = None,
        agent_role: Optional[str] = None,
        limit: int = 10,
        query_embedding: Optional[List[float]] = None,
        include_expired: bool = False,
    ) -> List[Memory]:
        """
        Retrieve memories. When ``query_embedding`` is supplied, candidates are
        ranked by cosine similarity blended with importance (semantic search).
        Otherwise falls back to keyword (LIKE) matching.
        """
        conn = self._require_conn()
        try:
            sql = "SELECT * FROM memories WHERE 1=1"
            params: list = []

            if not include_expired:
                sql += " AND (expires_at IS NULL OR expires_at > ?)"
                params.append(datetime.now(timezone.utc).isoformat())

            if memory_type:
                sql += " AND type = ?"
                params.append(memory_type.value)

            if agent_role:
                # Allow the specific role or the globally-shared STRATEGY memories.
                sql += " AND (agent_role = ? OR type = ?)"
                params.extend([agent_role, MemoryType.STRATEGY.value])

            if query_embedding:
                # Semantic path: pull a candidate pool, rank in Python.
                cursor = await conn.execute(sql, tuple(params))
                rows = await cursor.fetchall()
                memories = [self._row_to_memory(row) for row in rows]
                return self._rank_by_similarity(memories, query_embedding, limit)

            # Keyword path.
            if query:
                sql += " AND content LIKE ?"
                params.append(f"%{query}%")

            sql += " ORDER BY importance DESC, confidence DESC LIMIT ?"
            params.append(limit)

            cursor = await conn.execute(sql, tuple(params))
            rows = await cursor.fetchall()
            return [self._row_to_memory(row) for row in rows]
        except Exception as e:
            raise StorageError(f"Failed to search memories: {e}")

    @staticmethod
    def _rank_by_similarity(
        memories: List[Memory], query_embedding: List[float], limit: int
    ) -> List[Memory]:
        scored = []
        for mem in memories:
            if mem.embedding:
                sim = cosine_similarity(query_embedding, mem.embedding)
            else:
                sim = 0.0
            # Blend semantic relevance with the memory's own importance.
            score = 0.75 * sim + 0.25 * mem.importance
            scored.append((score, mem))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [mem for _, mem in scored[:limit]]

    async def find_similar_memory(
        self,
        embedding: List[float],
        memory_type: Optional[MemoryType] = None,
        agent_role: Optional[str] = None,
        threshold: float = 0.95,
    ) -> Optional[Memory]:
        """Return an existing near-duplicate memory above ``threshold``, if any."""
        if not embedding:
            return None
        candidates = await self.search_memories(
            memory_type=memory_type,
            agent_role=agent_role,
            query_embedding=embedding,
            limit=1,
        )
        if not candidates:
            return None
        best = candidates[0]
        if best.embedding and cosine_similarity(embedding, best.embedding) >= threshold:
            return best
        return None

    async def update_memory_feedback(
        self, memory_id: UUID, success: bool, alpha: float = 0.2
    ) -> Optional[Memory]:
        """
        Reinforce or weaken a memory based on a real outcome. Confidence moves
        toward 1.0 on success and 0.0 on failure via an exponential moving
        average, and reinforcement counters are incremented.
        """
        conn = self._require_conn()
        memory = await self.get_memory(memory_id)
        if memory is None:
            return None

        target = 1.0 if success else 0.0
        memory.confidence = round(
            min(1.0, max(0.0, memory.confidence + alpha * (target - memory.confidence))),
            4,
        )
        memory.reinforcement_count += 1
        if success:
            memory.success_count += 1
        memory.updated_at = datetime.now(timezone.utc)

        try:
            async with self._write_lock:
                await self._upsert_memory(conn, memory)
                await conn.commit()
        except Exception as e:
            raise StorageError(f"Failed to update memory feedback: {e}")
        return memory

    async def prune_expired(self) -> int:
        """Delete expired memories. Returns the number removed."""
        conn = self._require_conn()
        try:
            async with self._write_lock:
                cursor = await conn.execute(
                    "DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at <= ?",
                    (datetime.now(timezone.utc).isoformat(),),
                )
                await conn.commit()
                removed = cursor.rowcount or 0
            if removed:
                logger.info(f"Pruned {removed} expired memories.")
            return removed
        except Exception as e:
            raise StorageError(f"Failed to prune expired memories: {e}")

    @staticmethod
    def _row_to_memory(row) -> Memory:
        # Column order matches CREATE TABLE memories.
        return Memory(
            id=UUID(row[0]),
            content=row[1],
            type=MemoryType(row[2]),
            agent_role=row[3],
            confidence=row[4],
            importance=row[5],
            source=row[6],
            metadata=json.loads(row[7]) if row[7] else {},
            embedding=json.loads(row[8]) if row[8] else None,
            reinforcement_count=row[9] or 0,
            success_count=row[10] or 0,
            created_at=datetime.fromisoformat(row[11]),
            updated_at=datetime.fromisoformat(row[12]),
            expires_at=datetime.fromisoformat(row[13]) if row[13] else None,
        )
