import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator, List, Optional
from uuid import UUID

import aiosqlite

from experia.core.exceptions import StorageError
from experia.core.logging import logger
from experia.experience.models import ExperienceRecord, Lesson
from experia.memory.embeddings import cosine_similarity
from experia.memory.migrations import SchemaMigrator
from experia.memory.models import Memory, MemoryType
from experia.memory.retrieval import (
    RetrievalEngine,
    RetrievalQuery,
    rank_memories,
)
from experia.memory.serialization import (
    EncodedLesson,
    EncodedMemory,
    StorageSerializer,
)
from experia.memory.transactions import SQLiteTransactionManager
from experia.memory.vector_index import (
    VectorCandidateIndex,
    VectorIndexRebuildResult,
)

_MEMORY_COLUMNS = (
    "id, content, type, agent_role, confidence, importance, source, metadata, "
    "embedding, reinforcement_count, success_count, created_at, updated_at, "
    "expires_at"
)


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
        self._lifecycle_lock = asyncio.Lock()
        self._operations_drained = asyncio.Event()
        self._operations_drained.set()
        self._active_operations = 0
        self._lifecycle_state = "open"
        self._close_task: Optional[asyncio.Task[None]] = None
        self._serializer = StorageSerializer()
        self._transactions = SQLiteTransactionManager(
            self._require_conn, self._write_lock
        )
        self._vector_index = VectorCandidateIndex(
            self._require_conn,
            self._transactions,
        )
        self._retrieval_engine = RetrievalEngine(
            self._require_conn,
            self._serializer,
            self._vector_index,
        )
        self._migrator = SchemaMigrator(self._transactions)

    def _require_conn(self) -> aiosqlite.Connection:
        if self._lifecycle_state == "closed":
            raise StorageError(
                "Store is closed.",
                operation="lifecycle",
            )
        if self._conn is None:
            raise StorageError(
                "Store not initialized. Call `await store.initialize()`.",
            )
        return self._conn

    @asynccontextmanager
    async def _operation(self) -> AsyncIterator[None]:
        """Lease the connection lifecycle for one accepted public operation."""
        async with self._lifecycle_lock:
            if self._lifecycle_state != "open":
                raise StorageError(
                    "Store is closing or closed.",
                    operation="lifecycle",
                )
            self._active_operations += 1
            self._operations_drained.clear()

        try:
            yield
        finally:
            async with self._lifecycle_lock:
                self._active_operations -= 1
                if self._active_operations == 0:
                    self._operations_drained.set()

    async def initialize(self) -> None:
        """Open the connection and migrate its schema to the current version."""
        async with self._operation():
            try:
                if self._conn is None:
                    self._conn = await aiosqlite.connect(self.db_path)
                    # WAL improves read/write concurrency but is unsupported on some
                    # (e.g. networked) filesystems — enable it best-effort.
                    try:
                        await self._conn.execute("PRAGMA journal_mode=WAL")
                    except Exception as e:
                        logger.debug(
                            f"WAL mode unavailable, using default journal: {e}"
                        )
                    await self._conn.execute("PRAGMA foreign_keys=ON")

                await self._migrator.migrate(self._require_conn())
            except StorageError:
                raise
            except Exception as e:
                raise StorageError(
                    "Failed to initialize SQLite database.",
                    operation="initialize",
                    table="schema",
                ) from e

    async def close(self) -> None:
        """Close once, awaiting the same completion for every concurrent caller."""
        close_task = await self._coordinate_close()
        if close_task is not None:
            await asyncio.shield(close_task)

    async def _coordinate_close(self) -> Optional[asyncio.Task[None]]:
        async with self._lifecycle_lock:
            if self._lifecycle_state == "closed":
                return None
            if self._close_task is None:
                self._lifecycle_state = "closing"
                self._close_task = asyncio.create_task(self._close_when_drained())
            return self._close_task

    async def _close_when_drained(self) -> None:
        try:
            await self._operations_drained.wait()
            async with self._write_lock:
                conn = self._conn
                if conn is not None:
                    await conn.close()
        except BaseException:
            async with self._lifecycle_lock:
                if self._close_task is asyncio.current_task():
                    self._lifecycle_state = "open"
                    self._close_task = None
            raise

        async with self._lifecycle_lock:
            self._conn = None
            self._lifecycle_state = "closed"

    # --- Experience Methods ---

    async def save_experience(self, experience: ExperienceRecord) -> None:
        async with self._operation():
            encoded = self._serializer.encode_experience(experience)
            async with self._transactions.write(
                operation="save",
                table="experiences",
                record_ids=(encoded.id,),
            ) as conn:
                await conn.execute(
                    """
                    INSERT INTO experiences (id, task, action, result, agent_role, context, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    encoded.values(),
                )

    async def get_experience(self, experience_id: UUID) -> Optional[ExperienceRecord]:
        async with self._operation():
            conn = self._require_conn()
            try:
                cursor = await conn.execute(
                    "SELECT id, task, action, result, agent_role, context, created_at "
                    "FROM experiences WHERE id = ?",
                    (str(experience_id),),
                )
                row = await cursor.fetchone()
                return self._serializer.decode_experience(row) if row else None
            except StorageError:
                raise
            except Exception as e:
                raise StorageError(
                    "Failed to retrieve experience.",
                    operation="retrieve",
                    table="experiences",
                    record_ids=(experience_id,),
                ) from e

    async def get_recent_experiences(self, limit: int = 50) -> List[ExperienceRecord]:
        async with self._operation():
            conn = self._require_conn()
            try:
                cursor = await conn.execute(
                    "SELECT id, task, action, result, agent_role, context, created_at "
                    "FROM experiences ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                )
                rows = await cursor.fetchall()
                return [self._serializer.decode_experience(row) for row in rows]
            except StorageError:
                raise
            except Exception as e:
                raise StorageError(
                    "Failed to retrieve recent experiences.",
                    operation="retrieve",
                    table="experiences",
                ) from e

    # --- Lesson Methods ---

    async def save_lesson(self, lesson: Lesson) -> None:
        async with self._operation():
            encoded = self._serializer.encode_lesson(lesson)
            async with self._transactions.write(
                operation="save",
                table="lessons",
                record_ids=(encoded.id,),
            ) as conn:
                await self._insert_lesson(conn, encoded)

    async def save_lesson_and_memory(self, lesson: Lesson, memory: Memory) -> None:
        """Persist a lesson and its derived memory atomically in one transaction."""
        async with self._operation():
            encoded_lesson = self._serializer.encode_lesson(lesson)
            encoded_memory = self._serializer.encode_memory(memory)
            async with self._transactions.write(
                operation="save",
                table="lessons,memories",
                record_ids=(encoded_lesson.id, encoded_memory.id),
            ) as conn:
                await self._insert_lesson(conn, encoded_lesson)
                await self._upsert_memory(conn, encoded_memory)

    @staticmethod
    async def _insert_lesson(conn: aiosqlite.Connection, lesson: EncodedLesson) -> None:
        await conn.execute(
            """
            INSERT INTO lessons (id, experience_id, content, agent_role, root_cause, confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            lesson.values(),
        )

    # --- Memory Methods ---

    async def save_memory(self, memory: Memory) -> None:
        async with self._operation():
            encoded = self._serializer.encode_memory(memory)
            async with self._transactions.write(
                operation="save",
                table="memories",
                record_ids=(encoded.id,),
            ) as conn:
                await self._upsert_memory(conn, encoded)

    async def _upsert_memory(
        self,
        conn: aiosqlite.Connection,
        memory: EncodedMemory,
    ) -> None:
        await conn.execute(
            """
            INSERT OR REPLACE INTO memories
            (id, content, type, agent_role, confidence, importance, source, metadata,
             embedding, reinforcement_count, success_count, created_at, updated_at,
             expires_at, embedding_dimension)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (*memory.values(), memory.embedding_dimension),
        )
        await self._vector_index.maintain_encoded_memory(conn, memory)

    async def rebuild_vector_index(
        self,
        *,
        batch_size: int = 256,
        force: bool = False,
    ) -> VectorIndexRebuildResult:
        """Resume rebuilding derived vector bands from stored embeddings."""
        async with self._operation():
            return await self._vector_index.rebuild(
                batch_size=batch_size,
                force=force,
            )

    async def get_memory(self, memory_id: UUID) -> Optional[Memory]:
        async with self._operation():
            conn = self._require_conn()
            try:
                cursor = await conn.execute(
                    f"SELECT {_MEMORY_COLUMNS} FROM memories WHERE id = ?",
                    (str(memory_id),),
                )
                row = await cursor.fetchone()
                return self._serializer.decode_memory(row) if row else None
            except StorageError:
                raise
            except Exception as e:
                raise StorageError(
                    "Failed to retrieve memory.",
                    operation="retrieve",
                    table="memories",
                    record_ids=(memory_id,),
                ) from e

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
        retrieval_query = RetrievalQuery(
            text=query,
            memory_type=memory_type,
            agent_role=agent_role,
            limit=limit,
            query_embedding=(
                tuple(query_embedding) if query_embedding is not None else None
            ),
            include_expired=include_expired,
        )
        if retrieval_query.limit == 0:
            return []

        async with self._operation():
            return await self._search_memories(retrieval_query)

    async def _search_memories(self, query: RetrievalQuery) -> List[Memory]:
        """Delegate private callers and public search to the retrieval engine."""
        return await self._retrieval_engine.search(query)

    @staticmethod
    def _rank_by_similarity(
        memories: List[Memory], query_embedding: List[float], limit: int
    ) -> List[Memory]:
        """Compatibility helper for deterministic semantic reranking."""
        query = RetrievalQuery(
            limit=limit,
            query_embedding=tuple(query_embedding),
            include_expired=True,
        )
        return rank_memories(memories, query)

    async def find_similar_memory(
        self,
        embedding: List[float],
        memory_type: Optional[MemoryType] = None,
        agent_role: Optional[str] = None,
        threshold: float = 0.95,
    ) -> Optional[Memory]:
        """Return an existing near-duplicate memory above ``threshold``, if any."""
        async with self._operation():
            if not embedding:
                return None
            candidates = await self._search_memories(
                RetrievalQuery(
                    memory_type=memory_type,
                    agent_role=agent_role,
                    query_embedding=tuple(embedding),
                    limit=1,
                )
            )
            if not candidates:
                return None
            best = candidates[0]
            if (
                best.embedding
                and cosine_similarity(embedding, best.embedding) >= threshold
            ):
                return best
            return None

    async def update_memory_feedback(
        self, memory_id: UUID, success: bool, alpha: float = 0.2
    ) -> Optional[Memory]:
        """
        Reinforce or weaken a memory based on a real outcome. Confidence moves
        toward 1.0 on success and 0.0 on failure via an exponential moving
        average, and reinforcement counters are incremented atomically.
        """
        async with self._operation():
            target = 1.0 if success else 0.0
            feedback_sql = """
                UPDATE memories
                SET reinforcement_count = reinforcement_count + 1,
                    success_count = success_count + ?,
                    confidence = MIN(
                        1.0,
                        MAX(
                            0.0,
                            ROUND(confidence + ? * (? - confidence), 4)
                        )
                    ),
                    updated_at = ?
                WHERE id = ?
            """
            parameters = (
                int(bool(success)),
                alpha,
                target,
                datetime.now(timezone.utc).isoformat(),
                str(memory_id),
            )

            async with self._transactions.write(
                operation="update_feedback",
                table="memories",
                record_ids=(str(memory_id),),
            ) as conn:
                try:
                    cursor = await conn.execute(
                        f"{feedback_sql} RETURNING {_MEMORY_COLUMNS}",
                        parameters,
                    )
                    row = await cursor.fetchone()
                except aiosqlite.OperationalError as error:
                    if "returning" not in str(error).casefold():
                        raise
                    cursor = await conn.execute(feedback_sql, parameters)
                    if cursor.rowcount == 0:
                        row = None
                    else:
                        cursor = await conn.execute(
                            f"SELECT {_MEMORY_COLUMNS} FROM memories WHERE id = ?",
                            (str(memory_id),),
                        )
                        row = await cursor.fetchone()

                memory = self._serializer.decode_memory(row) if row else None
            return memory

    async def prune_expired(self) -> int:
        """Delete expired memories. Returns the number removed."""
        async with self._operation():
            async with self._transactions.write(
                operation="prune", table="memories"
            ) as conn:
                cursor = await conn.execute(
                    "DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at <= ?",
                    (datetime.now(timezone.utc).isoformat(),),
                )
                removed = cursor.rowcount or 0
            if removed:
                logger.info(f"Pruned {removed} expired memories.")
            return removed
