from typing import List, Optional, Protocol
from uuid import UUID

from experia.experience.models import ExperienceRecord, Lesson
from experia.memory.models import Memory, MemoryType


class MemoryStore(Protocol):
    """Protocol for memory and experience storage backends."""

    async def save_experience(self, experience: ExperienceRecord) -> None: ...

    async def get_experience(
        self, experience_id: UUID
    ) -> Optional[ExperienceRecord]: ...

    async def get_recent_experiences(
        self, limit: int = 50
    ) -> List[ExperienceRecord]: ...

    async def save_lesson(self, lesson: Lesson) -> None: ...

    async def save_lesson_and_memory(self, lesson: Lesson, memory: Memory) -> None: ...

    async def save_memory(self, memory: Memory) -> None: ...

    async def get_memory(self, memory_id: UUID) -> Optional[Memory]: ...

    async def search_memories(
        self,
        query: str = "",
        memory_type: Optional[MemoryType] = None,
        agent_role: Optional[str] = None,
        limit: int = 10,
        query_embedding: Optional[List[float]] = None,
        include_expired: bool = False,
    ) -> List[Memory]: ...

    async def find_similar_memory(
        self,
        embedding: List[float],
        memory_type: Optional[MemoryType] = None,
        agent_role: Optional[str] = None,
        threshold: float = 0.95,
    ) -> Optional[Memory]: ...

    async def update_memory_feedback(
        self, memory_id: UUID, success: bool, alpha: float = 0.2
    ) -> Optional[Memory]: ...

    async def prune_expired(self) -> int: ...


class Evaluator(Protocol):
    """Protocol for experience evaluators."""

    async def evaluate(self, experience: ExperienceRecord) -> Optional[Lesson]: ...
