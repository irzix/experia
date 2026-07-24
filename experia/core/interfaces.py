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

    async def save_lesson(self, lesson: Lesson) -> None: ...

    async def save_memory(self, memory: Memory) -> None: ...

    async def search_memories(
        self,
        query: str = "",
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
    ) -> List[Memory]: ...


class Evaluator(Protocol):
    """Protocol for experience evaluators."""

    async def evaluate(self, experience: ExperienceRecord) -> Optional[Lesson]: ...
