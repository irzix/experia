from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

from experia.context.builder import ContextBuilder
from experia.core.exceptions import ConfigurationError
from experia.core.interfaces import Evaluator, MemoryStore
from experia.core.logging import logger
from experia.experience.models import ExperienceRecord
from experia.improvement.rules import RuleGenerator
from experia.memory.models import Memory, MemoryType


class Learner:
    """
    The main asynchronous entry point for the Experia AI Cognitive Layer.
    Uses Dependency Injection to accept any MemoryStore and Evaluator implementations.
    """

    def __init__(
        self,
        store: MemoryStore,
        evaluator: Evaluator,
        rule_generator: Optional[RuleGenerator] = None,
        agent_role: str = "default",
    ):
        if not store or not evaluator:
            raise ConfigurationError("Learner requires both a store and an evaluator.")

        self.store = store
        self.evaluator = evaluator
        self.rule_generator = rule_generator
        self.agent_role = agent_role
        self.context_builder = ContextBuilder()
        logger.info(f"Experia Learner initialized successfully for role: {agent_role}.")

    async def record(
        self,
        task: str,
        action: str,
        result: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExperienceRecord:
        """Records an agent's action and its result asynchronously."""
        experience = ExperienceRecord(
            id=uuid4(),
            task=task,
            action=action,
            result=result,
            agent_role=self.agent_role,
            context=context or {},
            created_at=datetime.now(timezone.utc),
        )
        await self.store.save_experience(experience)
        logger.debug(f"Recorded experience {experience.id} for task '{task}'")

        await self._evaluate_and_remember(experience)
        return experience

    async def _evaluate_and_remember(self, experience: ExperienceRecord) -> None:
        """Internal asynchronous method to evaluate an experience and store its lesson."""
        lesson = await self.evaluator.evaluate(experience)
        if lesson:
            lesson.agent_role = self.agent_role
            await self.store.save_lesson(lesson)

            memory = Memory(
                id=uuid4(),
                content=lesson.content,
                type=MemoryType.LESSON,
                agent_role=self.agent_role,
                confidence=lesson.confidence,
                importance=0.7,
                source=f"experience_{experience.id}",
                metadata={"root_cause": lesson.root_cause}
                if hasattr(lesson, "root_cause") and lesson.root_cause
                else {},
            )
            await self.store.save_memory(memory)
            logger.info(
                f"Learned and remembered lesson from experience {experience.id}"
            )

            # Optionally generate a global rule
            if self.rule_generator:
                await self.rule_generator.consolidate_lesson(lesson)

    async def reflect(self, model: str = "gpt-4o", batch_size: int = 50) -> None:
        """
        Manually triggers the Reflection Engine to analyze past experiences and
        extract high-level STRATEGY memories. This is developer-controlled
        and not run automatically due to reasoning cost.
        """
        from experia.reflection.consolidation import ReflectionEngine

        engine = ReflectionEngine(store=self.store, model=model)
        await engine.reflect(batch_size=batch_size)

    async def retrieve_context(self, query: str = "", limit: int = 5) -> str:
        """Searches cognitive memory and builds a prompt string asynchronously."""
        memories = await self.store.search_memories(
            query=query, limit=limit, agent_role=self.agent_role
        )
        logger.debug(f"Retrieved {len(memories)} memories for context.")
        return self.context_builder.format_for_prompt(memories)

    async def remember(
        self, content: str, memory_type: MemoryType = MemoryType.FACT
    ) -> Memory:
        """Manually add a piece of knowledge to the memory store asynchronously."""
        memory = Memory(
            content=content,
            type=memory_type,
            importance=0.9,
            agent_role=self.agent_role,
        )
        await self.store.save_memory(memory)
        logger.debug(f"Manually remembered explicit knowledge: {memory.id}")
        return memory
