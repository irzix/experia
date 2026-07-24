import asyncio
from typing import Any, Dict, List, Optional, Set
from uuid import UUID

from experia.context.builder import ContextBuilder
from experia.core.exceptions import ConfigurationError
from experia.core.interfaces import Evaluator, MemoryStore
from experia.core.logging import logger
from experia.experience.models import ExperienceRecord, Lesson
from experia.improvement.rules import RuleGenerator
from experia.memory.embeddings import Embedder
from experia.memory.models import Memory, MemoryType


class Learner:
    """
    The main asynchronous entry point for the Experia AI Cognitive Layer.
    Uses Dependency Injection to accept any MemoryStore, Evaluator, and
    (optional) Embedder implementations.
    """

    def __init__(
        self,
        store: MemoryStore,
        evaluator: Evaluator,
        rule_generator: Optional[RuleGenerator] = None,
        embedder: Optional[Embedder] = None,
        agent_role: str = "default",
        background_evaluation: bool = True,
        dedup_threshold: float = 0.95,
    ):
        if not store or not evaluator:
            raise ConfigurationError("Learner requires both a store and an evaluator.")

        self.store = store
        self.evaluator = evaluator
        self.rule_generator = rule_generator
        self.embedder = embedder
        self.agent_role = agent_role
        self.background_evaluation = background_evaluation
        self.dedup_threshold = dedup_threshold
        self.context_builder = ContextBuilder()
        self._pending: Set[asyncio.Task] = set()
        logger.info(f"Experia Learner initialized successfully for role: {agent_role}.")

    async def record(
        self,
        task: str,
        action: str,
        result: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExperienceRecord:
        """
        Capture an agent's action and its result. The raw experience is
        persisted immediately and returned; the (potentially expensive)
        evaluation runs in the background unless ``background_evaluation`` is
        disabled. Use ``await learner.flush()`` to await pending evaluations.
        """
        experience = ExperienceRecord(
            task=task,
            action=action,
            result=result,
            agent_role=self.agent_role,
            context=context or {},
        )
        await self.store.save_experience(experience)
        logger.debug(f"Recorded experience {experience.id} for task '{task}'")

        if self.background_evaluation:
            self._spawn(self._evaluate_and_remember(experience))
        else:
            await self._evaluate_and_remember(experience)
        return experience

    def _spawn(self, coro) -> None:
        """Schedule background work, tracking it so ``flush()`` can await it."""
        task = asyncio.ensure_future(coro)
        self._pending.add(task)
        task.add_done_callback(self._on_task_done)

    def _on_task_done(self, task: asyncio.Task) -> None:
        self._pending.discard(task)
        if not task.cancelled():
            exc = task.exception()
            if exc:
                logger.error(f"Background evaluation failed: {exc}")

    async def flush(self) -> None:
        """Await all pending background evaluations."""
        while self._pending:
            await asyncio.gather(*list(self._pending), return_exceptions=True)

    async def _evaluate_and_remember(self, experience: ExperienceRecord) -> None:
        """Evaluate an experience and store its lesson + derived memory."""
        lesson = await self.evaluator.evaluate(experience)
        if not lesson:
            return

        lesson.agent_role = self.agent_role
        embedding = await self._embed(lesson.content)

        # De-duplicate: if a near-identical lesson memory already exists,
        # reinforce it instead of inserting a redundant copy.
        if embedding:
            existing = await self.store.find_similar_memory(
                embedding=embedding,
                memory_type=MemoryType.LESSON,
                agent_role=self.agent_role,
                threshold=self.dedup_threshold,
            )
            if existing:
                await self.store.save_lesson(lesson)
                await self.store.update_memory_feedback(existing.id, success=True)
                logger.info(
                    f"Reinforced existing memory {existing.id} from experience "
                    f"{experience.id} (deduplicated)."
                )
                await self._maybe_consolidate(lesson)
                return

        memory = self._build_lesson_memory(lesson, experience, embedding)
        await self.store.save_lesson_and_memory(lesson, memory)
        logger.info(f"Learned and remembered lesson from experience {experience.id}")
        await self._maybe_consolidate(lesson)

    def _build_lesson_memory(
        self,
        lesson: Lesson,
        experience: ExperienceRecord,
        embedding: Optional[List[float]],
    ) -> Memory:
        metadata = {}
        if getattr(lesson, "root_cause", None):
            metadata["root_cause"] = lesson.root_cause
        return Memory(
            content=lesson.content,
            type=MemoryType.LESSON,
            agent_role=self.agent_role,
            confidence=lesson.confidence,
            importance=0.7,
            source=f"experience_{experience.id}",
            metadata=metadata,
            embedding=embedding,
        )

    async def _maybe_consolidate(self, lesson: Lesson) -> None:
        if self.rule_generator:
            await self.rule_generator.consolidate_lesson(lesson)

    async def _embed(self, text: str) -> Optional[List[float]]:
        """Embed text if an embedder is configured; degrade gracefully on error."""
        if not self.embedder:
            return None
        try:
            return await self.embedder.embed_one(text)
        except Exception as e:
            logger.warning(f"Embedding failed, falling back to keyword search: {e}")
            return None

    async def reflect(self, model: str = "gpt-4o", batch_size: int = 50) -> None:
        """
        Manually triggers the Reflection Engine to analyze past experiences and
        extract high-level STRATEGY memories. Developer-controlled (not run
        automatically) due to reasoning cost.
        """
        from experia.reflection.consolidation import ReflectionEngine

        engine = ReflectionEngine(store=self.store, model=model)
        await engine.reflect(batch_size=batch_size)

    async def retrieve_context(self, query: str = "", limit: int = 5) -> str:
        """Search cognitive memory and build a prompt string."""
        query_embedding = await self._embed(query) if query else None
        memories = await self.store.search_memories(
            query=query,
            limit=limit,
            agent_role=self.agent_role,
            query_embedding=query_embedding,
        )
        logger.debug(f"Retrieved {len(memories)} memories for context.")
        return self.context_builder.format_for_prompt(memories)

    async def remember(
        self, content: str, memory_type: MemoryType = MemoryType.FACT
    ) -> Memory:
        """Manually add a piece of knowledge to the memory store."""
        embedding = await self._embed(content)

        if embedding:
            existing = await self.store.find_similar_memory(
                embedding=embedding,
                memory_type=memory_type,
                agent_role=self.agent_role,
                threshold=self.dedup_threshold,
            )
            if existing:
                updated = await self.store.update_memory_feedback(
                    existing.id, success=True
                )
                logger.debug(f"Deduplicated manual memory into {existing.id}")
                return updated or existing

        memory = Memory(
            content=content,
            type=memory_type,
            importance=0.9,
            agent_role=self.agent_role,
            embedding=embedding,
        )
        await self.store.save_memory(memory)
        logger.debug(f"Manually remembered explicit knowledge: {memory.id}")
        return memory

    async def reinforce(self, memory_id: UUID, success: bool) -> Optional[Memory]:
        """
        Close the feedback loop: report whether applying a memory led to a good
        outcome. Confidence is nudged toward 1.0 (success) or 0.0 (failure).
        """
        return await self.store.update_memory_feedback(memory_id, success=success)

    async def prune(self) -> int:
        """Remove expired memories. Returns the count pruned."""
        return await self.store.prune_expired()
