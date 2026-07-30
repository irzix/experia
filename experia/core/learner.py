import asyncio
import inspect
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from experia.context.builder import ContextBuilder
from experia.core.exceptions import (
    ConfigurationError,
    LifecycleError,
    SanitizationError,
)
from experia.core.interfaces import Evaluator, MemoryStore
from experia.core.logging import logger
from experia.core.work import (
    AsyncWorkManager,
    FlushReport,
    JobHandle,
    LifecycleObserver,
    OperationType,
    ShutdownPolicy,
    ShutdownReport,
    TerminalState,
    WorkManagerState,
)
from experia.experience.models import ExperienceRecord, Lesson
from experia.improvement.rules import RuleGenerator
from experia.memory.embeddings import Embedder
from experia.memory.models import Memory, MemoryType
from experia.security.protection import DataProtectionLayer


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
        *,
        embedding_failure: Literal["fallback", "raise"] = "fallback",
        data_protection: DataProtectionLayer | None = None,
        lifecycle_observer: LifecycleObserver | None = None,
    ):
        if not store or not evaluator:
            raise ConfigurationError("Learner requires both a store and an evaluator.")
        if embedding_failure not in {"fallback", "raise"}:
            raise ConfigurationError(
                "embedding_failure must be 'fallback' or 'raise'.",
                feature="Learner",
                parameter="embedding_failure",
            )

        self.store = store
        self.evaluator = evaluator
        self.rule_generator = rule_generator
        self.embedder = embedder
        self.agent_role = agent_role
        self.background_evaluation = background_evaluation
        self.dedup_threshold = dedup_threshold
        self.embedding_failure = embedding_failure
        self.data_protection = data_protection or DataProtectionLayer()
        if data_protection is not None:
            for sink in (self.evaluator, self.rule_generator, self.embedder):
                configure = getattr(sink, "_set_data_protection", None)
                if callable(configure):
                    configure(data_protection)
        self.context_builder = ContextBuilder()
        self._work_manager = AsyncWorkManager(observer=lifecycle_observer)
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
        self._assert_accepting("record")
        return await self._record(
            task=task,
            action=action,
            result=result,
            context=context,
        )

    async def _record(
        self,
        *,
        task: str,
        action: str,
        result: str,
        context: Optional[Dict[str, Any]],
        parent_job_id: UUID | None = None,
    ) -> ExperienceRecord:
        """Persist one accepted record request and schedule its evaluation."""

        experience = ExperienceRecord(
            task=task,
            action=action,
            result=result,
            agent_role=self.agent_role,
            context=context or {},
        )
        await self.store.save_experience(experience)
        logger.debug(f"Recorded experience {experience.id} for task '{task}'")

        evaluation = self._submit_evaluation(
            experience,
            parent_job_id=parent_job_id,
        )
        if not self.background_evaluation:
            if parent_job_id is None:
                await self.flush()
            else:
                await self._work_manager.wait(evaluation)
        return experience

    def _assert_accepting(self, operation: str) -> None:
        state = self._work_manager.state
        if state is not WorkManagerState.OPEN:
            raise LifecycleError(state=state.value, operation=operation)

    def _submit_callback_record(
        self,
        *,
        task: str,
        action: str,
        result: str,
        context: Optional[Dict[str, Any]],
        on_success: Any = None,
        on_failure: Any = None,
    ) -> JobHandle:
        """Register callback persistence as learner-owned background work."""

        self._assert_accepting("record")
        record_job_id: UUID | None = None

        async def record() -> None:
            if record_job_id is None:  # pragma: no cover - assigned before scheduling
                raise RuntimeError("Record job identity is unavailable")
            try:
                await self._record(
                    task=task,
                    action=action,
                    result=result,
                    context=context,
                    parent_job_id=record_job_id,
                )
            except Exception:
                if on_failure is not None:
                    on_failure()
                raise
            if on_success is not None:
                on_success()

        handle = self._work_manager.submit(OperationType.RECORD, record)
        record_job_id = handle.job_id
        return handle

    def _submit_evaluation(
        self,
        experience: ExperienceRecord,
        *,
        parent_job_id: UUID | None = None,
    ) -> JobHandle:
        evaluation_job_id: UUID | None = None

        async def evaluate() -> None:
            if (
                evaluation_job_id is None
            ):  # pragma: no cover - assigned before scheduling
                raise RuntimeError("Evaluation job identity is unavailable")
            await self._evaluate_and_remember(
                experience,
                parent_job_id=evaluation_job_id,
            )

        handle = self._work_manager.submit(
            OperationType.EVALUATION,
            evaluate,
            experience_id=experience.id,
            parent_job_id=parent_job_id,
        )
        evaluation_job_id = handle.job_id
        return handle

    async def flush(self) -> FlushReport:
        """Await the current background-work cutoff and return its outcomes."""
        return await self._work_manager.flush()

    async def shutdown(
        self, policy: ShutdownPolicy | str = ShutdownPolicy.DRAIN
    ) -> ShutdownReport:
        """Stop accepting evaluation work and drain or cancel accepted jobs."""
        return await self._work_manager.shutdown(policy)

    async def aclose(
        self,
        policy: ShutdownPolicy | str = ShutdownPolicy.DRAIN,
        *,
        close_store: bool = False,
    ) -> None:
        """Shut down owned work and optionally close the caller-owned store."""
        try:
            await self.shutdown(policy)
        finally:
            if close_store:
                await self._close_store()

    async def _close_store(self) -> None:
        close = getattr(self.store, "close", None)
        if not callable(close):
            raise ConfigurationError(
                "The configured store does not support close().",
                feature="Learner",
                parameter="close_store",
            )
        result = close()
        if inspect.isawaitable(result):
            await result

    async def _evaluate_and_remember(
        self,
        experience: ExperienceRecord,
        *,
        parent_job_id: UUID,
    ) -> None:
        """Evaluate an experience and store its lesson + derived memory."""
        lesson = await self.evaluator.evaluate(experience)
        if not lesson:
            return

        lesson.agent_role = self.agent_role
        embedding = await self._embed(
            lesson.content,
            experience_id=experience.id,
            parent_job_id=parent_job_id,
        )

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

    async def _embed(
        self,
        text: str,
        *,
        experience_id: UUID | None = None,
        parent_job_id: UUID | None = None,
    ) -> Optional[List[float]]:
        """Embed text, retaining fallback while tracking pipeline failures."""
        if not self.embedder:
            return None
        if parent_job_id is None:
            return await self._embed_direct(text)

        embedding: Optional[List[float]] = None
        embedding_error: Exception | None = None

        async def embed() -> None:
            nonlocal embedding, embedding_error
            try:
                embedding = await self._call_embedder(text)
            except Exception as error:
                embedding_error = error
                raise

        handle = self._work_manager.submit(
            OperationType.EMBEDDING,
            embed,
            experience_id=experience_id,
            parent_job_id=parent_job_id,
        )
        terminal_state = await self._work_manager.wait(handle)
        if terminal_state is TerminalState.SUCCESS:
            return embedding
        if terminal_state is TerminalState.CANCELLATION:
            raise asyncio.CancelledError
        if isinstance(embedding_error, SanitizationError):
            raise embedding_error
        if self.embedding_failure == "fallback":
            logger.warning("Embedding failed; falling back to keyword search.")
            return None
        if embedding_error is not None:
            raise embedding_error
        raise RuntimeError("Embedding failed without an available cause")

    async def _call_embedder(self, text: str) -> List[float]:
        if getattr(self.embedder, "_experia_protects_external", False):
            return await self.embedder.embed_one(text)

        fields, _metadata = self.data_protection.protect_sink(
            {"text": text},
            {
                "feature": "external_embedder",
                "operation": "embedding",
                "text_count": 1,
            },
        )
        return await self.embedder.embed_one(fields["text"])

    async def _embed_direct(self, text: str) -> Optional[List[float]]:
        try:
            return await self._call_embedder(text)
        except SanitizationError:
            raise
        except Exception:
            if self.embedding_failure == "raise":
                raise
            logger.warning("Embedding failed; falling back to keyword search.")
            return None

    async def reflect(self, model: str = "gpt-4o", batch_size: int = 50) -> None:
        """
        Manually triggers the Reflection Engine to analyze past experiences and
        extract high-level STRATEGY memories. Developer-controlled (not run
        automatically) due to reasoning cost.
        """
        from experia.reflection.consolidation import ReflectionEngine

        engine = ReflectionEngine(
            store=self.store,
            model=model,
            data_protection=self.data_protection,
        )
        await engine.reflect(batch_size=batch_size)

    async def retrieve_context(self, query: str = "", limit: int = 5) -> str:
        """Search cognitive memory and build a prompt string."""
        validator = getattr(self.context_builder, "validate_configuration", None)
        if callable(validator):
            validator()
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
