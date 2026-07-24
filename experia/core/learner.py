from typing import Any, Dict, Optional

from experia.context.builder import ContextBuilder
from experia.experience.evaluator import BaseEvaluator, SimpleHeuristicEvaluator
from experia.experience.models import ExperienceRecord
from experia.memory.models import Memory, MemoryType
from experia.memory.store import SQLiteStore


class Learner:
    """
    The main entry point for the Experia AI Cognitive Layer.
    Wraps around the storage, evaluation, and context-building logic.
    """

    def __init__(
        self,
        store: Optional[SQLiteStore] = None,
        evaluator: Optional[BaseEvaluator] = None,
    ):
        self.store = store or SQLiteStore()
        self.evaluator = evaluator or SimpleHeuristicEvaluator()
        self.context_builder = ContextBuilder()

    def record(
        self,
        task: str,
        action: str,
        result: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExperienceRecord:
        """
        Records an agent's action and its result.

        Args:
            task: The overall goal the agent is trying to achieve.
            action: The specific action the agent took.
            result: The outcome of the action (e.g., success, failure, error message).
            context: Optional dictionary of current state variables.

        Returns:
            The saved ExperienceRecord.
        """
        experience = ExperienceRecord(
            task=task, action=action, result=result, context=context or {}
        )
        self.store.save_experience(experience)

        # Immediate evaluation for the MVP.
        # In future versions, this can be moved to an async background worker.
        self._evaluate_and_remember(experience)

        return experience

    def _evaluate_and_remember(self, experience: ExperienceRecord) -> None:
        """
        Internal method to evaluate an experience and store its lesson as a memory.
        """
        lesson = self.evaluator.evaluate(experience)
        if lesson:
            self.store.save_lesson(lesson)

            # Convert the lesson into a long-term cognitive Memory object
            memory = Memory(
                content=lesson.content,
                type=MemoryType.LESSON,
                confidence=lesson.confidence,
                importance=0.7,  # Default importance for extracted lessons
                source=f"Experience:{experience.id}",
            )
            self.store.save_memory(memory)

    def retrieve_context(self, query: str = "", limit: int = 5) -> str:
        """
        Searches the cognitive memory for relevant information and returns it
        formatted as a string, ready to be injected into an LLM prompt.
        """
        memories = self.store.search_memories(query=query, limit=limit)
        return self.context_builder.format_for_prompt(memories)

    def remember(
        self, content: str, memory_type: MemoryType = MemoryType.FACT
    ) -> Memory:
        """
        Manually add a piece of knowledge to the memory store.
        Useful for storing user preferences or explicit facts.
        """
        memory = Memory(
            content=content,
            type=memory_type,
            importance=0.9,  # Explicit memories are highly important
        )
        self.store.save_memory(memory)
        return memory
