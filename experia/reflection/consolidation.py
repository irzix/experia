from typing import Optional

from experia.core.exceptions import EvaluationError
from experia.core.interfaces import MemoryStore
from experia.core.logging import logger
from experia.memory.models import Memory, MemoryType

try:
    import litellm
except ImportError:
    litellm = None


class ReflectionEngine:
    """
    Transforms raw experiences into reusable knowledge (Strategies) by analyzing multiple past experiences.
    Since reflection requires expensive reasoning over multiple experiences,
    it is designed to be developer-controlled (not automatically executed in the core runtime).
    """

    def __init__(self, store: MemoryStore, model: str = "gpt-4o"):
        self.store = store
        self.model = model
        self.system_prompt = (
            "You are a strategic Reflection Engine for an AI agent. "
            "You will be given a batch of recent experiences and lessons the agent learned. "
            "Your task is to identify broad patterns, merge similar lessons, and extract a unified STRATEGY for future tasks. "
            "If there are conflicting lessons, resolve them based on the success rates or outcomes. "
            "Return the consolidated strategy as a concise, high-level rule or guideline (e.g. 'For web deployment tasks, prefer Docker containers over bare-metal Nginx configs due to historical port conflict issues'). "
            "If the experiences are too diverse and no single strategy applies, return 'NONE'.\n"
            "Return ONLY the strategy string, or 'NONE'."
        )

    async def reflect(self, batch_size: int = 50) -> Optional[Memory]:
        """
        Retrieves the most recent experiences, analyzes them to find patterns,
        and generates a high-level STRATEGY memory.
        """
        if not litellm:
            raise EvaluationError(
                "litellm is not installed. Please install experia[llm] or litellm to use the ReflectionEngine."
            )

        logger.info(f"Starting reflection over the last {batch_size} experiences...")
        recent_experiences = await self.store.get_recent_experiences(limit=batch_size)

        if not recent_experiences:
            logger.info("No recent experiences found for reflection.")
            return None

        # Build prompt from experiences
        experiences_text = ""
        for i, exp in enumerate(recent_experiences):
            experiences_text += f"\nExperience {i + 1}:\n"
            experiences_text += (
                f"Task: {exp.task}\nAction: {exp.action}\nResult: {exp.result}\n"
            )

        user_prompt = f"Recent Experiences:\n{experiences_text}"

        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )

            strategy_text = response.choices[0].message.content.strip()

            if not strategy_text or strategy_text == "NONE":
                logger.info("Reflection did not yield any cohesive strategy.")
                return None

            strategy_memory = Memory(
                content=strategy_text,
                type=MemoryType.STRATEGY,
                importance=1.0,  # Strategies are highly important
                source="ReflectionEngine",
            )

            await self.store.save_memory(strategy_memory)
            logger.info(f"Generated new STRATEGY: {strategy_text}")

            return strategy_memory

        except Exception as e:
            logger.error(f"ReflectionEngine failed during reflection: {e}")
            raise EvaluationError(f"Reflection failed: {e}")
