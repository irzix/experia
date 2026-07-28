from typing import Optional

from experia.core.dependencies import require_optional_dependency
from experia.core.exceptions import EvaluationError
from experia.core.interfaces import MemoryStore
from experia.core.logging import logger
from experia.memory.models import Memory, MemoryType
from experia.security.protection import DataProtectionLayer

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

    _experia_protects_external = True

    def __init__(
        self,
        store: MemoryStore,
        model: str = "gpt-4o",
        *,
        data_protection: DataProtectionLayer | None = None,
    ):
        self.store = store
        self.model = model
        self._data_protection = data_protection or DataProtectionLayer()
        self.system_prompt = (
            "You are a strategic Reflection Engine for an AI agent. "
            "You will be given a batch of recent experiences and lessons the agent learned. "
            "Your task is to identify broad patterns, merge similar lessons, and extract a unified STRATEGY for future tasks. "
            "If there are conflicting lessons, resolve them based on the success rates or outcomes. "
            "Return the consolidated strategy as a concise, high-level rule or guideline (e.g. 'For web deployment tasks, prefer Docker containers over bare-metal Nginx configs due to historical port conflict issues'). "
            "If the experiences are too diverse and no single strategy applies, return 'NONE'.\n"
            "Return ONLY the strategy string, or 'NONE'."
        )

    def _set_data_protection(self, data_protection: DataProtectionLayer) -> None:
        self._data_protection = data_protection

    async def reflect(self, batch_size: int = 50) -> Optional[Memory]:
        """Analyze recent experiences and create a high-level STRATEGY memory."""
        require_optional_dependency(
            litellm is not None,
            feature="ReflectionEngine",
            extra="experia[llm]",
        )

        recent_experiences = await self.store.get_recent_experiences(limit=batch_size)
        if not recent_experiences:
            logger.info("No recent experiences found for reflection.")
            return None

        fields, metadata = self._data_protection.protect_sink(
            {
                "experiences": [
                    {
                        "task": experience.task,
                        "action": experience.action,
                        "result": experience.result,
                    }
                    for experience in recent_experiences
                ]
            },
            {
                "feature": "reflection_engine",
                "operation": "reflection",
                "batch_size": batch_size,
                "experience_count": len(recent_experiences),
                "model": self.model,
            },
        )

        experiences_text = ""
        for index, experience in enumerate(fields["experiences"]):
            experiences_text += f"\nExperience {index + 1}:\n"
            experiences_text += (
                f"Task: {experience['task']}\n"
                f"Action: {experience['action']}\n"
                f"Result: {experience['result']}\n"
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
                logger.info(
                    "Reflection did not yield a cohesive strategy.",
                    extra={"experia_metadata": metadata},
                )
                return None

            strategy_memory = Memory(
                content=strategy_text,
                type=MemoryType.STRATEGY,
                importance=1.0,
                source="ReflectionEngine",
            )

            await self.store.save_memory(strategy_memory)
            logger.info(
                "Generated a new reflection strategy.",
                extra={"experia_metadata": metadata},
            )
            return strategy_memory

        except Exception as error:
            logger.error(
                "Reflection failed.",
                extra={"experia_metadata": metadata},
            )
            raise EvaluationError("Reflection failed.") from error
