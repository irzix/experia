from typing import Optional

from experia.core.dependencies import require_optional_dependency
from experia.core.interfaces import MemoryStore
from experia.core.logging import logger
from experia.experience.models import Lesson
from experia.memory.models import Memory, MemoryType
from experia.security.protection import DataProtectionLayer

try:
    import litellm
except ImportError:
    litellm = None


class RuleGenerator:
    """
    Analyzes lessons and elevates them into permanent behavioral rules using an LLM.
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
            "You are a cognitive consolidator for an AI agent. "
            "You will be given a recent 'Lesson' the agent learned from an experience. "
            "Determine if this lesson represents a fundamental, reusable behavioral rule "
            "that the agent should ALWAYS follow. "
            "If yes, extract it as a concise, generalized RULE (e.g. 'Always do X before Y'). "
            "If no (it is too specific or trivial), return 'NONE'.\n"
            "Return ONLY the rule string, or 'NONE'."
        )

    def _set_data_protection(self, data_protection: DataProtectionLayer) -> None:
        self._data_protection = data_protection

    async def consolidate_lesson(self, lesson: Lesson) -> Optional[Memory]:
        """Evaluate a lesson and potentially create a RULE memory."""
        require_optional_dependency(
            litellm is not None,
            feature="RuleGenerator",
            extra="experia[llm]",
        )

        fields, metadata = self._data_protection.protect_sink(
            {
                "content": lesson.content,
                "root_cause": lesson.root_cause or "Unknown",
            },
            {
                "feature": "rule_generator",
                "operation": "rule_generation",
                "lesson_id": str(lesson.id),
                "experience_id": str(lesson.experience_id),
                "model": self.model,
            },
        )
        user_prompt = (
            f"Lesson Content: {fields['content']}\nRoot Cause: {fields['root_cause']}\n"
        )

        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
            )

            rule_text = response.choices[0].message.content.strip()
            if not rule_text or rule_text == "NONE":
                logger.debug(
                    "Lesson was not elevated to a rule.",
                    extra={"experia_metadata": metadata},
                )
                return None

            rule_memory = Memory(
                content=rule_text,
                type=MemoryType.RULE,
                importance=1.0,
                source=f"Lesson:{lesson.id}",
            )

            await self.store.save_memory(rule_memory)
            logger.info(
                "Generated a new rule from a lesson.",
                extra={"experia_metadata": metadata},
            )
            return rule_memory

        except Exception:
            logger.error(
                "Rule generation failed.",
                extra={"experia_metadata": metadata},
            )
            return None
