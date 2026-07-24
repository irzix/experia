from typing import Optional

from experia.core.interfaces import MemoryStore
from experia.core.logging import logger
from experia.experience.models import Lesson
from experia.memory.models import Memory, MemoryType

try:
    import litellm
except ImportError:
    litellm = None


class RuleGenerator:
    """
    Analyzes lessons and elevates them into permanent behavioral rules using an LLM.
    """

    def __init__(self, store: MemoryStore, model: str = "gpt-4o"):
        self.store = store
        self.model = model
        self.system_prompt = (
            "You are a cognitive consolidator for an AI agent. "
            "You will be given a recent 'Lesson' the agent learned from an experience. "
            "Determine if this lesson represents a fundamental, reusable behavioral rule "
            "that the agent should ALWAYS follow. "
            "If yes, extract it as a concise, generalized RULE (e.g. 'Always do X before Y'). "
            "If no (it is too specific or trivial), return 'NONE'.\n"
            "Return ONLY the rule string, or 'NONE'."
        )

    async def consolidate_lesson(self, lesson: Lesson) -> Optional[Memory]:
        """Evaluates a lesson and potentially creates a RULE memory."""
        if not litellm:
            logger.debug("litellm is not installed. Rule generation skipped.")
            return None

        user_prompt = (
            f"Lesson Content: {lesson.content}\n"
            f"Root Cause: {lesson.root_cause or 'Unknown'}\n"
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
                logger.debug(f"Lesson {lesson.id} was not elevated to a rule.")
                return None

            rule_memory = Memory(
                content=rule_text,
                type=MemoryType.RULE,
                importance=1.0,  # Rules are highly important
                source=f"Lesson:{lesson.id}",
            )

            await self.store.save_memory(rule_memory)
            logger.info(f"Generated new RULE from lesson {lesson.id}: {rule_text}")
            return rule_memory

        except Exception as e:
            logger.error(f"RuleGenerator failed for lesson {lesson.id}: {e}")
            return None
