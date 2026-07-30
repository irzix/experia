from typing import Optional

from experia.core.interfaces import Evaluator
from experia.core.logging import logger
from experia.experience.models import ExperienceRecord, Lesson


class SimpleHeuristicEvaluator(Evaluator):
    """
    An asynchronous heuristic-based evaluator for testing and basic usage.
    Implements the Evaluator Protocol.
    """

    def __init__(self) -> None:
        """Initialize an evaluator with no configurable constructor state."""

    async def evaluate(self, experience: ExperienceRecord) -> Optional[Lesson]:
        """Evaluates the experience asynchronously."""
        lower_result = experience.result.lower()

        if "fail" in lower_result or "error" in lower_result:
            content = (
                f"The action '{experience.action}' failed during "
                f"'{experience.task}'. Ensure prerequisites are met before retrying."
            )
            confidence = 0.6
            logger.info(f"Evaluator detected failure for experience {experience.id}")
        elif "success" in lower_result:
            content = (
                f"The action '{experience.action}' was successful for "
                f"'{experience.task}'. This strategy works."
            )
            confidence = 0.8
            logger.info(f"Evaluator detected success for experience {experience.id}")
        else:
            logger.debug(f"No actionable lesson found for experience {experience.id}")
            return None

        return Lesson(
            experience_id=experience.id, content=content, confidence=confidence
        )
