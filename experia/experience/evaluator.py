from abc import ABC, abstractmethod
from typing import Optional

from experia.experience.models import ExperienceRecord, Lesson


class BaseEvaluator(ABC):
    """
    Abstract base class for evaluating experiences and extracting lessons.
    """

    @abstractmethod
    def evaluate(self, experience: ExperienceRecord) -> Optional[Lesson]:
        """
        Analyzes the experience and returns a Lesson if there is something
        valuable to learn. Returns None if nothing significant was learned.
        """
        pass


class SimpleHeuristicEvaluator(BaseEvaluator):
    """
    A simple heuristic-based evaluator for testing and basic usage.
    In production, this should be replaced by an LLM-based evaluator.
    """

    def evaluate(self, experience: ExperienceRecord) -> Optional[Lesson]:
        # Very simple heuristic: if it failed, tell it to be careful next time.
        # This is just a placeholder logic to demonstrate the pipeline.

        lower_result = experience.result.lower()

        if "fail" in lower_result or "error" in lower_result:
            content = (
                f"The action '{experience.action}' failed during "
                f"'{experience.task}'. Ensure prerequisites are met before retrying."
            )
            confidence = 0.6
        elif "success" in lower_result:
            content = (
                f"The action '{experience.action}' was successful for "
                f"'{experience.task}'. This strategy works."
            )
            confidence = 0.8
        else:
            return None  # Nothing obvious to learn

        return Lesson(
            experience_id=experience.id, content=content, confidence=confidence
        )
