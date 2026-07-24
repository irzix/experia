import json
from typing import Optional

from pydantic import BaseModel, Field

from experia.core.exceptions import EvaluationError
from experia.core.interfaces import Evaluator
from experia.core.logging import logger
from experia.experience.models import ExperienceRecord, Lesson

try:
    import litellm
except ImportError:
    litellm = None


class EvaluatorResponseSchema(BaseModel):
    """Internal schema to force structured JSON output from the LLM."""

    lesson: str = Field(description="The extracted lesson or behavioral rule.")
    root_cause: str = Field(description="The root cause analysis of the outcome.")
    confidence: float = Field(description="Confidence from 0.0 to 1.0.")


class LLMEvaluator(Evaluator):
    """
    An advanced evaluator that uses an LLM via litellm to deeply analyze experiences.
    Requires litellm and a configured API key for the chosen model.
    """

    def __init__(self, model: str = "gpt-4o"):
        if not litellm:
            raise EvaluationError(
                "litellm is not installed. Please install experia[llm] or litellm to use the LLMEvaluator."
            )
        self.model = model
        self.system_prompt = (
            "You are an expert cognitive evaluator for an AI agent. "
            "Analyze the given agent's action and its result.\n"
            "1. Identify the root cause of why it succeeded or failed.\n"
            "2. Extract a concise, generalized lesson that the agent should remember for the future.\n"
            "Return the output in valid JSON format matching this schema:\n"
            '{"lesson": "string", "root_cause": "string", "confidence": float}'
        )

    async def evaluate(self, experience: ExperienceRecord) -> Optional[Lesson]:
        user_prompt = (
            f"Task: {experience.task}\n"
            f"Action Taken: {experience.action}\n"
            f"Result/Outcome: {experience.result}\n"
            f"Active Context: {json.dumps(experience.context)}\n"
        )

        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )

            raw_content = response.choices[0].message.content
            if not raw_content:
                logger.warning(
                    f"LLM returned empty content for experience {experience.id}"
                )
                return None

            parsed = EvaluatorResponseSchema.model_validate_json(raw_content)

            logger.info(f"LLM successfully evaluated experience {experience.id}")
            return Lesson(
                experience_id=experience.id,
                content=parsed.lesson,
                root_cause=parsed.root_cause,
                confidence=parsed.confidence,
            )

        except Exception as e:
            logger.error(
                f"LLMEvaluator failed to evaluate experience {experience.id}: {e}"
            )
            raise EvaluationError(f"LLM evaluation failed: {e}")
