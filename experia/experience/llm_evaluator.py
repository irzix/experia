import json
from typing import Optional

from pydantic import BaseModel, Field

from experia.core.dependencies import require_optional_dependency
from experia.core.exceptions import EvaluationError
from experia.core.interfaces import Evaluator
from experia.core.logging import logger
from experia.experience.models import ExperienceRecord, Lesson
from experia.security.protection import DataProtectionLayer

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

    _experia_protects_external = True

    def __init__(
        self,
        model: str = "gpt-4o",
        *,
        data_protection: DataProtectionLayer | None = None,
    ):
        require_optional_dependency(
            litellm is not None,
            feature="LLMEvaluator",
            extra="experia[llm]",
        )
        self.model = model
        self._data_protection = data_protection or DataProtectionLayer()
        self.system_prompt = (
            "You are an expert cognitive evaluator for an AI agent. "
            "Analyze the given agent's action and its result.\n"
            "1. Identify the root cause of why it succeeded or failed.\n"
            "2. Extract a concise, generalized lesson that the agent should remember for the future.\n"
            "Return the output in valid JSON format matching this schema:\n"
            '{"lesson": "string", "root_cause": "string", "confidence": float}'
        )

    def _set_data_protection(self, data_protection: DataProtectionLayer) -> None:
        self._data_protection = data_protection

    async def evaluate(self, experience: ExperienceRecord) -> Optional[Lesson]:
        require_optional_dependency(
            litellm is not None,
            feature="LLMEvaluator",
            extra="experia[llm]",
        )
        fields, metadata = self._data_protection.protect_sink(
            {
                "task": experience.task,
                "action": experience.action,
                "result": experience.result,
                "context": experience.context,
            },
            {
                "feature": "llm_evaluator",
                "operation": "evaluation",
                "experience_id": str(experience.id),
                "model": self.model,
            },
        )
        user_prompt = (
            f"Task: {fields['task']}\n"
            f"Action Taken: {fields['action']}\n"
            f"Result/Outcome: {fields['result']}\n"
            f"Active Context: {json.dumps(fields['context'], allow_nan=False)}\n"
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
                    "LLM evaluator returned empty content.",
                    extra={"experia_metadata": metadata},
                )
                return None

            parsed = EvaluatorResponseSchema.model_validate_json(raw_content)

            logger.info(
                "LLM evaluator completed successfully.",
                extra={"experia_metadata": metadata},
            )
            return Lesson(
                experience_id=experience.id,
                content=parsed.lesson,
                root_cause=parsed.root_cause,
                confidence=parsed.confidence,
            )

        except Exception as error:
            logger.error(
                "LLM evaluator failed.",
                extra={"experia_metadata": metadata},
            )
            raise EvaluationError("LLM evaluation failed.") from error
