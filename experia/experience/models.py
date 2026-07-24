from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ExperienceRecord(BaseModel):
    """
    Represents a raw experience captured from an agent's execution.
    It logs the exact task, action taken, and the subsequent outcome.
    """

    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for the experience record.",
    )
    task: str = Field(
        ...,
        description="The high-level task or goal the agent was attempting.",
        json_schema_extra={"example": "Deploy web application"},
    )
    action: str = Field(
        ...,
        description="The specific action or tool call executed.",
        json_schema_extra={"example": "Run command: docker restart nginx"},
    )
    result: str = Field(
        ...,
        description="The raw outcome, error message, or output of the action.",
        json_schema_extra={"example": "Failed: port 80 is already bound."},
    )
    context: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional state variables or memory context active during the action.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the experience occurred.",
    )


class Lesson(BaseModel):
    """
    Represents a discrete lesson extracted by the Evaluator engine from an experience.
    """

    id: UUID = Field(
        default_factory=uuid4, description="Unique identifier for the lesson."
    )
    experience_id: UUID = Field(
        ..., description="The ID of the experience this lesson was derived from."
    )
    content: str = Field(
        ...,
        description="The actual knowledge or rule extracted.",
        json_schema_extra={
            "example": "Always verify port availability using lsof -i :80 before starting the web server."
        },
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="The system's confidence in this lesson (0.0 to 1.0).",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the lesson was extracted.",
    )
