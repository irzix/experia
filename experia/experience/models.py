from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ExperienceRecord(BaseModel):
    """
    Represents a raw experience captured from an agent's execution.
    """
    id: UUID = Field(default_factory=uuid4)
    task: str
    action: str
    result: str
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Lesson(BaseModel):
    """
    Represents a lesson extracted from one or more experiences.
    """
    id: UUID = Field(default_factory=uuid4)
    experience_id: UUID
    content: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
