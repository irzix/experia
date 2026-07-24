from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """Enumeration of the different cognitive types of memory."""

    FACT = "fact"
    PREFERENCE = "preference"
    LESSON = "lesson"
    RULE = "rule"
    STRATEGY = "strategy"
    EXPERIENCE = "experience"


class Memory(BaseModel):
    """
    Represents a unit of knowledge stored in the cognitive memory layer.
    """

    id: UUID = Field(
        default_factory=uuid4, description="Unique identifier for the memory."
    )
    content: str = Field(
        ...,
        description="The actual knowledge stored.",
        json_schema_extra={"example": "User prefers short, concise answers."},
    )
    type: MemoryType = Field(..., description="The category/type of this memory.")
    agent_role: str = Field(
        default="default", description="The role of the agent that created the memory."
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence level in the accuracy of this memory.",
    )
    importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Importance weight used for retrieval ranking (0.0 to 1.0).",
    )
    source: Optional[str] = Field(
        default=None,
        description="Where this memory originated from (e.g., user input, experience ID).",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional custom data for advanced filtering.",
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the memory was first created.",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the memory was last modified or reinforced.",
    )
    expires_at: Optional[datetime] = Field(
        default=None, description="Optional expiration time for ephemeral memories."
    )
    reinforcement_count: int = Field(
        default=0,
        ge=0,
        description="How many times this memory has been validated by an outcome.",
    )
    success_count: int = Field(
        default=0,
        ge=0,
        description="How many of those validations were successful outcomes.",
    )

    # --- Semantic retrieval ---
    embedding: Optional[List[float]] = Field(
        default=None,
        exclude=True,
        repr=False,
        description="Vector embedding of `content`, used for semantic search.",
    )
