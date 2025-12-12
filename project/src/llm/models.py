"""Data models for LLM interactions."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Roles for chat messages."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    """A single message in a conversation."""

    role: MessageRole
    content: str

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary format for API calls."""
        return {"role": self.role.value, "content": self.content}


class ResponseMetadata(BaseModel):
    """Metadata about an LLM response."""

    model_name: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    cost: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)

    def __init__(self, **data):
        """Initialize and calculate total tokens if not provided."""
        super().__init__(**data)
        if self.total_tokens == 0:
            self.total_tokens = self.input_tokens + self.output_tokens


class LLMResponse(BaseModel):
    """Response from an LLM."""

    content: str
    metadata: ResponseMetadata
    raw_response: Optional[Any] = None

    class Config:
        arbitrary_types_allowed = True


class StreamChunk(BaseModel):
    """A chunk of streamed response."""

    content: str
    is_final: bool = False
    metadata: Optional[ResponseMetadata] = None

    class Config:
        arbitrary_types_allowed = True
