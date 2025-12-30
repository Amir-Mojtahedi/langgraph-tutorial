"""
Data schemas used by the agent and tools.

Separated to avoid circular imports and keep annotations clean.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Context(BaseModel):
    """Custom runtime context provided to tools at invocation time.

    Attributes:
        user_id: A caller- or session-specific identifier used by tools
                 (e.g., to infer location preferences).
    """

    user_id: str = Field(min_length=1, description="Caller/session identifier")


class ResponseFormat(BaseModel):
    """Structured response returned by the agent via ToolStrategy.

    Attributes:
        punny_response: The human-friendly response crafted with puns.
        weather_conditions: Optional summary of the weather conditions.
    """

    punny_response: str = Field(min_length=1, description="Pun-filled reply")
    weather_conditions: str | None = Field(
        default=None,
        min_length=1,
        description="Optional weather summary",
    )
