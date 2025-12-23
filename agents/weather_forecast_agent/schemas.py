"""
Data schemas used by the agent and tools.

Separating these helps avoid circular imports and keeps annotations clean.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Context:
    """Custom runtime context provided to tools at invocation time.

    Attributes:
        user_id: A caller- or session-specific identifier used by tools
                 (e.g., to infer location preferences).
    """

    user_id: str


@dataclass
class ResponseFormat:
    """Structured response returned by the agent via ToolStrategy.

    Attributes:
        punny_response: The human-friendly response crafted with puns.
        weather_conditions: Optional summary of the weather conditions.
    """

    punny_response: str
    weather_conditions: str | None = None
