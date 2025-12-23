"""
Tool definitions invoked by the agent.

Tools are small, deterministic functions the agent can call to fetch or
compute information. They may consume the runtime Context when needed.
"""

from __future__ import annotations

from langchain.tools import ToolRuntime, tool

from agents.weather_forecast_agent.schemas import Context


@tool
def get_weather_for_city(city: str) -> str:
    """Get the weather for a specific city.

    In a real app, this might call a weather API. For now, it's stubbed.
    """
    return f"The weather in {city} is sunny with a high of 75°F."


@tool
def get_user_city(runtime: ToolRuntime[Context]) -> str:
    """Infer the user's city from the provided runtime `Context`.

    Demonstrates how tools can access the per-request context passed to
    `agent.invoke(..., context=Context(...))`.
    """
    user_id = runtime.context.user_id
    return "Miami" if user_id == "1" else "San Francisco"
