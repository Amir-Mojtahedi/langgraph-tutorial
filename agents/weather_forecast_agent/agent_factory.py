"""
Agent construction.

This module composes the model, tools, schema, prompt, and checkpointer
into a runnable agent instance.
"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

from agents.weather_forecast_agent.config import SYSTEM_PROMPT, checkpointer, llm
from agents.weather_forecast_agent.schemas import Context, ResponseFormat
from agents.weather_forecast_agent.tools import get_user_city, get_weather_for_city


def build_agent():
    """Create and return the configured agent instance.

    Returns:
        A runnable agent supporting `invoke()` with messages, config,
        and an optional `context` matching the `Context` schema.
    """
    agent = create_agent(
        model=llm,
        tools=[get_weather_for_city, get_user_city],
        system_prompt=SYSTEM_PROMPT,
        context_schema=Context,
        response_format=ToolStrategy(ResponseFormat),
        checkpointer=checkpointer,
    )
    return agent
