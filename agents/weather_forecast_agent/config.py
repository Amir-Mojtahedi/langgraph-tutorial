"""
Configuration and LLM setup.

This module centralizes environment parsing, the system prompt, and
construction of shared runtime services (LLM client, checkpointer).
"""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import SecretStr

# Read environment variables for LLM configuration
LLM_MODEL = os.environ.get("LLM_MODEL")
LLM_BASE_URL = os.environ.get("LLM_URL")
OPENAI_API_KEY = SecretStr(os.environ.get("OPENAI_API_KEY", ""))

if not LLM_MODEL or not LLM_BASE_URL:
    raise ValueError("LLM_MODEL and LLM_URL environment variables must be set.")

# A playful system prompt guiding the agent's persona and tool usage
SYSTEM_PROMPT = """You are an expert weather forecaster, who speaks in puns.

You have access to two tools:

- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location

If a user asks you for the weather, make sure you know the location. If you can tell from the question that they mean wherever they are, use the get_user_location tool to find their location."""

# Construct the LLM client used by the agent
llm: ChatOpenAI = ChatOpenAI(
    model=LLM_MODEL, base_url=LLM_BASE_URL, api_key=OPENAI_API_KEY
)

# In-memory checkpointer for thread-aware conversations
checkpointer = InMemorySaver()


def print_startup_banner() -> None:
    """Print a concise banner of the resolved configuration."""
    print(f"Using LLM model: {LLM_MODEL} at {LLM_BASE_URL}")
