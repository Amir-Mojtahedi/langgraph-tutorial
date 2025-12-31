"""
Configuration and LLM setup.

This module centralizes environment parsing, the system prompt, and
construction of shared runtime services (LLM client, checkpointer).
"""

from __future__ import annotations

import os

from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import SecretStr

# Read environment variables for LLM configuration
LOCAL_LLM_MODEL = os.environ.get("LOCAL_LLM_MODEL", "")
LOCAL_LLM_PROVIDER = os.environ.get("LOCAL_LLM_PROVIDER", "")
LOCAL_LLM_URL = os.environ.get("LOCAL_LLM_URL", "")
API_KEY = SecretStr(os.environ.get("API_KEY", ""))

# A playful system prompt guiding the agent's persona and tool usage
SYSTEM_PROMPT = SystemMessage(
    content="""You are an expert weather forecaster, who speaks in puns.

You have access to three tools:

- get_coordinates_for_city: use this to get the latitude and longitude for a specific city
- get_weather_for_city: use this to get the weather for a specific location
- get_user_city: use this to get the user's location

If a user asks you for the weather, make sure you know the location. If you can tell from the question that they mean wherever they are, use the get_user_city tool to find their location."""
)

# Construct the LLM client used by the agent
local_model: BaseChatModel = init_chat_model(
    model=LOCAL_LLM_MODEL,
    model_provider=LOCAL_LLM_PROVIDER,
    base_url=LOCAL_LLM_URL,
    api_key=API_KEY,
)

# In-memory checkpointer for thread-aware conversations
checkpointer = InMemorySaver()


def print_startup_banner() -> None:
    """Print a concise banner of the resolved configuration."""
    print(f"Using LLM model: {LOCAL_LLM_MODEL} at {LOCAL_LLM_URL}")
