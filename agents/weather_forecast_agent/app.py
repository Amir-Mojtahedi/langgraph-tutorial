"""
Application entrypoint.

This module wires everything together and runs a simple
two-turn conversation with the weather agent. It deliberately
contains minimal logic so the rest of the codebase remains
testable and modular.
"""

# Import the composed agent and data models from local modules
from langchain_core.runnables import RunnableConfig

from agents.weather_forecast_agent.agent_factory import build_agent
from agents.weather_forecast_agent.config import print_startup_banner
from agents.weather_forecast_agent.schemas import Context


def main() -> None:
    """Run a small demo of the weather agent."""
    # Print a concise startup banner with resolved LLM configuration
    print_startup_banner()

    # Build the agent with tools, schema, system prompt, and checkpointer
    agent = build_agent()

    # RunnableConfig controls runtime behavior (e.g., thread_id for checkpointing)
    config: RunnableConfig = {"configurable": {"thread_id": "1"}}

    # Provide per-request runtime context used by tools (e.g., `get_user_city`)
    ctx = Context(user_id="1")

    # First user turn
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "what is the weather outside"}]},
        config=config,
        context=ctx,
    )
    print(response["structured_response"])  # ToolStrategy-structured output

if __name__ == "__main__":
    main()
