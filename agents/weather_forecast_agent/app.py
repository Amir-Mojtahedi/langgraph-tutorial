"""
Application entrypoint.

This module wires everything together and runs a simple
two-turn conversation with the weather agent. It deliberately
contains minimal logic so the rest of the codebase remains
testable and modular.
"""

# Import the composed agent and data models from local modules
from langchain.messages import AIMessage, HumanMessage
from langchain_core.messages import BaseMessage
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
    messages: list[BaseMessage] = []

    print("\nEnter your question (or 'exit' to quit):")
    while True:
        user_prompt = input("> ")

        if user_prompt.lower() == "exit":
            print("Exiting the weather forecast agent. Goodbye!")
            break

        messages.append(HumanMessage(content=user_prompt))

        response = agent.invoke(
            {"messages": messages},  # type: ignore
            config=config,
            context=ctx,
        )
        sr = response["structured_response"]

        content = (
            sr.weather_conditions
            if sr.weather_conditions is not None
            else sr.punny_response
        )

        print(sr)  # ToolStrategy-structured output
        messages.append(AIMessage(content=content))


if __name__ == "__main__":
    main()
