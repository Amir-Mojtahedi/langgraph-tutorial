"""
Tool definitions invoked by the agent.

Tools are small, deterministic functions the agent can call to fetch or
compute information. They may consume the runtime Context when needed.
"""

from __future__ import annotations

import os

import requests
from langchain.tools import ToolRuntime, tool

from agents.weather_forecast_agent.schemas import Context

WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY")
if not WEATHER_API_KEY:
    raise ValueError("WEATHER_API_KEY is not set in environment variables.")


@tool
def get_coordinates_for_city(city: str) -> dict[str, float]:
    """Resolve a city's geographic coordinates for downstream weather queries.

    Use when:
    - The user mentions a city by name (e.g., "weather in Paris").
    - You need precise latitude/longitude to call weather APIs.

    Input:
    - city: A single city name string (e.g., "San Francisco"). If ambiguous,
        prefer asking the user to clarify rather than guessing.

    Output:
    - dict with floats: {"lat": <latitude>, "lon": <longitude>}.

    Behavior:
    - Queries the OpenWeather Geocoding API and returns the first match.
    - Raises ValueError if the city cannot be resolved.
    """
    url = "http://api.openweathermap.org/geo/1.0/direct"
    response = requests.get(
        url=url,
        params={"q": city, "limit": 1, "appid": WEATHER_API_KEY},
        headers={"Content-Type": "application/json"},
    )
    data = response.json()
    if data:
        coords: dict[str, float] = {"lat": data[0]["lat"], "lon": data[0]["lon"]}
        return coords

    raise ValueError(f"Could not find coordinates for city: {city}")


@tool
def get_weather_for_city(city_latitude: float, city_longitude: float) -> str:
    """Fetch current weather conditions for the given coordinates.

    Use when:
    - You already obtained precise coordinates (via `get_coordinates_for_city`).
    - The user asks for the current weather conditions.

    Input:
    - city_latitude: float representing the latitude of the city.
    - city_longitude: float representing the longitude of the city.

    Output:
    - A concise, human-readable sentence including the city name and conditions.

    Behavior:
    - Calls the OpenWeather current weather endpoint with the provided coords.
    - Expects valid floats for lat/lon; may error if keys are missing/invalid.
    """
    url = "https://api.openweathermap.org/data/2.5/weather"
    response = requests.get(
        url,
        params={
            "lat": city_latitude,
            "lon": city_longitude,
            "appid": WEATHER_API_KEY,
        },
    )
    data = response.json()
    weather: dict[str, str] = {
        "city": data["name"],
        "weather_conditions": data["weather"][0]["description"],
    }
    return f"The weather in {weather['city']} is currently {weather['weather_conditions']}."


@tool
def get_user_city(runtime: ToolRuntime[Context]) -> str:
    """Infer the user's default city from the runtime `Context`.

    Use when:
    - The user did not specify a city and a personalized default is helpful.

    Input:
    - runtime: `ToolRuntime[Context]`, where `Context.user_id` is a non-empty string.

    Output:
    - A city name string to use for follow-up queries.

    Behavior:
    - Returns "Miami" for `user_id == "1"`; otherwise returns "San Francisco".
    - Prefer an explicit city mentioned in the user's prompt over this heuristic.
    """
    user_id = runtime.context.user_id
    return "Miami" if user_id == "1" else "San Francisco"
