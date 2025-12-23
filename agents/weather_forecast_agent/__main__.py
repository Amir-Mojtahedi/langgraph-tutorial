"""
Module entrypoint for `python -m apps.weather_forecast`.

This imports and runs `main()` from the local `app.py` so you can
start the application in module mode without referencing the file path.
"""

from agents.weather_forecast_agent.app import main

if __name__ == "__main__":
    main()
