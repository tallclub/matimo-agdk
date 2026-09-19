"""Google ADK example: one governed tool, one Gateway-routed model, via
MatimoPlugin registered once on the Runner.

    export MATIMO_API_KEY=me-live-...
    matimo-agdk register --name google-adk-demo --framework google-adk
    pip install matimo-agdk[google-adk]
    export MATIMO_GATEWAY_URL=http://localhost:8000/v1
    uv run python examples/google_adk_agent.py "What's the weather in Bengaluru?"
"""

from __future__ import annotations

import asyncio
import sys

from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types

from matimo_agdk import Governor
from matimo_agdk.adapters.google_adk import MatimoPlugin, gateway_model


def get_weather(city: str) -> dict:
    """Look up the current weather for a city."""
    return {"city": city, "forecast": "(pretend) sunny, 28C"}


async def main() -> None:
    question = " ".join(sys.argv[1:]) or "What's the weather in Bengaluru?"
    governor = Governor.from_env(agent_name="google-adk-demo")
    governor.start()

    try:
        agent = Agent(
            name="weather_agent",
            model=gateway_model(governor, model="gpt-4o-mini"),
            instruction="Answer using the get_weather tool when relevant.",
            tools=[get_weather],
        )
        # One line: registered once, governs every tool/model call this
        # Runner ever makes.
        runner = InMemoryRunner(agent=agent, plugins=[MatimoPlugin(governor)])
        session = await runner.session_service.create_session(
            app_name=runner.app_name, user_id="demo-user"
        )
        message = types.Content(role="user", parts=[types.Part(text=question)])
        async for event in runner.run_async(
            user_id="demo-user", session_id=session.id, new_message=message
        ):
            for part in event.content.parts if event.content else []:
                if part.text:
                    print(part.text)
    finally:
        governor.stop()


if __name__ == "__main__":
    asyncio.run(main())
