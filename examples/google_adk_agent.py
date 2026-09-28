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

# --- Matimo AGDK imports (same three roles in every example) ---------------
# 1. Governor: the core object. Holds this agent's registered identity, sends
#    telemetry to Matimo Gateway, and answers "is this tool call allowed?".
#    AsyncGovernor here, not the sync Governor: gateway_model() talks to
#    Gateway via the real openai SDK's AsyncOpenAI, which needs
#    AsyncGovernor.httpx_async_client() for full per-request signing --
#    see gateway_model()'s own docstring.
from matimo_agdk import AsyncGovernor

# 2. Google ADK adapter (matimo_agdk.adapters.google_adk):
from matimo_agdk.adapters.google_adk import (
    MatimoPlugin,  # telemetry + enforcement: an ADK plugin that policy-checks every tool call
    gateway_model,  # LLM routing: a BaseLlm (real openai SDK) already pointed at Matimo Gateway
)

# 3. Typed exception: a policy DENY on the LLM call itself (e.g. a model
#    allow/deny-list rule), an exhausted spend cap, or a rapid suspend all
#    raise this common base -- not the LLM SDK's own unstructured error --
#    see gateway_model()'s own docstring for why that wasn't always true.
from matimo_agdk.exceptions import GatewayError


def get_weather(city: str) -> dict:
    """Look up the current weather for a city."""
    return {"city": city, "forecast": "(pretend) sunny, 28C"}


async def main() -> None:
    question = " ".join(sys.argv[1:]) or "What's the weather in Bengaluru?"
    # Matimo: load the identity created by `matimo-agdk register` (plus the
    # API key / Gateway URL from env vars), then start background telemetry.
    governor = AsyncGovernor.from_env(agent_name="google_adk_agent")
    await governor.start()

    try:
        agent = Agent(
            name="weather_agent",
            # Matimo: the LLM call goes through Gateway, not straight to the provider.
            model=gateway_model(governor, model="gpt-4o-mini"),
            instruction="Answer using the get_weather tool when relevant.",
            tools=[get_weather],
        )
        # Matimo: one line, registered once -- governs every tool call and
        # records every model call this Runner ever makes. A DENY comes back
        # to the agent as the tool's result, so the run keeps going.
        runner = InMemoryRunner(agent=agent, plugins=[MatimoPlugin(governor)])
        session = await runner.session_service.create_session(
            app_name=runner.app_name, user_id="demo-user"
        )
        message = types.Content(role="user", parts=[types.Part(text=question)])
        try:
            async for event in runner.run_async(
                user_id="demo-user", session_id=session.id, new_message=message
            ):
                for part in event.content.parts if event.content else []:
                    if part.text:
                        print(part.text)
        except GatewayError as exc:
            # Matimo: a denied/blocked LLM call -- print the reason and stop
            # gracefully instead of letting a raw SDK traceback crash the
            # agent. Retrying this exact call will simply recur.
            print(f"Matimo Gateway blocked this call: {exc}")
    finally:
        # Matimo: flush any queued telemetry and stop the background task.
        await governor.stop()


if __name__ == "__main__":
    asyncio.run(main())
