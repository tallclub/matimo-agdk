"""AutoGen example: one governed tool, one Gateway-routed model client.

export MATIMO_API_KEY=me-live-...
matimo-agdk register --name autogen-demo --framework autogen
pip install matimo-agdk[autogen] && export MATIMO_GATEWAY_URL=http://localhost:8000/v1
uv run python examples/autogen_agent.py "What's 12 * 7?"
"""

from __future__ import annotations

import asyncio
import sys

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_core.tools import FunctionTool

# --- Matimo AGDK imports (same three roles in every example) ---------------
# 1. Governor: the core object. Holds this agent's registered identity, sends
#    telemetry to Matimo Gateway, and answers "is this tool call allowed?".
#    AutoGen is async-only, so this example uses AsyncGovernor -- the same
#    thing as Governor, but with awaitable start()/stop()/run().
from matimo_agdk import AsyncGovernor

# 2. AutoGen adapter (matimo_agdk.adapters.autogen):
from matimo_agdk.adapters.autogen import (
    gateway_model_client,  # LLM routing: an OpenAIChatCompletionClient pointed at Matimo Gateway
    govern_tools,  # enforcement: policy-checks each tool call before it runs
)


def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression, e.g. '12 * 7'."""
    return str(eval(expression, {"__builtins__": {}}))  # noqa: S307 - demo only


async def main() -> None:
    question = " ".join(sys.argv[1:]) or "What's 12 * 7?"
    # Matimo: load the identity created by `matimo-agdk register` (plus the
    # API key / Gateway URL from env vars), then start background telemetry.
    governor = AsyncGovernor.from_env(agent_name="autogen-demo")
    await governor.start()

    try:
        tool = FunctionTool(calculator, description="Evaluate an arithmetic expression.")
        # Matimo: wraps tool.run() in place; a policy DENY raises ToolDenied,
        # which AutoGen feeds back to the model as an error tool result.
        govern_tools([tool], governor)

        agent = AssistantAgent(
            name="calculator_agent",
            # Matimo: the LLM call goes through Gateway, not straight to OpenAI.
            model_client=gateway_model_client(governor, model="gpt-4o-mini"),
            tools=[tool],
        )
        # Matimo: groups every span below under one run in the Gateway UI.
        async with governor.run("autogen-demo-run"):
            response = await agent.on_messages(
                [TextMessage(content=question, source="user")], CancellationToken()
            )
            print(response.chat_message.content)
    finally:
        # Matimo: flush any queued telemetry and stop the background task.
        await governor.stop()


if __name__ == "__main__":
    asyncio.run(main())
