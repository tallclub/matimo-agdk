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

from matimo_agdk import AsyncGovernor
from matimo_agdk.adapters.autogen import gateway_model_client, govern_tools


def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression, e.g. '12 * 7'."""
    return str(eval(expression, {"__builtins__": {}}))  # noqa: S307 - demo only


async def main() -> None:
    question = " ".join(sys.argv[1:]) or "What's 12 * 7?"
    governor = AsyncGovernor.from_env(agent_name="autogen-demo")
    await governor.start()

    try:
        tool = FunctionTool(calculator, description="Evaluate an arithmetic expression.")
        govern_tools([tool], governor)  # wraps tool.run() in place

        agent = AssistantAgent(
            name="calculator_agent",
            model_client=gateway_model_client(governor, model="gpt-4o-mini"),
            tools=[tool],
        )
        async with governor.run("autogen-demo-run"):
            response = await agent.on_messages(
                [TextMessage(content=question, source="user")], CancellationToken()
            )
            print(response.chat_message.content)
    finally:
        await governor.stop()


if __name__ == "__main__":
    asyncio.run(main())
