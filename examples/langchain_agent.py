"""LangChain example: one governed tool, one Gateway-routed chat model.

export MATIMO_API_KEY=me-live-...
matimo-agdk register --name langchain-demo --framework langchain
pip install matimo-agdk[langchain]
export MATIMO_GATEWAY_URL=http://localhost:8000/v1
uv run python examples/langchain_agent.py "What's 12 * 7?"
"""

from __future__ import annotations

import sys

from langchain_core.tools import tool

from matimo_agdk import Governor
from matimo_agdk.adapters.langchain import MatimoCallbackHandler, gateway_chat_model, govern_tools


@tool
def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression, e.g. '12 * 7'."""
    return str(eval(expression, {"__builtins__": {}}))  # noqa: S307 - demo only


def main() -> None:
    question = " ".join(sys.argv[1:]) or "What's 12 * 7?"
    governor = Governor.from_env(agent_name="langchain-demo")
    governor.start()

    try:
        # Telemetry via the callback handler, enforcement via
        # govern_tools() -- see matimo_agdk.adapters.langchain's docstring
        # for why both are needed.
        handler = MatimoCallbackHandler(governor, mode="govern")
        tools = govern_tools([calculator], governor)
        model = gateway_chat_model(governor, model="gpt-4o-mini").bind_tools(tools)

        with governor.run("langchain-demo-run"):
            response = model.invoke(question, config={"callbacks": [handler]})
            print(f"Model response: {response.content!r}")
            if response.tool_calls:
                call = response.tool_calls[0]
                tool_by_name = {t.name: t for t in tools}
                result = tool_by_name[call["name"]].invoke(
                    call["args"], config={"callbacks": [handler]}
                )
                print(f"Tool result: {result}")
    finally:
        governor.stop()


if __name__ == "__main__":
    main()
