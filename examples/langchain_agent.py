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

# --- Matimo AGDK imports (same three roles in every example) ---------------
# 1. Governor: the core object. Holds this agent's registered identity, sends
#    telemetry to Matimo Gateway, and answers "is this tool call allowed?".
from matimo_agdk import Governor

# 2. LangChain adapter (matimo_agdk.adapters.langchain):
from matimo_agdk.adapters.langchain import (
    MatimoCallbackHandler,  # telemetry: records run/LLM/tool spans from LangChain's callbacks
    gateway_chat_model,  # LLM routing: a ChatOpenAI already pointed at Matimo Gateway
    govern_tools,  # enforcement: policy-checks each tool call before it runs
)


@tool
def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression, e.g. '12 * 7'."""
    return str(eval(expression, {"__builtins__": {}}))  # noqa: S307 - demo only


def main() -> None:
    question = " ".join(sys.argv[1:]) or "What's 12 * 7?"
    # Matimo: load the identity created by `matimo-agdk register` (plus the
    # API key / Gateway URL from env vars), then start background telemetry.
    governor = Governor.from_env(agent_name="langchain-demo")
    governor.start()

    try:
        # Matimo: telemetry via the callback handler, enforcement via
        # govern_tools() -- see matimo_agdk.adapters.langchain's docstring
        # for why both are needed.
        handler = MatimoCallbackHandler(governor, mode="govern")
        # Matimo: wraps the tools in place; a policy DENY now raises
        # ToolException instead of running the tool.
        tools = govern_tools([calculator], governor)
        # Matimo: the LLM call goes through Gateway, not straight to OpenAI.
        model = gateway_chat_model(governor, model="gpt-4o-mini").bind_tools(tools)

        # Matimo: groups every span below under one run in the Gateway UI.
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
        # Matimo: flush any queued telemetry and stop the background thread.
        governor.stop()


if __name__ == "__main__":
    main()
