"""CrewAI example: one governed tool, one Gateway-routed LLM, via
govern_crew() registered once after the crew is built.

    export MATIMO_API_KEY=me-live-... && matimo-agdk register --name crewai-demo --framework crewai
    pip install matimo-agdk[crewai] && export MATIMO_GATEWAY_URL=http://localhost:8000/v1
    uv run python examples/crewai_crew.py "Research the capital of India"
"""

from __future__ import annotations

import sys

from crewai import Agent, Crew, Task
from crewai.tools import tool

# --- Matimo AGDK imports (same three roles in every example) ---------------
# 1. Governor: the core object. Holds this agent's registered identity, sends
#    telemetry to Matimo Gateway, and answers "is this tool call allowed?".
from matimo_agdk import Governor

# 2. CrewAI adapter (matimo_agdk.adapters.crewai):
from matimo_agdk.adapters.crewai import (
    gateway_llm,  # LLM routing: a crewai LLM already pointed at Matimo Gateway
    govern_crew,  # enforcement: policy-checks every tool reachable from the crew
)


@tool("search")
def search(query: str) -> str:
    """Search the web for information."""
    return f"(pretend) search results for: {query}"


def main() -> None:
    mission = " ".join(sys.argv[1:]) or "Research the capital of India"
    # Matimo: load the identity created by `matimo-agdk register` (plus the
    # API key / Gateway URL from env vars), then start background telemetry.
    governor = Governor.from_env(agent_name="crewai-demo")
    governor.start()

    try:
        researcher = Agent(
            role="Researcher",
            goal="Answer the user's question using the search tool.",
            backstory="A careful research assistant.",
            tools=[search],
            # Matimo: the LLM call goes through Gateway, not straight to OpenAI.
            llm=gateway_llm(governor, model="gpt-4o-mini"),
        )
        task = Task(description=mission, expected_output="A short answer.", agent=researcher)
        crew = Crew(agents=[researcher], tasks=[task])
        # Matimo: wraps every tool reachable from the crew in place; a policy
        # DENY raises ToolDenied, which CrewAI feeds back to the agent.
        govern_crew(crew, governor)

        # Matimo: groups every span below under one run in the Gateway UI.
        with governor.run("crewai-demo-run"):
            print(crew.kickoff())
    finally:
        # Matimo: flush any queued telemetry and stop the background thread.
        governor.stop()


if __name__ == "__main__":
    main()
