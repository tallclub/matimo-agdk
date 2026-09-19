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

from matimo_agdk import Governor
from matimo_agdk.adapters.crewai import gateway_llm, govern_crew


@tool("search")
def search(query: str) -> str:
    """Search the web for information."""
    return f"(pretend) search results for: {query}"


def main() -> None:
    mission = " ".join(sys.argv[1:]) or "Research the capital of India"
    governor = Governor.from_env(agent_name="crewai-demo")
    governor.start()

    try:
        researcher = Agent(
            role="Researcher",
            goal="Answer the user's question using the search tool.",
            backstory="A careful research assistant.",
            tools=[search],
            llm=gateway_llm(governor, model="gpt-4o-mini"),
        )
        task = Task(description=mission, expected_output="A short answer.", agent=researcher)
        crew = Crew(agents=[researcher], tasks=[task])
        govern_crew(crew, governor)  # wraps every tool reachable from the crew

        with governor.run("crewai-demo-run"):
            print(crew.kickoff())
    finally:
        governor.stop()


if __name__ == "__main__":
    main()
