"""Plain-Python example: governor.guard() around a normal function, and an
OpenAI call routed through Matimo Gateway via governor.httpx_client().

Run once first:

    export MATIMO_API_KEY=me-live-...
    matimo-agdk register --name plain-python-demo --framework custom

Then:

    export MATIMO_GATEWAY_URL=http://localhost:8000/v1
    uv run python examples/plain_python.py "What is the capital of India?"
"""

from __future__ import annotations

import sys

# --- Matimo AGDK imports (same three roles in every example) ---------------
# 1. Governor: the core object. Holds this agent's registered identity, sends
#    telemetry to Matimo Gateway, and answers "is this tool call allowed?".
from matimo_agdk import Governor

# 2. No framework adapter here: with plain Python there is nothing to hook, so
#    the Governor is used directly. governor.guard() does the enforcement and
#    governor.httpx_client() does the LLM routing (see the calls below).
#    matimo_agdk.adapters.generic.govern() is the shortcut for guarding a whole
#    dict/list of tool functions at once.


def search(query: str) -> str:
    """A stand-in for a real tool -- governor.guard() never executes this
    itself, it only checks policy and records telemetry around it. The
    actual call always runs in this process."""
    return f"(pretend) search results for: {query}"


def main() -> None:
    mission = " ".join(sys.argv[1:]) or "What is the capital of India?"

    # Matimo: load the identity created by `matimo-agdk register` (plus the
    # API key / Gateway URL from env vars), then start background telemetry.
    governor = Governor.from_env(agent_name="plain-python-demo")
    governor.start()

    try:
        # Matimo: groups every span below under one run in the Gateway UI.
        with governor.run("plain-python-example"):
            # 1. Matimo enforcement -- a governed tool call: policy-checked,
            #    span-recorded, result reported. Raises ToolDenied if the
            #    policy says no.
            result = governor.guard(search, name="search", category="web")(query=mission)
            print(f"Tool result: {result}")

            # 2. Matimo LLM routing -- an LLM call routed through Gateway, not the real OpenAI
            #    API -- Gateway resolves the tenant's own BYOK credentials
            #    server-side, this process never sees them. Requires the
            #    `openai` package (not a matimo_agdk dependency):
            #    pip install openai
            try:
                import openai
            except ImportError:
                print("Install `openai` to also exercise the LLM call: pip install openai")
                return

            client = openai.OpenAI(
                base_url=governor.config.base_url,
                api_key=governor.config.api_key,
                # governor.httpx_client() signs every request with the
                # identity's private key and attaches the current session
                # token -- default_headers alone cannot do this, since the
                # signature covers each request's own body bytes.
                http_client=governor.httpx_client(),
            )
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": mission}],
            )
            answer = response.choices[0].message.content
            print(f"LLM answer: {answer}")

            # Matimo telemetry -- with no framework adapter to record LLM
            # calls automatically, report this one by hand.
            governor.llm_span(
                model="gpt-4o-mini",
                provider="openai",
                finish_reasons=[response.choices[0].finish_reason],
            )
    finally:
        # Matimo: flush any queued telemetry and stop the background thread.
        governor.stop()


if __name__ == "__main__":
    main()
