# Matimo AGDK User Manual

Matimo AGDK (Agent Governance Development Kit) is the Python SDK for running an AI agent you already have, built on any framework, under Matimo Gateway governance: a cryptographic identity per agent, every LLM call routed through Gateway, execution telemetry with a heartbeat, and a policy check on every tool call with human approval when a policy asks for it.

This manual is for the developer wiring an agent up. It assumes a Matimo Workbench tenant with an active Matimo Enterprise license and an admin who can create API keys and policies.

Version covered: matimo-agdk 0.1.0, Python 3.11 or newer.

## 1. How it fits together

| Piece | Where it runs | What it does |
|---|---|---|
| Your agent | Your process | LangChain, Google ADK, CrewAI, AutoGen, or plain Python. Unchanged, except for a few lines from this SDK. |
| Matimo AGDK | Your process | Holds the agent's identity, signs requests, ships telemetry, checks tools before they run. |
| Matimo Gateway | Workbench, `/v1` | Proxies LLM calls to the tenant's own provider keys, applies Guardrails and the Policy Engine, writes the call log. |
| Governance UI | Workbench admin | Where an admin registers keys, writes policies, approves pending tool calls, and suspends an agent. |

Three facts shape everything below:

- **No registration, no LLM.** An agent gets model access through Gateway only after it has an identity and a session. Registration is the enforcement lever.
- **AGDK never holds a provider key.** OpenAI or Anthropic credentials stay in Workbench (BYOK). Your agent sends its Matimo org API key and its own signature; Gateway resolves the provider server-side.
- **Tools run in your process.** `guard()` decides whether a call may happen and records what happened. It never executes the tool itself.

## 2. Install

```bash
pip install matimo-agdk                  # core: plain Python, OpenAI and Anthropic SDK routing
pip install "matimo-agdk[langchain]"     # plus the LangChain adapter
pip install "matimo-agdk[google-adk]"    # plus Google ADK (pulls litellm)
pip install "matimo-agdk[crewai]"
pip install "matimo-agdk[autogen]"       # AutoGen 0.7 (autogen-core, agentchat, ext)
pip install "matimo-agdk[all]"
```

Until the package is on PyPI, install from a checkout of the repository: `pip install -e ".[all]"` or `uv sync --all-extras`.

## 3. Before you start: what the admin gives you

1. **Gateway URL.** The Workbench backend plus `/v1`, for example `https://workbench.matimo.ai/v1` or `http://localhost:8000/v1` for local development.
2. **An org API key** created under Governance in Workbench with three scopes. A key missing any of them registers fine and then fails later with `insufficient_scope`.

| Scope | Needed for |
|---|---|
| `identity:manage` | `matimo register`, `rotate-key`, JWKS |
| `gateway:proxy` | sessions, LLM calls, telemetry |
| `agdk:check` | tool checks (`guard()`, adapters in govern mode) |

3. **A model your tenant allows**, for example `gpt-4o-mini`. Gateway rejects a model that is not configured for the tenant or allowed for the identity (`model_not_allowed`).

## 4. Register the agent

Registration happens once per agent, from the machine that will run it.

```bash
export MATIMO_GATEWAY_URL=http://localhost:8000/v1
export MATIMO_API_KEY=me-live-...
matimo-agdk register --name support-bot --framework langchain
```

`--framework` is one of `langchain`, `google-adk`, `crewai`, `autogen`, `custom`. Add `--tool-category web --tool-category crm` to declare the categories the agent may use; an admin can tighten these later.

The server generates an ECDSA P-256 keypair and returns the private key exactly once. The CLI stores it as two files, readable only by you, and never prints it:

```
~/.matimo/agents/support-bot.json   identity id, token, tenant, framework
~/.matimo/agents/support-bot.pem    private key, mode 0600
```

Windows cannot enforce the 0600 mode; keep the folder inside a user profile that other accounts cannot read.

Check the setup:

```bash
matimo-agdk doctor --name support-bot
```

`doctor` performs a real signed session handshake and a heartbeat, then prints the lifecycle status. If it passes, every code path below will reach Gateway.

## 5. Quickstart in plain Python

```python
from matimo_agdk import Governor

governor = Governor.from_env(agent_name="support-bot")
governor.start()                       # session handshake, telemetry exporter, heartbeat

def search(query: str) -> str:
    return my_search_backend(query)

try:
    with governor.run("answer-ticket"):
        # 1. A governed tool call. Raises ToolDenied if policy says no,
        #    blocks while an admin decides if policy says "approve first".
        hits = governor.guard(search, name="search", category="web")(query="refund policy")

        # 2. An LLM call through Gateway with the OpenAI SDK.
        from openai import OpenAI
        client = OpenAI(**governor.openai_client_kwargs(), http_client=governor.httpx_client())
        reply = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Summarise: {hits}"}],
        )
        governor.llm_span(model="gpt-4o-mini", provider="openai", status="completed")
finally:
    governor.stop()                    # flushes telemetry, closes the session
```

`Governor` also works as a context manager (`with Governor.from_env(...) as governor:`), and `AsyncGovernor` offers the same API with `await` on the network methods and `async with governor.run(...)`.

## 6. Configuration

Precedence, highest first: keyword arguments to `Governor(...)` or `Governor.from_env(...)`, then environment variables, then the credentials file written by `matimo register`.

| Environment variable | Purpose | Default |
|---|---|---|
| `MATIMO_GATEWAY_URL` | Gateway base URL | `http://localhost:8000/v1` |
| `MATIMO_API_KEY` | Org API key, sent as `Authorization: Bearer` | required |
| `MATIMO_AGENT_NAME` | Which credentials file to load, and the default display name | `matimo-agent` |
| `MATIMO_FRAMEWORK` | `langchain`, `google-adk`, `crewai`, `autogen`, `custom` | `custom` |
| `MATIMO_IDENTITY_TOKEN`, `MATIMO_IDENTITY_ID`, `MATIMO_TENANT_ID` | Identity without a credentials file (containers, CI) | from file |
| `MATIMO_PRIVATE_KEY` or `MATIMO_PRIVATE_KEY_FILE` | Private key PEM inline or by path | from file |

`GatewayConfig` fields you may set in code: `connect_timeout` (10 s), `read_timeout` (30 s), `telemetry_flush_interval` (5 s), `telemetry_batch_size` (50), `telemetry_queue_max` (2000), `heartbeat_interval` (derived from the server's staleness window, clamped to 15 s to 5 min), `fail_open_telemetry` (True: a telemetry outage never blocks the agent), `signing_enabled` (True), `credentials_dir`.

For a container, mount nothing: set `MATIMO_IDENTITY_TOKEN`, `MATIMO_IDENTITY_ID`, `MATIMO_TENANT_ID` and `MATIMO_PRIVATE_KEY` from your secret store.

## 7. Concepts

### Identity and signing

Every request that matters (session handshake, LLM calls, tool checks) carries `Matimo-Agent-Signature`, a compact ES256 JWS over a hash of the exact request bytes, signed with the agent's private key. Gateway verifies it against the public key it holds and publishes at `GET /v1/identities/:id/jwks`. Signing is always on in the SDK; an admin decides per identity whether Gateway requires it (`requireSignedRequests`). Telemetry is not signed by design.

### Session

`start()` performs `POST /v1/sessions` and keeps the session token in memory. Gateway requires it on every LLM and telemetry call. If the session expires or is deleted server-side, the SDK re-handshakes once and retries; you see nothing unless the second attempt fails (`SessionExpired`).

### Runs and spans

`governor.run(name)` opens a run id that the SDK attaches to LLM calls (`X-Matimo-Run-Id`), tool checks, and every span recorded inside it. `llm_span(...)` and `tool_span(...)` record what happened using OpenTelemetry GenAI attribute names (`gen_ai.request.model`, `gen_ai.tool.name`, token counts). Adapters record these for you.

### Telemetry and heartbeat

Spans are queued and flushed in batches to `POST /v1/telemetry/batch` every `telemetry_flush_interval` seconds. Every response carries a heartbeat: lifecycle status, emergency-stop flag, telemetry mode, staleness window, server time. An empty batch is a pure heartbeat and is sent even when the agent is idle.

Telemetry is the only signal Workbench has into tool execution it does not run. A tenant may set telemetry mode to `deny`: an agent whose last telemetry is older than the staleness window is refused at Gateway with `policy_denied` reason `telemetry_stale`. The fix is always the same: keep the exporter running (`start()` before work, `stop()` after).

### Rapid suspend

An admin can suspend or revoke an identity, or trip an org-wide emergency stop. Two things happen on two timescales:

- The next call Gateway receives from that agent is denied immediately, server-side, from a fresh database read.
- `governor.state`, `is_suspended()`, `raise_if_suspended()` and `on_suspend(callback)` reflect the last heartbeat. They can lag by up to one heartbeat interval. A loop that never calls Gateway and never checks state could keep running locally for that long.

Call `raise_if_suspended()` at the top of long loops, or register `on_suspend` to cancel work. There is no push channel in this version.

### Tool checks and human approval

`check_tool(name, args)` posts to `POST /v1/tools/check` and returns a `ToolDecision`:

| `decision` | Meaning | What `guard()` does |
|---|---|---|
| `ALLOW` | Policy permits the call | runs the function |
| `DENY` | Policy forbids it; `reason` says why | raises `ToolDenied`, function never runs |
| `PENDING` | A policy requires approval; `resume_token` identifies the request | polls `POST /v1/tools/check/status` until an admin approves (runs) or rejects (raises `ToolDenied`) |

Only the category Gateway resolves is trusted. `set_tool_category(name, category)` and the `category=` hint let you tell the server how to classify a tool; an admin can override it. After the function returns, `guard()` reports the outcome through `POST /v1/tools/result` and records a tool span.

## 8. Governor reference

| Method | Purpose |
|---|---|
| `Governor.from_env(**overrides)` | Build from env plus credentials file |
| `register(display_name=, framework=, allowed_tool_categories=, persist=True)` | Create an identity in code instead of the CLI |
| `start()` / `stop()` | Handshake and exporter lifecycle; also a context manager |
| `run(name)` | Context manager yielding a run id |
| `llm_span(**attrs)` / `tool_span(tool_name, **attrs)` | Record telemetry |
| `guard(fn, name=, category=)` | Policy-checked wrapper, usable as a decorator |
| `check_tool(name, args)` / `await_decision(resume_token)` | Manual tool check and wait |
| `set_tool_category(name, category)` | Classify a tool server-side |
| `httpx_client()` | Signed, session-aware `httpx.Client` for any OpenAI-compatible SDK |
| `openai_client_kwargs()` / `anthropic_client_kwargs()` | `base_url`, `api_key`, `default_headers` for those SDKs; pass `http_client=governor.httpx_client()` too |
| `request_headers(body=None)` | Live headers (and signature over `body`) for a client you build yourself |
| `bind_run_id(run_id)` | Attach a framework-owned run id without opening a span |
| `state` / `is_suspended()` / `raise_if_suspended()` / `on_suspend(cb)` | Heartbeat-driven governance state |
| `rotate_key()` | New keypair; the old key stops verifying immediately |
| `identity` | The bound `IdentityCredentials` |

`AsyncGovernor` mirrors this with `await` on `register`, `start`, `stop`, `check_tool`, `await_decision`, `set_tool_category`, `rotate_key`, `request_headers`, and `httpx_async_client()`.

## 9. Framework adapters

Each adapter has two halves: an observe half that records telemetry, and a govern half that checks tools. `mode="observe"` records only; `mode="govern"` (default) also enforces.

### LangChain

```python
from matimo_agdk.adapters.langchain import MatimoCallbackHandler, gateway_chat_model, govern_tools

tools = govern_tools([calculator, search], governor)          # policy check before each tool body
model = gateway_chat_model(governor, model="gpt-4o-mini").bind_tools(tools)
result = model.invoke(messages, config={"callbacks": [MatimoCallbackHandler(governor)]})
```

`gateway_chat_model(provider="openai")` signs every request. `provider="anthropic"` sends the session header only. A DENY raises LangChain's `ToolException`; set `handle_tool_error=True` on the tool to turn it into an observation string the agent can reason about. `AsyncMatimoCallbackHandler` exists for async chains.

### Google ADK

```python
from matimo_agdk.adapters.google_adk import MatimoPlugin, gateway_model

agent = Agent(name="weather", model=gateway_model(governor, model="gpt-4o-mini"), tools=[get_weather])
runner = InMemoryRunner(agent=agent, plugins=[MatimoPlugin(governor)])
```

One plugin governs every model and tool call the runner makes. A DENY is returned through ADK's own before-tool short-circuit, so the agent sees a structured refusal rather than a crash. LLM calls carry the live session token and the ADK invocation id as the run id; they are not signed per request (litellm builds the body after the hook), so keep `requireSignedRequests` off for ADK identities or route through a custom `BaseLlm` built on `governor.httpx_client()`.

### CrewAI

```python
from matimo_agdk.adapters.crewai import gateway_llm, govern_crew

researcher = Agent(role="Researcher", tools=[search], llm=gateway_llm(governor, model="gpt-4o-mini"))
crew = Crew(agents=[researcher], tasks=[task])
govern_crew(crew, governor)       # wraps every tool reachable from the crew
with governor.run("research"):
    crew.kickoff()
```

`gateway_llm()` installs a transport interceptor, so every CrewAI LLM request carries the live session token, the run id, and a per-request signature, and emits one LLM span per call (duration, status, model, token usage when the response isn't streamed) alongside the tool spans `govern_tool()`/`govern_crew()` already record — matching LangChain's/ADK's telemetry richness. `govern_tool(tool, governor)` governs a single tool. CrewAI exposes no per-`kickoff()` id, so `with governor.run(...):` (shown above) is what makes the LLM and tool spans of one crew execution share a run id in the Observability Hub timeline; without it, each call gets its own uncorrelated id.

### AutoGen 0.7

```python
from matimo_agdk.adapters.autogen import gateway_model_client, govern_tools

tool = FunctionTool(calculator, description="Evaluate arithmetic.")
govern_tools([tool], governor)
agent = AssistantAgent("calc", model_client=gateway_model_client(governor, model="gpt-4o-mini"), tools=[tool])
async with governor.run("calc-run"):
    await agent.on_messages([TextMessage(content=question, source="user")], CancellationToken())
```

Full per-request signing. Works with `Governor` (bridged through a thread) or `AsyncGovernor`. `gateway_model_client()` also wraps the model client's `create()`/`create_stream()` to emit one LLM span per call (duration, status, model, token usage/finish reason from AutoGen's own typed `CreateResult`), matching LangChain's/ADK's telemetry richness. AutoGen exposes no per-chat id to this wrapper, so `async with governor.run(...):` (shown above) is what makes the LLM and tool spans of one chat share a run id in the Observability Hub timeline; without it, each call gets its own uncorrelated id. Legacy AutoGen 0.2 (`pyautogen`) is not supported; use the generic adapter.

### Anything else

```python
from matimo_agdk.adapters.generic import govern

tools = govern({"search": search, "send_email": send_email}, governor, category="crm")
```

Takes a callable, a list, or a dict, and returns the same shape with each callable wrapped by `guard()`. This adapter only ever sees tool callables, never an LLM client, so it emits no LLM spans on its own -- call `governor.llm_span(...)` around your own LLM call site (section 5 above) if you want that call to show up alongside your `govern()`-wrapped tool calls in the Observability Hub.

## 10. CLI reference

```
matimo-agdk register   --name NAME --framework {langchain,google-adk,crewai,autogen,custom}
                  [--tool-category CAT ...] [--gateway-url URL] [--api-key KEY]
matimo-agdk status     [--name NAME]      print the current GovernanceState from a live heartbeat
matimo-agdk rotate-key [--name NAME]      new keypair, credentials file rewritten
matimo-agdk doctor     [--name NAME]      connectivity, handshake, signing, heartbeat
```

`--name` defaults to `MATIMO_AGENT_NAME`. `--gateway-url` and `--api-key` fall back to the environment variables.

## 11. Errors and what to do

| You see | Meaning | Fix |
|---|---|---|
| `GatewayError: This API key is missing the required scope: agdk:check` | Key lacks a scope | Ask the admin for a key with all three scopes (section 3) |
| `PolicyDenied` with `reason="telemetry_stale"` (`TelemetryStale`) | Tenant is in deny mode and your last telemetry is too old | Call `start()` before work; do not block the exporter thread |
| `AgentSuspended` (`agent_suspended`, `agent_revoked`, `emergency_stop_active`) | Admin action | Stop the agent; contact the admin |
| `AgentSuspendedLocally` | Raised by `raise_if_suspended()` from the last heartbeat | Same |
| `ToolDenied` | Policy said no, or an approval was rejected; `reason` explains | Handle in the agent loop; do not retry blindly |
| `ToolCheckTimeout` | A PENDING decision was not made in time | Surface to a human; retry creates a new request |
| `SessionExpired` | Re-handshake also failed | Check the key and the identity's lifecycle status |
| `SignatureRejected` | Gateway could not verify the signature | Rotate the key if the file was overwritten; check the clock |
| `RateLimited` | Per-identity or per-tenant rate limit | Back off; the exception carries `retry_after` when the server sends it |
| `GatewayUnavailable` | Network or 5xx | Retry with backoff; LLM calls fail closed, telemetry fails open |
| HTTP 400 `no_default_connection` | LLM call without `model` and the tenant has no default connection | Always pass `model=` |
| HTTP 403 `model_not_allowed` | Model not configured for the tenant or not allowed for the identity | Use a model the admin configured |
| HTTP 502 `upstream_error` | Every provider target failed | The tenant's provider key is wrong or the provider is down |
| Response with `finish_reason: "content_filter"` | A Guardrail blocked the prompt or the answer | HTTP 200; the content is the guardrail's configured message |

## 12. Operating an agent

- **Rotate keys** on a schedule or after any suspected exposure: `matimo rotate-key --name NAME`. The old key stops verifying the moment the server accepts the new one; running processes must restart to load it.
- **Watch the call log** in Workbench (Governance, Gateway, Activity) to see each call's policy outcome, guardrail outcome, model, run id, and Gateway overhead.
- **Suspend from the admin UI** when something looks wrong. The next Gateway call is refused immediately; the agent's own state follows within one heartbeat.
- **Multiple processes, one agent**: copy the credentials or pass them by environment; sessions are per process and that is fine.
- **Do not share one identity across different agents.** Policies, telemetry, and the audit trail are per identity.

## 13. Verifying an installation end to end

`scripts/live_check.py` in the repository runs 15 scenarios against a real Gateway: registration, doctor, heartbeat, telemetry masking, signature enforcement, a real signed completion, rapid suspend, tool checks in all four outcomes, the LangChain adapter, session expiry, key rotation, and the call log. It needs `MATIMO_API_KEY`, `MATIMO_TENANT_ID`, and a tenant-admin `MATIMO_ADMIN_TOKEN` from the same tenant, and optionally `MATIMO_GATEWAY_MODEL`. It never creates accounts, keys, licenses, or providers.

```bash
uv run --group dev python scripts/live_check.py
uv run --group dev python scripts/live_check.py --filter "tool check"
```

## 14. What AGDK does not do

- It does not hold or forward your LLM provider keys.
- It does not execute tools; it decides and records.
- It does not push a kill signal; suspension is enforced at Gateway and polled by the SDK.
- It does not sign telemetry.
- It does not support AutoGen 0.2, or multimodal message content parts through Gateway yet.
