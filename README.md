# Matimo AGDK

Matimo Agent Governance Development Kit. Govern any agent in three lines:

```python
from matimo_agdk import Governor

governor = Governor.from_env()
governor.start()

with governor.run("my-run"):
    result = governor.guard(my_tool_fn, name="search")(query="roaiq")
```

Matimo AGDK is the Python SDK half of Matimo Gateway (the other half is a
zero-code LLM proxy any HTTP client can point at directly, see the
`base_url`-swap section below). AGDK adds three things a proxy alone
cannot: a cryptographic agent identity signed locally, mandatory execution
telemetry so an unregistered or gone-quiet agent is provably absent (not
just unobserved), and a governed tool-call check-in with human-in-the-loop
support.

## Install

Not on PyPI yet: until it is, install from a checkout of this repository
(`pip install -e ".[all]"` or `uv sync --all-extras`). The commands below
show the intended PyPI form. Python 3.13 or newer is required.

```bash
pip install matimo-agdk
```

Framework adapters are optional extras -- see "Framework adapters" below
for what each one actually does:

```bash
pip install matimo-agdk[langchain]
pip install matimo-agdk[google-adk]
pip install matimo-agdk[crewai]
pip install matimo-agdk[autogen]
pip install matimo-agdk[all]
```

The full guide, from getting an API key to operating an agent, is [docs/USER-MANUAL.md](docs/USER-MANUAL.md).

## Quickstart

1. Register the agent once (per machine, not per run) and get an org API
   key from your Matimo Enterprise tenant admin:

   ```bash
   export MATIMO_API_KEY=me-live-...
   matimo-agdk register --name my-agent --framework custom
   ```

   This writes `~/.matimo/agents/my-agent.json` (identity metadata) and
   `~/.matimo/agents/my-agent.pem` (the private key, 0600 on POSIX). The
   private key is returned by the server exactly once, at registration --
   there is no way to retrieve it again. Losing it means registering fresh,
   or rotating the key if you still hold the identity and the org API key
   (`matimo-agdk rotate-key --name my-agent`). `register` refuses to run
   again for a name that already has credentials (a second registration
   would create a second identity and destroy the first one's key); pass
   `--force` only when you really mean to replace them.

2. Build a Governor from the persisted credentials:

   ```python
   from matimo_agdk import Governor

   governor = Governor.from_env(agent_name="my-agent")
   governor.start()  # starts the background telemetry/heartbeat exporter
   ```

3. Wrap the tools you want governed, and record LLM/tool activity inside a
   run:

   ```python
   def search(query: str) -> str: ...


   with governor.run("answer-a-question"):
       result = governor.guard(search, name="search", category="web")(query="roaiq")
       governor.llm_span(model="gpt-4o-mini", duration_ms=812)
   ```

4. Point your LLM client at Gateway instead of the real provider. Two
   ways, depending on whether you need per-request signing:

   ```python
   import openai

   # Quick: attaches the session token once, when the client is built. It is
   # not refreshed (a session lasts one hour by default) and nothing is
   # signed, so it stops working when the session expires, and it never works
   # for an identity whose tenant enforces requireSignedRequests. Fine for a
   # short script; not for a long-running agent.
   client = openai.OpenAI(**governor.openai_client_kwargs())

   # Recommended: attaches a live session token and signs every request, and
   # transparently re-handshakes if the session expires mid-run. Signing costs
   # nothing when the tenant does not enforce it.
   client = openai.OpenAI(
       base_url=governor.config.base_url,
       api_key=governor.config.api_key,
       http_client=governor.httpx_client(),
   )
   ```

   For Anthropic, use `governor.anthropic_http_client()` instead of
   `httpx_client()`: `anthropic` 1.6 and newer are built on `httpx2` and reject
   an `httpx.Client`. The helper returns whichever client the installed
   release accepts (`pip install httpx2` if it is missing).

   `default_headers=` alone cannot carry a per-request signature: the
   signature covers the exact bytes of each request's own body, which is
   only known once that specific request is being sent. `governor.httpx_client()`
   solves this with a request event hook that computes and attaches a fresh
   `Matimo-Agent-Signature` per call.

5. Async code uses the same shape with `AsyncGovernor` and `await`:

   ```python
   from matimo_agdk import AsyncGovernor

   governor = await AsyncGovernor.from_env(agent_name="my-agent").start()
   async with governor.run("answer-a-question"):
       result = await governor.guard(async_search, name="search")(query="roaiq")
   ```

## Framework adapters

Every adapter supports `mode="observe"` (telemetry only, never blocks) and
`mode="govern"` (default: also enforces DENY/PENDING and rapid suspend).
Each has one obvious registration point, but which piece actually
*enforces* a DENY differs by framework -- read each module's own docstring
before assuming attaching a callback/plugin alone is enough.

| Framework | Register once | LLM through Gateway | Enforcement point | Signing on LLM calls | Verified against |
|---|---|---|---|---|---|
| LangChain | `MatimoCallbackHandler`/`AsyncMatimoCallbackHandler` (telemetry) **+** `govern_tools()` (enforcement) | `gateway_chat_model(provider="openai"\|"anthropic")` | `govern_tools()`'s wrapped `_run`/`_arun` -- the callback alone can only crash the chain, not deny gracefully (see the module docstring) | Full (`openai`), session-header-only (`anthropic`) | langchain-core 1.6.3, langchain-openai 1.6.2, langchain-anthropic 1.7.2 |
| Google ADK | `MatimoPlugin(governor)` on `Runner(plugins=[...])` | `gateway_model()` (`LiteLlm`) | `MatimoPlugin.before_tool_callback`'s dict short-circuit -- ADK's own documented graceful-DENY contract | Live session token and run id per call (custom `LiteLLMClient`); no per-request signature | google-adk 2.9.1 (+ `extensions` extra) |
| CrewAI | `govern_crew(crew_or_agents, governor)` | `gateway_llm()` | The wrapped `_run`/`_arun` on each tool | Full: live session token, run id, and per-request JWS via CrewAI's transport interceptor (verified live with `requireSignedRequests=true`) | crewai 1.15.22 |
| AutoGen | `govern_tools(tools, governor)` | `gateway_model_client()` | The wrapped `run()` on each `BaseTool` | Full | autogen-core/-agentchat/-ext 0.7.5 (modern generation only -- see below) |
| Any other framework | `govern(callable_or_tools, governor)` | build your own client with `governor.openai_client_kwargs()`/`httpx_client()` | `governor.guard()` under the hood | Depends on your client | No framework dependency at all |

**Sync vs. async governors.** `gateway_chat_model()` (LangChain),
`gateway_model()` (ADK) and `gateway_llm()` (CrewAI) build their clients
synchronously, so they need a sync `Governor`; passing an `AsyncGovernor`
raises a `TypeError` that says so. A sync `Governor` still serves async
framework code (blocking calls are moved to a thread). `gateway_model_client()`
(AutoGen) is the reverse: it needs an `AsyncGovernor`. LangChain's `ChatOpenAI`
is wired with the live sync client only, so its async methods (`ainvoke`) fall
back to a session header fixed at construction and are not signed.

**"Session-header-only" means**: the session token is attached, but there
is no per-request `Matimo-Agent-Signature` (nonce + body hash) the way
LangChain's OpenAI path and AutoGen get, because the underlying client
(`ChatAnthropic`, ADK's `LiteLlm`, CrewAI's `LLM`) doesn't expose a
request-hook mechanism this SDK can use. Each `gateway_*`/`govern_*`
helper's own docstring explains exactly why, verified by reading the
installed package's source, not assumed. If your tenant might ever enable
`requireSignedRequests`, prefer a path marked "Full" for that traffic.

**AutoGen note**: `pyautogen` (the legacy 0.2-style `ConversableAgent`/
`register_function` package name) ships an **empty** `__init__.py` as of
its latest release on PyPI (0.10.0) -- verified directly, not assumed. This
adapter targets the actively maintained `autogen_core`/`autogen_agentchat`/
`autogen_ext` generation instead. If you're on the legacy API (now under
the community `ag2` package), wrap your registered functions with
`matimo_agdk.adapters.generic.govern()` directly -- it works with any
plain callable.

**Spans and runs, by design.** LangChain reports one tool span per call: when a
tool is wrapped by `govern_tools()` the wrapper is canonical (it sees the
decision) and the callback handler skips its own span for that call; a tool that
is not wrapped still gets the handler's span. A denied call's span is tied to
LangChain's run tree (its run and span ids) only when a Matimo callback handler
is attached, since that is what tells the wrapper which run it is inside. CrewAI
and AutoGen give the SDK no per-crew or per-chat id, so they cannot group a crew
or chat into one run on their own: wrap the call in `governor.run()` (or
`async with governor.run()`), which is the grouping mechanism. LangChain LLM
spans carry `gen_ai.provider.name` (`openai`, `anthropic`, `google`) derived from
the client class; an unknown class gives no provider.

See `examples/{langchain,google_adk,crewai,autogen}_agent.py` for a
runnable end-to-end demo of each, and each adapter module's own docstring
for the full detail this table compresses.

## What AGDK does not do

- **AGDK never executes your tools.** `governor.guard()` checks policy and
  records telemetry around your own callable; the actual tool code always
  runs in your process, on your infrastructure. A DENY simply means the
  wrapped call never happens.
- **AGDK never sees your real LLM provider credentials.** Once registered,
  your agent's LLM traffic goes through Matimo Gateway using Gateway's own
  session token, not your OpenAI/Anthropic key. Gateway resolves the
  actual BYOK provider credentials server-side.
- **AGDK is not a zero-code option.** If you don't need a cryptographic
  identity, telemetry, or tool governance, you can point any OpenAI/
  Anthropic-compatible client directly at Gateway's `base_url` with just
  an API key and skip AGDK entirely -- that "base_url swap" path is real
  and supported, just without the identity/telemetry/tool-check layer
  this SDK adds.

## Rapid suspend: what it really means

An admin can flip an identity's `lifecycle_status` to `suspended` or
`revoked`, or trip an org-wide emergency stop. Two different things happen
next, on two different timescales:

- **The next call Gateway itself receives (LLM proxy or tool check) is
  denied immediately, server-side, regardless of AGDK.** Lifecycle status
  is read fresh from the database on every such call, never from a stale
  cache. This is real and immediate.
- **AGDK's own `governor.state`/`is_suspended()`/`raise_if_suspended()`/
  `on_suspend(callback)` are POLLED, not pushed.** They reflect whatever
  the last telemetry heartbeat response said, which can lag the true
  server state by up to one heartbeat interval (sized automatically from
  the server's reported staleness window, clamped to 15s-5min, or set
  explicitly via `heartbeat_interval`). This is why the product calls this
  "rapid suspend," never "instant kill": a long-running loop that never
  calls Gateway again and never checks `governor.state` in between could,
  in principle, keep running locally for up to that interval even after
  being suspended. There is no push-based kill channel in v1 -- see
  `docs/SERVER-CONTRACT.md` section 10.

## When Gateway is unreachable

A tool check needs Gateway. What a tool call does when Gateway cannot answer at
all (a connection error, a timeout, a 5xx) is your choice, per agent:

| `tool_check_failure_mode` | What happens |
|---|---|
| `fail_closed` (default) | The tool does not run. `guard()` raises `ToolCheckUnavailable` (a `GatewayUnavailable`). Every adapter returns it as a recoverable tool error, the way it returns a DENY: LangChain a `ToolException` (an observation when the tool has `handle_tool_error=True`), ADK an `{"error": ...}` dict from `before_tool_callback`, CrewAI and AutoGen a raised exception their own tool loop turns into an error result. |
| `fail_open_bounded` | The tool runs, but only while Gateway was last heard from within `fail_open_max_stale_seconds` (default 300, and **hard-capped at 300**: a larger value is rejected when the config loads). Otherwise it behaves as `fail_closed`. |

"Heard from" means a successful tool check or a telemetry heartbeat (so call
`governor.start()`; without it only successful checks count). A call that ran
this way is marked: `ToolDecision.degraded` is True, and its tool span carries
`matimo.degraded_mode=true` and `matimo.degraded_cache_age_seconds`.

**Circuit breaker.** After `tool_check_breaker_threshold` (default 3)
consecutive transport failures the circuit opens for
`tool_check_breaker_cooldown` seconds (default 30) and checks fail fast, without
touching the network, instead of waiting out the transport's retries on every
call. After the cooldown one probe goes through; any real answer from Gateway
closes the circuit, a failed probe re-opens it. The first N failing calls each
still wait out the retry budget (about 2 to 4 seconds when the connection is
refused, much longer if packets are silently dropped, because each attempt can
wait for `connect_timeout` or `read_timeout`).

**Never softened, whatever the mode:**

- an explicit `DENY`, and an unrecognized decision (already a `DENY`);
- any 4xx: a bad key, a missing scope, a rejected signature, a suspended or
  revoked agent, a rate limit. These raise as before and do not trip the breaker;
- a suspended or emergency-stopped state the last heartbeat reported;
- a `PENDING` awaiting approval: polling failures raise, and so does an outage
  during the re-check that follows a `PENDING` with no resume token.

**What `fail_open_bounded` cannot know.** The SDK does not hold your policy and
does not replay Gateway's last answer for a tool. Beyond the freshness rule it
adds one guard: it will not fail open for a tool whose most recent decision in
this process was `DENY` or `PENDING`, so a tool that needs approval is never
waved through by an outage. A tool it has not checked before is allowed if
Gateway was heard from recently. If that is not acceptable for your tools, keep
`fail_closed`. This concerns tool checks only: LLM calls go through Gateway and
fail closed with it, and reporting failures are governed separately by
`fail_open_telemetry`.

## Configuration

`GatewayConfig` loads with this precedence, highest wins: explicit kwargs
passed to `Governor(...)` / `Governor.from_env(...)` > environment
variables > the credentials file written by `matimo-agdk register`.

| Env var | Purpose |
|---|---|
| `MATIMO_GATEWAY_URL` | Gateway base URL (default `http://localhost:8000/v1`) |
| `MATIMO_API_KEY` | Org API key (`Authorization: Bearer ...`). It needs three scopes: `gateway:proxy` (LLM calls), `identity:manage` (register, rotate-key, JWKS), and `agdk:check` (tool checks). A key without `agdk:check` registers fine and then fails on the first `guard()` with `missing the required scope: agdk:check`. |
| `MATIMO_IDENTITY_TOKEN` | Bearer identity token, if not loading from a credentials file |
| `MATIMO_IDENTITY_ID` | Identity UUID, ditto |
| `MATIMO_TENANT_ID` | Tenant UUID, ditto |
| `MATIMO_PRIVATE_KEY` | Private key PEM inline |
| `MATIMO_PRIVATE_KEY_FILE` | Path to a private key PEM file |
| `MATIMO_AGENT_NAME` | Agent name (credentials file key, and default displayName) |
| `MATIMO_FRAMEWORK` | `langchain` \| `google-adk` \| `crewai` \| `autogen` \| `custom` |
| `MATIMO_TOOL_CHECK_FAILURE_MODE` | `fail_closed` (default) \| `fail_open_bounded`; see "When Gateway is unreachable" |
| `MATIMO_FAIL_OPEN_MAX_STALE_SECONDS` | How stale `fail_open_bounded` may be, in seconds (default and maximum 300) |

Code-only settings on `GatewayConfig`: `tool_check_breaker_threshold` (3),
`tool_check_breaker_cooldown` (30 s), and `capture_tool_results` (False).

**Tool results are not sent by default.** With `capture_tool_results=True`,
every adapter and `guard()` put the tool's result (or, when the tool raised, its
error text) on the tool span as `gen_ai.tool.call.result`: redacted the same way
as arguments, then cut to 500 characters. Off by default because a result can
hold anything the tool read. Google ADK and the LangChain callback handler used
to send it unconditionally; that changed, see the CHANGELOG.

## CLI

```bash
matimo-agdk register --name my-agent --framework langchain
matimo-agdk status --name my-agent
matimo-agdk rotate-key --name my-agent
matimo-agdk doctor --name my-agent
```

## Testing

Three levels, from fastest/no-dependencies to a full live pass against a
real Matimo Gateway. Run them in order -- each one proves something the
last one couldn't.

### 1. Unit tests -- mocked network, no server needed

```bash
uv sync --all-extras --group dev
uv run pytest -q
uv run ruff check .
uv run mypy matimo_agdk
```

No test touches the network: HTTP is mocked with `respx` or in-process fakes
(see `CONTRIBUTING.md`), so this proves the client builds correct requests and
handles every scripted response shape correctly -- signing, retries,
redaction, adapter wiring -- without needing Gateway, Postgres, or a license.
This is the check to run on every change, and the one CI runs.

### 2. CLI sanity check -- a real Gateway, no code

Once a local Matimo Gateway is reachable (part of the Matimo Workbench
backend, default `http://localhost:8000/v1`) and you have an org API key from a
tenant admin:

```bash
export MATIMO_API_KEY=me-live-...
export MATIMO_GATEWAY_URL=http://localhost:8000/v1   # default, can omit
matimo-agdk register --name test-agent --framework custom
matimo-agdk doctor --name test-agent
```

`doctor` performs a real signed session handshake and a heartbeat, then
prints the lifecycle status. `doctor: all checks passed` with
`lifecycle_status=active` means the identity, signing, and session path
are genuinely wired end to end -- the fastest way to answer "is AGDK
actually working" without writing any code.

### 3. Full live check -- every governance path, against a real tenant

`scripts/live_check.py` runs 15 end-to-end scenarios (register, doctor,
heartbeat, telemetry, signature enforcement, a real signed completion,
rapid suspend, tool checks ALLOW / DENY / PENDING approved / PENDING
rejected, the LangChain adapter, session expiry, key rotation, call log)
against a real Matimo Gateway. It never provisions accounts, keys,
licenses, or LLM providers: an admin creates those in Matimo Workbench
first and the script only consumes them.

```bash
export MATIMO_API_KEY=...       # org API key with gateway:proxy, identity:manage, agdk:check
export MATIMO_TENANT_ID=...     # the tenant that key belongs to
export MATIMO_ADMIN_TOKEN=...   # a tenant-admin session JWT, used only for the admin APIs
export MATIMO_GATEWAY_URL=http://localhost:8000/v1   # default
export MATIMO_GATEWAY_MODEL=gpt-4o-mini              # optional; used for raw chat calls
uv run --group dev python scripts/live_check.py
uv run --group dev python scripts/live_check.py --filter "tool check"
```

The admin token must belong to the same tenant as the key; the script refuses to start otherwise, because every admin API call would silently act on another tenant. `DATABASE_URL` is optional and only enables three sub-assertions that no
admin API exposes yet (`last_telemetry_at`, raw telemetry event
attributes, the runs row); they are skipped without it. The signed
completion scenario fails, not skips, when the tenant's configured
provider has no working key, because that is a real fact about the tenant.

## Development

See `CONTRIBUTING.md`.

## License

MIT, copyright 2026 ROAIQ Technologies. See `LICENSE`.
