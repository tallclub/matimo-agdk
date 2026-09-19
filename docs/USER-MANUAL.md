# Matimo AGDK User Manual

Matimo AGDK (Agent Governance Development Kit) is a small Python library. You add it to an AI agent you have already built, and it connects that agent to **Matimo Gateway**, where your company can see what the agent does, set rules for it, and stop it if needed.

Version covered: matimo-agdk 0.1.0, Python 3.13 or newer.

**How to read this manual**

| If you are... | Read |
|---|---|
| New to AGDK | Parts 1 to 4, in order. About 15 minutes. You will finish with a working governed agent. |
| Adding AGDK to LangChain, Google ADK, CrewAI or AutoGen | Part 5, then the matching file in `examples/` |
| Looking something up | Part 6 (words), Part 7 (errors), then the Reference parts (8 to 14) |

---

## Part 1. The idea in plain words

### The problem

You built an agent. It calls an LLM (like GPT) and runs tools (search the web, send an email, update a CRM). Now your company asks:

- *Which agents exist, and who owns them?*
- *Which models and tools is each one allowed to use?*
- *Can we stop one right now if it misbehaves?*
- *Can we see what it actually did?*

### What AGDK does about it

AGDK adds four things to your agent:

| What | In plain words |
|---|---|
| **An identity** | The agent is registered once and gets an ID badge that cannot be forged. Without a registered identity it cannot use the LLM at all. |
| **LLM calls through Gateway** | Instead of talking to OpenAI directly, the agent talks to Gateway. Gateway holds the real OpenAI key, so your agent never sees it. |
| **A permission check before every tool** | Before the agent runs a tool, AGDK asks Gateway "may I?". The answer is **Allow**, **Deny**, or **Pending** (a human must approve first). |
| **Reporting** | AGDK tells Gateway what the agent did and sends a regular "still alive" signal. Admins see it in Workbench, and can suspend the agent. |

### Where things run

```
 YOUR MACHINE                                        MATIMO WORKBENCH (your company)
+-------------------------------+                  +---------------------------------+
| Your agent code               |  "here is my     | Gateway                         |
|   + AGDK                      |   LLM request" ->|   - checks the agent's identity |--> OpenAI,
|                               |                  |   - applies your company's rules|    Anthropic...
|   Your tools run HERE         |  "may I run      |   - keeps a log of everything   |
|   (search, send_email, ...)   |   send_email?" ->|                                 |
|                               | <- Allow / Deny  | Admin screens                   |
+-------------------------------+                  |   - create API keys, set rules  |
                                                   |   - approve pending tool calls  |
                                                   |   - suspend an agent            |
                                                   +---------------------------------+
```

Three facts to remember:

1. **No registration, no LLM.** An agent only gets model access through Gateway after it has registered.
2. **AGDK never holds your OpenAI or Anthropic key.** Those stay in Workbench. Your agent uses a Matimo key instead.
3. **Your tools always run on your machine.** AGDK only asks permission and records the result. It never runs the tool for you.

---

## Part 2. Before you start

You need three things from whoever administers Matimo Workbench at your company. Your tenant (your company's space in Workbench) must have an active Matimo Enterprise license, or Gateway will refuse every call.

| You need | Example | Notes |
|---|---|---|
| **Gateway URL** | `http://localhost:8000/v1` or `https://workbench.matimo.ai/v1` | The Workbench backend address plus `/v1` |
| **An org API key** | `me-live-...` | Created under Governance in Workbench. It must have all three scopes listed below. |
| **A model name** | `gpt-4o-mini` | A model your admin has already set up for your tenant |

The API key needs three scopes (permissions). A key missing one still lets you register, then fails later, which is confusing, so check them now:

| Scope | Lets AGDK do this |
|---|---|
| `identity:manage` | Register the agent, rotate its key |
| `gateway:proxy` | Open a session, make LLM calls, send reports |
| `agdk:check` | Ask permission before running a tool |

**A message you can send your admin:**

> Please give me (1) the Matimo Gateway URL, (2) an org API key with the scopes `identity:manage`, `gateway:proxy` and `agdk:check`, and (3) the name of a model my tenant is allowed to use. I'm going to register a new agent called `<name>`.

You also need Python 3.13 or newer.

---

## Part 3. Your first governed agent (10 minutes)

This walkthrough uses no AI framework, only plain Python. You will register an agent, check the connection, and run a tool call through the permission check.

### Step 1. Install

AGDK is not on PyPI yet, so install from a checkout of this repository, from the repository folder:

```bash
uv sync --all-extras          # or:  pip install -e ".[all]"
```

Once it is published, this becomes `pip install matimo-agdk`.

### Step 2. Tell AGDK where Gateway is and who you are

Set two environment variables in your terminal. They last only for that terminal window.

Bash / Git Bash / macOS / Linux:
```bash
export MATIMO_GATEWAY_URL=http://localhost:8000/v1
export MATIMO_API_KEY=me-live-...
```

Windows PowerShell:
```powershell
$env:MATIMO_GATEWAY_URL = "http://localhost:8000/v1"
$env:MATIMO_API_KEY = "me-live-..."
```

### Step 3. Register your agent (once)

```bash
matimo-agdk register --name my-first-agent --framework custom
```

> The command is `matimo-agdk` (not `matimo`). `--framework custom` means "plain Python". Other values are `langchain`, `google-adk`, `crewai` and `autogen`.

You will see something like:

```
Registered identity 4b9a4d53-... (my-first-agent)
  identity_token: me-id-...
  framework:      custom
  metadata:       ~/.matimo/agents/my-first-agent.json
  private key:    ~/.matimo/agents/my-first-agent.pem (never printed, never retrievable again)
```

What just happened: Gateway created an identity for your agent and generated a **private key**, a secret that proves later requests really come from this agent. AGDK saved it to the `.pem` file. **The server shows the private key exactly once, so do not delete that file.** If you lose it you must register again or rotate the key (Part 8).

**Remember the name you chose (`my-first-agent`).** Everything else uses it to find your files. This is the most common beginner mistake, so it is covered again in Part 7.

### Step 4. Check that it all works

```bash
matimo-agdk doctor --name my-first-agent
```

Expected output:

```
Gateway URL: http://localhost:8000/v1
  [ok] org API key present
  [ok] identity loaded: 4b9a4d53-...
  [ok] session handshake succeeded (token prefix: ...)
  [ok] telemetry heartbeat succeeded: lifecycle_status=active
doctor: all checks passed
```

If you see `[FAIL]`, go to Part 7. Do not go further until `doctor` passes.

### Step 5. Run a governed tool call

Save this as `hello_agdk.py`:

```python
from matimo_agdk import Governor, ToolDenied

# 1. Load the identity you registered. The name must match --name from Step 3.
governor = Governor.from_env(agent_name="my-first-agent")

# 2. Connect to Gateway and start reporting in the background.
governor.start()

# A normal Python function. This is the "tool" we want to control.
def search(query: str) -> str:
    return f"(pretend) search results for: {query}"

try:
    # 3. A "run" groups everything below into one story in the Workbench log.
    with governor.run("hello-run"):
        try:
            # 4. guard() asks Gateway "may this agent call 'search'?" first.
            #    Allowed: the function runs. Denied: ToolDenied is raised and
            #    the function never runs. Pending: waits for a human to decide.
            safe_search = governor.guard(search, name="search", category="web")
            print("Allowed:", safe_search(query="refund policy"))
        except ToolDenied as denied:
            print("Denied by policy:", denied.reason)
finally:
    # 5. Always stop: sends any last reports before your program exits.
    governor.stop()
```

Run it:

```bash
uv run python hello_agdk.py
```

Expected output, if your tenant has no rule against this tool:

```
Allowed: (pretend) search results for: refund policy
```

If your admin has set a rule that forbids `search`, you will see `Denied by policy: ...` instead. That is the governance working, not an error.

### Step 6. See it in Workbench

Ask your admin, or open Workbench yourself if you have access: **Governance, Gateway, Activity**. Your `hello-run` should appear with the tool call and the policy decision.

### Step 7. Add an LLM call through Gateway (optional)

To send an LLM request through Gateway instead of straight to OpenAI, add this inside the `with governor.run(...)` block. It needs the `openai` package (`uv pip install openai`) and a model your tenant allows.

```python
from openai import OpenAI

client = OpenAI(
    base_url=governor.config.base_url,   # Gateway, not api.openai.com
    api_key=governor.config.api_key,     # your Matimo key, not an OpenAI key
    http_client=governor.httpx_client(), # signs each request as your agent
)
reply = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say hello in five words."}],
)
print(reply.choices[0].message.content)

# Tell Gateway about the call so it shows up in the run.
governor.llm_span(model="gpt-4o-mini", provider="openai", status="completed")
```

`examples/plain_python.py` in this repository is the same thing as a complete script.

---

## Part 4. What you just did, in one picture

```
register (once)  ->  Governor.from_env()  ->  governor.start()  ->  governor.run() { guard(tool) ... }  ->  governor.stop()
 get an identity      load that identity      open a session,        group the work; ask permission          flush the
                                              begin reporting        before each tool                        reports
```

The same five moves apply to every framework. Only the wiring in the middle changes, and that is Part 5.

---

## Part 5. Using AGDK with your framework

Every framework example in `examples/` uses the same three building blocks:

| Role | What it does | Name in each framework |
|---|---|---|
| **Governor** | The core object: identity, reporting, permission checks | `Governor` (`AsyncGovernor` for AutoGen) |
| **Tool enforcement** | Makes the framework ask permission before each tool | see table |
| **LLM routing** | Points the framework's model at Gateway | see table |

| Framework | Tool enforcement | LLM routing | Example |
|---|---|---|---|
| LangChain | `govern_tools(...)`, plus `MatimoCallbackHandler` for reporting | `gateway_chat_model(...)` | `examples/langchain_agent.py` |
| Google ADK | `MatimoPlugin(governor)` | `gateway_model(...)` | `examples/google_adk_agent.py` |
| CrewAI | `govern_crew(crew, governor)` | `gateway_llm(...)` | `examples/crewai_crew.py` |
| AutoGen | `govern_tools(...)` | `gateway_model_client(...)` | `examples/autogen_agent.py` |
| Anything else | `govern(...)` from `matimo_agdk.adapters.generic` | your own client, as in Step 7 | `examples/plain_python.py` |

Each example is short and commented line by line, and starts with the exact commands to run it. Two things to know before running one:

- **Each example registers under its own name**, for instance `langchain-demo`, so run the `matimo-agdk register --name ...` line from the top of that file first. Your `my-first-agent` will not be found by it.
- **Install the framework's extra**, for instance `uv run --extra langchain python examples/langchain_agent.py`. The extras are `langchain`, `google-adk`, `crewai` and `autogen`.

Details and caveats for each framework are in Part 12.

---

## Part 6. Words you will see

| Word | Meaning |
|---|---|
| **Matimo Gateway** | The server that your agent's LLM calls and permission checks go to. It lives inside Workbench, under `/v1`. |
| **Workbench** | The Matimo web application where admins manage keys, rules and approvals. |
| **Tenant** | Your company's own space inside Workbench. |
| **Org API key** | A secret (`me-live-...`) that proves a request comes from your company. Shared by the agents you register. |
| **Identity** | One registered agent: an ID, a token, and a private key. One identity per agent. |
| **Private key** | The secret file (`.pem`) created at registration. Keep it safe. It is shown once. |
| **Signature** | Proof attached to a request that it really came from your agent, made with the private key. AGDK does this for you. |
| **Session** | A short-lived pass your agent gets when it starts. AGDK opens and renews it automatically. |
| **Governor** | The AGDK object you create in your code. Everything goes through it. |
| **Run** | A named group of activity, like "answer-ticket". Makes the log readable. |
| **Span / telemetry** | A record of one thing that happened (an LLM call, a tool call). Sent to Gateway in the background. |
| **Heartbeat** | The regular "still alive" signal. Gateway replies with the agent's current status, for example `active` or `suspended`. |
| **Policy** | A rule an admin writes, such as "this agent may not use `send_email`". |
| **Guardrail** | A rule that blocks or changes a prompt or an answer. |
| **BYOK** | "Bring your own key": your company stores its OpenAI or Anthropic key in Workbench, not in your code. |
| **Adapter** | A small piece of AGDK code for one framework (LangChain, ADK, ...). |
| **Scope** | A permission attached to an API key, for example `agdk:check`. |
| **Allow / Deny / Pending** | The three answers to "may I run this tool?". Pending means a human must approve first. |

---

## Part 7. When something goes wrong

### First-run problems

| You see | What it means | Fix |
|---|---|---|
| `error: no identity found. Run matimo-agdk register first.` | AGDK looked for your files under a different name. Commands like `status` and `doctor` use `MATIMO_AGENT_NAME`, then the default `matimo-agent`, unless you pass `--name`. | Add `--name <the name you registered>`, or set `MATIMO_AGENT_NAME` for the terminal session. Files live in `~/.matimo/agents/`. |
| `matimo: command not found` | The command is `matimo-agdk`. | Use `matimo-agdk`. If that is also missing, install the package (Step 1) and activate your virtual environment. |
| `[FAIL] no org API key configured` | `MATIMO_API_KEY` is not set in this terminal. | Repeat Step 2. Environment variables do not carry over to a new terminal window. |
| Error mentioning `invalid_api_key` (HTTP 401) | The key is wrong, expired or revoked. | Ask your admin for a new one. |
| Error mentioning `license_required` (HTTP 403) | Your tenant has no active Matimo Enterprise license. | Ask your admin. |
| Error mentioning `insufficient_scope` or `missing the required scope` | The key lacks a scope. | Ask for a key with all three scopes (Part 2). |
| `Governor has no identity: call governor.register(...) once, or run matimo-agdk register ...` | Your code asked for an agent name that has no saved identity. | The `agent_name=` in your code must match the `--name` you registered. |
| Connection refused, or timeout | Gateway is not running or the URL is wrong. | Check `MATIMO_GATEWAY_URL`, including `/v1`. |

### Errors while your agent runs

| You see | Meaning | What to do |
|---|---|---|
| `ToolDenied` | Policy said no, or an approval was rejected. `reason` explains. | Handle it in your agent loop. Do not retry blindly. |
| `ToolCheckTimeout` | A Pending decision was not made in time. | Tell a human. A retry creates a new request. |
| `AgentSuspended` (`agent_suspended`, `agent_revoked`, `emergency_stop_active`) | An admin acted. | Stop the agent and contact the admin. |
| `AgentSuspendedLocally` | Raised by `raise_if_suspended()` from the last heartbeat. | Same. |
| `PolicyDenied` with reason `telemetry_stale` (`TelemetryStale`) | Your tenant blocks agents that have gone quiet, and yours has. | Call `governor.start()` before working, and do not block the background reporter. |
| `SessionExpired` | The automatic renewal also failed. | Check the API key and the identity's status. |
| `SignatureRejected` | Gateway could not verify the signature. | Rotate the key if the `.pem` file was overwritten. Also check your computer's clock. |
| `RateLimited` | Too many requests. | Back off. The exception has `retry_after` when the server sends it. |
| `GatewayUnavailable` | Network problem or a server 5xx. | Retry with backoff. LLM calls fail closed, and reporting fails open. |
| HTTP 400 `no_default_connection` | LLM call without `model=` and the tenant has no default model. | Always pass `model=`. |
| HTTP 403 `model_not_allowed` | The model is not set up for the tenant or not allowed for this agent. | Use a model your admin configured. |
| HTTP 502 `upstream_error` | Every model provider target failed. | The tenant's provider key is wrong or the provider is down. |
| Response with `finish_reason: "content_filter"` | A guardrail blocked the prompt or the answer. | This is HTTP 200. The content is the guardrail's configured message. |

---

# Reference

Everything from here on is for looking things up. You do not need it for a first run.

## 8. Registering and managing agents

### Register

```bash
export MATIMO_GATEWAY_URL=http://localhost:8000/v1
export MATIMO_API_KEY=me-live-...
matimo-agdk register --name support-bot --framework langchain
```

`--framework` is one of `langchain`, `google-adk`, `crewai`, `autogen`, `custom`. Add `--tool-category web --tool-category crm` to declare the tool categories the agent may use. An admin can tighten these later.

The server generates the key pair (ECDSA P-256) and returns the private key exactly once. The CLI stores two files, readable only by you, and never prints the key:

```
~/.matimo/agents/support-bot.json   identity id, token, tenant, framework
~/.matimo/agents/support-bot.pem    private key, mode 0600
```

Windows cannot enforce the 0600 mode. Keep the folder inside a user profile that other accounts cannot read.

You can also register from code: `governor.register(display_name=..., framework=...)`.

### CLI

```
matimo-agdk register   --name NAME --framework {langchain,google-adk,crewai,autogen,custom}
                  [--tool-category CAT ...] [--gateway-url URL] [--api-key KEY]
matimo-agdk status     [--name NAME]      print the current GovernanceState from a live heartbeat
matimo-agdk rotate-key [--name NAME]      new keypair, credentials file rewritten
matimo-agdk doctor     [--name NAME]      connectivity, handshake, signing, heartbeat
```

`--name` is required for `register`. For the other commands it falls back to `MATIMO_AGENT_NAME`, then to `matimo-agent`. `--gateway-url` and `--api-key` fall back to the environment variables.

### Operating an agent

- **Rotate keys** on a schedule or after any suspected exposure: `matimo-agdk rotate-key --name NAME`. The old key stops verifying the moment the server accepts the new one, and running processes must restart to load it.
- **Watch the call log** in Workbench (Governance, Gateway, Activity) to see each call's policy outcome, guardrail outcome, model, run id, and Gateway overhead.
- **Suspend from the admin UI** when something looks wrong. The next Gateway call is refused immediately, and the agent's own state follows within one heartbeat.
- **Several processes, one agent**: copy the credentials or pass them by environment. Sessions are per process and that is fine.
- **Do not share one identity across different agents.** Policies, telemetry and the audit trail are per identity.

## 9. Configuration

Precedence, highest first: keyword arguments to `Governor(...)` or `Governor.from_env(...)`, then environment variables, then the credentials file written by `matimo-agdk register`.

| Environment variable | Purpose | Default |
|---|---|---|
| `MATIMO_GATEWAY_URL` | Gateway base URL | `http://localhost:8000/v1` |
| `MATIMO_API_KEY` | Org API key, sent as `Authorization: Bearer` | required |
| `MATIMO_AGENT_NAME` | Which credentials file to load, and the default display name | `matimo-agent` |
| `MATIMO_FRAMEWORK` | `langchain`, `google-adk`, `crewai`, `autogen`, `custom` | `custom` |
| `MATIMO_IDENTITY_TOKEN`, `MATIMO_IDENTITY_ID`, `MATIMO_TENANT_ID` | Identity without a credentials file (containers, CI) | from file |
| `MATIMO_PRIVATE_KEY` or `MATIMO_PRIVATE_KEY_FILE` | Private key PEM inline or by path | from file |

You normally never set `MATIMO_TENANT_ID` yourself. It is saved in the credentials file at registration. It is only needed when you skip that file and supply the whole identity through environment variables.

`GatewayConfig` fields you may set in code: `connect_timeout` (10 s), `read_timeout` (30 s), `telemetry_flush_interval` (5 s), `telemetry_batch_size` (50), `telemetry_queue_max` (2000), `heartbeat_interval` (derived from the server's staleness window, clamped to 15 s to 5 min), `fail_open_telemetry` (True: a reporting outage never blocks the agent), `signing_enabled` (True), `credentials_dir`.

For a container, mount nothing: set `MATIMO_IDENTITY_TOKEN`, `MATIMO_IDENTITY_ID`, `MATIMO_TENANT_ID` and `MATIMO_PRIVATE_KEY` from your secret store.

`Governor` also works as a context manager (`with Governor.from_env(...) as governor:`). `AsyncGovernor` offers the same API with `await` on the network methods and `async with governor.run(...)`.

## 10. How it works underneath

### Identity and signing

Every request that matters (session handshake, LLM calls, tool checks) carries `Matimo-Agent-Signature`, a compact ES256 JWS over a hash of the exact request bytes, signed with the agent's private key. Gateway verifies it against the public key it holds and publishes at `GET /v1/identities/:id/jwks`. Signing is always on in the SDK, and an admin decides per identity whether Gateway requires it (`requireSignedRequests`). Telemetry is not signed by design.

### Session

`start()` performs `POST /v1/sessions` and keeps the session token in memory. Gateway requires it on every LLM and telemetry call. If the session expires or is deleted server-side, the SDK re-handshakes once and retries. You see nothing unless the second attempt fails (`SessionExpired`).

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

Only the category Gateway resolves is trusted. `set_tool_category(name, category)` and the `category=` hint let you tell the server how to classify a tool, and an admin can override it. After the function returns, `guard()` reports the outcome through `POST /v1/tools/result` and records a tool span.

## 11. Governor reference

| Method | Purpose |
|---|---|
| `Governor.from_env(**overrides)` | Build from env plus credentials file |
| `register(display_name=, framework=, allowed_tool_categories=, persist=True)` | Create an identity in code instead of the CLI |
| `start()` / `stop()` | Handshake and exporter lifecycle; also a context manager |
| `run(name)` | Context manager yielding a run id |
| `llm_span(**attrs)` / `tool_span(tool_name, **attrs)` | Record telemetry |
| `guard(fn, name=, category=)` | Policy-checked wrapper, usable as a decorator |
| `check_and_wait(name, args)` | Tool check that always ends in a final ALLOW or DENY (polls a PENDING for you) |
| `check_tool(name, args)` / `await_decision(resume_token)` | The two steps by hand. `check_tool` can return PENDING with no `resume_token`; treat that as "not approved yet", never as "go ahead" |
| `set_tool_category(name, category)` | Classify a tool server-side |
| `httpx_client()` | Signed, session-aware `httpx.Client` for any OpenAI-compatible SDK |
| `openai_client_kwargs()` / `anthropic_client_kwargs()` | `base_url`, `api_key`, `default_headers` for those SDKs; pass `http_client=governor.httpx_client()` too |
| `request_headers(body=None)` | Live headers (and signature over `body`) for a client you build yourself |
| `bind_run_id(run_id)` | Attach a framework-owned run id without opening a span |
| `state` / `is_suspended()` / `raise_if_suspended()` / `on_suspend(cb)` | Heartbeat-driven governance state |
| `rotate_key()` | New keypair; the old key stops verifying immediately |
| `identity` | The bound `IdentityCredentials` |

`AsyncGovernor` mirrors this with `await` on `register`, `start`, `stop`, `check_tool`, `await_decision`, `set_tool_category`, `rotate_key`, `request_headers`, and `httpx_async_client()`.

## 12. Framework adapters in detail

Each adapter has two halves: an observe half that records telemetry, and a govern half that checks tools. `mode="observe"` records only, and `mode="govern"` (default) also enforces.

### LangChain

```python
from matimo_agdk.adapters.langchain import MatimoCallbackHandler, gateway_chat_model, govern_tools

tools = govern_tools([calculator, search], governor)          # policy check before each tool body
model = gateway_chat_model(governor, model="gpt-4o-mini").bind_tools(tools)
result = model.invoke(messages, config={"callbacks": [MatimoCallbackHandler(governor)]})
```

`gateway_chat_model(provider="openai")` signs every request. `provider="anthropic"` sends the session header only. A DENY raises LangChain's `ToolException`. Set `handle_tool_error=True` on the tool to turn it into an observation string the agent can reason about. `AsyncMatimoCallbackHandler` exists for async chains.

### Google ADK

```python
from matimo_agdk.adapters.google_adk import MatimoPlugin, gateway_model

agent = Agent(name="weather", model=gateway_model(governor, model="gpt-4o-mini"), tools=[get_weather])
runner = InMemoryRunner(agent=agent, plugins=[MatimoPlugin(governor)])
```

One plugin governs every model and tool call the runner makes. A DENY is returned through ADK's own before-tool short-circuit, so the agent sees a structured refusal rather than a crash. LLM calls carry the live session token and the ADK invocation id as the run id. They are not signed per request (litellm builds the body after the hook), so keep `requireSignedRequests` off for ADK identities or route through a custom `BaseLlm` built on `governor.httpx_client()`.

### CrewAI

```python
from matimo_agdk.adapters.crewai import gateway_llm, govern_crew

researcher = Agent(role="Researcher", tools=[search], llm=gateway_llm(governor, model="gpt-4o-mini"))
crew = Crew(agents=[researcher], tasks=[task])
govern_crew(crew, governor)       # wraps every tool reachable from the crew
with governor.run("research"):
    crew.kickoff()
```

`gateway_llm()` installs a transport interceptor, so every CrewAI LLM request carries the live session token, the run id, and a per-request signature. It also emits one LLM span per call (duration, status, model, token usage when the response isn't streamed) alongside the tool spans `govern_tool()`/`govern_crew()` already record. `govern_tool(tool, governor)` governs a single tool. CrewAI exposes no per-`kickoff()` id, so `with governor.run(...):` (shown above) is what makes the LLM and tool spans of one crew execution share a run id in the Observability Hub timeline. Without it, each call gets its own uncorrelated id.

### AutoGen 0.7

```python
from matimo_agdk.adapters.autogen import gateway_model_client, govern_tools

tool = FunctionTool(calculator, description="Evaluate arithmetic.")
govern_tools([tool], governor)
agent = AssistantAgent("calc", model_client=gateway_model_client(governor, model="gpt-4o-mini"), tools=[tool])
async with governor.run("calc-run"):
    await agent.on_messages([TextMessage(content=question, source="user")], CancellationToken())
```

Full per-request signing. `gateway_model_client()` needs an `AsyncGovernor`, because AutoGen's model clients are async-only. `govern_tools()` works with either kind of governor. `gateway_model_client()` also wraps the model client's `create()`/`create_stream()` to emit one LLM span per call (duration, status, model, token usage/finish reason from AutoGen's own typed `CreateResult`). AutoGen exposes no per-chat id to this wrapper, so `async with governor.run(...):` (shown above) is what makes the LLM and tool spans of one chat share a run id in the Observability Hub timeline. Without it, each call gets its own uncorrelated id. Legacy AutoGen 0.2 (`pyautogen`) is not supported. Use the generic adapter.

### Anything else

```python
from matimo_agdk.adapters.generic import govern

tools = govern({"search": search, "send_email": send_email}, governor, category="crm")
```

Takes a callable, a list, or a dict, and returns the same shape with each callable wrapped by `guard()`. This adapter only ever sees tool callables, never an LLM client, so it emits no LLM spans on its own. Call `governor.llm_span(...)` around your own LLM call site (Step 7 above) if you want that call to show up alongside your `govern()`-wrapped tool calls in the Observability Hub.

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
