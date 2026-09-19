# Changelog

All notable changes to this project are documented in this file.

## [0.1.0] - 2026-09-18

Initial core SDK build. Not yet published to PyPI.

### Changed (2026-09-19, Python floor)

- `requires-python` raised from 3.11 to 3.13; classifiers list 3.13 and 3.14.
  CI runs 3.13 with every extra and 3.14 with every extra except `crewai`:
  CrewAI cannot be imported on Python 3.14 yet (verified with crewai 1.15.22:
  chromadb's `pydantic.v1` models raise `ConfigError`), so the `crewai`
  adapter is untested on 3.14. ruff and mypy now target 3.13.

### Fixed (2026-09-18, telemetry richness parity across adapters)

- CrewAI and AutoGen now emit LLM spans, not just tool spans -- found live
  against the Gateway Observability Hub: a CrewAI run showed only the
  gateway's own flat call-log entry (no purple LLM-span/green tool-span
  markers), unlike LangChain's and Google ADK's richly-correlated
  timelines, even though every adapter shared the same
  `emit_llm_span()`/`emit_tool_span()` primitives -- CrewAI's and AutoGen's
  LLM call paths simply never called the former.
- `matimo_agdk.adapters.crewai`: `gateway_llm()`'s `MatimoInterceptor`
  (`make_interceptor()`) now times every outbound LLM request and emits an
  LLM span -- duration, status, the requested model always, plus the
  response's own model and `gen_ai.usage.*` token counts when the response
  isn't a streamed `text/event-stream` body.
- `matimo_agdk.adapters.autogen`: `gateway_model_client()` now wraps the
  returned `OpenAIChatCompletionClient`'s `create()`/`create_stream()` and
  emits an LLM span per call, reading `usage`/`finish_reason` off AutoGen's
  own typed `CreateResult` rather than re-parsing a response body.
- Neither CrewAI nor AutoGen exposes a natural per-kickoff/per-chat id to
  these call sites (unlike LangChain's run_id/parent_run_id tree or ADK's
  invocation_id), so LLM and tool spans there correlate only when the
  caller wraps the run in `governor.run(...)` -- already the documented
  usage pattern for both adapters, now documented as load-bearing for
  telemetry correlation, not just as a style choice.
- `matimo_agdk.adapters.generic` and its docs now say explicitly that it
  emits no LLM spans (it only ever sees tool callables) and point at
  calling `governor.llm_span()` directly around your own LLM call site.

### Fixed (2026-09-18, external code review)

- Run ids are context-local (`contextvars.ContextVar`): concurrent
  `governor.run()` blocks in different tasks or threads no longer see or
  reset each other's id. New `matimo_agdk.governor.current_run_id()`.
- `guard()` outside a `governor.run()` block now opens an implicit run
  named `tool:<name>` instead of executing the tool and then raising.
- Redaction is recursive and word-based (`matimo_agdk._redact`): secrets
  nested inside tool arguments are masked in both the tool-check body and
  telemetry; `keyword`, `monkey` and `tokenizer` are no longer masked.
- `fail_open_telemetry=False` no longer kills the exporter thread: the
  error is logged, stored, and raised on the caller's next
  `submit()`/`flush_now()`/`stop()`.
- The heartbeat interval is re-sized from the server-reported staleness
  window on every heartbeat, and the first loop tick is a forced
  heartbeat; `connect_timeout` is now applied to the HTTP clients.
- Anthropic clients authenticate with `Authorization: Bearer`
  (`auth_token=` in `anthropic_client_kwargs()`, an explicit header in
  `gateway_chat_model(provider="anthropic")`); `x-api-key` was never read
  by Gateway.
- A DENY now records a tool span with status `denied`.
- LangChain and CrewAI tool wrappers include positional inputs in the
  check body (`arg0`, `arg1`, ...), so distinct calls no longer collapse
  onto one server-side dedup key.
- Signed requests are re-signed with a fresh nonce on every retry.
- Credential files are created with mode 0600 from the first byte.
- `Governor.stop()` no longer closes the HTTP client; `close()` /
  `aclose()` (and the context managers) do, so `start()` can be called
  again after `stop()`.
- The ADK plugin restores the previous run id after each model call.
- CLI renamed back to `matimo-agdk` to avoid colliding with Matimo OSS.
- `requires-python` raised to 3.11 to match the documentation; `py.typed`
  shipped; CI workflow and commitlint configuration added.

### Fixed (2026-09-18, after running every example against a live Gateway)

- `rotate_key()` no longer writes a credentials file for an identity that
  was registered with `persist=False`; it only overwrites a file that
  already exists (found when the live check leaked identities into
  `~/.matimo/agents`).
- CrewAI: `gateway_llm()` now installs a transport interceptor so every
  request carries the live session token, the current run id, and a
  per-request `Matimo-Agent-Signature`; verified live with
  `requireSignedRequests=true`.
- Google ADK: `gateway_model()` now uses a custom `LiteLLMClient` that
  injects the live session token and run id per call, and `MatimoPlugin`
  binds the ADK invocation id as the governor's current run, so ADK LLM
  calls correlate in the call log. No per-request signature (litellm
  serialises the body after the hook).
- New public `Governor.request_headers(body=None)` and
  `Governor.bind_run_id()` for adapters that own the outbound request.
- `scripts/live_check.py`: no passwords, no provider keys; requires
  `MATIMO_API_KEY`, `MATIMO_TENANT_ID`, `MATIMO_ADMIN_TOKEN` (tenant checked
  against the JWT); optional `MATIMO_GATEWAY_MODEL`.

### Added

- `Governor` / `AsyncGovernor`: the single entry point. Registration,
  session handshake, telemetry export with heartbeat, tool governance,
  run/span recording, `guard()` decorator, and `httpx_client()` /
  `openai_client_kwargs()` / `anthropic_client_kwargs()` for pointing any
  LLM SDK at Matimo Gateway.
- `GatewayConfig`: kwargs > env > credentials-file precedence.
- Compact ES256 JWS signing (`matimo_agdk.identity`), matching the exact
  envelope Matimo Gateway verifies (header `{alg: ES256, kid}`, claims
  `{iss, sub, tenant_id, external_framework?, nonce, iat, exp, body_hash}`).
- `matimo-agdk` CLI: `register`, `status`, `rotate-key`, `doctor`. (The bare
  `matimo` name is taken by Matimo OSS's own CLI package, so this one keeps
  the package name.)
- Typed exceptions mapped from Gateway's flat `{error, message?}` envelope:
  `PolicyDenied`, `TelemetryStale`, `AgentSuspended`, `SessionExpired`,
  `SignatureRejected`, `RateLimited`, `GatewayUnavailable`, `ToolDenied`,
  `ToolCheckTimeout`, `AgentSuspendedLocally`, `SigningError`.
- Framework adapters (`matimo_agdk.adapters.*`), each an optional extra:
  `langchain` (`MatimoCallbackHandler`/`AsyncMatimoCallbackHandler`,
  `govern_tools()`, `gateway_chat_model()`), `google_adk` (`MatimoPlugin`,
  `gateway_model()`), `crewai` (`govern_crew()`/`govern_tool()`,
  `gateway_llm()`), `autogen` (`govern_tools()`, `gateway_model_client()`,
  modern `autogen_core`/`autogen_agentchat`/`autogen_ext` generation only
  -- see the adapter's own docstring on why legacy `pyautogen` isn't
  supported), and `generic` (`govern()`, no framework dependency at all).
  Every adapter supports `mode="observe"`/`mode="govern"`. See the README's
  "Framework adapters" section and `examples/*_agent.py`.

### Known gaps

- No PyPI package published yet; install from a checkout.
- Google ADK and LangChain-with-Anthropic paths send the session header
  but no per-request signature (see the adapter docstrings).
- Rapid suspend is polled at the heartbeat interval, never pushed.
