# Changelog

All notable changes to this project are documented in this file.

## [0.1.0] - 2026-09-18

Initial core SDK build. Not yet published to PyPI.

### Fixed (2026-09-20, end-to-end review; every item reproduced by script first)

Governance and correctness:

- **Fail closed on an unrecognized tool-check decision.** Only `ALLOW`, `DENY`
  and `PENDING` exist in the contract; anything else (or a missing `decision`)
  used to fall through `guard()` and every adapter and *ran the tool*. It now
  resolves to `DENY` with reason `unrecognized_tool_check_decision`.
- **Generic adapter dropped positional arguments** whenever any keyword argument
  was present, so `f(1, b=2)` and `f(99, b=2)` shared one server-side dedup key
  and could reuse each other's cached decision. Fixed with the shared
  `call_args_from()`.
- **LangChain: one call raised two policy checks** (and would have asked a human
  to approve twice) for any tool without its own coroutine, because the default
  `_arun` routes back through the wrapped `_run`. Checked once now. `govern_tools()`
  is also idempotent (LangChain and AutoGen), and CrewAI's default `_arun`
  (which only raises `NotImplementedError`) is no longer wrapped.
- **LangChain callback handlers leaked memory** on failed chains: `on_chain_error`
  was not implemented, so run-tree entries were never released.
- **LangChain callback handlers left runs `running` in Gateway.** Every parentless
  `.invoke()` (a bare LLM call, a bare tool call) became its own run under
  LangChain's own run id, and nothing ever sent the terminal `kind:"run"` span
  Gateway needs to end a run. The `langchain_agent.py` example showed one
  completed run plus two `running` ones. Inside `governor.run()` the handlers now
  join that run; outside one they open a run and close it `completed`/`failed`.
- **Every other adapter had the same stuck-`running` bug outside `governor.run()`.**
  `emit_llm_span()`/`emit_tool_span()` fell back to a fresh run id that nothing
  closed, which hit CrewAI, AutoGen, the generic adapter's observe mode and async
  bridge, and LangChain's tool wrapper (and every denied-tool span). The fallback
  now wraps the span in a one-span run it opens and closes, as `Governor.guard()`
  already did. `tests/adapters/test_run_lifecycle.py` holds every adapter to one
  invariant: each run opens once and closes once, with the close last.
- **ADK: a denied tool call landed in a run of its own** instead of the
  invocation's run (`bind_run_id()` only covers model calls). It now uses the
  invocation id. `MatimoPlugin` also takes `category=` like every other adapter.
- **Generic `govern()` was not idempotent** (LangChain, CrewAI and AutoGen skip an
  already-governed tool), so governing twice policy-checked one call twice.
- **`rotate_key()` left existing clients signing with the revoked key.** Any
  `httpx_client()` created before a rotation, plus adapters, the telemetry
  exporter and `guard()`-wrapped callables, held the old signer. Identity is now
  re-bound in place (`JWSSigner.rekey`, `SessionManager.rebind`, `ToolGovernor.rebind`).
- **`guard()` could discard a tool's result after the tool had run.** With
  `fail_open_telemetry=False`, a stored telemetry flush error was raised from the
  span emission in `guard()`'s `finally`, replacing the tool's return value (or its
  own exception) and skipping the result report. The error is now raised *before*
  the check and the tool run (fail closed, no side effect); after the tool has
  run it is logged and the span is still recorded. A DENY always raises
  `ToolDenied`. Sync and async.
- **`governor.run()` never closed the run on cancellation or `KeyboardInterrupt`**
  (only on `Exception`), leaving it `running` server-side until the sweep. It now
  closes with `cancelled`.

Data loss and resilience:

- `stop()` / `flush_now()` sent one batch only, silently discarding the rest of
  the queue (302 queued events, 50 delivered). They now drain everything, and the
  background loop no longer caps throughput at one batch per flush interval.
- The exporter thread died silently on any non-`GatewayError` (a malformed session
  response, a raising `on_suspend` callback), ending heartbeats and rapid-suspend
  polling. Both are now contained and logged; a malformed session response is a
  `GatewayError`.
- A batch the server rejects outright (400/413/422) was re-queued and resent
  forever, wedging telemetry; it is now dropped and counted in `dropped_count`.
- `stop()` unregisters its `atexit` hook, and `on_suspend` callbacks survive
  `stop()` then `start()`.
- `register()` and `rotate_key()` bind the new identity before writing it to disk,
  and credentials are written atomically (temp file, fsync, replace), so a failed
  write can no longer destroy the only copy of a private key.
- `status` polling and `check` now retry transient 5xx/connection errors: a
  human-approval wait can last hours and a single 502 aborted it.

Security:

- `repr()` of `GatewayConfig` and `IdentityCredentials` no longer includes the API
  key, identity token or private key PEM.
- `set_tool_category()` percent-encodes the tool name; `"../identities/x"` used to
  be normalised by the HTTP client into a different route, sent with the org key.
- `Retry-After` is capped (`RetryPolicy.max_retry_after`, default 60s) and
  `nan`/`inf`/negative values are ignored; a hostile header could park a thread
  for a day. An exhausted 429 now exposes `RateLimited.retry_after`.
- Redaction also scrubs string *values* for PEM private keys, `Bearer` credentials
  and Matimo/OpenAI/GitHub/AWS key shapes, and is applied to the error text sent
  to `/tools/result`. It remains a backstop, not a classifier.
- `GatewayConfig` warns when credentials would cross plain `http` to a non-loopback
  host.

Behavior changes you may notice:

- `Governor.register()` / `AsyncGovernor.register()` refuse to overwrite existing
  credentials for the same name, before any network call (`overwrite=True`; CLI
  `register --force`). Registering twice used to orphan the first identity.
- `GatewayConfig` rejects unknown options and non-positive timeouts, batch sizes
  and intervals instead of silently ignoring them; an unreadable
  `MATIMO_PRIVATE_KEY_FILE` is an error instead of being ignored.
- `register()`, `rotate_key()` and `start()` fail fast without an org API key.
- `gateway_chat_model()`, `gateway_llm()` and `gateway_model()` raise a clear
  `TypeError` for an `AsyncGovernor` (they previously crashed with "'coroutine'
  object is not subscriptable", and the CrewAI docstring wrongly promised support).
- CLI: `--version`, `register --force`; `doctor` and the other commands report a
  malformed key or bad configuration as an `error:` line, not a traceback.
- New public `Governor.flush()` / `AsyncGovernor.flush()`; the CLI no longer reaches
  into private attributes. `AgentSuspendedLocally` is picklable.

Found by the live check against a real Gateway:

- **`anthropic` >= 1.6 rejected `governor.httpx_client()`** (`TypeError: ... this SDK
  uses httpx2`). That release is built on `httpx2`, a separate library whose classes
  are unrelated to `httpx`'s, so a live, signed client was impossible for Anthropic.
  New `Governor.httpx2_client()` / `AsyncGovernor.httpx2_async_client()` (same live
  session token, per-request signature and transparent re-handshake, as `httpx2`
  clients) and `anthropic_http_client()`, which picks the client the installed
  `anthropic` accepts by reading its declared requirements. `openai` still accepts
  `httpx`; it now also ships an optional `httpx2` extra, so the same helper pattern
  applies if it ever requires it.

Tooling and docs: fixed 13 ruff errors and 8 unformatted files that would have
failed CI; README no longer claims the header-only client works with
`requireSignedRequests`; `CONTRIBUTING.md` documents the real setup and gate;
added Dependabot for `uv` and GitHub Actions.

### Fixed (2026-09-19, Google ADK runs stayed `running` in Gateway Observability)

- `MatimoPlugin` never emitted a `kind:"run"` span, and Gateway only ends a
  run on an explicit terminal run span (SERVER-CONTRACT section 7.3), so every
  ADK run stayed `running` until the staleness sweep. The plugin now opens the
  run in `before_run_callback` and closes it `completed` (`after_run_callback`)
  or `failed` (`on_run_error_callback`), keyed by ADK's invocation id. A run
  whose event stream the caller abandons early still falls back to the sweep,
  because ADK skips `after_run_callback` in that case.
- Added public `Governor.run_span()` / `AsyncGovernor.run_span()` for adapters
  whose framework owns the run and cannot use `governor.run()`.
- Every ADK and LangChain `llm` and `tool` span was silently dropped:
  the adapters pass `span_id` (and LangChain `parent_span_id`), but
  `telemetry.llm_span()` / `tool_span()` did not accept them, so each call
  raised a `TypeError` that the adapters' never-break-the-agent handler
  swallowed. Both builders now forward `span_id` / `parent_span_id`. The
  existing tests used a `MagicMock` governor, which accepts any kwargs, so
  they could not see it; new regression tests run the real builders. CrewAI,
  AutoGen and the generic adapter were audited and were not affected.

### Changed (2026-09-19, Python floor)

- `requires-python` raised from 3.11 to 3.13; classifiers list 3.13 and 3.14.
  CI runs 3.13 with every extra and 3.14 with every extra except `crewai`:
  CrewAI cannot be imported on Python 3.14 yet (verified with crewai 1.15.22:
  chromadb's `pydantic.v1` models raise `ConfigError`), so the `crewai`
  adapter is untested on 3.14. ruff and mypy now target 3.13.

### Fixed (2026-09-19, tool check could run a tool that policy held for approval)

- Gateway answers `PENDING` with **no `resumeToken`** when an identical tool
  check is already in flight (`reason: "duplicate_check_in_flight"`, raised
  by the server's duplicate-pending race handling in
  `GatewayToolCheckService.checkTool`). Every call site only polled when a
  token was present and only refused on `DENY`, so a tokenless `PENDING`
  fell straight through and the tool ran without the approval a policy
  required -- in `Governor.guard()`, `AsyncGovernor.guard()`, and all of the
  framework adapters (they share `adapters/_shared.py`). Reproduced with a
  mocked tokenless `PENDING`: the guarded function executed.
- New `ToolGovernor.check_and_wait()` / `AsyncToolGovernor.check_and_wait()`
  (also `Governor.check_and_wait()` / `AsyncGovernor.check_and_wait()`) is
  now the single place a check is resolved to a final `ALLOW`/`DENY`: a
  `PENDING` with a token is polled as before; a tokenless `PENDING` is
  re-checked after 0.5s, 1s, 2s and 4s (the in-flight check's answer is
  cached server-side for 15 minutes, so the retry normally returns the
  token) and, if it still has none, becomes `DENY` with reason
  `tool_check_pending_without_resume_token` -- fail closed. `guard()` and
  every adapter now call it. `check_tool()`/`await_decision()` are
  unchanged for callers who drive the two steps by hand; those callers must
  handle a tokenless `PENDING` themselves.
- Corrected two stale comments in `tools.py` (initial poll interval is 3s,
  not 2s; the jitter is a uniform +/-10%, not "decorrelated").

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
- `gateway_chat_model(provider="openai")` wires the sync `http_client` only, so
  `ChatOpenAI` async methods use a session header fixed at construction and are
  not signed. `gateway_*` helpers for LangChain, ADK and CrewAI need a sync
  `Governor`; there is no async-native equivalent yet.
- ADK plugin bookkeeping (`_open_runs`, `_pending_tool`) is not pruned for runs
  the caller abandons mid-stream.
