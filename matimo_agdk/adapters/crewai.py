"""CrewAI adapter. Verified against crewai==1.15.22.

Install with `pip install matimo-agdk[crewai]`.

Two pieces:

- `govern_crew(crew_or_agents, governor, mode=...)` -- wraps every tool
  reachable from a `Crew` (its agents' tools, plus any task-level tool
  overrides), a bare list of `Agent`s, or a bare list of tools, in place.
  Also exposed as `govern_tool(tool, governor, mode=...)` for a single
  tool. Same monkeypatch-the-instance technique as the LangChain adapter's
  `govern_tools()`: CrewAI's `BaseTool` is a pydantic v2 model, and setting
  an underscore-prefixed instance attribute (`tool._run = ...`) on one is
  verified to work (pydantic v2 treats it as a private attribute, not a
  field needing validation).
- `gateway_llm(governor, model=...)` -- a CrewAI `LLM` pointed at Gateway.
  **Session-header-only, signing off** -- see its own docstring.

## What enforces what

`_run`/`_arun` wrapping is the only enforcement mechanism for CrewAI (there
is no callback-based veto point analogous to a LangChain
`BaseCallbackHandler` in CrewAI's tool-dispatch path). `mode="govern"`
(default) raises `ToolDenied` from inside the wrapped `_run`/`_arun` on a
policy DENY; CrewAI's own tool-execution wrapper
(`crewai.tools.tool_usage.ToolUsage`) catches a raised exception from a
tool call and feeds the error back to the agent as an observation (the
same graceful-recovery shape LangChain's `handle_tool_error` provides),
rather than crashing the whole crew run. `mode="observe"` only records tool
spans, never calls `check_tool()`.

Rapid suspend (`mode="govern"` only): `governor.raise_if_suspended()` is
called before each wrapped tool's real body runs.

## LLM spans and gen_ai.* attributes

`gateway_llm()`'s `MatimoInterceptor` (see `make_interceptor()`) times every
outbound LLM request and emits one LLM span per call: `duration_ms`,
`status` (`"completed"`/`"error"`, from the response status code), the
requested `model` always, plus the response's own `model` and
`gen_ai.usage.*` token counts when the response body is JSON (a streaming
`text/event-stream` response is left untouched -- this interceptor never
reads its body, so CrewAI's own streaming behavior isn't affected, at the
cost of no token-usage enrichment for a streamed call).

**Run-id correlation.** Unlike LangChain's run_id/parent_run_id tree or
ADK's invocation_id, CrewAI exposes no natural per-`kickoff()` id this
interceptor or `govern_tool()`'s tool wrapper can see. Every span here (LLM
and tool alike) uses the ambient `governor.run()` id when one is active,
and otherwise gives each call a one-span run of its own, opened and closed
around it -- the same fallback `emit_llm_span()`/`emit_tool_span()` use
everywhere in this SDK. **Wrap `crew.kickoff()` in
`with governor.run("my-crew-run"):`** to get one correlated run per crew
execution in the Gateway Observability Hub; without it, every LLM call and
every tool call in the crew shows up as its own separate, uncorrelated
(but completed, not stuck `running`) run.
"""

from __future__ import annotations

import functools
import json
import time
from contextvars import ContextVar
from typing import Any

from ..exceptions import ToolDenied
from ._shared import (
    Mode,
    async_check_and_wait,
    async_raise_if_suspended,
    call_args_from,
    check_mode,
    default_llm_headers,
    emit_llm_span,
    emit_tool_span,
    sync_check_and_wait,
    sync_raise_if_suspended,
)

try:
    import crewai  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised only when the extra is missing
    raise ImportError(
        "matimo_agdk.adapters.crewai requires the 'crewai' extra: pip install matimo-agdk[crewai]"
    ) from exc


def _now() -> float:
    return time.monotonic()


# Bridges on_outbound -> on_inbound (and aon_outbound -> aon_inbound): CrewAI's
# HTTPTransport/AsyncHTTPTransport call these two hooks back-to-back within one
# synchronous call stack (per request), but the framework gives on_inbound only
# the httpx.Response -- not the originating httpx.Request -- so there is no
# object to key a dict by. A ContextVar (thread-local for the sync transport,
# task-local for the async one, same mechanism governor.py's `_current_run`
# already relies on) is scoped correctly for both a threaded sync interceptor
# and concurrent async calls sharing one MatimoInterceptor instance.
_PENDING_LLM_CALL: ContextVar[tuple[float, str | None] | None] = ContextVar(
    "matimo_agdk_crewai_pending_llm_call", default=None
)


def _model_from_request_body(content: bytes | None) -> str | None:
    if not content:
        return None
    try:
        data = json.loads(content)
    except (ValueError, TypeError):
        return None
    model = data.get("model") if isinstance(data, dict) else None
    return str(model) if model else None


def _usage_and_model_from_body(body: bytes) -> tuple[dict[str, Any], str | None]:
    """Best-effort `gen_ai.usage.*` attributes and the response's own
    `model` field from an OpenAI-compatible chat-completions JSON body.
    Never raises -- an unparseable or unexpected body just yields no
    enrichment; the LLM span itself (duration/status/requested model)
    still emits either way."""
    try:
        data = json.loads(body)
    except (ValueError, TypeError):
        return {}, None
    if not isinstance(data, dict):
        return {}, None
    attrs: dict[str, Any] = {}
    usage = data.get("usage")
    if isinstance(usage, dict):
        if usage.get("prompt_tokens") is not None:
            attrs["gen_ai.usage.input_tokens"] = usage["prompt_tokens"]
        if usage.get("completion_tokens") is not None:
            attrs["gen_ai.usage.output_tokens"] = usage["completion_tokens"]
    model = data.get("model")
    return attrs, (str(model) if model else None)


def _wrap_sync_run(
    inner: Any, tool_name: str, governor: Any, mode: Mode, category: str | None
) -> Any:
    @functools.wraps(inner)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        call_args = call_args_from(args, kwargs)
        if mode == "govern":
            sync_raise_if_suspended(governor)
            decision = sync_check_and_wait(governor, tool_name, call_args, category=category)
            if decision.denied:
                raise ToolDenied(decision.reason)
        started = _now()
        status = "completed"
        try:
            return inner(*args, **kwargs)
        except Exception:
            status = "error"
            raise
        finally:
            emit_tool_span(
                governor,
                tool_name,
                status=status,
                duration_ms=int((_now() - started) * 1000),
                arguments=call_args,
            )

    return wrapper


def _wrap_async_run(
    inner: Any, tool_name: str, governor: Any, mode: Mode, category: str | None
) -> Any:
    @functools.wraps(inner)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        call_args = call_args_from(args, kwargs)
        if mode == "govern":
            await async_raise_if_suspended(governor)
            decision = await async_check_and_wait(governor, tool_name, call_args, category=category)
            if decision.denied:
                raise ToolDenied(decision.reason)
        started = _now()
        status = "completed"
        try:
            return await inner(*args, **kwargs)
        except Exception:
            status = "error"
            raise
        finally:
            emit_tool_span(
                governor,
                tool_name,
                status=status,
                duration_ms=int((_now() - started) * 1000),
                arguments=call_args,
            )

    return wrapper


def govern_tool(
    tool: Any, governor: Any, *, mode: Mode = "govern", category: str | None = None
) -> Any:
    """Wraps one CrewAI `BaseTool` instance's `_run` (and `_arun`, if it
    overrides the base class's default `NotImplementedError` stub) in
    place. Returns the same instance for convenience/chaining."""
    check_mode(mode)
    if getattr(tool, "_matimo_governed", False):
        return tool
    name = getattr(tool, "name", None) or type(tool).__name__
    base_run = getattr(tool, "_run", None)
    if base_run is not None:
        tool._run = _wrap_sync_run(base_run, name, governor, mode, category)  # noqa: SLF001

    # CrewAI's default `_arun` just raises NotImplementedError. Wrapping it would
    # run a policy check (and possibly open a human-approval request) for a call
    # that can never execute, so only a real override is wrapped.
    from crewai.tools import BaseTool as CrewBaseTool

    overrides_arun = getattr(type(tool), "_arun", None) is not CrewBaseTool._arun
    base_arun = getattr(tool, "_arun", None)
    if overrides_arun and base_arun is not None:
        tool._arun = _wrap_async_run(base_arun, name, governor, mode, category)  # noqa: SLF001

    try:
        tool._matimo_governed = True  # noqa: SLF001
    except Exception:  # noqa: BLE001 - a stricter pydantic config could reject this; harmless
        pass
    return tool


def _tools_of(agent: Any) -> list[Any]:
    return list(getattr(agent, "tools", None) or [])


def govern_crew(
    crew_or_agents: Any,
    governor: Any,
    *,
    mode: Mode = "govern",
    category: str | None = None,
) -> Any:
    """Wraps every tool reachable from `crew_or_agents`: a `Crew` (its
    `.agents`' tools plus any `.tasks`' own tool overrides), a bare list of
    `Agent`s, or a bare list of tools. Call this once, right after building
    the crew/agents, before `crew.kickoff()`. Returns `crew_or_agents`
    unchanged (tools are patched in place) for convenience.
    """
    check_mode(mode)
    tools: list[Any] = []

    if hasattr(crew_or_agents, "agents"):  # Crew-shaped
        for agent in getattr(crew_or_agents, "agents", None) or []:
            tools.extend(_tools_of(agent))
        for task in getattr(crew_or_agents, "tasks", None) or []:
            task_tools = getattr(task, "tools", None)
            if task_tools:
                tools.extend(task_tools)
    elif isinstance(crew_or_agents, (list, tuple)):
        for item in crew_or_agents:
            if hasattr(item, "tools"):  # Agent-shaped
                tools.extend(_tools_of(item))
            else:  # a bare tool
                tools.append(item)
    else:  # a single Agent
        tools.extend(_tools_of(crew_or_agents))

    seen: set[int] = set()
    for t in tools:
        if id(t) in seen:
            continue
        seen.add(id(t))
        govern_tool(t, governor, mode=mode, category=category)
    return crew_or_agents


def gateway_llm(governor: Any, *, model: str | None = None, **kwargs: Any) -> Any:
    """Returns a `crewai.LLM` pointed at Gateway's OpenAI-compatible
    endpoint (`custom_openai=True` forces CrewAI's native OpenAI provider
    path rather than a LiteLLM passthrough).

    **Per-request headers and signing, via CrewAI's transport interceptor
    (2026-09-18).** CrewAI's `OpenAICompletion` accepts an `interceptor`
    (`crewai.llms.hooks.base.BaseInterceptor`) that sees every outbound
    `httpx.Request`. `MatimoInterceptor` uses it to attach the live session
    token, the current `governor.run()` id, and, when signing is enabled, a
    `Matimo-Agent-Signature` JWS over that request's exact body bytes, so
    CrewAI LLM calls correlate to runs in the call log and survive
    `requireSignedRequests=true`. The same interceptor also emits an LLM
    span per call -- see this module's docstring, "LLM spans and gen_ai.*
    attributes". `additional_params.extra_headers` stays as a static
    fallback. Needs a sync `Governor` (an `AsyncGovernor` raises a clear
    `TypeError`: its session handshake must be awaited, and this builder is
    synchronous); a sync `Governor` still serves async CrewAI code. Session
    expiry is not retried here; the default session TTL is one hour.
    """
    headers = default_llm_headers(governor, "gateway_llm")
    model_name = model or "matimo/auto"
    from crewai import LLM

    interceptor = make_interceptor(governor)

    return LLM(  # type: ignore[call-arg]
        # `custom_openai` is a real, working kwarg at runtime -- it is
        # popped out of **kwargs by LLM.__new__ before routing to the
        # native OpenAI provider (see this function's docstring), which is
        # why it isn't on LLM's own typed constructor signature.
        model=f"openai/{model_name}",
        base_url=governor.config.base_url,
        api_key=governor.config.api_key,
        custom_openai=True,
        additional_params={"extra_headers": headers},
        interceptor=interceptor,
        **kwargs,
    )


def make_interceptor(governor: Any) -> Any:
    """A CrewAI `BaseInterceptor` that attaches `governor.request_headers()`
    (live session token, run id, body signature) to every outbound request,
    and emits one LLM span per call (see this module's docstring, "LLM
    spans and gen_ai.* attributes", for the exact shape and the run-id
    correlation caveat). Public so it can be reused on a crewai `LLM` you
    construct yourself.

    Response-body reading happens here only for a non-streaming response
    (`content-type` without `text/event-stream`), and only *after* the
    transport has handed the response back to this interceptor -- calling
    `.read()`/`.aread()` here is safe because httpx caches the body on
    first read (`Response._content`), so CrewAI's own subsequent read of
    the same response reuses that cache rather than re-touching the
    network (verified against httpx's `Response.read()`/`iter_bytes()`).
    """
    import httpx
    from crewai.llms.hooks.base import BaseInterceptor

    class MatimoInterceptor(BaseInterceptor[httpx.Request, httpx.Response]):  # type: ignore[misc]
        def on_outbound(self, message: httpx.Request) -> httpx.Request:
            for key, value in governor.request_headers(message.content or b"").items():
                message.headers[key] = value
            _PENDING_LLM_CALL.set((_now(), _model_from_request_body(message.content)))
            return message

        def on_inbound(self, message: httpx.Response) -> httpx.Response:
            started, request_model = _PENDING_LLM_CALL.get() or (_now(), None)
            attrs: dict[str, Any] = {}
            response_model = None
            if "text/event-stream" not in message.headers.get("content-type", ""):
                try:
                    message.read()
                    attrs, response_model = _usage_and_model_from_body(message.content)
                except Exception:  # noqa: BLE001 -- telemetry must never break the real call
                    pass
            emit_llm_span(
                governor,
                model=response_model or request_model,
                provider="openai",
                status="completed" if message.is_success else "error",
                duration_ms=int((_now() - started) * 1000),
                attributes=attrs or None,
            )
            return message

        async def aon_outbound(self, message: httpx.Request) -> httpx.Request:
            return self.on_outbound(message)

        async def aon_inbound(self, message: httpx.Response) -> httpx.Response:
            started, request_model = _PENDING_LLM_CALL.get() or (_now(), None)
            attrs: dict[str, Any] = {}
            response_model = None
            if "text/event-stream" not in message.headers.get("content-type", ""):
                try:
                    await message.aread()
                    attrs, response_model = _usage_and_model_from_body(message.content)
                except Exception:  # noqa: BLE001
                    pass
            emit_llm_span(
                governor,
                model=response_model or request_model,
                provider="openai",
                status="completed" if message.is_success else "error",
                duration_ms=int((_now() - started) * 1000),
                attributes=attrs or None,
            )
            return message

    return MatimoInterceptor()
