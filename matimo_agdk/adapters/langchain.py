"""LangChain adapter. Verified against langchain-core==1.6.3,
langchain-openai==1.6.2, langchain-anthropic==1.7.2.

Install with `pip install matimo-agdk[langchain]`.

Three pieces, used together:

- `MatimoCallbackHandler` (sync chains/agents) / `AsyncMatimoCallbackHandler`
  (async chains/agents, `.ainvoke()`/`.arun()`/an async agent loop) --
  registered once via `callbacks=[handler]` at chain/agent construction (or
  passed per-call). Records `run`/`llm`/`tool` telemetry spans automatically
  for every chain, LLM, and tool invocation LangChain's own callback system
  already fires. **Telemetry (and, in `mode="govern"`, a hard rapid-suspend
  stop) only -- see "What enforces what" below.**
- `govern_tools(tools, governor)` -- wraps each tool's `_run`/`_arun` so a
  policy DENY is raised *from inside the tool call*, as langchain_core's
  own `ToolException` (not this SDK's framework-agnostic `ToolDenied` --
  see below for why that distinction matters), which LangChain's
  `handle_tool_error` mechanism can catch and turn into a recoverable
  "observation" the agent's own reasoning loop can react to (try a
  different tool, ask a clarifying question, etc.), rather than an
  uncaught exception that crashes the whole chain. **The tool must be
  constructed with (or have) `handle_tool_error=True`** -- without it,
  `ToolException` still propagates just like any other exception; this is
  standard LangChain configuration, not something `govern_tools()` sets on
  your behalf, since it would be surprising for this SDK to silently change
  a tool's own configured error-handling policy.
- `gateway_chat_model(governor, model=..., provider="openai"|"anthropic")`
  -- a `ChatOpenAI`/`ChatAnthropic` already pointed at Gateway.

## What enforces what (read this before wiring only the callback handler)

**`MatimoCallbackHandler`/`AsyncMatimoCallbackHandler` alone cannot veto a
tool call gracefully.** LangChain *can* technically raise from
`on_tool_start()` to prevent a tool's `_run` from ever executing (verified
by reading `langchain_core/tools/base.py`: `BaseTool.run()` calls
`callback_manager.on_tool_start()` *before* entering the `try:` block that
wraps `_run`), but an exception raised there is never caught by the tool's
own `handle_tool_error`/`handle_validation_error` config -- it propagates
straight past `BaseTool.run()` entirely, typically crashing the whole
chain rather than giving the agent a recoverable "tool call denied"
observation to reason about. That is what "LangChain callbacks cannot
[gracefully] veto a tool call" means in practice, and why `govern_tools()`
exists as a separate, required step for real enforcement: raising
langchain_core's own `ToolException` from *inside* `_run`/`_arun` **is**
caught by `BaseTool.run()`'s normal error path (when the tool has
`handle_tool_error=True`), so a well-configured agent recovers gracefully
instead of crashing. **Verified live against a real Gateway that this only
holds for `ToolException` specifically** -- `BaseTool.run()`
(`langchain_core/tools/base.py`) special-cases exactly `ToolException` and
pydantic's `ValidationError`; every other exception type (this SDK's own
`ToolDenied` included) falls into its generic
`except (Exception, KeyboardInterrupt)` branch and is unconditionally
re-raised regardless of `handle_tool_error`. An earlier version of this
adapter raised `ToolDenied` here and the "IS caught" claim above did not
actually hold -- see CHANGELOG.md, 2026-09-18 live verification.

So: attach the callback handler for telemetry (and, in `mode="govern"`,
for a hard rapid-suspend stop at each LLM/tool boundary -- see below), and
always also call `govern_tools()` if you want DENY/PENDING to actually be
enforced rather than merely observed.

**Rapid suspend** (`mode="govern"` only): the callback handler calls
`governor.raise_if_suspended()` in `on_llm_start`/`on_chat_model_start`/
`on_tool_start`, all of which (verified) run before the corresponding
LLM/tool call, so a locally-known suspended state raises there and stops
the chain immediately (see the core's own honesty note on what "rapid" --
polled, not pushed -- really means). `mode="observe"` never calls this.

## Sync vs. async: use the matching handler

LangChain's synchronous callback dispatch (`handle_event`, used by
`.invoke()`/`.run()`) silently swallows exceptions raised from an `async
def` callback method -- confirmed by reading `langchain_core.callbacks
.manager._run_coros`'s own comment: "exceptions raised by these coroutines
are always logged and swallowed here ... regardless of the handler's
`raise_error` setting." That means an all-async handler used in a sync
chain would look like it's blocking, but a rapid-suspend exception would
be silently dropped. To avoid that trap, this module ships two separate
handler classes rather than one that tries to do both:

- `MatimoCallbackHandler(BaseCallbackHandler)` -- sync methods, use with
  `.invoke()`/`.run()`.
- `AsyncMatimoCallbackHandler(AsyncCallbackHandler)` -- async methods, use
  with `.ainvoke()`/`.arun()`/an async agent loop.

Both set `raise_error = True` so a raised exception (rapid suspend) is not
swallowed by LangChain's default log-and-continue behavior.

## LLM spans and gen_ai.* attributes

Both handlers report `gen_ai.usage.input_tokens`/`gen_ai.usage.output_tokens`
when the underlying `LLMResult.llm_output` carries a `token_usage` dict
(true for the OpenAI/Anthropic integrations), plus `model`/`finish_reasons`
when available. Spans correlate via the framework's own `run_id`/
`parent_run_id` UUID chain: the root of that chain becomes the emitted
`run_id`, and each node's own `run_id` becomes its `span_id` -- so an
agent's chain-start, its LLM calls, and its tool calls all land under one
`run_id` in Matimo's telemetry, without the caller needing to wrap
anything in `governor.run()` (though doing so still works and is
respected for spans emitted outside these callbacks).
"""

from __future__ import annotations

import functools
import time
from typing import Any
from uuid import UUID

from ._shared import (
    Mode,
    async_check_and_wait,
    async_raise_if_suspended,
    call_args_from,
    check_mode,
    emit_llm_span,
    emit_tool_span,
    sync_check_and_wait,
    sync_raise_if_suspended,
    truncate,
)

try:
    from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler
    from langchain_core.tools import ToolException
except ImportError as exc:  # pragma: no cover - exercised only when the extra is missing
    raise ImportError(
        "matimo_agdk.adapters.langchain requires the 'langchain' extra: "
        "pip install matimo-agdk[langchain]"
    ) from exc


def _now() -> float:
    return time.monotonic()


def _usage_from_llm_output(llm_output: dict[str, Any] | None) -> dict[str, Any]:
    if not llm_output:
        return {}
    usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
    out: dict[str, Any] = {}
    if usage.get("prompt_tokens") is not None:
        out["gen_ai.usage.input_tokens"] = usage["prompt_tokens"]
    if usage.get("completion_tokens") is not None:
        out["gen_ai.usage.output_tokens"] = usage["completion_tokens"]
    return out


class _RunTree:
    """Tracks LangChain's own run_id/parent_run_id tree so LLM/tool spans
    can carry a stable `run_id` (the tree's root) plus this node's own id
    as `span_id`/`parent_span_id`. Bounded: an entry is dropped as soon as
    that node's *_end/*_error fires."""

    def __init__(self) -> None:
        self._root_of: dict[UUID, str] = {}
        self._started_at: dict[UUID, float] = {}

    def start(self, run_id: UUID, parent_run_id: UUID | None) -> str:
        root = self._root_of.get(parent_run_id) if parent_run_id else None
        root = root or str(run_id)
        self._root_of[run_id] = root
        self._started_at[run_id] = _now()
        return root

    def finish(self, run_id: UUID) -> tuple[str, int]:
        root = self._root_of.pop(run_id, str(run_id))
        started = self._started_at.pop(run_id, _now())
        return root, int((_now() - started) * 1000)


class _LangChainSpans:
    """Span-emission logic shared by the sync and async handler, kept as
    composition (not a common base class) so each handler's MRO stays a
    clean single inheritance from the LangChain base class it needs."""

    def __init__(self, governor: Any, mode: Mode) -> None:
        self.governor = governor
        self.mode: Mode = check_mode(mode)
        self.tree = _RunTree()

    def on_chain_start(self, run_id: UUID, parent_run_id: UUID | None) -> None:
        self.tree.start(run_id, parent_run_id)

    def on_chain_end(self, run_id: UUID) -> None:
        self.tree.finish(run_id)

    def on_llm_start(self, run_id: UUID, parent_run_id: UUID | None) -> None:
        self.tree.start(run_id, parent_run_id)

    def on_llm_end(
        self,
        run_id: UUID,
        parent_run_id: UUID | None,
        *,
        model: str | None,
        llm_output: dict[str, Any] | None,
        finish_reasons: list[str] | None,
        status: str,
    ) -> None:
        root, duration_ms = self.tree.finish(run_id)
        attrs = _usage_from_llm_output(llm_output)
        emit_llm_span(
            self.governor,
            run_id=root,
            span_id=str(run_id),
            parent_span_id=str(parent_run_id) if parent_run_id else None,
            model=model,
            finish_reasons=finish_reasons,
            status=status,
            duration_ms=duration_ms,
            attributes=attrs or None,
        )

    def on_tool_start(self, run_id: UUID, parent_run_id: UUID | None) -> None:
        self.tree.start(run_id, parent_run_id)

    def on_tool_end(
        self,
        run_id: UUID,
        parent_run_id: UUID | None,
        *,
        tool_name: str,
        status: str,
        result: Any = None,
    ) -> None:
        root, duration_ms = self.tree.finish(run_id)
        emit_tool_span(
            self.governor,
            tool_name,
            run_id=root,
            span_id=str(run_id),
            parent_span_id=str(parent_run_id) if parent_run_id else None,
            status=status,
            duration_ms=duration_ms,
            call_id=str(run_id),
            result=truncate(result) if result is not None else None,
        )


def _extract_finish_reasons(response: Any) -> list[str] | None:
    try:
        generations = response.generations[0]
        reasons = [
            g.generation_info.get("finish_reason")
            for g in generations
            if getattr(g, "generation_info", None)
        ]
        return [r for r in reasons if r] or None
    except (AttributeError, IndexError, KeyError, TypeError):
        return None


class MatimoCallbackHandler(BaseCallbackHandler):  # type: ignore[misc]
    """Sync LangChain callback handler. Use with `.invoke()`/`.run()`.

    Telemetry (and, in `mode="govern"`, a hard rapid-suspend stop) only --
    see this module's docstring, "What enforces what": pair with
    `govern_tools()` for actual DENY/PENDING enforcement.
    """

    raise_error = True

    def __init__(self, governor: Any, mode: Mode = "govern") -> None:
        super().__init__()
        self._impl = _LangChainSpans(governor, mode)
        self.governor = governor
        self.mode: Mode = self._impl.mode

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._impl.on_chain_start(run_id, parent_run_id)

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._impl.on_chain_end(run_id)

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        if self.mode == "govern":
            sync_raise_if_suspended(self.governor)
        self._impl.on_llm_start(run_id, parent_run_id)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        if self.mode == "govern":
            sync_raise_if_suspended(self.governor)
        self._impl.on_llm_start(run_id, parent_run_id)

    def on_llm_end(
        self, response: Any, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
    ) -> None:
        llm_output = getattr(response, "llm_output", None)
        model = (llm_output or {}).get("model_name")
        self._impl.on_llm_end(
            run_id,
            parent_run_id,
            model=model,
            llm_output=llm_output,
            finish_reasons=_extract_finish_reasons(response),
            status="completed",
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._impl.on_llm_end(
            run_id, parent_run_id, model=None, llm_output=None, finish_reasons=None, status="error"
        )

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        if self.mode == "govern":
            sync_raise_if_suspended(self.governor)
        self._impl.on_tool_start(run_id, parent_run_id)

    def on_tool_end(
        self, output: Any, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
    ) -> None:
        tool_name = str(kwargs.get("name") or "tool")
        self._impl.on_tool_end(
            run_id, parent_run_id, tool_name=tool_name, status="completed", result=output
        )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        tool_name = str(kwargs.get("name") or "tool")
        self._impl.on_tool_end(
            run_id, parent_run_id, tool_name=tool_name, status="error", result=str(error)
        )


class AsyncMatimoCallbackHandler(AsyncCallbackHandler):  # type: ignore[misc]
    """Async twin of MatimoCallbackHandler. Use with `.ainvoke()`/`.arun()`
    or any async agent loop."""

    raise_error = True

    def __init__(self, governor: Any, mode: Mode = "govern") -> None:
        super().__init__()
        self._impl = _LangChainSpans(governor, mode)
        self.governor = governor
        self.mode: Mode = self._impl.mode

    async def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._impl.on_chain_start(run_id, parent_run_id)

    async def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._impl.on_chain_end(run_id)

    async def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        if self.mode == "govern":
            await async_raise_if_suspended(self.governor)
        self._impl.on_llm_start(run_id, parent_run_id)

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        if self.mode == "govern":
            await async_raise_if_suspended(self.governor)
        self._impl.on_llm_start(run_id, parent_run_id)

    async def on_llm_end(
        self, response: Any, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
    ) -> None:
        llm_output = getattr(response, "llm_output", None)
        model = (llm_output or {}).get("model_name")
        self._impl.on_llm_end(
            run_id,
            parent_run_id,
            model=model,
            llm_output=llm_output,
            finish_reasons=_extract_finish_reasons(response),
            status="completed",
        )

    async def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._impl.on_llm_end(
            run_id, parent_run_id, model=None, llm_output=None, finish_reasons=None, status="error"
        )

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        if self.mode == "govern":
            await async_raise_if_suspended(self.governor)
        self._impl.on_tool_start(run_id, parent_run_id)

    async def on_tool_end(
        self, output: Any, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
    ) -> None:
        tool_name = str(kwargs.get("name") or "tool")
        self._impl.on_tool_end(
            run_id, parent_run_id, tool_name=tool_name, status="completed", result=output
        )

    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        tool_name = str(kwargs.get("name") or "tool")
        self._impl.on_tool_end(
            run_id, parent_run_id, tool_name=tool_name, status="error", result=str(error)
        )


def _wrap_sync_run(
    inner: Any, tool_name: str, governor: Any, mode: Mode, category: str | None
) -> Any:
    @functools.wraps(inner)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        call_args = call_args_from(args, kwargs, exclude=("run_manager", "config"))
        if mode == "govern":
            sync_raise_if_suspended(governor)
            decision = sync_check_and_wait(governor, tool_name, call_args, category=category)
            if decision.denied:
                # Raise langchain_core's own ToolException, not the
                # framework-agnostic ToolDenied. Found live-testing against
                # a real Gateway: BaseTool.run()'s handle_tool_error/
                # handle_validation_error machinery (langchain_core/
                # tools/base.py) only special-cases ToolException and
                # pydantic's ValidationError -- every other exception type,
                # ToolDenied included, always falls into the generic
                # `except (Exception, KeyboardInterrupt)` branch and is
                # unconditionally re-raised regardless of
                # handle_tool_error's value. A bare ToolDenied here would
                # crash the whole chain exactly like an unhandled callback
                # exception would -- the opposite of this function's whole
                # purpose. See CHANGELOG.md, 2026-09-18 live verification.
                raise ToolException(decision.reason)
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
        call_args = call_args_from(args, kwargs, exclude=("run_manager", "config"))
        if mode == "govern":
            await async_raise_if_suspended(governor)
            decision = await async_check_and_wait(governor, tool_name, call_args, category=category)
            if decision.denied:
                # See _wrap_sync_run's identical comment: ToolException, not
                # ToolDenied, is what BaseTool.run()/arun()'s error-handling
                # actually special-cases.
                raise ToolException(decision.reason)
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


def govern_tools(
    tools: list[Any],
    governor: Any,
    *,
    mode: Mode = "govern",
    category: str | None = None,
) -> list[Any]:
    """Wraps each tool's `_run` (and `_arun`, if the tool overrides it) in
    place so a policy check runs before the tool's real body, and a tool
    span is recorded after. Mutates and returns the same list of tool
    objects -- call this once, right after building your tool list, before
    handing it to an agent/executor.

    `mode="observe"`: records tool spans, never calls `check_tool()`.
    `mode="govern"` (default): also calls `check_tool()`/`await_decision()`
    and raises langchain_core's own `ToolException` on DENY (caught by
    LangChain's own `handle_tool_error` machinery only if the tool has that
    configured -- see this module's docstring for why it must be
    `ToolException` specifically, not this SDK's own `ToolDenied`), plus
    `raise_if_suspended()` before each call.

    Uses `functools.wraps` on the replacement so `inspect.signature()`
    still resolves through to the original `_run`/`_arun` -- required
    because `BaseTool.run()` inspects `_run`'s own signature to decide
    whether to inject `run_manager`/`config` kwargs; without this, a tool
    whose `_run` declares either parameter would silently stop receiving
    it once wrapped (verified against langchain-core==1.6.3).
    """
    check_mode(mode)
    for t in tools:
        name = getattr(t, "name", None) or type(t).__name__
        base_run = getattr(t, "_run", None)
        if base_run is not None:
            t._run = _wrap_sync_run(base_run, name, governor, mode, category)  # noqa: SLF001

        # Only wrap _arun if the tool overrides the BaseTool default (the
        # default _arun already delegates to _run via a thread, so
        # wrapping both would double-check the same call).
        defines_own_arun = "_arun" in type(t).__dict__ or any(
            "_arun" in base.__dict__ for base in type(t).__mro__[1:-1]
        )
        base_arun = getattr(t, "_arun", None)
        if defines_own_arun and base_arun is not None:
            t._arun = _wrap_async_run(base_arun, name, governor, mode, category)  # noqa: SLF001
    return tools


def gateway_chat_model(
    governor: Any,
    *,
    model: str | None = None,
    provider: str = "openai",
    **kwargs: Any,
) -> Any:
    """Returns a `ChatOpenAI` (provider="openai", default) or
    `ChatAnthropic` (provider="anthropic") already pointed at Gateway.

    `provider="openai"` -> full support: the session token *and* a fresh
    `Matimo-Agent-Signature` are attached per request, via
    `governor.httpx_client()`'s request event hook
    (`ChatOpenAI.http_client`/`http_async_client` are real, documented
    constructor kwargs in langchain-openai>=1.6, verified directly against
    the installed package's pydantic fields).

    `provider="anthropic"` -> **session-header-only, signing effectively
    off.** `langchain-anthropic==1.7.2`'s `ChatAnthropic` builds its own
    internal `httpx.Client`/`AsyncClient` (via private `_client`/
    `_async_client` cached properties) and does not accept a
    caller-supplied `http_client`/`http_async_client` -- verified by
    reading `langchain_anthropic/chat_models.py` directly, not assumed.
    The session token is attached via `default_headers` (computed once, at
    construction time, so it works with Gateway's default hour-long
    session TTL but is not refreshed mid-process on renewal) and no
    `Matimo-Agent-Signature` is ever sent on this path. This is a real,
    disclosed limitation, not "coming soon": if your tenant might ever
    flip on `requireSignedRequests`, route that traffic through
    `gateway_chat_model(provider="openai")` against Gateway's
    OpenAI-compatible endpoint instead (Gateway accepts either wire format
    for the same underlying model), or build a raw `anthropic.Client`
    yourself using `governor.anthropic_client_kwargs()` plus your own
    signing.
    """
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model or "matimo/auto",
            base_url=governor.config.base_url,
            api_key=governor.config.api_key,
            default_headers=governor.openai_client_kwargs()["default_headers"],
            http_client=governor.httpx_client(),
            **kwargs,
        )
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        # ChatAnthropic exposes api_key (sent as x-api-key, which Gateway
        # ignores) but not auth_token, so the org key travels in an explicit
        # Authorization header instead; api_key is a placeholder that keeps
        # the SDK's own "credentials present" check happy.
        headers = dict(governor.anthropic_client_kwargs()["default_headers"])
        headers["Authorization"] = f"Bearer {governor.config.api_key}"
        return ChatAnthropic(  # type: ignore[call-arg]
            # `model` is a real, working kwarg at runtime (a pydantic
            # populate_by_name alias) -- mypy's view of ChatAnthropic's
            # generated __init__ signature doesn't expose the alias.
            model=model or "matimo/auto",
            base_url=governor.config.base_url,
            api_key="matimo-gateway",  # type: ignore[arg-type]
            default_headers=headers,
            **kwargs,
        )
    raise ValueError(f"provider must be 'openai' or 'anthropic', got {provider!r}")
