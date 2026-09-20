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
`run_id` in Matimo's telemetry.

## One span per tool call

A tool wrapped by `govern_tools()` and also seen by a callback handler would
otherwise be reported twice. The governing wrapper is canonical, because it is
the only one that sees the decision: it emits the single tool span, with the
call's arguments, status (`completed`/`error`/`denied`), the result when
`capture_tool_results` is on, a `matimo.degraded_mode` marker when Gateway was
unreachable and the call ran fail-open, and LangChain's own `run_id`/
`parent_run_id` as `span_id`/`parent_span_id`, so it sits in the same run as
the chain's LLM spans. The handler notices that the wrapper already reported the
call and skips its own span. A tool that is *not* wrapped still gets the
handler's span.

The wrapper finds its own LangChain run through `langchain_core`'s
`var_child_runnable_config`, the contextvar `BaseTool.run()`/`arun()` set around
`_run`/`_arun` for nested runnables. If the wrapper cannot find it (the tool's
`_run` called directly, outside `BaseTool.run()`), it still emits its span, just
without those ids.

A denied call's span (`denied`) is emitted by the wrapper as well and carries the
same ids, so with a handler attached it is tied to the run tree; without a handler
there is no run tree to tie it to. LangChain reports a `ToolException` that
`handle_tool_error` absorbed as an ordinary `on_tool_end`; the handler does not try
to reclassify that, which is one more reason the wrapper's span is the canonical one.

A Gateway outage that leaves a tool check unanswered (fail-closed) is raised as a
`ToolException` too, so a tool with `handle_tool_error=True` hands the agent a
recoverable observation instead of crashing the chain.

## LLM provider

An LLM span's `gen_ai.provider.name` is derived from the serialized class name
LangChain passes to `on_llm_start` (`ChatOpenAI` gives `openai`, `ChatAnthropic`
gives `anthropic`, `ChatGoogleGenerativeAI` gives `google`); an unknown class
gives no provider. It names the client class, which is not always the upstream
provider when the call is routed through Gateway.

## Run lifecycle

Gateway only ends a run on an explicit terminal `kind:"run"` span
(docs/SERVER-CONTRACT.md 7.3), so who owns the run matters:

- Inside a `governor.run()` block, every span the handler emits joins that
  run (`governor.run()` opens and closes it). Separate `.invoke()` calls in
  one block therefore show up as one run, not one run each.
- Outside one, the handler opens a run for each parentless chain/LLM/tool
  call (`status="running"`) and closes it `completed`, or `failed` if that
  call raises. Without this, each such call left a run `running` until the
  staleness sweep.
"""

from __future__ import annotations

import functools
import time
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from .._outage import degraded_attributes
from ..exceptions import ToolCheckUnavailable
from ..governor import current_run_id
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
    from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler
    from langchain_core.tools import ToolException
except ImportError as exc:  # pragma: no cover - exercised only when the extra is missing
    raise ImportError(
        "matimo_agdk.adapters.langchain requires the 'langchain' extra: "
        "pip install matimo-agdk[langchain]"
    ) from exc


def _now() -> float:
    return time.monotonic()


# The id of the tool whose governed call is in progress in this context. A tool
# without its own coroutine has its async call routed by LangChain through the
# (already wrapped) `_run` in an executor thread, which copies this context; the
# inner wrapper sees its own id here and skips, so one call is checked once and
# a PENDING approval is requested once, not twice.
_GOVERNED_TOOL: ContextVar[int | None] = ContextVar("matimo_agdk_langchain_governed", default=None)


@dataclass
class _ToolCall:
    """A tool call a callback handler has started and not yet finished. `emitted`
    flips to True when a governing wrapper has reported the call, so the handler
    does not report it a second time."""

    root: str
    parent_run_id: UUID | None
    emitted: bool = False


# LangChain tool run id -> its in-flight call. Keyed by that id (unique per call, and
# the same id the wrapper reads back from `BaseTool.run()`'s config context) rather
# than by tool name, so two concurrent calls of one tool cannot be confused.
_TOOL_CALLS: dict[UUID, _ToolCall] = {}


def _current_tool_call() -> tuple[UUID, _ToolCall] | None:
    """The in-flight call the running code is inside, if a Matimo handler
    registered it. `BaseTool.run()`/`arun()` set `var_child_runnable_config` around
    `_run`/`_arun` with a child callback manager whose `parent_run_id` is the tool's
    own run id."""
    try:
        from langchain_core.runnables.config import var_child_runnable_config

        config = var_child_runnable_config.get()
        run_id = getattr(config.get("callbacks") if config else None, "parent_run_id", None)
    except Exception:  # noqa: BLE001 -- a langchain change must not break tool calls
        return None
    if run_id is None:
        return None
    call = _TOOL_CALLS.get(run_id)
    return (run_id, call) if call is not None else None


def _span_ids(current: tuple[UUID, _ToolCall] | None) -> dict[str, Any]:
    """The framework-side identity of a tool call, as `emit_tool_span()` kwargs
    (`run_id` is popped by the caller, the rest passed through)."""
    if current is None:
        return {}
    run_id, call = current
    ids: dict[str, Any] = {"run_id": call.root, "span_id": str(run_id), "call_id": str(run_id)}
    if call.parent_run_id is not None:
        ids["parent_span_id"] = str(call.parent_run_id)
    return ids


def _mark_emitted(current: tuple[UUID, _ToolCall] | None) -> None:
    if current is not None:
        current[1].emitted = True


# Class name in the serialized id LangChain passes to `on_llm_start` -> the
# `gen_ai.provider.name` of its span. Anything not listed gets no provider.
_PROVIDER_BY_CLASS = {
    "ChatOpenAI": "openai",
    "OpenAI": "openai",
    "ChatAnthropic": "anthropic",
    "AnthropicLLM": "anthropic",
    "ChatGoogleGenerativeAI": "google",
    "GoogleGenerativeAI": "google",
    "ChatVertexAI": "google",
    "VertexAI": "google",
}


def _provider_from_serialized(serialized: dict[str, Any] | None) -> str | None:
    ident = (serialized or {}).get("id")
    if isinstance(ident, list) and ident:
        return _PROVIDER_BY_CLASS.get(str(ident[-1]))
    return None


def _node_name(serialized: dict[str, Any] | None, kwargs: dict[str, Any]) -> str | None:
    """A readable name for the run a root LangChain node opens: the explicit
    `name` LangChain passes, else the serialized `name`, else its class name."""
    name = kwargs.get("name") or (serialized or {}).get("name")
    if not name:
        ident = (serialized or {}).get("id")
        if isinstance(ident, list) and ident:
            name = ident[-1]
    return str(name) if name else None


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

    def start(
        self, run_id: UUID, parent_run_id: UUID | None, ambient: str | None = None
    ) -> tuple[str, bool]:
        """Returns `(root, owned)`. A node under a known parent joins that
        parent's run; a parentless node joins the caller's ambient
        `governor.run()` if there is one; otherwise it is the root of a run
        of its own (`owned`), which the caller must open and close."""
        root = self._root_of.get(parent_run_id) if parent_run_id else None
        owned = root is None and ambient is None
        root = root or ambient or str(run_id)
        self._root_of[run_id] = root
        self._started_at[run_id] = _now()
        return root, owned

    def root_of(self, run_id: UUID) -> str:
        return self._root_of.get(run_id, str(run_id))

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
        # Runs this handler opened itself (no parent, no ambient governor.run()),
        # keyed by the root node's LangChain run_id -> (name, started).
        self._owned: dict[UUID, tuple[str, float]] = {}
        self._provider: dict[UUID, str | None] = {}

    def _start(self, run_id: UUID, parent_run_id: UUID | None, name: str | None) -> None:
        root, owned = self.tree.start(run_id, parent_run_id, ambient=current_run_id())
        if not owned:
            return
        run_name = name or "langchain-run"
        self._owned[run_id] = (run_name, _now())
        try:
            self.governor.run_span(root, status="running", name=run_name)
        except Exception:  # noqa: BLE001 -- telemetry must never break the caller's agent
            pass

    def _close(self, run_id: UUID, status: str) -> None:
        # Gateway only ends a run on an explicit terminal `kind:"run"` span
        # (docs/SERVER-CONTRACT.md 7.3); without this a run this handler opened
        # stays `running` until the staleness sweep.
        opened = self._owned.pop(run_id, None)
        if opened is None:
            return
        name, started = opened
        try:
            self.governor.run_span(
                str(run_id),
                status="completed" if status == "completed" else "failed",
                name=name,
                duration_ms=int((_now() - started) * 1000),
            )
        except Exception:  # noqa: BLE001
            pass

    def on_chain_start(
        self, run_id: UUID, parent_run_id: UUID | None, name: str | None = None
    ) -> None:
        self._start(run_id, parent_run_id, name)

    def on_chain_end(self, run_id: UUID, status: str = "completed") -> None:
        self.tree.finish(run_id)
        self._close(run_id, status)

    def on_llm_start(
        self,
        run_id: UUID,
        parent_run_id: UUID | None,
        name: str | None = None,
        provider: str | None = None,
    ) -> None:
        self._provider[run_id] = provider
        self._start(run_id, parent_run_id, name)

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
            provider=self._provider.pop(run_id, None),
            finish_reasons=finish_reasons,
            status=status,
            duration_ms=duration_ms,
            attributes=attrs or None,
        )
        self._close(run_id, status)

    def on_tool_start(
        self, run_id: UUID, parent_run_id: UUID | None, name: str | None = None
    ) -> None:
        self._start(run_id, parent_run_id, name)
        _TOOL_CALLS[run_id] = _ToolCall(self.tree.root_of(run_id), parent_run_id)

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
        call = _TOOL_CALLS.pop(run_id, None)
        # A governing wrapper that already reported this call is canonical (it saw
        # the decision); reporting it again would show one call as two spans.
        if call is None or not call.emitted:
            emit_tool_span(
                self.governor,
                tool_name,
                run_id=root,
                span_id=str(run_id),
                parent_span_id=str(parent_run_id) if parent_run_id else None,
                status=status,
                duration_ms=duration_ms,
                call_id=str(run_id),
                result=result,
            )
        self._close(run_id, status)


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
        self._impl.on_chain_start(run_id, parent_run_id, _node_name(serialized, kwargs))

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._impl.on_chain_end(run_id)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        # A chain that raises fires on_chain_error, never on_chain_end; without
        # this its run-tree entry would be retained for the life of the process.
        self._impl.on_chain_end(run_id, status="error")

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
        self._impl.on_llm_start(
            run_id,
            parent_run_id,
            _node_name(serialized, kwargs),
            _provider_from_serialized(serialized),
        )

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
        self._impl.on_llm_start(
            run_id,
            parent_run_id,
            _node_name(serialized, kwargs),
            _provider_from_serialized(serialized),
        )

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
        self._impl.on_tool_start(run_id, parent_run_id, _node_name(serialized, kwargs))

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
        self._impl.on_chain_start(run_id, parent_run_id, _node_name(serialized, kwargs))

    async def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._impl.on_chain_end(run_id)

    async def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._impl.on_chain_end(run_id, status="error")

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
        self._impl.on_llm_start(
            run_id,
            parent_run_id,
            _node_name(serialized, kwargs),
            _provider_from_serialized(serialized),
        )

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
        self._impl.on_llm_start(
            run_id,
            parent_run_id,
            _node_name(serialized, kwargs),
            _provider_from_serialized(serialized),
        )

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
        self._impl.on_tool_start(run_id, parent_run_id, _node_name(serialized, kwargs))

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
    inner: Any, tool_name: str, governor: Any, mode: Mode, category: str | None, tool_id: int
) -> Any:
    @functools.wraps(inner)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if _GOVERNED_TOOL.get() == tool_id:
            return inner(*args, **kwargs)  # already governed by the async wrapper above us
        call_args = call_args_from(args, kwargs, exclude=("run_manager", "config"))
        current = _current_tool_call()
        ids = _span_ids(current)
        span_run_id = ids.pop("run_id", None)
        decision = None
        if mode == "govern":
            sync_raise_if_suspended(governor)
            try:
                decision = sync_check_and_wait(
                    governor,
                    tool_name,
                    call_args,
                    category=category,
                    run_id=span_run_id,
                    span_extra=ids,
                )
            except ToolCheckUnavailable as exc:
                # Gateway could not answer and the failure mode is fail-closed. Like a
                # DENY, this must be a ToolException for LangChain's
                # `handle_tool_error` to turn it into a recoverable observation.
                raise ToolException(str(exc)) from exc
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
                _mark_emitted(current)
                raise ToolException(decision.reason)
        started = _now()
        status = "completed"
        span_result: Any = None
        try:
            result = inner(*args, **kwargs)
            span_result = result
            return result
        except Exception as exc:
            status = "error"
            span_result = str(exc)
            raise
        finally:
            emit_tool_span(
                governor,
                tool_name,
                run_id=span_run_id,
                status=status,
                duration_ms=int((_now() - started) * 1000),
                arguments=call_args,
                result=span_result,
                attributes=degraded_attributes(decision),
                **ids,
            )
            _mark_emitted(current)

    return wrapper


def _wrap_async_run(
    inner: Any, tool_name: str, governor: Any, mode: Mode, category: str | None, tool_id: int
) -> Any:
    @functools.wraps(inner)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        if _GOVERNED_TOOL.get() == tool_id:
            return await inner(*args, **kwargs)
        call_args = call_args_from(args, kwargs, exclude=("run_manager", "config"))
        current = _current_tool_call()
        ids = _span_ids(current)
        span_run_id = ids.pop("run_id", None)
        decision = None
        if mode == "govern":
            await async_raise_if_suspended(governor)
            try:
                decision = await async_check_and_wait(
                    governor,
                    tool_name,
                    call_args,
                    category=category,
                    run_id=span_run_id,
                    span_extra=ids,
                )
            except ToolCheckUnavailable as exc:
                # See _wrap_sync_run: a recoverable ToolException, like a DENY.
                raise ToolException(str(exc)) from exc
            if decision.denied:
                # See _wrap_sync_run's identical comment: ToolException, not
                # ToolDenied, is what BaseTool.run()/arun()'s error-handling
                # actually special-cases.
                _mark_emitted(current)
                raise ToolException(decision.reason)
        started = _now()
        status = "completed"
        span_result: Any = None
        marker = _GOVERNED_TOOL.set(tool_id)
        try:
            result = await inner(*args, **kwargs)
            span_result = result
            return result
        except Exception as exc:
            status = "error"
            span_result = str(exc)
            raise
        finally:
            _GOVERNED_TOOL.reset(marker)
            emit_tool_span(
                governor,
                tool_name,
                run_id=span_run_id,
                status=status,
                duration_ms=int((_now() - started) * 1000),
                arguments=call_args,
                result=span_result,
                attributes=degraded_attributes(decision),
                **ids,
            )
            _mark_emitted(current)

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
        if getattr(t, "_matimo_governed", False):
            continue  # govern_tools() twice must not stack two checks on one call
        name = getattr(t, "name", None) or type(t).__name__
        tool_id = id(t)
        base_run = getattr(t, "_run", None)
        if base_run is not None:
            t._run = _wrap_sync_run(base_run, name, governor, mode, category, tool_id)  # noqa: SLF001
        base_arun = getattr(t, "_arun", None)
        if base_arun is not None:
            # Wrapped unconditionally; the _GOVERNED_TOOL marker makes the inner
            # `_run` wrapper step aside when LangChain's default `_arun` (or a
            # StructuredTool without a coroutine) routes back through it.
            t._arun = _wrap_async_run(base_arun, name, governor, mode, category, tool_id)  # noqa: SLF001
        try:
            t._matimo_governed = True  # noqa: SLF001
        except Exception:  # noqa: BLE001 -- a stricter model config may reject this; harmless
            pass
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

    `provider="openai"` -> full support for synchronous calls: the session
    token *and* a fresh `Matimo-Agent-Signature` are attached per request, via
    `governor.httpx_client()`'s request event hook (`ChatOpenAI.http_client`
    is a real, documented constructor kwarg in langchain-openai>=1.6, verified
    directly against the installed package's pydantic fields). Only the sync
    `http_client` is wired: `ChatOpenAI`'s async methods (`ainvoke`, `astream`)
    use its own async client with a session header fixed at construction, and
    no signature. Needs a sync `Governor`.

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

        default_headers = default_llm_headers(governor, "gateway_chat_model")
        return ChatOpenAI(
            model=model or "matimo/auto",
            base_url=governor.config.base_url,
            api_key=governor.config.api_key,
            default_headers=default_headers,
            http_client=governor.httpx_client(),
            **kwargs,
        )
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        # ChatAnthropic exposes api_key (sent as x-api-key, which Gateway
        # ignores) but not auth_token, so the org key travels in an explicit
        # Authorization header instead; api_key is a placeholder that keeps
        # the SDK's own "credentials present" check happy.
        default_llm_headers(governor, "gateway_chat_model")  # rejects an AsyncGovernor clearly
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
