"""The universal fallback adapter: no framework import at all. If your
agent framework isn't LangChain, Google ADK, CrewAI, or AutoGen -- or it's
a bespoke in-house agent loop, a raw MCP tool dict, or anything else that
ultimately dispatches tool calls as plain Python callables -- this module
is how "any external framework" stays literally true.

One entry point:

    governed = govern(my_tools, governor)   # dict, list, or a single callable

`mode="govern"` (default) delegates straight to `governor.guard()` (or
`AsyncGovernor.guard()`, auto-selected by which `Governor` type you pass)
-- the same audited core enforcement path every other adapter in this
package is built on top of, not a reimplementation. `mode="observe"`
records a tool span around each call without ever calling `check_tool()`,
which core `guard()` does not support on its own (it always governs), so
this module adds a thin telemetry-only wrapper for that case.

## LLM spans: this module intentionally does not emit them

There is no framework here to hook -- `govern()` only ever sees tool
callables, never an LLM client or request/response pair, so unlike every
other adapter in this package there is no wrapper this module could add to
get LLM-span parity for free. If your agent loop makes its own LLM calls
(through `governor.openai_client_kwargs()`/`anthropic_client_kwargs()`/
`httpx_client()`, or any client you point at Gateway yourself), call
`governor.llm_span(model=..., provider=..., status=..., duration_ms=...)`
around that call site the same way the quickstart in
docs/USER-MANUAL.md section 5 does -- one span per call, using the same
`gen_ai.*` shape every framework adapter's spans use. Without that call,
tool calls governed through `govern()` still show up richly in the Gateway
Observability Hub; LLM calls simply won't have a span at all (not a
flattened or degraded one -- none).
"""

from __future__ import annotations

import time
from typing import Any

from ._shared import (
    Mode,
    async_check_and_wait,
    async_raise_if_suspended,
    check_mode,
    emit_tool_span,
    is_async_governor,
)


def _now() -> float:
    return time.monotonic()


def _observe_sync(fn: Any, name: str, governor: Any) -> Any:
    import functools

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        started = _now()
        status = "completed"
        try:
            return fn(*args, **kwargs)
        except Exception:
            status = "error"
            raise
        finally:
            emit_tool_span(
                governor,
                name,
                status=status,
                duration_ms=int((_now() - started) * 1000),
                arguments=kwargs or {f"arg{i}": v for i, v in enumerate(args)},
            )

    return wrapper


def _observe_async(fn: Any, name: str, governor: Any) -> Any:
    import functools

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        started = _now()
        status = "completed"
        try:
            return await fn(*args, **kwargs)
        except Exception:
            status = "error"
            raise
        finally:
            emit_tool_span(
                governor,
                name,
                status=status,
                duration_ms=int((_now() - started) * 1000),
                arguments=kwargs or {f"arg{i}": v for i, v in enumerate(args)},
            )

    return wrapper


def _govern_one(
    fn: Any, governor: Any, *, mode: Mode, category: str | None, name: str | None
) -> Any:
    import inspect

    tool_name = name or getattr(fn, "__name__", None) or "tool"

    if mode == "observe":
        if inspect.iscoroutinefunction(fn):
            return _observe_async(fn, tool_name, governor)
        return _observe_sync(fn, tool_name, governor)

    # mode == "govern": reuse the audited core Governor.guard()/
    # AsyncGovernor.guard() -- do not reimplement check/wait/span/report.
    if inspect.iscoroutinefunction(fn):
        if not is_async_governor(governor):
            # Bridge: run the sync Governor's blocking guard() logic (via
            # its own check_tool/await_decision) around the async callable
            # ourselves, since Governor.guard() only wraps sync callables.
            import functools

            @functools.wraps(fn)
            async def bridged(*args: Any, **kwargs: Any) -> Any:
                call_args = kwargs or {f"arg{i}": v for i, v in enumerate(args)}
                await async_raise_if_suspended(governor)
                decision = await async_check_and_wait(
                    governor, tool_name, call_args, category=category
                )
                from ..exceptions import ToolDenied

                if decision.denied:
                    raise ToolDenied(decision.reason)
                started = _now()
                status = "completed"
                try:
                    return await fn(*args, **kwargs)
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

            return bridged
        return governor.guard(fn, name=tool_name, category=category)

    # fn is sync.
    if is_async_governor(governor):
        raise TypeError(
            "an AsyncGovernor was passed to govern() for a synchronous callable -- "
            "use a sync Governor for sync callables, or make the callable async"
        )
    return governor.guard(fn, name=tool_name, category=category)


def govern(
    callable_or_tools: Any,
    governor: Any,
    *,
    mode: Mode = "govern",
    category: str | None = None,
) -> Any:
    """Governs a single callable, or every callable in a `dict`/`list`/
    `tuple` of them (dict keys and function `__name__`s are used as the
    tool name for policy checks and telemetry). Returns the same shape
    back -- a dict in, a dict out; a list in, a list out; a bare callable
    in, a bare callable out -- so this is a drop-in replacement wherever
    your framework or in-house loop expects a plain callable.
    """
    check_mode(mode)
    if isinstance(callable_or_tools, dict):
        return {
            key: _govern_one(fn, governor, mode=mode, category=category, name=str(key))
            for key, fn in callable_or_tools.items()
        }
    if isinstance(callable_or_tools, (list, tuple)):
        governed = [
            _govern_one(fn, governor, mode=mode, category=category, name=None)
            for fn in callable_or_tools
        ]
        return type(callable_or_tools)(governed)
    if callable(callable_or_tools):
        return _govern_one(callable_or_tools, governor, mode=mode, category=category, name=None)
    raise TypeError(
        "govern() expects a callable, or a dict/list/tuple of callables, "
        f"got {type(callable_or_tools).__name__}"
    )
