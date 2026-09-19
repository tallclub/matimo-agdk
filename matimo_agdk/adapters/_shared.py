"""Internal helpers shared by every framework adapter.

Not a public module -- nothing here is exported from `matimo_agdk` or
`matimo_agdk.adapters`. Import a specific adapter module instead
(`matimo_agdk.adapters.langchain`, `.google_adk`, `.crewai`, `.autogen`,
`.generic`).

Every adapter in this package supports two modes, both driven by the enum
below:

- `"observe"`: telemetry only. Run/LLM/tool spans are recorded, nothing is
  ever blocked, `governor.check_tool()` is never called.
- `"govern"` (default): additionally calls `governor.check_tool()` (and
  `await_decision()` on a `PENDING` outcome) before every tool call the
  framework dispatches, and calls `governor.raise_if_suspended()` at each
  LLM/tool boundary for rapid suspend.

Two supporting design decisions apply across every adapter:

1. **A `Governor` (sync) works from both sync and async framework code.**
   Every adapter accepts either a sync `Governor` or an `AsyncGovernor`.
   When a sync `Governor` is used from an async call site (Google ADK is
   async-only; LangChain's `_arun`/async callbacks; AutoGen's async tool
   dispatch), the blocking calls are bridged via `asyncio.to_thread` so the
   event loop is never blocked. When an `AsyncGovernor` is used from a sync
   call site, that is a programming error -- there is no way to await from
   inside a synchronous function -- and a clear `TypeError` is raised
   rather than deadlocking or silently doing nothing.
2. **Span correlation degrades gracefully.** `Governor.llm_span()`/
   `tool_span()` need either an explicit `run_id` or an ambient
   `governor.run()` block. Adapters always pass an explicit `run_id`
   derived from the framework's own run/invocation identifier when one is
   available (so LLM and tool spans across one agent turn correlate
   automatically, matching the server's `runId`/`spanId`/`parentSpanId`
   shape), and fall back to a freshly generated id per call otherwise --
   telemetry is emitted either way, just without cross-call correlation in
   that fallback case.
"""

from __future__ import annotations

import asyncio
import inspect
import uuid
from typing import Any, Literal

from ..tools import ToolDecision

Mode = Literal["observe", "govern"]

VALID_MODES = ("observe", "govern")


def check_mode(mode: str) -> Mode:
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
    return mode  # type: ignore[return-value]


def is_async_governor(governor: Any) -> bool:
    """True if `governor` is an AsyncGovernor (or anything else whose
    check_tool() is a coroutine function) -- duck-typed rather than an
    isinstance check so a test double works without importing the real
    class."""
    return inspect.iscoroutinefunction(getattr(governor, "check_tool", None))


def new_run_id() -> str:
    return uuid.uuid4().hex


def call_args_from(
    args: tuple[Any, ...], kwargs: dict[str, Any], *, exclude: tuple[str, ...] = ()
) -> dict[str, Any]:
    """The check-body/telemetry view of a tool call. Positional inputs get
    synthetic names (arg0, arg1, ...) so a single-input tool called
    positionally no longer hashes to an empty dict, which collapsed
    distinct calls onto one server-side dedup key."""
    merged: dict[str, Any] = {f"arg{i}": v for i, v in enumerate(args)}
    merged.update({k: v for k, v in kwargs.items() if k not in exclude})
    return merged


def sync_check_and_wait(
    governor: Any, tool_name: str, args: dict[str, Any], *, category: str | None = None
) -> ToolDecision:
    """Blocking check-then-await-PENDING, for a sync Governor called from
    sync framework code. Raises TypeError if handed an AsyncGovernor."""
    if is_async_governor(governor):
        raise TypeError(
            "an AsyncGovernor was passed to a synchronous governed call site -- "
            "use a sync Governor here, or use this adapter's async entry point"
        )
    decision = governor.check_and_wait(tool_name, args, category_hint=category)
    if decision.denied:
        emit_tool_span(governor, tool_name, status="denied", duration_ms=0, arguments=args)
    return decision


async def async_check_and_wait(
    governor: Any, tool_name: str, args: dict[str, Any], *, category: str | None = None
) -> ToolDecision:
    """Non-blocking check-then-await-PENDING for async framework code.
    Awaits directly if `governor` is an AsyncGovernor; otherwise bridges
    the sync Governor's blocking calls via asyncio.to_thread so the event
    loop is never blocked."""
    if is_async_governor(governor):
        decision = await governor.check_and_wait(tool_name, args, category_hint=category)
    else:
        decision = await asyncio.to_thread(
            governor.check_and_wait, tool_name, args, category_hint=category
        )
    if decision.denied:
        emit_tool_span(governor, tool_name, status="denied", duration_ms=0, arguments=args)
    return decision


def sync_raise_if_suspended(governor: Any) -> None:
    if is_async_governor(governor):
        raise TypeError(
            "an AsyncGovernor was passed to a synchronous governed call site -- "
            "use a sync Governor here, or use this adapter's async entry point"
        )
    governor.raise_if_suspended()


async def async_raise_if_suspended(governor: Any) -> None:
    if is_async_governor(governor):
        governor.raise_if_suspended()
    else:
        await asyncio.to_thread(governor.raise_if_suspended)


def emit_llm_span(governor: Any, *, run_id: str | None = None, **kwargs: Any) -> None:
    """governor.llm_span() with a graceful fallback to a fresh run_id when
    no explicit one is given and no `governor.run()` block is active."""
    try:
        governor.llm_span(run_id=run_id, **kwargs)
    except Exception:  # noqa: BLE001 -- telemetry must never break the caller's agent
        if run_id is None:
            try:
                governor.llm_span(run_id=new_run_id(), **kwargs)
            except Exception:  # noqa: BLE001
                pass


def emit_tool_span(
    governor: Any, tool_name: str, *, run_id: str | None = None, **kwargs: Any
) -> None:
    try:
        governor.tool_span(tool_name, run_id=run_id, **kwargs)
    except Exception:  # noqa: BLE001
        if run_id is None:
            try:
                governor.tool_span(tool_name, run_id=new_run_id(), **kwargs)
            except Exception:  # noqa: BLE001
                pass


def truncate(value: Any, limit: int = 500) -> str:
    text = value if isinstance(value, str) else str(value)
    return text if len(text) <= limit else text[:limit] + "...[TRUNCATED]"


class MissingExtra(ImportError):
    """Raised when a framework adapter is imported without its optional
    extra installed. A plain, greppable message: `pip install
    matimo-agdk[<extra>]`."""

    def __init__(self, extra: str, package_hint: str) -> None:
        super().__init__(
            f"matimo_agdk.adapters.{extra.replace('-', '_')} requires the '{extra}' extra "
            f"({package_hint} not importable): pip install matimo-agdk[{extra}]"
        )
