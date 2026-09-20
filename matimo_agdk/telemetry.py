"""Async-safe background telemetry exporter, heartbeat, and rapid suspend.

Wire shape: docs/SERVER-CONTRACT.md section 7.1 (Shape A, the bespoke
envelope) and AGDK-SERVER-HEARTBEAT-REPORT.md (the heartbeat now riding
every POST /v1/telemetry/batch response, including a pure {"events": []}
poll, which never touches last_telemetry_at server-side).

Rapid suspend is POLLED, not pushed (docs/SERVER-CONTRACT.md section 10:
"No push-based kill switch"). Name and document this honestly: a suspend
or emergency stop takes up to one heartbeat interval to be observed
locally, even though the *next Gateway call* is denied immediately
server-side regardless of this local state.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from ._redact import redact, scrub_string
from .exceptions import AgentSuspendedLocally, GatewayError
from .transport import AsyncGatewayHTTP, GatewayHTTP

TELEMETRY_SESSION_HEADER = "X-Matimo-Session-Token"
_log = logging.getLogger("matimo_agdk.telemetry")

Kind = Literal["run", "llm", "tool", "log", "error"]

_MAX_ATTRIBUTE_VALUE_LEN = 2000

# 4xx statuses that mean "this request as sent will never be accepted".
# 401/403 (session/auth/policy), 408 and 429 are excluded: those are about the
# sender's state or a passing condition, so the same batch may succeed later.
_TRANSIENT_4XX = frozenset({401, 403, 408, 425, 429})


def _is_permanent_rejection(exc: GatewayError) -> bool:
    """True when the server rejected the batch itself (400 validation error,
    413 too large, 422, ...). Retrying that same batch forever would wedge the
    exporter behind one poison event, so it is dropped instead."""
    status = exc.status_code
    return status is not None and 400 <= status < 500 and status not in _TRANSIENT_4XX


# ---------------------------------------------------------------------------
# GovernanceState
# ---------------------------------------------------------------------------


@dataclass
class GovernanceState:
    """The SDK's local, polled view of this identity's governance status,
    refreshed from every telemetry batch response's `heartbeat` field.

    This is inherently as-of-last-poll, not real-time -- see this module's
    docstring and README.md's "what rapid suspend really means" section.
    """

    lifecycle_status: str = "unknown"
    emergency_stop: bool = False
    telemetry_mode: str = "advisory"
    telemetry_staleness_minutes: float = 30.0
    server_time: str | None = None
    last_polled_monotonic: float = field(default_factory=time.monotonic)
    # Unlike last_polled_monotonic (which starts at "now"), this is None until a
    # heartbeat has really arrived, so "never heard from Gateway" is distinguishable
    # from "heard just now". The tool-check fail-open rule reads it.
    last_heartbeat_monotonic: float | None = None

    def update_from_heartbeat(self, heartbeat: dict[str, Any]) -> None:
        self.lifecycle_status = heartbeat.get("lifecycleStatus", self.lifecycle_status)
        self.emergency_stop = bool(heartbeat.get("emergencyStop", self.emergency_stop))
        self.telemetry_mode = heartbeat.get("telemetryMode", self.telemetry_mode)
        self.telemetry_staleness_minutes = heartbeat.get(
            "telemetryStalenessMinutes", self.telemetry_staleness_minutes
        )
        self.server_time = heartbeat.get("serverTime", self.server_time)
        self.last_polled_monotonic = time.monotonic()
        self.last_heartbeat_monotonic = self.last_polled_monotonic

    @property
    def is_suspended(self) -> bool:
        return self.emergency_stop or self.lifecycle_status in ("suspended", "revoked")


# ---------------------------------------------------------------------------
# Event construction + gen_ai.* helpers
# ---------------------------------------------------------------------------


def redact_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    """Local, recursive, key-based redaction before send, on top of (never
    instead of) Gateway's own server-side masking (docs/SERVER-CONTRACT.md
    section 7.1). Nested dicts such as a tool span's
    `gen_ai.tool.call.arguments` are walked too, so `{"password": ...}`
    inside an arguments dict is masked before it leaves the process."""
    return redact(attributes, max_string=_MAX_ATTRIBUTE_VALUE_LEN)


def build_event(
    *,
    run_id: str,
    kind: Kind,
    session_id: str | None = None,
    span_id: str | None = None,
    parent_span_id: str | None = None,
    name: str | None = None,
    status: str | None = None,
    started_at: str | None = None,
    duration_ms: int | None = None,
    attributes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {"runId": run_id, "kind": kind}
    if session_id is not None:
        event["sessionId"] = session_id
    if span_id is not None:
        event["spanId"] = span_id
    if parent_span_id is not None:
        event["parentSpanId"] = parent_span_id
    if name is not None:
        event["name"] = name
    if status is not None:
        event["status"] = status
    if started_at is not None:
        event["startedAt"] = started_at
    if duration_ms is not None:
        event["durationMs"] = duration_ms
    if attributes:
        event["attributes"] = redact_attributes(attributes)
    return event


def run_span(
    run_id: str,
    *,
    name: str | None = None,
    status: str | None = None,
    started_at: str | None = None,
    duration_ms: int | None = None,
    session_id: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    attrs = dict(attributes or {})
    attrs.setdefault("gen_ai.operation.name", "invoke_agent")
    return build_event(
        run_id=run_id,
        kind="run",
        name=name,
        status=status,
        started_at=started_at,
        duration_ms=duration_ms,
        session_id=session_id,
        attributes=attrs,
    )


def llm_span(
    run_id: str,
    *,
    name: str = "chat_completion",
    status: str | None = None,
    started_at: str | None = None,
    duration_ms: int | None = None,
    session_id: str | None = None,
    span_id: str | None = None,
    parent_span_id: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    finish_reasons: list[str] | None = None,
    operation_name: str | None = None,
    guardrail_blocked: bool | None = None,
    attributes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """`operation_name` lets a caller send its own, more specific
    gen_ai.operation.name (e.g. "generate_content" for Gemini) --
    docs/SERVER-CONTRACT.md section 7.1 says the server only fills a
    coarse per-kind default when the caller didn't already set one."""
    attrs = dict(attributes or {})
    attrs.setdefault("gen_ai.operation.name", operation_name or "chat")
    if model:
        attrs.setdefault("gen_ai.request.model", model)
    if provider:
        attrs.setdefault("gen_ai.provider.name", provider)
    if finish_reasons:
        attrs.setdefault("gen_ai.response.finish_reasons", finish_reasons)
    if guardrail_blocked is not None:
        attrs.setdefault("matimo.guardrail_blocked", guardrail_blocked)
    return build_event(
        run_id=run_id,
        kind="llm",
        name=name,
        status=status,
        started_at=started_at,
        duration_ms=duration_ms,
        session_id=session_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        attributes=attrs,
    )


MAX_TOOL_RESULT_LEN = 500


def result_text(result: Any) -> str:
    """A tool result as span text: key-redacted if it is a dict or list, scrubbed
    for secret-shaped strings, then cut to MAX_TOOL_RESULT_LEN characters.
    Redaction runs on the whole value before the cut, so a secret straddling the
    limit is masked rather than half-sent."""
    text = str(redact(result))
    return scrub_string(text)[:MAX_TOOL_RESULT_LEN]


def tool_span(
    run_id: str,
    tool_name: str,
    *,
    status: str | None = None,
    started_at: str | None = None,
    duration_ms: int | None = None,
    session_id: str | None = None,
    span_id: str | None = None,
    parent_span_id: str | None = None,
    call_id: str | None = None,
    arguments: Any | None = None,
    result: Any | None = None,
    attributes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    attrs = dict(attributes or {})
    attrs.setdefault("gen_ai.operation.name", "execute_tool")
    attrs.setdefault("gen_ai.tool.name", tool_name)
    if call_id:
        attrs.setdefault("gen_ai.tool.call.id", call_id)
    if arguments is not None:
        attrs.setdefault("gen_ai.tool.call.arguments", arguments)
    if result is not None:
        attrs.setdefault("gen_ai.tool.call.result", result_text(result))
    return build_event(
        run_id=run_id,
        kind="tool",
        name=tool_name,
        status=status,
        started_at=started_at,
        duration_ms=duration_ms,
        session_id=session_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        attributes=attrs,
    )


# ---------------------------------------------------------------------------
# Sync (thread-based) exporter
# ---------------------------------------------------------------------------


class TelemetryExporter:
    """Background thread that batches telemetry events and flushes them to
    POST /v1/telemetry/batch, sending a pure heartbeat poll ({"events": []})
    whenever nothing else is pending and the heartbeat interval has
    elapsed. Bounded queue with drop-oldest overflow behavior.
    """

    def __init__(
        self,
        http: GatewayHTTP,
        session_manager: Any,
        *,
        flush_interval: float = 5.0,
        batch_size: int = 50,
        queue_max: int = 2000,
        heartbeat_interval: float = 60.0,
        fail_open: bool = True,
        on_suspend: Callable[[GovernanceState], None] | None = None,
        heartbeat_resolver: Callable[[float], float] | None = None,
    ) -> None:
        self._http = http
        self._session_manager = session_manager
        self._flush_interval = flush_interval
        self._batch_size = batch_size
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=queue_max)
        self._heartbeat_interval = heartbeat_interval
        # Re-sizes the heartbeat interval from the staleness window the
        # server reports on every heartbeat (a function of minutes ->
        # seconds); None keeps the constructor value forever.
        self._heartbeat_resolver = heartbeat_resolver
        self._fail_open = fail_open
        self._on_suspend = on_suspend
        self.state = GovernanceState()
        self.dropped_count = 0
        # With fail_open=False the background thread never dies on a
        # GatewayError: it stores the error here and the caller's next
        # submit()/flush_now()/stop() raises it (once).
        self.last_error: GatewayError | None = None
        self._first_tick = True
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_flush_monotonic = time.monotonic()
        self._suspend_notified = False
        self._state_lock = threading.Lock()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="matimo-agdk-telemetry", daemon=True)
        self._thread.start()
        atexit.register(self.stop)

    def stop(self, *, timeout: float = 5.0) -> None:
        if self._thread is None:
            return
        # Drop the atexit reference: it would otherwise pin this exporter (and
        # its HTTP client) for the life of the process and fire again at exit.
        atexit.unregister(self.stop)
        self._stop_event.set()
        self._thread.join(timeout=timeout)
        self._thread = None
        self._flush_all()
        self._raise_pending()

    def set_on_suspend(self, callback: Callable[[GovernanceState], None] | None) -> None:
        self._on_suspend = callback

    def _raise_pending(self) -> None:
        if self._fail_open or self.last_error is None:
            return
        exc, self.last_error = self.last_error, None
        raise exc

    def submit(self, event: dict[str, Any]) -> None:
        self._raise_pending()
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self.dropped_count += 1
            try:
                self._queue.put_nowait(event)
            except queue.Full:
                self.dropped_count += 1

    def flush_now(self) -> None:
        """Synchronous, on-demand flush of everything queued (always at least
        one request, so it doubles as a heartbeat) -- for tests, the CLI and
        `governor.stop()`'s final drain."""
        self._flush_all()
        self._raise_pending()

    def _flush_all(self) -> None:
        """Sends batches until the queue is empty or a send fails. A single
        `_flush` sends at most `batch_size` events, so a one-shot call would
        silently strand the rest of a backlog at shutdown."""
        ok = self._flush(force=True)
        while ok and not self._queue.empty():
            ok = self._flush(force=True)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            backlog = False
            try:
                # The first tick is a forced heartbeat so the server-reported
                # staleness window sizes the interval before the first idle wait.
                ok = self._flush(force=self._first_tick)
                backlog = ok and not self._queue.empty()
            except Exception:  # noqa: BLE001 -- the exporter thread must never die
                _log.exception("telemetry exporter iteration failed; will retry")
            self._first_tick = False
            # A backlog is drained back to back; only an idle queue waits.
            if not backlog:
                self._stop_event.wait(min(self._flush_interval, self._heartbeat_interval))

    def _drain(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        while len(events) < self._batch_size:
            try:
                events.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return events

    def _flush(self, *, force: bool = False) -> bool:
        """Sends one batch. Returns False if the send failed (the batch is
        re-queued, or dropped when the server rejected it outright)."""
        now = time.monotonic()
        events = self._drain()
        due_for_heartbeat = (now - self._last_flush_monotonic) >= self._heartbeat_interval
        if not events and not due_for_heartbeat and not force:
            return True
        self._last_flush_monotonic = now
        try:
            # call_with_retry() re-handshakes exactly once if the session
            # was invalidated behind this process's back (e.g. an admin/
            # another process called DELETE /v1/sessions, or the cached
            # token's server-side TTL was cut short) -- found live-testing
            # that nothing previously called this method at all, so a
            # session invalidated out-of-band was silently unrecoverable
            # until the next 80%-of-TTL proactive renewal. See
            # the 2026-09-18 live verification (CHANGELOG.md).
            resp = self._session_manager.call_with_retry(
                lambda token: self._http.request(
                    "POST",
                    "/telemetry/batch",
                    json_body={"events": events},
                    headers={TELEMETRY_SESSION_HEADER: token},
                    sign=False,  # never checked on this route -- see contract Drift #2
                    idempotent=True,
                )
            )
        except GatewayError as exc:
            # Re-queue what we drained so it is not silently lost on a
            # transient failure. With fail_open the agent's own work is
            # never blocked; without it the error is stored and raised on
            # the caller's next submit()/flush_now()/stop() instead of
            # killing this thread (which silently stopped heartbeats).
            permanent = _is_permanent_rejection(exc)
            if permanent:
                # Re-sending a batch the server refuses on its merits would wedge
                # the exporter behind it forever; drop it and keep going.
                self.dropped_count += len(events)
                _log.warning("telemetry batch of %d rejected and dropped: %s", len(events), exc)
            else:
                for event in events:
                    self._requeue(event)
            if not self._fail_open:
                _log.warning("telemetry flush failed (fail_open=False): %s", exc)
                self.last_error = exc
            return permanent
        except Exception:  # noqa: BLE001 -- e.g. a malformed response; never lose the batch
            _log.exception("telemetry flush failed unexpectedly; batch re-queued")
            for event in events:
                self._requeue(event)
            return False
        heartbeat = (resp.data or {}).get("heartbeat") if isinstance(resp.data, dict) else None
        if heartbeat:
            self.state.update_from_heartbeat(heartbeat)
            if self._heartbeat_resolver is not None:
                self._heartbeat_interval = self._heartbeat_resolver(
                    float(self.state.telemetry_staleness_minutes)
                )
            self._maybe_notify_suspend()
        return True

    def _requeue(self, event: dict[str, Any]) -> None:
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            self.dropped_count += 1

    def _maybe_notify_suspend(self) -> None:
        with self._state_lock:
            if not self.state.is_suspended:
                self._suspend_notified = False
                return
            if self._suspend_notified:
                return
            self._suspend_notified = True
            callback = self._on_suspend
        # Outside the lock, and isolated: a user callback must not be able to
        # deadlock or kill the exporter thread.
        if callback:
            try:
                callback(self.state)
            except Exception:  # noqa: BLE001
                _log.exception("on_suspend callback raised")

    def is_suspended(self) -> bool:
        return self.state.is_suspended

    def raise_if_suspended(self) -> None:
        if self.state.is_suspended:
            raise AgentSuspendedLocally(self.state.lifecycle_status, self.state.emergency_stop)


# ---------------------------------------------------------------------------
# Async (asyncio-task-based) exporter
# ---------------------------------------------------------------------------


class AsyncTelemetryExporter:
    """Async twin of TelemetryExporter, using an asyncio.Queue and an
    asyncio.Task instead of a thread. The queue/stop-event are bound lazily
    to whichever event loop calls start()/submit() first."""

    def __init__(
        self,
        http: AsyncGatewayHTTP,
        session_manager: Any,
        *,
        flush_interval: float = 5.0,
        batch_size: int = 50,
        queue_max: int = 2000,
        heartbeat_interval: float = 60.0,
        fail_open: bool = True,
        on_suspend: Callable[[GovernanceState], None] | None = None,
        heartbeat_resolver: Callable[[float], float] | None = None,
    ) -> None:
        self._http = http
        self._session_manager = session_manager
        self._flush_interval = flush_interval
        self._batch_size = batch_size
        self._queue_max = queue_max
        self._queue: asyncio.Queue[dict[str, Any]] | None = None
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat_resolver = heartbeat_resolver
        self._fail_open = fail_open
        self._on_suspend = on_suspend
        self.state = GovernanceState()
        self.dropped_count = 0
        self.last_error: GatewayError | None = None
        self._first_tick = True
        self._task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None
        self._last_flush_monotonic = time.monotonic()
        self._suspend_notified = False

    def _ensure_bound(self) -> None:
        if self._queue is None:
            self._queue = asyncio.Queue(maxsize=self._queue_max)
        if self._stop_event is None:
            self._stop_event = asyncio.Event()

    async def start(self) -> None:
        self._ensure_bound()
        if self._task is not None:
            return
        assert self._stop_event is not None
        self._stop_event.clear()
        self._task = asyncio.ensure_future(self._run())

    async def stop(self, *, timeout: float = 5.0) -> None:
        if self._task is None:
            return
        assert self._stop_event is not None
        self._stop_event.set()
        try:
            await asyncio.wait_for(self._task, timeout=timeout)
        except TimeoutError:
            self._task.cancel()
        self._task = None
        await self._flush_all()
        self._raise_pending()

    def set_on_suspend(self, callback: Callable[[GovernanceState], None] | None) -> None:
        self._on_suspend = callback

    def _raise_pending(self) -> None:
        if self._fail_open or self.last_error is None:
            return
        exc, self.last_error = self.last_error, None
        raise exc

    def submit(self, event: dict[str, Any]) -> None:
        self._raise_pending()
        self._ensure_bound()
        assert self._queue is not None
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self.dropped_count += 1
            try:
                self._queue.put_nowait(event)
            except asyncio.QueueFull:
                self.dropped_count += 1

    async def flush_now(self) -> None:
        await self._flush_all()
        self._raise_pending()

    async def _flush_all(self) -> None:
        """See TelemetryExporter._flush_all()."""
        self._ensure_bound()
        assert self._queue is not None
        ok = await self._flush(force=True)
        while ok and not self._queue.empty():
            ok = await self._flush(force=True)

    async def _run(self) -> None:
        assert self._stop_event is not None
        assert self._queue is not None
        while not self._stop_event.is_set():
            backlog = False
            try:
                ok = await self._flush(force=self._first_tick)
                backlog = ok and not self._queue.empty()
            except Exception:  # noqa: BLE001 -- the exporter task must never die
                _log.exception("telemetry exporter iteration failed; will retry")
            self._first_tick = False
            if backlog:
                continue
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=min(self._flush_interval, self._heartbeat_interval),
                )
            except TimeoutError:
                pass

    def _drain(self) -> list[dict[str, Any]]:
        assert self._queue is not None
        events: list[dict[str, Any]] = []
        while len(events) < self._batch_size:
            try:
                events.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return events

    async def _flush(self, *, force: bool = False) -> bool:
        """See TelemetryExporter._flush()."""
        self._ensure_bound()
        now = time.monotonic()
        events = self._drain()
        due_for_heartbeat = (now - self._last_flush_monotonic) >= self._heartbeat_interval
        if not events and not due_for_heartbeat and not force:
            return True
        self._last_flush_monotonic = now
        try:
            # See the sync exporter's _flush() for why this goes through
            # call_with_retry() rather than a bare get_token().
            resp = await self._session_manager.call_with_retry(
                lambda token: self._http.request(
                    "POST",
                    "/telemetry/batch",
                    json_body={"events": events},
                    headers={TELEMETRY_SESSION_HEADER: token},
                    sign=False,
                    idempotent=True,
                )
            )
        except GatewayError as exc:
            permanent = _is_permanent_rejection(exc)
            if permanent:
                self.dropped_count += len(events)
                _log.warning("telemetry batch of %d rejected and dropped: %s", len(events), exc)
            else:
                self._requeue(events)
            if not self._fail_open:
                _log.warning("telemetry flush failed (fail_open=False): %s", exc)
                self.last_error = exc
            return permanent
        except Exception:  # noqa: BLE001 -- e.g. a malformed response; never lose the batch
            _log.exception("telemetry flush failed unexpectedly; batch re-queued")
            self._requeue(events)
            return False
        heartbeat = (resp.data or {}).get("heartbeat") if isinstance(resp.data, dict) else None
        if heartbeat:
            self.state.update_from_heartbeat(heartbeat)
            if self._heartbeat_resolver is not None:
                self._heartbeat_interval = self._heartbeat_resolver(
                    float(self.state.telemetry_staleness_minutes)
                )
            self._maybe_notify_suspend()
        return True

    def _requeue(self, events: list[dict[str, Any]]) -> None:
        assert self._queue is not None
        for event in events:
            try:
                self._queue.put_nowait(event)
            except asyncio.QueueFull:
                self.dropped_count += 1

    def _maybe_notify_suspend(self) -> None:
        if not self.state.is_suspended:
            self._suspend_notified = False
            return
        if self._suspend_notified:
            return
        self._suspend_notified = True
        if self._on_suspend:
            try:
                self._on_suspend(self.state)
            except Exception:  # noqa: BLE001 -- a user callback must not kill the exporter
                _log.exception("on_suspend callback raised")

    def is_suspended(self) -> bool:
        return self.state.is_suspended

    def raise_if_suspended(self) -> None:
        if self.state.is_suspended:
            raise AgentSuspendedLocally(self.state.lifecycle_status, self.state.emergency_stop)
