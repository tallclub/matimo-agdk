"""Push channel for rapid suspend and emergency stop.

`GET /v1/control/stream` (docs/SERVER-CONTRACT.md section 7.3) is a
Server-Sent Events stream on the agent's session. Gateway publishes suspend,
restore, revoke and emergency-stop changes on it within about a second, where
polling the heartbeat alone takes 15 s to 5 min. This module is the client half:
a background consumer that keeps that stream open and feeds it into the same
`GovernanceState` the heartbeat poll maintains, so `guard()`, tool checks and
`governor.is_suspended()` see a suspend right away.

Design rules, in order of importance:

1. **Polling stays the guarantee.** Nothing here replaces or weakens the
   heartbeat poll. A push is at-most-once and may be late or out of order, so it
   is a hint, and every push also triggers an immediate heartbeat poll that sets
   the authoritative state.
2. **A push can only tighten.** Suspended, revoked and emergency-stop-on apply at
   once (`GovernanceState.tighten_from_hint`); restore, emergency-stop-off and an
   "active" snapshot change nothing locally and only cause that heartbeat poll.
   So a forged, replayed or misordered event can at worst make the agent stop
   briefly until the next poll corrects it, never let a suspended agent run.
3. **It never blocks or breaks the agent.** It lives on its own daemon thread
   (`ControlStreamConsumer`) or task (`AsyncControlStreamConsumer`), swallows
   every error, and reconnects with capped exponential backoff and jitter. A
   server without the route (older Gateway: 404 or 405, or 501), a refused
   license or scope, a Redis outage on the server, a proxy that drops the
   stream: each just leaves the SDK on polling, with one quiet log line.
4. **Reconnect needs no replay.** Every connect begins with a `ready` frame
   carrying a current snapshot; if it differs from local state the consumer
   triggers a heartbeat poll. There is no Last-Event-ID because state, not an
   event log, is what a reconnecting client needs.

Disable with `GatewayConfig(control_stream_enabled=False)` or
`MATIMO_CONTROL_STREAM=0`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import socket
import threading
import time
from collections.abc import Callable
from typing import Any, Literal

import httpx

from .exceptions import GatewayError, RateLimited, SessionExpired
from .telemetry import TELEMETRY_SESSION_HEADER

_log = logging.getLogger("matimo_agdk.control_stream")

CONTROL_STREAM_PATH = "/control/stream"

# Reconnect policy. A connection that lasted at least _STABLE_AFTER_SECONDS resets
# the backoff, so a flapping server cannot pin the delay at its minimum forever.
_MIN_BACKOFF_SECONDS = 1.0
_MAX_BACKOFF_SECONDS = 60.0
_STABLE_AFTER_SECONDS = 10.0
# Older server (no route) or a refusal that will not fix itself soon: try again
# rarely. A server upgrade is picked up within this window without a restart.
_UNSUPPORTED_RETRY_SECONDS = 600.0
_REFUSED_RETRY_SECONDS = 300.0
# A revoked identity gets `ready` then `closing`; reconnecting fast would loop.
_REVOKED_RETRY_SECONDS = 300.0
_MAX_RETRY_AFTER_SECONDS = 300.0
# One SSE frame is a few hundred bytes. Anything past this is not ours.
_MAX_FRAME_BYTES = 64 * 1024

Status = Literal["stopped", "connecting", "connected", "backoff", "unsupported"]


# ---------------------------------------------------------------------------
# SSE parsing
# ---------------------------------------------------------------------------


class SSEParser:
    """Incremental parser for the subset of Server-Sent Events Gateway emits:
    `event:` and `data:` fields, blank-line dispatch, `:` comment lines (the
    keepalive). `id:` and `retry:` are ignored on purpose (see module doc, rule 4).

    Feed it one line at a time, without the trailing newline, as httpx's
    `iter_lines()` yields them. `feed()` returns `(event_name, data)` when a
    frame completes, else None. `data` is the parsed JSON object, or None if the
    frame carried no data or not valid JSON.
    """

    def __init__(self) -> None:
        self._event: str | None = None
        self._data: list[str] = []
        self._size = 0

    def feed(self, line: str) -> tuple[str, Any] | None:
        if line == "":
            return self._dispatch()
        if line.startswith(":"):
            return None
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            self._event = value
        elif field == "data":
            self._size += len(value)
            if self._size <= _MAX_FRAME_BYTES:
                self._data.append(value)
        return None

    def _dispatch(self) -> tuple[str, Any] | None:
        if self._event is None and not self._data:
            return None
        name = self._event or "message"
        raw = "\n".join(self._data)
        oversized = self._size > _MAX_FRAME_BYTES
        self._event = None
        self._data = []
        self._size = 0
        if oversized:
            return name, None
        data: Any = None
        if raw:
            try:
                data = json.loads(raw)
            except ValueError:
                data = None
        return name, data


# ---------------------------------------------------------------------------
# Frame handling (shared by the sync and async consumers)
# ---------------------------------------------------------------------------


class _FrameHandler:
    """Turns one parsed frame into an effect on the exporter. Pure policy, no I/O."""

    def __init__(self, exporter: Any, identity_id: Callable[[], str | None]) -> None:
        self._exporter = exporter
        self._identity_id = identity_id

    def handle(self, name: str, data: Any) -> str | None:
        """Returns the close reason when the frame is `closing`, else None.
        Unknown event names are ignored, so a newer server can add events."""
        if name == "closing":
            reason = data.get("reason") if isinstance(data, dict) else None
            return reason if isinstance(reason, str) else "unknown"
        if name not in ("ready", "lifecycle", "emergency_stop"):
            return None
        if not isinstance(data, dict):
            return None
        identity_id = self._identity_id()
        self._exporter.apply_control_hint(name, data, identity_id)
        if name == "ready":
            if self._snapshot_differs(data, identity_id):
                self._exporter.request_refresh()
            return None
        # A lifecycle or emergency-stop event always confirms against the
        # heartbeat, whatever its direction: that is what makes a relaxing event
        # (restore, stop off) safe, and it corrects a wrongly-applied tighten.
        if name == "lifecycle" and data.get("identityId") != identity_id:
            return None
        self._exporter.request_refresh(jitter=name == "emergency_stop")
        return None

    def _snapshot_differs(self, data: dict[str, Any], identity_id: str | None) -> bool:
        if identity_id is None or data.get("identityId") != identity_id:
            return False
        state = self._exporter.state
        if data.get("lifecycleStatus") != state.lifecycle_status:
            return True
        if bool(data.get("emergencyStop")) != state.emergency_stop:
            return True
        version = data.get("configVersion")
        return (
            isinstance(version, int)
            and not isinstance(version, bool)
            and state.config_version is not None
            and version != state.config_version
        )


def _backoff_delay(attempt: int) -> float:
    raw = min(_MAX_BACKOFF_SECONDS, _MIN_BACKOFF_SECONDS * (2**attempt))
    return random.uniform(raw / 2, raw)


class _Policy:
    """Maps how a connection ended to what to do next. Shared by both consumers."""

    def __init__(self) -> None:
        self.status: Status = "stopped"
        self._logged_unsupported = False

    def after_error(self, exc: BaseException, attempt: int) -> float:
        if isinstance(exc, RateLimited):
            retry_after = exc.retry_after if exc.retry_after is not None else 30.0
            return min(max(retry_after, _backoff_delay(attempt)), _MAX_RETRY_AFTER_SECONDS)
        status_code = exc.status_code if isinstance(exc, GatewayError) else None
        if status_code in (404, 405, 501):
            self.status = "unsupported"
            if not self._logged_unsupported:
                self._logged_unsupported = True
                _log.info(
                    "Gateway has no control stream (HTTP %s); staying on heartbeat polling",
                    status_code,
                )
            return _UNSUPPORTED_RETRY_SECONDS
        if status_code is not None and 400 <= status_code < 500 and status_code != 401:
            # Scope, license or similar: a refusal that will not clear in seconds.
            _log.debug("control stream refused (HTTP %s): %s", status_code, exc)
            self.status = "backoff"
            return _REFUSED_RETRY_SECONDS
        self.status = "backoff"
        return _backoff_delay(attempt)

    def after_close(self, reason: str | None, lasted: float | None, attempt: int) -> float:
        self.status = "backoff"
        if reason == "identity_revoked":
            return _REVOKED_RETRY_SECONDS
        if lasted is not None and lasted >= _STABLE_AFTER_SECONDS:
            return random.uniform(_MIN_BACKOFF_SECONDS / 2, _MIN_BACKOFF_SECONDS)
        return _backoff_delay(attempt)

    @staticmethod
    def next_attempt(attempt: int, lasted: float | None) -> int:
        if lasted is not None and lasted >= _STABLE_AFTER_SECONDS:
            return 0
        return attempt + 1


def _shutdown_response_socket(response: httpx.Response) -> None:
    """Best-effort `SHUT_RDWR` on a streaming response's underlying socket.

    httpx doesn't expose this, so it's reached through the `network_stream`
    extension httpcore attaches to the response; any failure (older httpx, a
    non-socket transport, a socket already gone) is swallowed -- the caller's
    `response.close()` is still the primary shutdown path everywhere shutdown()
    isn't needed or available.
    """
    try:
        network_stream = response.extensions.get("network_stream")
        if network_stream is None:
            return
        sock = network_stream.get_extra_info("socket")
        if isinstance(sock, socket.socket):
            sock.shutdown(socket.SHUT_RDWR)
    except Exception:  # noqa: BLE001 -- best-effort only
        pass


# ---------------------------------------------------------------------------
# Sync consumer (a daemon thread)
# ---------------------------------------------------------------------------


class ControlStreamConsumer:
    """Keeps `GET /v1/control/stream` open on a background daemon thread."""

    def __init__(
        self,
        http: Any,
        session_manager: Any,
        exporter: Any,
        identity_id: Callable[[], str | None],
        *,
        read_timeout: float = 45.0,
    ) -> None:
        self._http = http
        self._session = session_manager
        self._handler = _FrameHandler(exporter, identity_id)
        self._read_timeout = read_timeout
        self._policy = _Policy()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._response: httpx.Response | None = None

    @property
    def status(self) -> Status:
        return self._policy.status

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._policy.status = "connecting"
        self._thread = threading.Thread(
            target=self._run, name="matimo-agdk-control-stream", daemon=True
        )
        self._thread.start()

    def stop(self, *, timeout: float = 2.0) -> None:
        if self._thread is None:
            return
        self._stop.set()
        response = self._response
        if response is not None:
            # On an idle stream the reader thread is parked in a blocking socket
            # recv(); response.close() alone doesn't wake it on Linux (unlike
            # Windows/macOS), since POSIX close() from another thread doesn't
            # interrupt a concurrent blocking read on the same fd. shutdown()
            # does, so it runs first.
            _shutdown_response_socket(response)
            try:
                response.close()  # unblocks a reader parked in iter_lines()
            except Exception:  # noqa: BLE001 -- already closing; nothing to salvage
                pass
        self._thread.join(timeout=timeout)
        self._thread = None
        self._policy.status = "stopped"

    def _run(self) -> None:
        attempt = 0
        while not self._stop.is_set():
            connected_at: float | None = None
            reason: str | None = None
            try:
                token = self._session.get_token()
                with self._http.stream_get(
                    CONTROL_STREAM_PATH,
                    headers={TELEMETRY_SESSION_HEADER: token},
                    read_timeout=self._read_timeout,
                ) as response:
                    self._response = response
                    self._policy.status = "connecting"
                    parser = SSEParser()
                    for line in response.iter_lines():
                        if self._stop.is_set():
                            break
                        frame = parser.feed(line)
                        if frame is None:
                            continue
                        name, data = frame
                        if name == "ready" and connected_at is None:
                            connected_at = time.monotonic()
                            self._policy.status = "connected"
                        reason = self._handler.handle(name, data)
                        if reason is not None:
                            break
                lasted = None if connected_at is None else time.monotonic() - connected_at
                delay = self._policy.after_close(reason, lasted, attempt)
                attempt = self._policy.next_attempt(attempt, lasted)
            except SessionExpired:
                # Deleted or expired session: drop it so the next connect
                # re-handshakes (get_token() then mints a fresh one).
                self._invalidate_session()
                delay = _MIN_BACKOFF_SECONDS
            except Exception as exc:  # noqa: BLE001 -- the consumer must never die
                if self._stop.is_set():
                    break
                lasted = None if connected_at is None else time.monotonic() - connected_at
                delay = self._policy.after_error(exc, attempt)
                attempt = self._policy.next_attempt(attempt, lasted)
            finally:
                self._response = None
            if self._stop.wait(delay):
                break
        self._policy.status = "stopped"

    def _invalidate_session(self) -> None:
        try:
            self._session.invalidate()
        except Exception:  # noqa: BLE001
            _log.debug("session invalidate failed", exc_info=True)


# ---------------------------------------------------------------------------
# Async consumer (an asyncio task)
# ---------------------------------------------------------------------------


class AsyncControlStreamConsumer:
    """Async twin of ControlStreamConsumer, an asyncio task on the caller's loop."""

    def __init__(
        self,
        http: Any,
        session_manager: Any,
        exporter: Any,
        identity_id: Callable[[], str | None],
        *,
        read_timeout: float = 45.0,
    ) -> None:
        self._http = http
        self._session = session_manager
        self._handler = _FrameHandler(exporter, identity_id)
        self._read_timeout = read_timeout
        self._policy = _Policy()
        self._task: asyncio.Task[None] | None = None
        self._stop: asyncio.Event | None = None

    @property
    def status(self) -> Status:
        return self._policy.status

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stop = asyncio.Event()
        self._policy.status = "connecting"
        self._task = asyncio.ensure_future(self._run())

    async def stop(self, *, timeout: float = 2.0) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        if self._stop is not None:
            self._stop.set()
        task.cancel()
        try:
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=timeout)
        except TimeoutError:
            pass
        self._policy.status = "stopped"

    async def _run(self) -> None:
        assert self._stop is not None
        stop = self._stop
        attempt = 0
        while not stop.is_set():
            connected_at: float | None = None
            reason: str | None = None
            try:
                token = await self._session.get_token()
                async with self._http.stream_get(
                    CONTROL_STREAM_PATH,
                    headers={TELEMETRY_SESSION_HEADER: token},
                    read_timeout=self._read_timeout,
                ) as response:
                    self._policy.status = "connecting"
                    parser = SSEParser()
                    async for line in response.aiter_lines():
                        frame = parser.feed(line)
                        if frame is None:
                            continue
                        name, data = frame
                        if name == "ready" and connected_at is None:
                            connected_at = time.monotonic()
                            self._policy.status = "connected"
                        reason = self._handler.handle(name, data)
                        if reason is not None:
                            break
                lasted = None if connected_at is None else time.monotonic() - connected_at
                delay = self._policy.after_close(reason, lasted, attempt)
                attempt = self._policy.next_attempt(attempt, lasted)
            except SessionExpired:
                try:
                    await self._session.invalidate()
                except Exception:  # noqa: BLE001
                    _log.debug("session invalidate failed", exc_info=True)
                delay = _MIN_BACKOFF_SECONDS
            except Exception as exc:  # noqa: BLE001 -- the consumer must never die
                # CancelledError is a BaseException, so stop() still cancels us.
                _log.debug("control stream iteration failed", exc_info=True)
                lasted = None if connected_at is None else time.monotonic() - connected_at
                delay = self._policy.after_error(exc, attempt)
                attempt = self._policy.next_attempt(attempt, lasted)
            try:
                await asyncio.wait_for(stop.wait(), timeout=delay)
            except TimeoutError:
                continue
            break
        self._policy.status = "stopped"
