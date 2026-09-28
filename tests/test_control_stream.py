"""The control-stream push channel (matimo_agdk.control_stream).

The consumer tests run against a real local HTTP server that speaks
Server-Sent Events, so they cover the actual streaming path (httpx `stream()`,
line iteration, closing a blocked reader), not a mock of it.
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx
import pytest
import respx

from matimo_agdk import control_stream as cs
from matimo_agdk.config import GatewayConfig
from matimo_agdk.control_stream import (
    AsyncControlStreamConsumer,
    ControlStreamConsumer,
    SSEParser,
    _FrameHandler,
)
from matimo_agdk.exceptions import (
    GatewayError,
    GatewayUnavailable,
    SessionExpired,
)
from matimo_agdk.governor import AsyncGovernor, Governor
from matimo_agdk.telemetry import AsyncTelemetryExporter, GovernanceState, TelemetryExporter
from matimo_agdk.transport import AsyncGatewayHTTP, GatewayHTTP, GatewayResponse

from .conftest import BASE_URL, make_session_response, wait_until

ME = "11111111-1111-1111-1111-111111111111"
OTHER = "99999999-9999-9999-9999-999999999999"


def frame(name: str, data: dict[str, Any] | None = None) -> str:
    return f"event: {name}\ndata: {json.dumps(data or {})}\n\n"


def ready(status: str = "active", stop: bool = False, version: int | None = 5) -> str:
    return frame(
        "ready",
        {
            "type": "ready",
            "identityId": ME,
            "lifecycleStatus": status,
            "emergencyStop": stop,
            "configVersion": version,
            "serverTime": "t",
            "keepaliveSeconds": 15,
        },
    )


def lifecycle(status: str, identity: str = ME) -> str:
    return frame(
        "lifecycle",
        {
            "type": "lifecycle",
            "identityId": identity,
            "lifecycleStatus": status,
            "configVersion": 6,
            "serverTime": "t",
        },
    )


def emergency_stop(active: bool) -> str:
    return frame(
        "emergency_stop",
        {
            "type": "emergency_stop",
            "identityId": None,
            "emergencyStop": active,
            "configVersion": 6,
            "serverTime": "t",
        },
    )


# ---------------------------------------------------------------------------
# SSE parsing
# ---------------------------------------------------------------------------


def parse_all(text: str) -> list[tuple[str, Any]]:
    parser = SSEParser()
    out = []
    for line in text.split("\n"):
        result = parser.feed(line)
        if result is not None:
            out.append(result)
    return out


def test_parser_reads_frames_and_ignores_comments_and_unknown_fields() -> None:
    text = (
        ": keepalive\n\n"
        'retry: 5000\nid: 9\nevent: lifecycle\ndata: {"a": 1}\n\n'
        ": another comment\n"
        'event: closing\ndata: {"reason": "x"}\n\n'
    )
    assert parse_all(text) == [("lifecycle", {"a": 1}), ("closing", {"reason": "x"})]


def test_parser_joins_multiline_data_and_defaults_the_event_name() -> None:
    assert parse_all('data: {"a":\ndata: 1}\n\n') == [("message", {"a": 1})]


def test_parser_returns_none_data_for_bad_json_and_keeps_going() -> None:
    assert parse_all("event: x\ndata: not json\n\n" + frame("y", {"ok": True})) == [
        ("x", None),
        ("y", {"ok": True}),
    ]


def test_parser_drops_an_oversized_frame_without_losing_the_next_one() -> None:
    big = "event: big\ndata: " + "a" * (cs._MAX_FRAME_BYTES + 10) + "\n\n"
    assert parse_all(big + frame("small", {"n": 1})) == [("big", None), ("small", {"n": 1})]


def test_parser_ignores_a_blank_line_with_nothing_pending() -> None:
    assert parse_all("\n\n\n") == []


# ---------------------------------------------------------------------------
# GovernanceState.tighten_from_hint: an event can only tighten
# ---------------------------------------------------------------------------


def hint(name: str, **fields: Any) -> dict[str, Any]:
    return {"identityId": ME, **fields}


def test_a_suspend_or_revoke_tightens_at_once() -> None:
    s = GovernanceState(lifecycle_status="active")
    assert s.tighten_from_hint("lifecycle", hint("lifecycle", lifecycleStatus="suspended"), ME)
    assert s.is_suspended and s.lifecycle_status == "suspended"
    assert s.tighten_from_hint("lifecycle", hint("lifecycle", lifecycleStatus="revoked"), ME)
    assert s.lifecycle_status == "revoked"


def test_an_emergency_stop_on_tightens_at_once() -> None:
    s = GovernanceState(lifecycle_status="active")
    assert s.tighten_from_hint("emergency_stop", {"emergencyStop": True}, ME)
    assert s.emergency_stop and s.is_suspended


@pytest.mark.parametrize(
    ("start", "name", "data"),
    [
        ("suspended", "lifecycle", {"identityId": ME, "lifecycleStatus": "active"}),
        ("revoked", "lifecycle", {"identityId": ME, "lifecycleStatus": "suspended"}),
        ("revoked", "lifecycle", {"identityId": ME, "lifecycleStatus": "active"}),
        ("suspended", "ready", {"identityId": ME, "lifecycleStatus": "active"}),
    ],
)
def test_an_event_never_relaxes_lifecycle(start: str, name: str, data: dict[str, Any]) -> None:
    s = GovernanceState(lifecycle_status=start)
    assert s.tighten_from_hint(name, data, ME) is False
    assert s.lifecycle_status == start


def test_an_emergency_stop_off_or_a_clean_snapshot_never_relaxes_the_stop() -> None:
    s = GovernanceState(lifecycle_status="active", emergency_stop=True)
    assert s.tighten_from_hint("emergency_stop", {"emergencyStop": False}, ME) is False
    assert s.tighten_from_hint("ready", {"identityId": ME, "emergencyStop": False}, ME) is False
    assert s.emergency_stop is True


def test_events_for_another_identity_or_malformed_events_change_nothing() -> None:
    s = GovernanceState(lifecycle_status="active")
    assert not s.tighten_from_hint(
        "lifecycle", hint("lifecycle", lifecycleStatus="suspended") | {"identityId": OTHER}, ME
    )
    assert not s.tighten_from_hint("lifecycle", {"lifecycleStatus": "suspended"}, ME)
    assert not s.tighten_from_hint(
        "lifecycle", hint("lifecycle", lifecycleStatus="suspended"), None
    )
    assert not s.tighten_from_hint("lifecycle", hint("lifecycle", lifecycleStatus="bogus"), ME)
    assert not s.tighten_from_hint("lifecycle", "suspended", ME)
    assert not s.tighten_from_hint("something_new", {"identityId": ME, "emergencyStop": True}, ME)
    assert not s.tighten_from_hint("emergency_stop", {"emergencyStop": "yes"}, ME)
    assert s.lifecycle_status == "active" and not s.emergency_stop


def test_a_hint_never_touches_the_heartbeat_owned_fields() -> None:
    s = GovernanceState(lifecycle_status="active", config_version=3)
    s.tighten_from_hint(
        "lifecycle", hint("lifecycle", lifecycleStatus="suspended", configVersion=99), ME
    )
    assert s.config_version == 3
    assert s.last_heartbeat_monotonic is None


# ---------------------------------------------------------------------------
# _FrameHandler policy
# ---------------------------------------------------------------------------


class FakeExporter:
    def __init__(self, state: GovernanceState | None = None) -> None:
        self.state = state or GovernanceState(lifecycle_status="active", config_version=5)
        self.hints: list[tuple[str, Any]] = []
        self.refreshes: list[bool] = []

    def apply_control_hint(self, name: str, data: Any, identity_id: str | None) -> bool:
        self.hints.append((name, data))
        return self.state.tighten_from_hint(name, data, identity_id)

    def request_refresh(self, *, jitter: bool = False) -> None:
        self.refreshes.append(jitter)


def handler_for(state: GovernanceState | None = None) -> tuple[_FrameHandler, FakeExporter]:
    exporter = FakeExporter(state)
    return _FrameHandler(exporter, lambda: ME), exporter


def test_lifecycle_events_apply_and_always_request_a_refresh_whatever_their_direction() -> None:
    h, ex = handler_for()
    h.handle("lifecycle", json.loads(lifecycle("suspended").split("data: ")[1]))
    assert ex.state.is_suspended and ex.refreshes == [False]
    h.handle("lifecycle", json.loads(lifecycle("active").split("data: ")[1]))
    assert ex.state.is_suspended  # the relaxing event alone changed nothing
    assert ex.refreshes == [False, False]  # but it did ask for the authoritative answer


def test_a_tenant_wide_event_requests_a_jittered_refresh() -> None:
    h, ex = handler_for()
    h.handle("emergency_stop", {"emergencyStop": True, "identityId": None})
    assert ex.state.emergency_stop and ex.refreshes == [True]


def test_a_lifecycle_event_for_another_identity_does_nothing_at_all() -> None:
    h, ex = handler_for()
    h.handle("lifecycle", {"identityId": OTHER, "lifecycleStatus": "suspended"})
    assert not ex.state.is_suspended and ex.refreshes == []


def test_ready_only_refreshes_when_the_snapshot_disagrees_with_local_state() -> None:
    h, ex = handler_for(GovernanceState(lifecycle_status="active", config_version=5))
    h.handle(
        "ready",
        {"identityId": ME, "lifecycleStatus": "active", "emergencyStop": False, "configVersion": 5},
    )
    assert ex.refreshes == []
    h.handle(
        "ready",
        {"identityId": ME, "lifecycleStatus": "active", "emergencyStop": False, "configVersion": 8},
    )
    assert ex.refreshes == [False]  # config changed while we were disconnected
    h.handle(
        "ready",
        {
            "identityId": ME,
            "lifecycleStatus": "suspended",
            "emergencyStop": False,
            "configVersion": 8,
        },
    )
    assert ex.state.is_suspended and len(ex.refreshes) == 2
    ex.state.lifecycle_status = "suspended"
    ex.refreshes.clear()
    # Local says suspended, server now says active: never trusted, so refresh.
    h.handle(
        "ready",
        {"identityId": ME, "lifecycleStatus": "active", "emergencyStop": False, "configVersion": 8},
    )
    assert ex.state.is_suspended and ex.refreshes == [False]


def test_ready_ignores_a_null_config_version_and_a_snapshot_for_someone_else() -> None:
    h, ex = handler_for(GovernanceState(lifecycle_status="active", config_version=5))
    h.handle(
        "ready",
        {
            "identityId": ME,
            "lifecycleStatus": "active",
            "emergencyStop": False,
            "configVersion": None,
        },
    )
    h.handle("ready", {"identityId": OTHER, "lifecycleStatus": "suspended", "emergencyStop": True})
    assert ex.refreshes == [] and not ex.state.is_suspended


def test_closing_returns_its_reason_and_unknown_frames_are_ignored() -> None:
    h, ex = handler_for()
    assert h.handle("closing", {"reason": "identity_revoked"}) == "identity_revoked"
    assert h.handle("closing", None) == "unknown"
    assert h.handle("brand_new_event", {"x": 1}) is None
    assert h.handle("lifecycle", "not a dict") is None
    assert ex.refreshes == []


# ---------------------------------------------------------------------------
# A real local SSE server
# ---------------------------------------------------------------------------


class Connection:
    def __init__(self) -> None:
        self.frames: queue.Queue[str | None] = queue.Queue()

    def push(self, text: str) -> None:
        self.frames.put(text)

    def close(self) -> None:
        self.frames.put(None)


class ServerState:
    def __init__(self) -> None:
        self.script: list[Any] = []  # per connection: an int status, or a list of initial frames
        self.default: Any = [ready()]
        self.requests: list[dict[str, Any]] = []
        self.connections: list[Connection] = []
        self.lock = threading.Lock()

    def next_mode(self) -> Any:
        with self.lock:
            return self.script.pop(0) if self.script else self.default

    @property
    def current(self) -> Connection:
        assert wait_until(lambda: bool(self.connections), timeout=3), "no client connected"
        return self.connections[-1]


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:  # silence
        pass

    def do_GET(self) -> None:  # noqa: N802
        state: ServerState = self.server.state  # type: ignore[attr-defined]
        state.requests.append({"path": self.path, "headers": dict(self.headers)})
        mode = state.next_mode()
        if isinstance(mode, int):
            body = json.dumps(
                {"error": "session_expired"} if mode == 401 else {"error": "nope"}
            ).encode()
            self.send_response(mode)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        conn = Connection()
        state.connections.append(conn)
        try:
            for text in mode:
                self.wfile.write(text.encode())
            self.wfile.flush()
            while True:
                item = conn.frames.get()
                if item is None:
                    return
                self.wfile.write(item.encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            return


@pytest.fixture()
def sse_server() -> Iterator[tuple[ServerState, str]]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.daemon_threads = True
    state = ServerState()
    server.state = state  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state, f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        for conn in state.connections:
            conn.close()
        server.shutdown()
        server.server_close()


class FakeSession:
    def __init__(self) -> None:
        self.token = "tok-1"
        self.invalidations = 0

    def get_token(self) -> str:
        return self.token

    def invalidate(self) -> None:
        self.invalidations += 1
        self.token = f"tok-{self.invalidations + 1}"

    def call_with_retry(self, fn: Any) -> Any:
        return fn(self.get_token())


class AsyncFakeSession(FakeSession):
    async def get_token(self) -> str:  # type: ignore[override]
        return self.token

    async def invalidate(self) -> None:  # type: ignore[override]
        self.invalidations += 1
        self.token = f"tok-{self.invalidations + 1}"

    async def call_with_retry(self, fn: Any) -> Any:  # type: ignore[override]
        return await fn(await self.get_token())


class HeartbeatHTTP:
    """Stands in for the telemetry transport: answers every poll with the next
    scripted heartbeat, holding the last one."""

    def __init__(self, heartbeats: list[dict[str, Any]]) -> None:
        self.heartbeats = list(heartbeats)
        self.polls = 0

    def _next(self) -> GatewayResponse:
        self.polls += 1
        hb = self.heartbeats.pop(0) if len(self.heartbeats) > 1 else self.heartbeats[0]
        return GatewayResponse(200, {"accepted": 0, "failed": [], "heartbeat": hb}, httpx.Headers())

    def request(self, *a: Any, **k: Any) -> GatewayResponse:
        return self._next()


class AsyncHeartbeatHTTP(HeartbeatHTTP):
    async def request(self, *a: Any, **k: Any) -> GatewayResponse:  # type: ignore[override]
        return self._next()


def heartbeat(status: str = "active", stop: bool = False, version: int = 5) -> dict[str, Any]:
    return {
        "lifecycleStatus": status,
        "emergencyStop": stop,
        "telemetryMode": "advisory",
        "telemetryStalenessMinutes": 30,
        "configVersion": version,
        "serverTime": "t",
    }


class FastReconnect:
    """Shrinks the reconnect delays so a test does not wait seconds."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cs, "_MIN_BACKOFF_SECONDS", 0.05)
        monkeypatch.setattr(cs, "_MAX_BACKOFF_SECONDS", 0.1)


@pytest.fixture()
def fast(monkeypatch: pytest.MonkeyPatch) -> FastReconnect:
    from matimo_agdk import telemetry

    monkeypatch.setattr(telemetry, "_TENANT_WIDE_REFRESH_JITTER_SECONDS", 0.0)
    return FastReconnect(monkeypatch)


def build_sync(
    base_url: str, heartbeats: list[dict[str, Any]] | None = None
) -> tuple[TelemetryExporter, ControlStreamConsumer, FakeSession, HeartbeatHTTP, GatewayHTTP]:
    hb_http = HeartbeatHTTP(heartbeats or [heartbeat()])
    session = FakeSession()
    exporter = TelemetryExporter(
        hb_http,  # type: ignore[arg-type]
        session,
        flush_interval=3600.0,  # the interval poll must play no part in these tests
        heartbeat_interval=3600.0,
    )
    http = GatewayHTTP(base_url, "key")
    consumer = ControlStreamConsumer(http, session, exporter, lambda: ME, read_timeout=5.0)
    return exporter, consumer, session, hb_http, http


def test_a_pushed_suspend_reaches_state_within_a_second_with_polling_effectively_off(
    sse_server: tuple[ServerState, str], fast: FastReconnect
) -> None:
    state, base = sse_server
    exporter, consumer, _session, hb_http, http = build_sync(
        base, [heartbeat("active"), heartbeat("suspended", version=6)]
    )
    fired: list[str] = []
    exporter.set_on_suspend(lambda s: fired.append(s.lifecycle_status))
    exporter.start()
    consumer.start()
    try:
        assert wait_until(lambda: consumer.status == "connected", timeout=3)
        assert wait_until(lambda: exporter.state.lifecycle_status == "active", timeout=3)
        polls_before = hb_http.polls

        t0 = time.monotonic()
        state.current.push(lifecycle("suspended"))
        assert wait_until(exporter.is_suspended, timeout=1.0)
        assert time.monotonic() - t0 < 1.0
        assert fired == ["suspended"]  # on_suspend fired from the push, not a poll
        # ... and the confirming poll was triggered right away, well inside the 3600 s interval.
        assert wait_until(lambda: hb_http.polls > polls_before, timeout=2)
        assert wait_until(lambda: exporter.state.config_version == 6, timeout=2)
        with pytest.raises(Exception, match="suspended"):
            exporter.raise_if_suspended()
        # The stream's request carried the session token and asked for SSE.
        req = state.requests[0]
        assert req["path"] == "/v1/control/stream"
        assert req["headers"]["X-Matimo-Session-Token"] == "tok-1"
        assert req["headers"]["Accept"] == "text/event-stream"
        assert req["headers"]["Authorization"] == "Bearer key"
    finally:
        consumer.stop()
        exporter.stop()
        http.close()


def test_a_pushed_emergency_stop_tightens_at_once(
    sse_server: tuple[ServerState, str], fast: FastReconnect
) -> None:
    state, base = sse_server
    exporter, consumer, _s, _hb, http = build_sync(
        base, [heartbeat("active"), heartbeat(stop=True)]
    )
    exporter.start()
    consumer.start()
    try:
        assert wait_until(lambda: consumer.status == "connected", timeout=3)
        state.current.push(emergency_stop(True))
        assert wait_until(lambda: exporter.state.emergency_stop, timeout=1.0)
    finally:
        consumer.stop()
        exporter.stop()
        http.close()


def test_a_forged_restore_cannot_relax_state_the_heartbeat_still_says_is_suspended(
    sse_server: tuple[ServerState, str], fast: FastReconnect
) -> None:
    state, base = sse_server
    # Heartbeats: active, then suspended forever (the truth).
    exporter, consumer, _s, hb_http, http = build_sync(
        base, [heartbeat("active"), heartbeat("suspended")]
    )
    exporter.start()
    consumer.start()
    try:
        assert wait_until(lambda: consumer.status == "connected", timeout=3)
        state.current.push(lifecycle("suspended"))
        assert wait_until(exporter.is_suspended, timeout=1.0)
        assert wait_until(
            lambda: exporter.state.last_polled_monotonic > 0 and hb_http.polls >= 2, timeout=2
        )
        polls = hb_http.polls
        # A forged (or stale) "restore" arrives. It must not relax anything by itself,
        # and the poll it triggers re-confirms the truth: still suspended.
        state.current.push(lifecycle("active"))
        assert wait_until(lambda: hb_http.polls > polls, timeout=2)
        time.sleep(0.1)
        assert exporter.is_suspended()
    finally:
        consumer.stop()
        exporter.stop()
        http.close()


def test_a_real_restore_takes_effect_only_through_the_heartbeat_it_triggers(
    sse_server: tuple[ServerState, str], fast: FastReconnect
) -> None:
    state, base = sse_server
    exporter, consumer, _s, hb_http, http = build_sync(
        base, [heartbeat("active"), heartbeat("suspended"), heartbeat("active")]
    )
    exporter.start()
    consumer.start()
    try:
        assert wait_until(lambda: consumer.status == "connected", timeout=3)
        state.current.push(lifecycle("suspended"))
        assert wait_until(exporter.is_suspended, timeout=1.0)
        state.current.push(lifecycle("active"))
        # Not relaxed by the event itself, but relaxed by the confirming poll shortly after.
        assert wait_until(lambda: not exporter.is_suspended(), timeout=2)
        assert exporter.state.lifecycle_status == "active"
    finally:
        consumer.stop()
        exporter.stop()
        http.close()


def test_reconnect_starts_with_a_snapshot_that_catches_a_change_missed_while_disconnected(
    sse_server: tuple[ServerState, str], fast: FastReconnect
) -> None:
    state, base = sse_server
    # First connection: clean, then dropped. Second connection's snapshot says suspended.
    state.script = [[ready("active")], [ready("suspended")]]
    exporter, consumer, _s, _hb, http = build_sync(
        base, [heartbeat("active"), heartbeat("suspended")]
    )
    exporter.start()
    consumer.start()
    try:
        assert wait_until(lambda: consumer.status == "connected", timeout=3)
        state.current.close()  # the first stream drops
        assert wait_until(lambda: len(state.requests) >= 2, timeout=3)  # reconnected
        assert wait_until(exporter.is_suspended, timeout=2)  # the ready snapshot tightened it
    finally:
        consumer.stop()
        exporter.stop()
        http.close()


def test_an_older_server_without_the_route_leaves_polling_alone_and_is_not_hammered(
    sse_server: tuple[ServerState, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    state, base = sse_server
    state.script = [404]
    exporter, consumer, _s, hb_http, http = build_sync(base)
    exporter.start()
    consumer.start()
    try:
        assert wait_until(lambda: consumer.status == "unsupported", timeout=3)
        time.sleep(0.3)
        assert len(state.requests) == 1  # one attempt, then a long wait (600 s), not a retry storm
        assert consumer._thread is not None and consumer._thread.is_alive()
        assert wait_until(lambda: hb_http.polls >= 1, timeout=2)  # polling is untouched
    finally:
        consumer.stop()
        exporter.stop()
        http.close()


@pytest.mark.parametrize("code", [405, 501, 403])
def test_405_501_and_403_are_treated_as_a_quiet_refusal_not_a_crash(
    sse_server: tuple[ServerState, str], code: int
) -> None:
    state, base = sse_server
    state.script = [code]
    exporter, consumer, _s, _hb, http = build_sync(base)
    consumer.start()
    try:
        assert wait_until(lambda: len(state.requests) >= 1, timeout=3)
        assert wait_until(lambda: consumer.status in ("unsupported", "backoff"), timeout=3)
        assert consumer._thread is not None and consumer._thread.is_alive()
    finally:
        consumer.stop()
        http.close()


def test_a_401_session_expired_drops_the_session_and_reconnects_with_a_fresh_one(
    sse_server: tuple[ServerState, str], fast: FastReconnect
) -> None:
    state, base = sse_server
    state.script = [401]
    exporter, consumer, session, _hb, http = build_sync(base)
    consumer.start()
    try:
        assert wait_until(lambda: consumer.status == "connected", timeout=5)
        assert session.invalidations == 1
        assert [r["headers"]["X-Matimo-Session-Token"] for r in state.requests] == [
            "tok-1",
            "tok-2",
        ]
    finally:
        consumer.stop()
        http.close()


def test_a_revoked_identity_gets_a_long_delay_instead_of_a_reconnect_loop(
    sse_server: tuple[ServerState, str], fast: FastReconnect
) -> None:
    state, base = sse_server
    state.default = [
        ready("revoked"),
        frame("closing", {"type": "closing", "reason": "identity_revoked"}),
    ]
    exporter, consumer, _s, _hb, http = build_sync(base)
    consumer.start()
    try:
        assert wait_until(lambda: len(state.requests) >= 1, timeout=3)
        time.sleep(0.4)
        assert len(state.requests) == 1  # would be ~8 with the fast test backoff if it looped
        assert exporter.state.lifecycle_status == "revoked"  # tightened from the ready snapshot
    finally:
        consumer.stop()
        http.close()


def test_stop_returns_promptly_even_while_blocked_reading_an_idle_stream(
    sse_server: tuple[ServerState, str],
) -> None:
    state, base = sse_server
    exporter, consumer, _s, _hb, http = build_sync(base)
    consumer.start()
    assert wait_until(lambda: consumer.status == "connected", timeout=3)
    time.sleep(0.1)
    t0 = time.monotonic()
    consumer.stop(timeout=3.0)
    assert time.monotonic() - t0 < 2.5
    assert consumer._thread is None and consumer.status == "stopped"
    http.close()


def test_a_read_timeout_on_a_silent_stream_reconnects(
    sse_server: tuple[ServerState, str], fast: FastReconnect
) -> None:
    state, base = sse_server
    exporter, consumer, session, _hb, http = build_sync(base)
    consumer._read_timeout = 0.3  # the server never sends a keepalive here
    consumer.start()
    try:
        assert wait_until(lambda: len(state.requests) >= 2, timeout=5)
    finally:
        consumer.stop()
        http.close()


def test_a_connection_refused_never_raises_into_the_agent_and_keeps_retrying(
    fast: FastReconnect,
) -> None:
    exporter, consumer, _s, _hb, http = build_sync("http://127.0.0.1:1/v1")  # nothing listens there
    consumer.start()
    try:
        assert wait_until(lambda: consumer.status == "backoff", timeout=3)
        time.sleep(0.3)
        assert consumer._thread is not None and consumer._thread.is_alive()
    finally:
        consumer.stop()
        http.close()


# ---------------------------------------------------------------------------
# Async twin
# ---------------------------------------------------------------------------


async def _await_true(predicate: Any, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return bool(predicate())


def build_async(
    base_url: str, heartbeats: list[dict[str, Any]] | None = None
) -> tuple[
    AsyncTelemetryExporter,
    AsyncControlStreamConsumer,
    AsyncFakeSession,
    AsyncHeartbeatHTTP,
    AsyncGatewayHTTP,
]:
    hb_http = AsyncHeartbeatHTTP(heartbeats or [heartbeat()])
    session = AsyncFakeSession()
    exporter = AsyncTelemetryExporter(
        hb_http,  # type: ignore[arg-type]
        session,
        flush_interval=3600.0,
        heartbeat_interval=3600.0,
    )
    http = AsyncGatewayHTTP(base_url, "key")
    consumer = AsyncControlStreamConsumer(http, session, exporter, lambda: ME, read_timeout=5.0)
    return exporter, consumer, session, hb_http, http


async def test_async_a_pushed_suspend_tightens_at_once_and_triggers_the_confirming_poll(
    sse_server: tuple[ServerState, str], fast: FastReconnect
) -> None:
    state, base = sse_server
    exporter, consumer, _s, hb_http, http = build_async(
        base, [heartbeat("active"), heartbeat("suspended", version=6)]
    )
    await exporter.start()
    await consumer.start()
    try:
        assert await _await_true(lambda: consumer.status == "connected")
        assert await _await_true(lambda: exporter.state.lifecycle_status == "active")
        polls = hb_http.polls
        state.current.push(lifecycle("suspended"))
        assert await _await_true(exporter.is_suspended, timeout=1.0)
        assert await _await_true(lambda: hb_http.polls > polls)
        assert await _await_true(lambda: exporter.state.config_version == 6)
    finally:
        await consumer.stop()
        await exporter.stop()
        await http.aclose()


async def test_async_an_event_cannot_relax_state_and_a_restore_arrives_via_the_poll(
    sse_server: tuple[ServerState, str], fast: FastReconnect
) -> None:
    state, base = sse_server
    exporter, consumer, _s, hb_http, http = build_async(
        base, [heartbeat("active"), heartbeat("suspended"), heartbeat("active")]
    )
    await exporter.start()
    await consumer.start()
    try:
        assert await _await_true(lambda: consumer.status == "connected")
        state.current.push(lifecycle("suspended"))
        assert await _await_true(exporter.is_suspended, timeout=1.0)
        state.current.push(lifecycle("active"))
        assert await _await_true(lambda: not exporter.is_suspended(), timeout=2.0)
    finally:
        await consumer.stop()
        await exporter.stop()
        await http.aclose()


async def test_async_404_is_a_quiet_unsupported_and_stop_is_prompt(
    sse_server: tuple[ServerState, str],
) -> None:
    state, base = sse_server
    state.script = [404]
    exporter, consumer, _s, _hb, http = build_async(base)
    await exporter.start()
    await consumer.start()
    try:
        assert await _await_true(lambda: consumer.status == "unsupported")
        await asyncio.sleep(0.2)
        assert len(state.requests) == 1
    finally:
        t0 = time.monotonic()
        await consumer.stop()
        assert time.monotonic() - t0 < 2.5
        await exporter.stop()
        await http.aclose()


async def test_async_stop_cancels_a_reader_blocked_on_an_idle_stream(
    sse_server: tuple[ServerState, str],
) -> None:
    state, base = sse_server
    exporter, consumer, _s, _hb, http = build_async(base)
    await consumer.start()
    assert await _await_true(lambda: consumer.status == "connected")
    t0 = time.monotonic()
    await consumer.stop()
    assert time.monotonic() - t0 < 2.5
    assert consumer.status == "stopped"
    await http.aclose()


async def test_async_reconnect_snapshot_catches_a_missed_change(
    sse_server: tuple[ServerState, str], fast: FastReconnect
) -> None:
    state, base = sse_server
    state.script = [[ready("active")], [ready("suspended")]]
    exporter, consumer, _s, _hb, http = build_async(
        base, [heartbeat("active"), heartbeat("suspended")]
    )
    await exporter.start()
    await consumer.start()
    try:
        assert await _await_true(lambda: consumer.status == "connected")
        state.current.close()
        assert await _await_true(exporter.is_suspended, timeout=3.0)
    finally:
        await consumer.stop()
        await exporter.stop()
        await http.aclose()


# ---------------------------------------------------------------------------
# stream_get transport behaviour
# ---------------------------------------------------------------------------


def test_stream_get_maps_error_statuses_to_the_same_typed_errors_request_does() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers["x-case"] == "401":
            return httpx.Response(401, json={"error": "session_expired"})
        return httpx.Response(404, text="<html>nope</html>")

    http = GatewayHTTP(
        BASE_URL,
        "k",
        client=httpx.Client(base_url=BASE_URL, transport=httpx.MockTransport(handler)),
    )
    with (
        pytest.raises(SessionExpired),
        http.stream_get("/control/stream", headers={"x-case": "401"}),
    ):
        pass
    with (
        pytest.raises(GatewayError) as excinfo,
        http.stream_get("/control/stream", headers={"x-case": "404"}),
    ):
        pass
    assert excinfo.value.status_code == 404


def test_stream_get_yields_lines_and_does_not_send_a_json_content_type() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        return httpx.Response(200, content=ready().encode() + b": keepalive\n\n")

    http = GatewayHTTP(
        BASE_URL,
        "k",
        client=httpx.Client(base_url=BASE_URL, transport=httpx.MockTransport(handler)),
    )
    with http.stream_get("/control/stream", headers={"X-Matimo-Session-Token": "t"}) as resp:
        lines = list(resp.iter_lines())
    assert "event: ready" in lines
    assert seen["accept"] == "text/event-stream"
    assert "content-type" not in seen
    assert seen["x-matimo-session-token"] == "t"


def test_stream_get_turns_a_transport_failure_into_gateway_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    http = GatewayHTTP(
        BASE_URL,
        "k",
        client=httpx.Client(base_url=BASE_URL, transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(GatewayUnavailable), http.stream_get("/control/stream"):
        pass


async def test_async_stream_get_maps_errors_and_yields_lines() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers.get("x-case") == "401":
            return httpx.Response(401, json={"error": "session_expired"})
        return httpx.Response(200, content=ready().encode())

    http = AsyncGatewayHTTP(
        BASE_URL,
        "k",
        client=httpx.AsyncClient(base_url=BASE_URL, transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(SessionExpired):
        async with http.stream_get("/control/stream", headers={"x-case": "401"}):
            pass
    async with http.stream_get("/control/stream") as resp:
        lines = [line async for line in resp.aiter_lines()]
    assert "event: ready" in lines
    await http.aclose()


# ---------------------------------------------------------------------------
# Governor wiring and the disable flag
# ---------------------------------------------------------------------------


def governor_config(identity: Any, credentials_dir: Any, **over: Any) -> GatewayConfig:
    return GatewayConfig(
        base_url=BASE_URL,
        api_key="k",
        identity_token=identity.identity_token,
        identity_id=identity.identity_id,
        tenant_id=identity.tenant_id,
        private_key_pem=identity.private_key_pem,
        agent_name=identity.display_name,
        credentials_dir=credentials_dir,
        **over,
    )


@respx.mock
def test_the_flag_turns_the_push_channel_off_and_it_never_touches_the_route(
    identity: Any, credentials_dir: Any
) -> None:
    respx.post(f"{BASE_URL}/sessions").mock(
        return_value=httpx.Response(201, json=make_session_response())
    )
    respx.post(f"{BASE_URL}/telemetry/batch").mock(
        return_value=httpx.Response(200, json={"data": {"heartbeat": heartbeat()}})
    )
    stream = respx.get(f"{BASE_URL}/control/stream").mock(
        return_value=httpx.Response(200, text=ready())
    )
    gov = Governor(governor_config(identity, credentials_dir, control_stream_enabled=False))
    gov.start()
    try:
        assert gov.control_stream_status == "disabled"
        time.sleep(0.2)
        assert stream.call_count == 0
    finally:
        gov.close()


@respx.mock
def test_a_governor_against_a_server_without_the_route_reports_unsupported_and_keeps_polling(
    identity: Any, credentials_dir: Any
) -> None:
    respx.post(f"{BASE_URL}/sessions").mock(
        return_value=httpx.Response(201, json=make_session_response())
    )
    respx.post(f"{BASE_URL}/telemetry/batch").mock(
        return_value=httpx.Response(200, json={"data": {"heartbeat": heartbeat("suspended")}})
    )
    respx.get(f"{BASE_URL}/control/stream").mock(
        return_value=httpx.Response(404, text="Cannot GET")
    )
    gov = Governor(governor_config(identity, credentials_dir))
    gov.start()
    try:
        assert wait_until(lambda: gov.control_stream_status == "unsupported", timeout=3)
        assert wait_until(gov.is_suspended, timeout=3)  # the poll still works
    finally:
        gov.close()
    assert gov.control_stream_status == "disabled"  # cleared on stop


@respx.mock
def test_governor_stop_then_start_restarts_the_push_channel(
    identity: Any, credentials_dir: Any
) -> None:
    respx.post(f"{BASE_URL}/sessions").mock(
        return_value=httpx.Response(201, json=make_session_response())
    )
    respx.post(f"{BASE_URL}/telemetry/batch").mock(
        return_value=httpx.Response(200, json={"data": {"heartbeat": heartbeat()}})
    )
    route = respx.get(f"{BASE_URL}/control/stream").mock(return_value=httpx.Response(404, text=""))
    gov = Governor(governor_config(identity, credentials_dir))
    gov.start()
    assert wait_until(lambda: route.call_count == 1, timeout=3)
    gov.stop()
    assert gov.control_stream_status == "disabled"
    gov.start()
    try:
        assert wait_until(lambda: route.call_count == 2, timeout=3)
    finally:
        gov.close()


@respx.mock
async def test_async_governor_wiring_and_flag(identity: Any, credentials_dir: Any) -> None:
    respx.post(f"{BASE_URL}/sessions").mock(
        return_value=httpx.Response(201, json=make_session_response())
    )
    respx.post(f"{BASE_URL}/telemetry/batch").mock(
        return_value=httpx.Response(200, json={"data": {"heartbeat": heartbeat()}})
    )
    route = respx.get(f"{BASE_URL}/control/stream").mock(return_value=httpx.Response(404, text=""))

    off = AsyncGovernor(governor_config(identity, credentials_dir, control_stream_enabled=False))
    await off.start()
    assert off.control_stream_status == "disabled"
    await off.aclose()
    assert route.call_count == 0

    on = AsyncGovernor(governor_config(identity, credentials_dir))
    await on.start()
    try:
        assert await _await_true(lambda: on.control_stream_status == "unsupported")
    finally:
        await on.aclose()


def test_the_env_var_disables_the_push_channel(
    monkeypatch: pytest.MonkeyPatch, credentials_dir: Any
) -> None:
    for value, expected in (
        ("0", False),
        ("false", False),
        ("off", False),
        ("1", True),
        ("yes", True),
    ):
        monkeypatch.setenv("MATIMO_CONTROL_STREAM", value)
        assert (
            GatewayConfig.load(credentials_dir=credentials_dir).control_stream_enabled is expected
        )
    monkeypatch.delenv("MATIMO_CONTROL_STREAM")
    assert GatewayConfig.load(credentials_dir=credentials_dir).control_stream_enabled is True


# ---------------------------------------------------------------------------
# on_suspend registered after the agent was already found suspended
# ---------------------------------------------------------------------------


def test_a_callback_registered_after_a_push_suspend_still_fires_exactly_once() -> None:
    exporter = TelemetryExporter(HeartbeatHTTP([heartbeat()]), FakeSession())  # type: ignore[arg-type]
    exporter.state.lifecycle_status = "active"
    assert exporter.apply_control_hint(
        "lifecycle", {"identityId": ME, "lifecycleStatus": "suspended"}, ME
    )
    fired: list[str] = []
    exporter.set_on_suspend(lambda s: fired.append(s.lifecycle_status))
    assert fired == ["suspended"]
    exporter.set_on_suspend(
        lambda s: fired.append("again")
    )  # a second registration does not re-fire
    assert fired == ["suspended"]


async def test_async_a_callback_registered_after_a_push_suspend_still_fires_exactly_once() -> None:
    exporter = AsyncTelemetryExporter(AsyncHeartbeatHTTP([heartbeat()]), AsyncFakeSession())  # type: ignore[arg-type]
    assert exporter.apply_control_hint("emergency_stop", {"emergencyStop": True}, ME)
    fired: list[bool] = []
    exporter.set_on_suspend(lambda s: fired.append(s.emergency_stop))
    assert fired == [True]


def test_the_callback_fires_again_for_a_new_suspension_after_the_agent_was_restored() -> None:
    exporter = TelemetryExporter(HeartbeatHTTP([heartbeat()]), FakeSession())  # type: ignore[arg-type]
    fired: list[int] = []
    exporter.set_on_suspend(lambda s: fired.append(len(fired)))
    exporter.apply_control_hint("lifecycle", {"identityId": ME, "lifecycleStatus": "suspended"}, ME)
    exporter.state.update_from_heartbeat(heartbeat("active"))
    exporter._maybe_notify_suspend()  # what a heartbeat does after updating state
    exporter.apply_control_hint("lifecycle", {"identityId": ME, "lifecycleStatus": "suspended"}, ME)
    assert fired == [0, 1]
