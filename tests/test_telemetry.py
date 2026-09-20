from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from matimo_agdk.exceptions import AgentSuspendedLocally, GatewayError, SessionExpired
from matimo_agdk.telemetry import (
    GovernanceState,
    TelemetryExporter,
    build_event,
    llm_span,
    redact_attributes,
    run_span,
    tool_span,
)
from matimo_agdk.transport import GatewayResponse


class FakeHTTP:
    """A minimal stand-in for GatewayHTTP that records every telemetry
    batch it receives and returns a scripted sequence of responses."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def request(
        self, method: str, path: str, *, json_body: Any = None, **kwargs: Any
    ) -> GatewayResponse:
        with self._lock:
            self.calls.append({"method": method, "path": path, "json_body": json_body})
            if not self._responses:
                return GatewayResponse(
                    status_code=200, data={"accepted": 0, "failed": []}, headers={}
                )
            outcome = self._responses.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return GatewayResponse(status_code=200, data=outcome, headers={})


class FakeSessionManager:
    """Mirrors the real SessionManager.call_with_retry() contract closely
    enough to prove _flush() actually goes through it now (the 2026-09-18
    live verification, see CHANGELOG.md, found this call site previously
    bypassed call_with_retry() entirely, calling get_token() directly -- a
    session invalidated behind the SDK's back was then silently
    unrecoverable via telemetry)."""

    def __init__(self, token: str = "tok") -> None:
        self.token = token
        self.invalidate_count = 0

    def get_token(self) -> str:
        return self.token

    def invalidate(self) -> None:
        self.invalidate_count += 1
        self.token = f"{self.token}-renewed"

    def call_with_retry(self, fn: Any) -> Any:
        token = self.get_token()
        try:
            return fn(token)
        except SessionExpired:
            self.invalidate()
            token = self.get_token()
            return fn(token)


def heartbeat(
    lifecycle: str = "active",
    emergency: bool = False,
    mode: str = "advisory",
    staleness: float = 30.0,
) -> dict[str, Any]:
    return {
        "lifecycleStatus": lifecycle,
        "emergencyStop": emergency,
        "telemetryMode": mode,
        "telemetryStalenessMinutes": staleness,
        "serverTime": "2026-09-18T00:00:00Z",
    }


# ---------------------------------------------------------------------------
# GovernanceState
# ---------------------------------------------------------------------------


def test_governance_state_defaults_not_suspended() -> None:
    state = GovernanceState()
    assert state.is_suspended is False


def test_governance_state_suspended_on_lifecycle() -> None:
    state = GovernanceState()
    state.update_from_heartbeat(heartbeat(lifecycle="suspended"))
    assert state.is_suspended is True


def test_governance_state_suspended_on_emergency_stop() -> None:
    state = GovernanceState()
    state.update_from_heartbeat(heartbeat(emergency=True))
    assert state.is_suspended is True


def test_governance_state_revoked_is_suspended() -> None:
    state = GovernanceState()
    state.update_from_heartbeat(heartbeat(lifecycle="revoked"))
    assert state.is_suspended is True


# ---------------------------------------------------------------------------
# Event builders
# ---------------------------------------------------------------------------


def test_build_event_minimal() -> None:
    event = build_event(run_id="r1", kind="log")
    assert event == {"runId": "r1", "kind": "log"}


def test_run_span_sets_gen_ai_operation_name() -> None:
    event = run_span("r1", name="my-run", status="running")
    assert event["attributes"]["gen_ai.operation.name"] == "invoke_agent"


def test_llm_span_operation_name_override() -> None:
    event = llm_span("r1", operation_name="generate_content", model="gemini-pro")
    assert event["attributes"]["gen_ai.operation.name"] == "generate_content"
    assert event["attributes"]["gen_ai.request.model"] == "gemini-pro"


def test_tool_span_fields() -> None:
    event = tool_span("r1", "search", call_id="call-1", arguments={"q": "x"}, result="answer")
    attrs = event["attributes"]
    assert attrs["gen_ai.tool.name"] == "search"
    assert attrs["gen_ai.tool.call.id"] == "call-1"
    assert attrs["gen_ai.tool.call.arguments"] == {"q": "x"}
    assert attrs["gen_ai.tool.call.result"] == "answer"


def test_redact_attributes_masks_secret_like_keys() -> None:
    attrs = redact_attributes({"api_key": "sk-123", "note": "fine", "password": "hunter2"})
    assert attrs["api_key"] == "[REDACTED]"
    assert attrs["password"] == "[REDACTED]"
    assert attrs["note"] == "fine"


def test_redact_attributes_truncates_long_strings() -> None:
    long_value = "x" * 3000
    attrs = redact_attributes({"note": long_value})
    assert len(attrs["note"]) < 3000
    assert attrs["note"].endswith("...[TRUNCATED]")


# ---------------------------------------------------------------------------
# TelemetryExporter: batching, heartbeat, overflow, suspend transition
# ---------------------------------------------------------------------------


def test_flush_now_sends_queued_events_in_one_batch() -> None:
    http = FakeHTTP([{"accepted": 2, "failed": [], "heartbeat": heartbeat()}])
    exporter = TelemetryExporter(http, FakeSessionManager(), heartbeat_interval=9999)
    exporter.submit(build_event(run_id="r1", kind="llm"))
    exporter.submit(build_event(run_id="r1", kind="tool"))
    exporter.flush_now()

    assert len(http.calls) == 1
    assert len(http.calls[0]["json_body"]["events"]) == 2


def test_flush_sends_empty_heartbeat_when_due() -> None:
    http = FakeHTTP([{"accepted": 0, "failed": [], "heartbeat": heartbeat()}])
    exporter = TelemetryExporter(http, FakeSessionManager(), heartbeat_interval=0.0)
    exporter.flush_now()
    assert http.calls[0]["json_body"]["events"] == []


def test_flush_skips_when_nothing_pending_and_not_due() -> None:
    http = FakeHTTP([])
    exporter = TelemetryExporter(http, FakeSessionManager(), heartbeat_interval=9999)
    exporter._flush()  # not forced, nothing queued, heartbeat not due
    assert len(http.calls) == 0


def test_flush_updates_governance_state_from_heartbeat() -> None:
    http = FakeHTTP([{"accepted": 0, "failed": [], "heartbeat": heartbeat(lifecycle="suspended")}])
    exporter = TelemetryExporter(http, FakeSessionManager(), heartbeat_interval=0.0)
    exporter.flush_now()
    assert exporter.state.lifecycle_status == "suspended"
    assert exporter.is_suspended() is True


def test_raise_if_suspended_raises_after_suspend_heartbeat() -> None:
    http = FakeHTTP([{"accepted": 0, "failed": [], "heartbeat": heartbeat(emergency=True)}])
    exporter = TelemetryExporter(http, FakeSessionManager(), heartbeat_interval=0.0)
    exporter.flush_now()
    with pytest.raises(AgentSuspendedLocally):
        exporter.raise_if_suspended()


def test_on_suspend_callback_fires_once_on_transition() -> None:
    http = FakeHTTP(
        [
            {"accepted": 0, "failed": [], "heartbeat": heartbeat(lifecycle="active")},
            {"accepted": 0, "failed": [], "heartbeat": heartbeat(lifecycle="suspended")},
            {"accepted": 0, "failed": [], "heartbeat": heartbeat(lifecycle="suspended")},
            {"accepted": 0, "failed": [], "heartbeat": heartbeat(lifecycle="active")},
            {"accepted": 0, "failed": [], "heartbeat": heartbeat(lifecycle="suspended")},
        ]
    )
    calls: list[GovernanceState] = []
    exporter = TelemetryExporter(
        http, FakeSessionManager(), heartbeat_interval=0.0, on_suspend=lambda s: calls.append(s)
    )
    for _ in range(5):
        exporter.flush_now()
    # Suspended fired on transition into suspended (2nd and 5th flush), not
    # on the repeated 3rd suspended flush.
    assert len(calls) == 2


def test_overflow_drops_oldest_event() -> None:
    http = FakeHTTP([])
    exporter = TelemetryExporter(http, FakeSessionManager(), queue_max=2, heartbeat_interval=9999)
    exporter.submit(build_event(run_id="r1", kind="log", name="first"))
    exporter.submit(build_event(run_id="r1", kind="log", name="second"))
    exporter.submit(build_event(run_id="r1", kind="log", name="third"))
    assert exporter.dropped_count == 1

    drained = exporter._drain()
    names = [e.get("name") for e in drained]
    assert "first" not in names
    assert "second" in names
    assert "third" in names


def test_fail_open_requeues_events_on_gateway_error() -> None:
    http = FakeHTTP([GatewayError("boom")])
    exporter = TelemetryExporter(
        http, FakeSessionManager(), heartbeat_interval=9999, fail_open=True
    )
    exporter.submit(build_event(run_id="r1", kind="log"))
    exporter.flush_now()  # first attempt fails and re-queues
    assert exporter._queue.qsize() == 1


def test_fail_closed_raises_on_gateway_error() -> None:
    http = FakeHTTP([GatewayError("boom")])
    exporter = TelemetryExporter(
        http, FakeSessionManager(), heartbeat_interval=9999, fail_open=False
    )
    exporter.submit(build_event(run_id="r1", kind="log"))
    with pytest.raises(GatewayError):
        exporter.flush_now()


def test_flush_reauthenticates_once_on_session_expired_then_succeeds() -> None:
    """Regression test for a real bug found live-verifying against Gateway
    (2026-09-18 live verification, see CHANGELOG.md): _flush() used to call
    session_manager.get_token() directly, so a session invalidated behind the
    SDK's back (e.g. another
    process calling DELETE /v1/sessions) produced an infinite loop of
    SessionExpired -> re-queue -> SessionExpired again, since nothing ever
    told the SessionManager to drop its stale cached token. Routing through
    call_with_retry() (previously defined but never called anywhere) fixes
    this: one SessionExpired triggers exactly one re-handshake, and the
    retried call succeeds."""
    session = FakeSessionManager(token="stale-token")
    http = FakeHTTP(
        [
            SessionExpired("session expired", status_code=401, code="session_expired"),
            {"accepted": 1, "failed": [], "heartbeat": heartbeat()},
        ]
    )
    exporter = TelemetryExporter(http, session, heartbeat_interval=9999)
    exporter.submit(build_event(run_id="r1", kind="log"))
    exporter.flush_now()

    assert session.invalidate_count == 1
    assert len(http.calls) == 2  # the failed attempt, then the retried one
    assert exporter._queue.qsize() == 0  # nothing left re-queued -- it succeeded
    assert exporter.state.lifecycle_status == "active"  # heartbeat from the retried call landed


def test_batch_size_limits_events_per_flush() -> None:
    http = FakeHTTP([{"accepted": 3, "failed": []}, {"accepted": 2, "failed": []}])
    exporter = TelemetryExporter(http, FakeSessionManager(), batch_size=3, heartbeat_interval=9999)
    for i in range(5):
        exporter.submit(build_event(run_id="r1", kind="log", name=str(i)))
    exporter.flush_now()
    assert len(http.calls[0]["json_body"]["events"]) == 3

    exporter.flush_now()
    assert len(http.calls[1]["json_body"]["events"]) == 2


def test_background_thread_flushes_on_interval() -> None:
    http = FakeHTTP([{"accepted": 1, "failed": []}] * 5)
    exporter = TelemetryExporter(
        http, FakeSessionManager(), flush_interval=0.05, heartbeat_interval=9999
    )
    exporter.start()
    try:
        exporter.submit(build_event(run_id="r1", kind="log"))
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not http.calls:
            time.sleep(0.02)
        assert len(http.calls) >= 1
    finally:
        exporter.stop()
