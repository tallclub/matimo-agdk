"""Async twin of test_telemetry.py's exporter tests (review finding
2026-09-18: AsyncTelemetryExporter had no direct tests)."""

from __future__ import annotations

from typing import Any

import pytest

from matimo_agdk.exceptions import AgentSuspendedLocally, GatewayError, SessionExpired
from matimo_agdk.telemetry import AsyncTelemetryExporter, build_event
from matimo_agdk.transport import GatewayResponse

from .test_telemetry import heartbeat


class AsyncFakeHTTP:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def request(
        self, method: str, path: str, *, json_body: Any = None, **kwargs: Any
    ) -> GatewayResponse:
        self.calls.append({"method": method, "path": path, "json_body": json_body})
        if not self._responses:
            return GatewayResponse(status_code=200, data={"accepted": 0, "failed": []}, headers={})
        outcome = self._responses.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return GatewayResponse(status_code=200, data=outcome, headers={})


class AsyncFakeSessionManager:
    def __init__(self, token: str = "tok") -> None:
        self.token = token
        self.invalidate_count = 0

    async def get_token(self) -> str:
        return self.token

    async def invalidate(self) -> None:
        self.invalidate_count += 1
        self.token = f"{self.token}-renewed"

    async def call_with_retry(self, fn: Any) -> Any:
        token = await self.get_token()
        try:
            return await fn(token)
        except SessionExpired:
            await self.invalidate()
            token = await self.get_token()
            return await fn(token)


async def test_flush_now_sends_queued_events_in_one_batch() -> None:
    http = AsyncFakeHTTP([{"accepted": 2, "failed": [], "heartbeat": heartbeat()}])
    exporter = AsyncTelemetryExporter(http, AsyncFakeSessionManager(), heartbeat_interval=9999)
    exporter.submit(build_event(run_id="r1", kind="llm"))
    exporter.submit(build_event(run_id="r1", kind="tool"))
    await exporter.flush_now()
    assert len(http.calls) == 1
    assert len(http.calls[0]["json_body"]["events"]) == 2


async def test_heartbeat_updates_state_and_raise_if_suspended() -> None:
    http = AsyncFakeHTTP([{"accepted": 0, "failed": [], "heartbeat": heartbeat(lifecycle="suspended")}])
    exporter = AsyncTelemetryExporter(http, AsyncFakeSessionManager(), heartbeat_interval=0.0)
    await exporter.flush_now()
    assert exporter.state.lifecycle_status == "suspended"
    with pytest.raises(AgentSuspendedLocally):
        exporter.raise_if_suspended()


async def test_fail_open_requeues_events_on_gateway_error() -> None:
    http = AsyncFakeHTTP([GatewayError("boom")])
    exporter = AsyncTelemetryExporter(http, AsyncFakeSessionManager(), heartbeat_interval=9999)
    exporter.submit(build_event(run_id="r1", kind="log"))
    await exporter.flush_now()
    assert exporter._queue is not None
    assert exporter._queue.qsize() == 1
    assert exporter.last_error is None


async def test_fail_closed_stores_error_and_raises_on_next_call() -> None:
    http = AsyncFakeHTTP([GatewayError("boom")])
    exporter = AsyncTelemetryExporter(
        http, AsyncFakeSessionManager(), heartbeat_interval=9999, fail_open=False
    )
    exporter.submit(build_event(run_id="r1", kind="log"))
    await exporter._flush(force=True)  # what the background loop does: never raises
    assert isinstance(exporter.last_error, GatewayError)
    with pytest.raises(GatewayError):
        exporter.submit(build_event(run_id="r1", kind="log"))
    assert exporter.last_error is None  # raised exactly once


async def test_flush_reauthenticates_once_on_session_expired() -> None:
    session = AsyncFakeSessionManager(token="stale")
    http = AsyncFakeHTTP(
        [
            SessionExpired("session expired", status_code=401, code="session_expired"),
            {"accepted": 1, "failed": [], "heartbeat": heartbeat()},
        ]
    )
    exporter = AsyncTelemetryExporter(http, session, heartbeat_interval=9999)
    exporter.submit(build_event(run_id="r1", kind="log"))
    await exporter.flush_now()
    assert session.invalidate_count == 1
    assert len(http.calls) == 2
    assert exporter.state.lifecycle_status == "active"


async def test_heartbeat_interval_resized_from_server_staleness() -> None:
    http = AsyncFakeHTTP([{"accepted": 0, "failed": [], "heartbeat": heartbeat(staleness=3.0)}])
    exporter = AsyncTelemetryExporter(
        http,
        AsyncFakeSessionManager(),
        heartbeat_interval=300.0,
        heartbeat_resolver=lambda minutes: max(15.0, min(300.0, minutes * 60.0 / 3.0)),
    )
    await exporter.flush_now()
    assert exporter._heartbeat_interval == 60.0


async def test_background_task_first_tick_is_a_heartbeat() -> None:
    http = AsyncFakeHTTP([{"accepted": 0, "failed": [], "heartbeat": heartbeat()}])
    exporter = AsyncTelemetryExporter(
        http, AsyncFakeSessionManager(), flush_interval=0.05, heartbeat_interval=9999
    )
    await exporter.start()
    try:
        import asyncio

        for _ in range(40):
            if http.calls:
                break
            await asyncio.sleep(0.02)
        assert http.calls and http.calls[0]["json_body"]["events"] == []
    finally:
        await exporter.stop()
