"""Governor behaviours fixed in the 2026-09-18 review: guard() outside a
run, DENY spans, stop()/start() again, Anthropic auth, and the async
guard() path (previously untested)."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from matimo_agdk.config import GatewayConfig
from matimo_agdk.exceptions import ToolDenied
from matimo_agdk.governor import AsyncGovernor, Governor
from matimo_agdk.identity import IdentityCredentials

from .conftest import BASE_URL, future_iso


class RecordingExporter:
    """Stands in for the telemetry exporter: records every submitted event."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def submit(self, event: dict[str, Any]) -> None:
        self.events.append(event)

    def stop(self) -> None:
        pass


class AsyncRecordingExporter(RecordingExporter):
    async def stop(self) -> None:  # type: ignore[override]
        pass


def _config(identity: IdentityCredentials, **overrides: Any) -> GatewayConfig:
    return GatewayConfig(
        base_url=BASE_URL,
        api_key="org-key",
        identity_token=identity.identity_token,
        identity_id=identity.identity_id,
        tenant_id=identity.tenant_id,
        private_key_pem=identity.private_key_pem,
        agent_name=identity.display_name,
        telemetry_flush_interval=9999,
        heartbeat_interval=9999,
        **overrides,
    )


def _mock_session() -> None:
    respx.post(f"{BASE_URL}/sessions").mock(
        return_value=httpx.Response(
            201,
            json={"data": {"sessionToken": "tok", "expiresAt": future_iso(3600), "identityId": "x"}},
        )
    )


@respx.mock
def test_guard_outside_run_opens_an_implicit_run(identity: IdentityCredentials) -> None:
    _mock_session()
    respx.post(f"{BASE_URL}/tools/check").mock(
        return_value=httpx.Response(200, json={"data": {"decision": "ALLOW"}})
    )
    governor = Governor(_config(identity))
    recorder = RecordingExporter()
    governor._telemetry = recorder  # type: ignore[assignment]  # noqa: SLF001
    calls: list[int] = []

    @governor.guard(name="add")
    def add(a: int, b: int) -> int:
        calls.append(1)
        return a + b

    assert add(a=2, b=3) == 5
    assert calls == [1]
    kinds = [(e["kind"], e.get("status"), e.get("name")) for e in recorder.events]
    assert ("run", "running", "tool:add") in kinds
    assert ("tool", "completed", "add") in kinds
    assert ("run", "completed", "tool:add") in kinds
    run_ids = {e["runId"] for e in recorder.events}
    assert len(run_ids) == 1


@respx.mock
def test_deny_records_a_tool_span_and_never_runs_the_tool(identity: IdentityCredentials) -> None:
    _mock_session()
    respx.post(f"{BASE_URL}/tools/check").mock(
        return_value=httpx.Response(200, json={"data": {"decision": "DENY", "reason": "nope"}})
    )
    governor = Governor(_config(identity))
    recorder = RecordingExporter()
    governor._telemetry = recorder  # type: ignore[assignment]  # noqa: SLF001
    ran: list[int] = []

    def wire(amount: int) -> str:
        ran.append(1)
        return "sent"

    with governor.run("r"), pytest.raises(ToolDenied):
        governor.guard(wire, name="wire")(amount=1)
    assert ran == []
    denied = [e for e in recorder.events if e["kind"] == "tool"]
    assert len(denied) == 1
    assert denied[0]["status"] == "denied"
    assert denied[0]["attributes"]["gen_ai.tool.call.arguments"] == {"amount": 1}


@respx.mock
def test_stop_then_start_again_works(identity: IdentityCredentials) -> None:
    _mock_session()
    batch = respx.post(f"{BASE_URL}/telemetry/batch").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "accepted": 0,
                    "failed": [],
                    "heartbeat": {"lifecycleStatus": "active", "telemetryStalenessMinutes": 30},
                }
            },
        )
    )
    governor = Governor(_config(identity))
    governor.start()
    governor.stop()
    governor.start()
    assert governor._telemetry is not None  # noqa: SLF001
    governor._telemetry.flush_now()  # noqa: SLF001
    governor.stop()
    assert batch.call_count >= 1
    governor.close()


@respx.mock
def test_anthropic_kwargs_use_auth_token(identity: IdentityCredentials) -> None:
    _mock_session()
    kwargs = Governor(_config(identity)).anthropic_client_kwargs()
    assert kwargs["auth_token"] == "org-key"
    assert "api_key" not in kwargs
    assert kwargs["base_url"] == BASE_URL


@respx.mock
async def test_async_guard_allow_and_deny(identity: IdentityCredentials) -> None:
    _mock_session()
    route = respx.post(f"{BASE_URL}/tools/check")
    route.side_effect = [
        httpx.Response(200, json={"data": {"decision": "ALLOW"}}),
        httpx.Response(200, json={"data": {"decision": "DENY", "reason": "no"}}),
    ]
    governor = AsyncGovernor(_config(identity))
    recorder = AsyncRecordingExporter()
    governor._telemetry = recorder  # type: ignore[assignment]  # noqa: SLF001

    @governor.guard(name="lookup")
    async def lookup(q: str) -> str:
        return f"hit:{q}"

    async with governor.run("async-run"):
        assert await lookup(q="x") == "hit:x"
        with pytest.raises(ToolDenied):
            await lookup(q="y")
    statuses = [e.get("status") for e in recorder.events if e["kind"] == "tool"]
    assert statuses == ["completed", "denied"]
    await governor.aclose()
