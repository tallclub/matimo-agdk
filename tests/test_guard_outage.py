"""Governor.guard() / check_tool() during a Gateway outage: config wiring, the
degraded marker on the span, and the circuit breaker seen from the public API."""

from __future__ import annotations

import dataclasses
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
import respx

from matimo_agdk.exceptions import PolicyDenied, ToolCheckUnavailable
from matimo_agdk.governor import AsyncGovernor, Governor
from matimo_agdk.identity import IdentityCredentials
from matimo_agdk.telemetry import GovernanceState
from matimo_agdk.transport import RetryPolicy

from .conftest import BASE_URL
from .test_governor import bound_config

CHECK = f"{BASE_URL}/tools/check"


def _down() -> Any:
    return httpx.ConnectError("gateway down")


def _allow() -> httpx.Response:
    return httpx.Response(200, json={"data": {"decision": "ALLOW"}})


def make_governor(identity: IdentityCredentials, credentials_dir: Any, **config: Any) -> Governor:
    base = bound_config(identity, credentials_dir)
    governor = Governor(base.model_copy(update=config))
    governor._http.retry_policy = RetryPolicy(max_retries=0)  # noqa: SLF001
    governor._telemetry = MagicMock()  # noqa: SLF001
    governor._telemetry.state = GovernanceState()  # noqa: SLF001
    return governor


def make_async_governor(
    identity: IdentityCredentials, credentials_dir: Any, **config: Any
) -> AsyncGovernor:
    base = bound_config(identity, credentials_dir)
    governor = AsyncGovernor(base.model_copy(update=config))
    governor._http.retry_policy = RetryPolicy(max_retries=0)  # noqa: SLF001
    governor._telemetry = MagicMock()  # noqa: SLF001
    governor._telemetry.state = GovernanceState()  # noqa: SLF001
    return governor


def tool_spans(governor: Any) -> list[dict[str, Any]]:
    submitted = [c.args[0] for c in governor._telemetry.submit.call_args_list]  # noqa: SLF001
    return [e for e in submitted if e["kind"] == "tool"]


def test_config_is_wired_into_the_tool_governor(
    identity: IdentityCredentials, credentials_dir: Any
) -> None:
    governor = make_governor(
        identity,
        credentials_dir,
        tool_check_failure_mode="fail_open_bounded",
        fail_open_max_stale_seconds=120.0,
        tool_check_breaker_threshold=5,
        tool_check_breaker_cooldown=7.0,
    )
    outage = governor._tools.outage  # noqa: SLF001
    assert outage.failure_mode == "fail_open_bounded"
    assert outage.max_stale_seconds == 120.0
    assert outage.breaker.threshold == 5
    assert outage.breaker.cooldown == 7.0


def test_a_rebound_identity_keeps_its_outage_state(
    identity: IdentityCredentials, credentials_dir: Any, keypair: Any
) -> None:
    governor = make_governor(identity, credentials_dir)
    before = governor._tools.outage  # noqa: SLF001
    rotated = dataclasses.replace(identity, private_key_pem=keypair[0])
    governor._bind_identity(rotated)  # noqa: SLF001
    assert governor._tools.outage is before  # noqa: SLF001


@respx.mock
def test_guard_fail_closed_by_default_never_runs_the_tool(
    identity: IdentityCredentials, credentials_dir: Any
) -> None:
    governor = make_governor(identity, credentials_dir)
    respx.post(CHECK).mock(side_effect=_down())
    ran: list[int] = []

    def search(q: str) -> str:
        ran.append(1)
        return q

    with governor.run("r"), pytest.raises(ToolCheckUnavailable):
        governor.guard(search, name="search")(q="x")
    assert ran == []


@respx.mock
def test_guard_fail_open_bounded_runs_the_tool_and_marks_the_span(
    identity: IdentityCredentials, credentials_dir: Any
) -> None:
    governor = make_governor(identity, credentials_dir, tool_check_failure_mode="fail_open_bounded")
    route = respx.post(CHECK)
    route.side_effect = [_allow(), _down()]

    with governor.run("r"):
        assert governor.guard(lambda: "warm", name="warm")() == "warm"
        assert governor.guard(lambda q: f"got {q}", name="search")(q="x") == "got x"

    warm, degraded = tool_spans(governor)
    assert "matimo.degraded_mode" not in (warm.get("attributes") or {})
    attrs = degraded["attributes"]
    assert attrs["matimo.degraded_mode"] is True
    assert attrs["matimo.degraded_cache_age_seconds"] >= 0
    assert degraded["status"] == "completed"


@respx.mock
def test_a_real_heartbeat_is_enough_contact_to_fail_open(
    identity: IdentityCredentials, credentials_dir: Any
) -> None:
    governor = make_governor(identity, credentials_dir, tool_check_failure_mode="fail_open_bounded")
    governor._telemetry.state.update_from_heartbeat(  # noqa: SLF001
        {"lifecycleStatus": "active", "emergencyStop": False}
    )
    respx.post(CHECK).mock(side_effect=_down())
    decision = governor.check_tool("search", {})
    assert decision.allowed and decision.degraded


@respx.mock
def test_a_suspended_heartbeat_never_fails_open(
    identity: IdentityCredentials, credentials_dir: Any
) -> None:
    governor = make_governor(identity, credentials_dir, tool_check_failure_mode="fail_open_bounded")
    governor._telemetry.state.update_from_heartbeat(  # noqa: SLF001
        {"lifecycleStatus": "suspended"}
    )
    respx.post(CHECK).mock(side_effect=_down())
    with pytest.raises(ToolCheckUnavailable):
        governor.check_tool("search", {})


@respx.mock
def test_guard_circuit_opens_and_fails_fast(
    identity: IdentityCredentials, credentials_dir: Any
) -> None:
    governor = make_governor(identity, credentials_dir, tool_check_breaker_threshold=2)
    route = respx.post(CHECK).mock(side_effect=_down())
    guarded = governor.guard(lambda: "x", name="t")
    with governor.run("r"):
        for _ in range(2):
            with pytest.raises(ToolCheckUnavailable):
                guarded()
        assert route.call_count == 2
        with pytest.raises(ToolCheckUnavailable) as info:
            guarded()
    assert info.value.circuit_open
    assert route.call_count == 2


@respx.mock
def test_guard_policy_denied_is_not_an_outage(
    identity: IdentityCredentials, credentials_dir: Any
) -> None:
    governor = make_governor(identity, credentials_dir, tool_check_failure_mode="fail_open_bounded")
    respx.post(CHECK).mock(
        return_value=httpx.Response(
            403, json={"error": "policy_denied", "message": "tool_category_not_allowed"}
        )
    )
    governor._tools.outage.record_contact()  # noqa: SLF001
    with governor.run("r"), pytest.raises(PolicyDenied):
        governor.guard(lambda: "x", name="t")()


@pytest.mark.asyncio
@respx.mock
async def test_async_guard_fail_closed_and_fail_open(
    identity: IdentityCredentials, credentials_dir: Any
) -> None:
    closed = make_async_governor(identity, credentials_dir)
    respx.post(CHECK).mock(side_effect=_down())

    async def fetch(q: str) -> str:
        return f"got {q}"

    async with closed.run("r"):
        with pytest.raises(ToolCheckUnavailable):
            await closed.guard(fetch, name="fetch")(q="x")

    bounded = make_async_governor(
        identity, credentials_dir, tool_check_failure_mode="fail_open_bounded"
    )
    route = respx.post(CHECK)
    route.side_effect = [_allow(), _down()]
    async with bounded.run("r"):
        await bounded.guard(fetch, name="warm")(q="w")
        assert await bounded.guard(fetch, name="fetch")(q="x") == "got x"
    _warm, degraded = tool_spans(bounded)
    assert degraded["attributes"]["matimo.degraded_mode"] is True
