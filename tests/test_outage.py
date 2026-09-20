"""Tool-check outage behaviour (matimo_agdk._outage): the circuit breaker, the
`fail_closed` / `fail_open_bounded` modes, and every case that must never fail
open. Sync and async tool governors share one implementation, so each scenario
that matters is exercised through both."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx
from pydantic import ValidationError

from matimo_agdk._outage import CircuitBreaker, OutageGuard, degraded_attributes
from matimo_agdk.config import GatewayConfig
from matimo_agdk.exceptions import (
    AgentSuspended,
    GatewayError,
    GatewayUnavailable,
    PolicyDenied,
    RateLimited,
    SignatureRejected,
    ToolCheckUnavailable,
)
from matimo_agdk.identity import IdentityCredentials
from matimo_agdk.telemetry import GovernanceState
from matimo_agdk.tools import (
    UNRECOGNIZED_DECISION_DENY_REASON,
    AsyncToolGovernor,
    ToolGovernor,
)
from matimo_agdk.transport import AsyncGatewayHTTP, GatewayHTTP, RetryPolicy

from .conftest import BASE_URL

CHECK = f"{BASE_URL}/tools/check"
STATUS = f"{BASE_URL}/tools/check/status"

ALLOW = httpx.Response(200, json={"data": {"decision": "ALLOW"}})
DENY = httpx.Response(200, json={"data": {"decision": "DENY", "reason": "nope"}})
PENDING_NO_TOKEN = httpx.Response(200, json={"data": {"decision": "PENDING"}})
PENDING = httpx.Response(200, json={"data": {"decision": "PENDING", "resumeToken": "rt-1"}})


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _guard(
    clock: FakeClock,
    *,
    mode: str = "fail_closed",
    threshold: int = 3,
    cooldown: float = 30.0,
    max_stale: float = 300.0,
    state: GovernanceState | None = None,
) -> OutageGuard:
    return OutageGuard(
        failure_mode=mode,
        max_stale_seconds=max_stale,
        breaker_threshold=threshold,
        breaker_cooldown=cooldown,
        state_provider=(lambda: state) if state is not None else None,
        clock=clock,
    )


def sync_tools(identity: IdentityCredentials, guard: OutageGuard) -> ToolGovernor:
    return ToolGovernor(
        GatewayHTTP(BASE_URL, "org-key", retry_policy=RetryPolicy(max_retries=0)),
        identity_token=identity.identity_token,
        identity_id=identity.identity_id,
        tenant_id=identity.tenant_id,
        external_framework=identity.external_framework,
        poll_interval=0.001,
        poll_max_interval=0.002,
        max_wait_seconds=1.0,
        recheck_delays=(0.0,),
        outage=guard,
    )


def async_tools(identity: IdentityCredentials, guard: OutageGuard) -> AsyncToolGovernor:
    return AsyncToolGovernor(
        AsyncGatewayHTTP(BASE_URL, "org-key", retry_policy=RetryPolicy(max_retries=0)),
        identity_token=identity.identity_token,
        identity_id=identity.identity_id,
        tenant_id=identity.tenant_id,
        external_framework=identity.external_framework,
        poll_interval=0.001,
        poll_max_interval=0.002,
        max_wait_seconds=1.0,
        recheck_delays=(0.0,),
        outage=guard,
    )


def _down() -> Any:
    return httpx.ConnectError("gateway down")


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------


def test_breaker_opens_after_threshold_and_half_opens_after_cooldown() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(threshold=3, cooldown=30.0, clock=clock)
    assert breaker.state == "closed"
    for _ in range(2):
        assert breaker.acquire()
        breaker.record_failure()
    assert breaker.state == "closed"
    assert breaker.acquire()
    breaker.record_failure()
    assert breaker.state == "open"

    assert not breaker.acquire()  # open: fail fast
    assert breaker.retry_in() == pytest.approx(30.0)
    clock.advance(29.9)
    assert not breaker.acquire()

    clock.advance(0.2)
    assert breaker.acquire()  # cooldown over: one probe
    assert breaker.state == "half_open"
    assert not breaker.acquire()  # ...and only one
    assert not breaker.acquire()

    breaker.record_success()
    assert breaker.state == "closed"
    assert breaker.acquire()


def test_breaker_failed_probe_reopens_for_a_full_cooldown() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(threshold=1, cooldown=10.0, clock=clock)
    breaker.acquire()
    breaker.record_failure()
    clock.advance(10)
    assert breaker.acquire()  # the probe
    breaker.record_failure()  # ...fails
    assert breaker.state == "open"
    assert not breaker.acquire()
    clock.advance(9.9)
    assert not breaker.acquire()
    clock.advance(0.2)
    assert breaker.acquire()


def test_breaker_success_resets_the_consecutive_count() -> None:
    breaker = CircuitBreaker(threshold=3, cooldown=30.0, clock=FakeClock())
    for _ in range(2):
        breaker.record_failure()
    breaker.record_success()
    for _ in range(2):
        breaker.record_failure()
    assert breaker.state == "closed"  # 2 + 2 with a success between is not 3 in a row
    breaker.record_failure()
    assert breaker.state == "open"


def test_breaker_released_probe_lets_the_next_caller_probe() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(threshold=1, cooldown=5.0, clock=clock)
    breaker.record_failure()
    clock.advance(5)
    assert breaker.acquire()
    breaker.release_probe()  # e.g. the probing task was cancelled
    assert breaker.acquire()


# ---------------------------------------------------------------------------
# fail_closed (the default): raises, and the circuit fails fast
# ---------------------------------------------------------------------------


@respx.mock
def test_fail_closed_raises_and_circuit_opens_then_recovers(identity: IdentityCredentials) -> None:
    clock = FakeClock()
    tools = sync_tools(identity, _guard(clock, threshold=3, cooldown=30.0))
    route = respx.post(CHECK)
    route.side_effect = [_down(), _down(), _down()]

    for _ in range(3):
        with pytest.raises(ToolCheckUnavailable) as info:
            tools.check("search", {})
        assert not info.value.circuit_open
        assert isinstance(info.value, GatewayUnavailable)  # existing handlers still match
    assert route.call_count == 3

    # Open: no request goes out at all.
    with pytest.raises(ToolCheckUnavailable) as info:
        tools.check("search", {})
    assert info.value.circuit_open
    assert route.call_count == 3
    with pytest.raises(ToolCheckUnavailable):
        tools.check_and_wait("search", {})
    assert route.call_count == 3

    # After the cooldown one probe goes out; a real answer closes the circuit.
    clock.advance(31)
    route.side_effect = None
    route.mock(return_value=ALLOW)
    assert tools.check("search", {}).allowed
    assert route.call_count == 4
    assert tools.outage.breaker.state == "closed"
    assert tools.check("search", {}).allowed
    assert route.call_count == 5


@respx.mock
def test_fail_closed_failed_probe_reopens(identity: IdentityCredentials) -> None:
    clock = FakeClock()
    tools = sync_tools(identity, _guard(clock, threshold=1, cooldown=30.0))
    route = respx.post(CHECK)
    route.side_effect = _down()

    with pytest.raises(ToolCheckUnavailable):
        tools.check("t", {})
    clock.advance(31)
    with pytest.raises(ToolCheckUnavailable) as probe:
        tools.check("t", {})  # the probe, really sent
    assert not probe.value.circuit_open
    assert route.call_count == 2
    with pytest.raises(ToolCheckUnavailable) as after:
        tools.check("t", {})
    assert after.value.circuit_open
    assert route.call_count == 2


@respx.mock
def test_fail_closed_never_fails_open_even_with_fresh_contact(
    identity: IdentityCredentials,
) -> None:
    clock = FakeClock()
    tools = sync_tools(identity, _guard(clock))
    route = respx.post(CHECK)
    route.side_effect = [ALLOW, _down()]
    assert tools.check("warm", {}).allowed
    with pytest.raises(ToolCheckUnavailable):
        tools.check("search", {})


@pytest.mark.parametrize("status", [500, 502, 503, 504])
@respx.mock
def test_5xx_counts_as_a_transport_failure(identity: IdentityCredentials, status: int) -> None:
    tools = sync_tools(identity, _guard(FakeClock(), threshold=2))
    route = respx.post(CHECK).mock(return_value=httpx.Response(status, json={"error": "boom"}))
    for _ in range(2):
        with pytest.raises(ToolCheckUnavailable) as info:
            tools.check("t", {})
        assert info.value.status_code == status
    assert tools.outage.breaker.state == "open"
    assert route.call_count == 2


@respx.mock
def test_timeout_counts_as_a_transport_failure(identity: IdentityCredentials) -> None:
    tools = sync_tools(identity, _guard(FakeClock(), threshold=1))
    respx.post(CHECK).mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(ToolCheckUnavailable):
        tools.check("t", {})
    assert tools.outage.breaker.state == "open"


# ---------------------------------------------------------------------------
# Never fail open: 4xx, DENY, unrecognized, PENDING
# ---------------------------------------------------------------------------

_FOUR_XX: list[tuple[int, dict[str, str], type[GatewayError]]] = [
    (403, {"error": "policy_denied", "message": "tool_category_not_allowed"}, PolicyDenied),
    (403, {"error": "policy_denied", "message": "agent_suspended"}, AgentSuspended),
    (403, {"error": "signature_required"}, SignatureRejected),
    (429, {"error": "rate_limit_exceeded"}, RateLimited),
    (401, {"error": "unauthorized"}, GatewayError),
    (400, {"error": "invalid_request"}, GatewayError),
    (404, {"error": "not_found"}, GatewayError),
]


@pytest.mark.parametrize(("status", "body", "expected"), _FOUR_XX)
@respx.mock
def test_4xx_is_never_softened_and_never_trips_the_breaker(
    identity: IdentityCredentials, status: int, body: dict[str, str], expected: type[GatewayError]
) -> None:
    clock = FakeClock()
    tools = sync_tools(identity, _guard(clock, mode="fail_open_bounded", threshold=2))
    respx.post(CHECK).mock(return_value=httpx.Response(status, json=body))
    tools.outage.record_contact()  # fresh contact: the strongest case for failing open
    for _ in range(5):
        with pytest.raises(expected) as info:
            tools.check("search", {})
        assert not isinstance(info.value, ToolCheckUnavailable)
    assert tools.outage.breaker.state == "closed"


@respx.mock
def test_explicit_deny_is_returned_as_deny_in_fail_open_mode(
    identity: IdentityCredentials,
) -> None:
    tools = sync_tools(identity, _guard(FakeClock(), mode="fail_open_bounded"))
    respx.post(CHECK).mock(return_value=DENY)
    decision = tools.check("search", {})
    assert decision.denied
    assert not decision.degraded


@respx.mock
def test_unrecognized_decision_is_a_deny_and_not_fresh_contact(
    identity: IdentityCredentials,
) -> None:
    tools = sync_tools(identity, _guard(FakeClock(), mode="fail_open_bounded"))
    route = respx.post(CHECK)
    route.side_effect = [httpx.Response(200, json={"data": {"decision": "MAYBE"}}), _down()]
    decision = tools.check("search", {})
    assert decision.denied
    assert decision.reason == UNRECOGNIZED_DECISION_DENY_REASON
    # A garbled reply is not a successful check, so it is no basis for failing open.
    with pytest.raises(ToolCheckUnavailable):
        tools.check("search", {})


@respx.mock
def test_pending_recheck_outage_never_fails_open(identity: IdentityCredentials) -> None:
    """PENDING with no resume token is re-checked; if Gateway dies in between,
    the call must not become an ALLOW."""
    tools = sync_tools(identity, _guard(FakeClock(), mode="fail_open_bounded"))
    route = respx.post(CHECK)
    route.side_effect = [PENDING_NO_TOKEN, _down()]
    with pytest.raises(ToolCheckUnavailable):
        tools.check_and_wait("send_email", {})


@respx.mock
def test_status_polling_outage_raises_and_never_fails_open(identity: IdentityCredentials) -> None:
    tools = sync_tools(identity, _guard(FakeClock(), mode="fail_open_bounded"))
    respx.post(CHECK).mock(return_value=PENDING)
    respx.post(STATUS).mock(return_value=httpx.Response(503, json={"error": "down"}))
    with pytest.raises(GatewayError) as info:
        tools.check_and_wait("send_email", {})
    assert info.value.status_code == 503


# ---------------------------------------------------------------------------
# fail_open_bounded
# ---------------------------------------------------------------------------


@respx.mock
def test_fail_open_allows_when_contact_is_fresh(identity: IdentityCredentials) -> None:
    clock = FakeClock()
    tools = sync_tools(identity, _guard(clock, mode="fail_open_bounded"))
    route = respx.post(CHECK)
    route.side_effect = [ALLOW, _down()]
    assert tools.check("warm", {}).allowed
    clock.advance(42)

    decision = tools.check("search", {})
    assert decision.allowed
    assert decision.degraded
    assert decision.degraded_age_seconds == pytest.approx(42.0)
    assert decision.resume_token is None
    assert degraded_attributes(decision) == {
        "matimo.degraded_mode": True,
        "matimo.degraded_cache_age_seconds": 42.0,
    }
    # check_and_wait takes the same path and does not poll or re-check.
    route.side_effect = [_down()]
    assert tools.check_and_wait("search", {}).degraded


@respx.mock
def test_fail_open_is_bounded_by_the_staleness_limit(identity: IdentityCredentials) -> None:
    clock = FakeClock()
    tools = sync_tools(identity, _guard(clock, mode="fail_open_bounded", max_stale=300.0))
    route = respx.post(CHECK)
    route.side_effect = [ALLOW] + [_down()] * 3
    tools.check("warm", {})

    clock.advance(299)
    assert tools.check("search", {}).degraded
    clock.advance(2)  # 301s since the last successful check
    with pytest.raises(ToolCheckUnavailable):
        tools.check("search", {})


@respx.mock
def test_fail_open_with_no_contact_at_all_fails_closed(identity: IdentityCredentials) -> None:
    tools = sync_tools(identity, _guard(FakeClock(), mode="fail_open_bounded"))
    respx.post(CHECK).mock(side_effect=_down())
    with pytest.raises(ToolCheckUnavailable):
        tools.check("search", {})


@respx.mock
def test_a_fresh_heartbeat_counts_as_contact(identity: IdentityCredentials) -> None:
    clock = FakeClock()
    state = GovernanceState(lifecycle_status="active", last_heartbeat_monotonic=clock() - 10)
    tools = sync_tools(identity, _guard(clock, mode="fail_open_bounded", state=state))
    respx.post(CHECK).mock(side_effect=_down())

    decision = tools.check("search", {})  # no successful check yet, only a heartbeat
    assert decision.degraded
    assert decision.degraded_age_seconds == pytest.approx(10.0)

    state.last_heartbeat_monotonic = clock() - 400  # too old
    with pytest.raises(ToolCheckUnavailable):
        tools.check("search", {})


@respx.mock
def test_a_state_that_never_had_a_heartbeat_is_not_fresh(identity: IdentityCredentials) -> None:
    """GovernanceState.last_polled_monotonic starts at "now"; the fail-open rule
    must read last_heartbeat_monotonic, which is None until a heartbeat lands."""
    tools = sync_tools(
        identity, _guard(FakeClock(), mode="fail_open_bounded", state=GovernanceState())
    )
    respx.post(CHECK).mock(side_effect=_down())
    with pytest.raises(ToolCheckUnavailable):
        tools.check("search", {})


@pytest.mark.parametrize(
    "state",
    [
        GovernanceState(lifecycle_status="suspended"),
        GovernanceState(lifecycle_status="revoked"),
        GovernanceState(lifecycle_status="active", emergency_stop=True),
    ],
)
@respx.mock
def test_locally_known_suspension_never_fails_open(
    identity: IdentityCredentials, state: GovernanceState
) -> None:
    clock = FakeClock()
    state.last_heartbeat_monotonic = clock()  # perfectly fresh
    tools = sync_tools(identity, _guard(clock, mode="fail_open_bounded", state=state))
    respx.post(CHECK).mock(side_effect=_down())
    with pytest.raises(ToolCheckUnavailable):
        tools.check("search", {})


@pytest.mark.parametrize("last", [DENY, PENDING, PENDING_NO_TOKEN])
@respx.mock
def test_a_tool_last_denied_or_pending_is_not_waved_through(
    identity: IdentityCredentials, last: httpx.Response
) -> None:
    """The SDK cannot see the policy, so it will not fail open on a tool whose
    latest decision was DENY or needed approval."""
    tools = sync_tools(identity, _guard(FakeClock(), mode="fail_open_bounded"))
    route = respx.post(CHECK)
    route.side_effect = [last, _down(), ALLOW, _down(), _down()]

    assert not tools.check("risky", {}).allowed
    with pytest.raises(ToolCheckUnavailable):
        tools.check("risky", {})  # outage: still not allowed
    assert tools.check("harmless", {}).allowed
    assert tools.check("harmless", {}).degraded is True  # another tool is unaffected


@respx.mock
def test_a_later_allow_clears_an_earlier_deny_for_the_same_tool(
    identity: IdentityCredentials,
) -> None:
    tools = sync_tools(identity, _guard(FakeClock(), mode="fail_open_bounded"))
    route = respx.post(CHECK)
    route.side_effect = [DENY, ALLOW, _down()]
    tools.check("t", {})
    tools.check("t", {})
    assert tools.check("t", {}).degraded


@respx.mock
def test_open_circuit_in_fail_open_mode_allows_without_a_request(
    identity: IdentityCredentials,
) -> None:
    clock = FakeClock()
    tools = sync_tools(
        identity, _guard(clock, mode="fail_open_bounded", threshold=1, cooldown=30.0)
    )
    route = respx.post(CHECK)
    route.side_effect = [ALLOW, _down()]
    tools.check("warm", {})
    assert tools.check("search", {}).degraded  # the failure that opens the circuit
    assert tools.outage.breaker.state == "open"
    calls = route.call_count
    assert tools.check("search", {}).degraded  # circuit open: no request, still bounded
    assert route.call_count == calls
    clock.advance(301)
    route.side_effect = _down()  # the cooldown is over too, so this call is the probe
    with pytest.raises(ToolCheckUnavailable):
        tools.check("search", {})


# ---------------------------------------------------------------------------
# The async twin
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_async_circuit_opens_fails_fast_and_recovers(identity: IdentityCredentials) -> None:
    clock = FakeClock()
    tools = async_tools(identity, _guard(clock, threshold=2, cooldown=30.0))
    route = respx.post(CHECK)
    route.side_effect = [_down(), _down()]
    for _ in range(2):
        with pytest.raises(ToolCheckUnavailable):
            await tools.check("t", {})
    with pytest.raises(ToolCheckUnavailable) as info:
        await tools.check("t", {})
    assert info.value.circuit_open
    assert route.call_count == 2

    clock.advance(31)
    route.side_effect = None
    route.mock(return_value=ALLOW)
    assert (await tools.check("t", {})).allowed
    assert tools.outage.breaker.state == "closed"


@pytest.mark.asyncio
@respx.mock
async def test_async_fail_open_bounded(identity: IdentityCredentials) -> None:
    clock = FakeClock()
    tools = async_tools(identity, _guard(clock, mode="fail_open_bounded"))
    route = respx.post(CHECK)
    route.side_effect = [ALLOW, _down(), _down()]
    await tools.check("warm", {})
    clock.advance(10)
    decision = await tools.check_and_wait("search", {})
    assert decision.degraded
    assert decision.degraded_age_seconds == pytest.approx(10.0)
    clock.advance(400)
    with pytest.raises(ToolCheckUnavailable):
        await tools.check_and_wait("search", {})


@pytest.mark.asyncio
@respx.mock
async def test_async_never_fails_open_on_4xx_deny_pending_or_suspension(
    identity: IdentityCredentials,
) -> None:
    clock = FakeClock()
    state = GovernanceState(lifecycle_status="active", last_heartbeat_monotonic=clock())
    tools = async_tools(identity, _guard(clock, mode="fail_open_bounded", state=state))
    route = respx.post(CHECK)

    route.mock(return_value=httpx.Response(403, json={"error": "policy_denied", "message": "x"}))
    with pytest.raises(PolicyDenied):
        await tools.check("search", {})

    route.mock(return_value=DENY)
    assert (await tools.check("risky", {})).denied
    route.mock(side_effect=_down())
    with pytest.raises(ToolCheckUnavailable):
        await tools.check("risky", {})  # last decision DENY

    state.lifecycle_status = "suspended"
    with pytest.raises(ToolCheckUnavailable):
        await tools.check("search", {})  # suspended

    state.lifecycle_status = "active"
    route.side_effect = [PENDING_NO_TOKEN, _down()]
    with pytest.raises(ToolCheckUnavailable):
        await tools.check_and_wait("other", {})  # PENDING re-check outage


@pytest.mark.asyncio
@respx.mock
async def test_async_status_polling_outage_raises(identity: IdentityCredentials) -> None:
    tools = async_tools(identity, _guard(FakeClock(), mode="fail_open_bounded"))
    respx.post(CHECK).mock(return_value=PENDING)
    respx.post(STATUS).mock(return_value=httpx.Response(500, json={"error": "down"}))
    with pytest.raises(GatewayError):
        await tools.check_and_wait("send_email", {})


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_defaults_are_fail_closed_with_a_300s_cap() -> None:
    config = GatewayConfig()
    assert config.tool_check_failure_mode == "fail_closed"
    assert config.fail_open_max_stale_seconds == 300.0
    assert config.tool_check_breaker_threshold == 3
    assert config.tool_check_breaker_cooldown == 30.0
    assert config.capture_tool_results is False


def test_config_accepts_a_staleness_up_to_the_cap() -> None:
    assert GatewayConfig(fail_open_max_stale_seconds=300).fail_open_max_stale_seconds == 300
    assert GatewayConfig(fail_open_max_stale_seconds=1).fail_open_max_stale_seconds == 1


@pytest.mark.parametrize("value", [300.01, 301, 3600])
def test_config_rejects_a_staleness_over_the_cap(value: float) -> None:
    with pytest.raises(ValidationError, match="hard-capped at 300"):
        GatewayConfig(fail_open_max_stale_seconds=value)


@pytest.mark.parametrize("value", [0, -5])
def test_config_rejects_a_non_positive_staleness(value: float) -> None:
    with pytest.raises(ValidationError):
        GatewayConfig(fail_open_max_stale_seconds=value)


def test_config_rejects_an_unknown_failure_mode() -> None:
    with pytest.raises(ValidationError):
        GatewayConfig(tool_check_failure_mode="fail_open")  # type: ignore[arg-type]


def test_config_load_reads_the_env_vars(
    monkeypatch: pytest.MonkeyPatch, credentials_dir: Any
) -> None:
    monkeypatch.setenv("MATIMO_TOOL_CHECK_FAILURE_MODE", "fail_open_bounded")
    monkeypatch.setenv("MATIMO_FAIL_OPEN_MAX_STALE_SECONDS", "120")
    config = GatewayConfig.load(credentials_dir=credentials_dir)
    assert config.tool_check_failure_mode == "fail_open_bounded"
    assert config.fail_open_max_stale_seconds == 120.0

    monkeypatch.setenv("MATIMO_FAIL_OPEN_MAX_STALE_SECONDS", "900")
    with pytest.raises(ValidationError, match="hard-capped"):
        GatewayConfig.load(credentials_dir=credentials_dir)


def test_degraded_attributes_ignores_non_degraded_and_test_doubles() -> None:
    from unittest.mock import MagicMock

    from matimo_agdk.tools import ToolDecision

    assert degraded_attributes(ToolDecision(decision="ALLOW")) is None
    assert degraded_attributes(MagicMock()) is None  # a MagicMock's `.degraded` is not True
    assert degraded_attributes(ToolDecision(decision="ALLOW", degraded=True)) == {
        "matimo.degraded_mode": True
    }
