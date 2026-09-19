from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest
import respx

from matimo_agdk.exceptions import AgentSuspendedLocally, ToolDenied
from matimo_agdk.identity import IdentityCredentials
from matimo_agdk.tools import NO_RESUME_TOKEN_DENY_REASON, ToolDecision

from ..conftest import BASE_URL, future_iso
from .conftest import bound_async_governor, bound_governor


def add(a: int, b: int) -> int:
    return a + b


async def async_add(a: int, b: int) -> int:
    return a + b


def _mock_sessions_and_telemetry() -> None:
    respx.post(f"{BASE_URL}/sessions").mock(
        return_value=httpx.Response(
            201,
            json={
                "data": {"sessionToken": "tok", "expiresAt": future_iso(3600), "identityId": "x"}
            },
        )
    )
    respx.post(f"{BASE_URL}/telemetry/batch").mock(
        return_value=httpx.Response(200, json={"data": {"accepted": 1, "failed": []}})
    )


def test_observe_mode_never_calls_check_tool_single_callable() -> None:
    from matimo_agdk.adapters.generic import govern

    gov = MagicMock()
    gov.check_tool = MagicMock()
    gov.check_and_wait = MagicMock()
    gov.tool_span = MagicMock()

    governed = govern(add, gov, mode="observe")
    assert governed(1, 2) == 3
    gov.check_tool.assert_not_called()
    gov.check_and_wait.assert_not_called()
    gov.tool_span.assert_called_once()


def test_observe_mode_dict_of_tools() -> None:
    from matimo_agdk.adapters.generic import govern

    gov = MagicMock()
    gov.check_tool = MagicMock()
    gov.check_and_wait = MagicMock()
    gov.tool_span = MagicMock()

    governed = govern({"add": add}, gov, mode="observe")
    assert governed["add"](a=1, b=2) == 3
    gov.check_tool.assert_not_called()
    gov.check_and_wait.assert_not_called()
    assert gov.tool_span.call_args.args[0] == "add"


@respx.mock
def test_govern_mode_allow_passes_through(identity: IdentityCredentials, credentials_dir) -> None:
    from matimo_agdk.adapters.generic import govern

    _mock_sessions_and_telemetry()
    respx.post(f"{BASE_URL}/tools/check").mock(
        return_value=httpx.Response(200, json={"data": {"decision": "ALLOW"}})
    )
    governor = bound_governor(identity, credentials_dir)
    governed = govern(add, governor, mode="govern")

    # governor.guard() (what mode="govern" delegates to for a sync
    # Governor + sync callable) needs an active governor.run() block or an
    # explicit run_id -- this is a real, existing core requirement, not
    # something generic.govern() papers over.
    with governor.run("test-run"):
        assert governed(a=3, b=4) == 7


@respx.mock
def test_govern_mode_deny_raises_tool_denied_not_a_crash(
    identity: IdentityCredentials, credentials_dir
) -> None:
    from matimo_agdk.adapters.generic import govern

    _mock_sessions_and_telemetry()
    respx.post(f"{BASE_URL}/tools/check").mock(
        return_value=httpx.Response(
            200, json={"data": {"decision": "DENY", "reason": "tool_category_not_allowed"}}
        )
    )
    governor = bound_governor(identity, credentials_dir)
    governed = govern(add, governor, mode="govern")

    with pytest.raises(ToolDenied) as excinfo:
        governed(a=1, b=1)
    assert excinfo.value.reason == "tool_category_not_allowed"


@respx.mock
def test_govern_mode_pending_then_approved_proceeds(
    identity: IdentityCredentials, credentials_dir
) -> None:
    from matimo_agdk.adapters.generic import govern

    _mock_sessions_and_telemetry()
    respx.post(f"{BASE_URL}/tools/check").mock(
        return_value=httpx.Response(
            200, json={"data": {"decision": "PENDING", "resumeToken": "rt-1"}}
        )
    )
    status_route = respx.post(f"{BASE_URL}/tools/check/status")
    status_route.side_effect = [
        httpx.Response(200, json={"data": {"decision": "PENDING"}}),
        httpx.Response(200, json={"data": {"decision": "ALLOW"}}),
    ]
    respx.post(f"{BASE_URL}/tools/result").mock(
        return_value=httpx.Response(202, json={"data": {"accepted": True}})
    )
    governor = bound_governor(identity, credentials_dir)
    governor._tools.poll_interval = 0.01  # noqa: SLF001
    governor._tools.poll_max_interval = 0.02  # noqa: SLF001
    governed = govern(add, governor, mode="govern")

    with governor.run("test-run"):
        assert governed(a=10, b=20) == 30


def _mock_tokenless_pending():
    return respx.post(f"{BASE_URL}/tools/check").mock(
        return_value=httpx.Response(
            200, json={"data": {"decision": "PENDING", "reason": "duplicate_check_in_flight"}}
        )
    )


@respx.mock
async def test_govern_async_callable_tokenless_pending_is_denied_with_async_governor(
    identity: IdentityCredentials, credentials_dir
) -> None:
    from matimo_agdk.adapters.generic import govern

    _mock_sessions_and_telemetry()
    _mock_tokenless_pending()
    governor = bound_async_governor(identity, credentials_dir)
    await governor.start()
    try:
        governor._tools.recheck_delays = (0.0,)  # noqa: SLF001
        ran: list[int] = []

        async def tool(a: int, b: int) -> int:
            ran.append(a)
            return a + b

        governed = govern(tool, governor, mode="govern")
        async with governor.run("test-run"):
            with pytest.raises(ToolDenied) as excinfo:
                await governed(a=1, b=1)
        assert excinfo.value.reason == NO_RESUME_TOKEN_DENY_REASON
        assert ran == []
    finally:
        await governor.stop()


@respx.mock
async def test_govern_async_callable_tokenless_pending_is_denied_with_sync_governor(
    identity: IdentityCredentials, credentials_dir
) -> None:
    """Covers adapters._shared.async_check_and_wait's sync-Governor branch
    (the path Google ADK's plugin and AutoGen's tools take)."""
    from matimo_agdk.adapters.generic import govern

    _mock_sessions_and_telemetry()
    _mock_tokenless_pending()
    governor = bound_governor(identity, credentials_dir)
    governor._tools.recheck_delays = (0.0,)  # noqa: SLF001
    ran: list[int] = []

    async def tool(a: int, b: int) -> int:
        ran.append(a)
        return a + b

    governed = govern(tool, governor, mode="govern")
    with governor.run("test-run"):
        with pytest.raises(ToolDenied) as excinfo:
            await governed(a=1, b=1)
    assert excinfo.value.reason == NO_RESUME_TOKEN_DENY_REASON
    assert ran == []


def test_suspended_state_stops_before_dispatch() -> None:
    from matimo_agdk.adapters.generic import govern

    def guard_impl(fn, name=None, category=None):
        def wrapper(*a, **kw):
            raise AgentSuspendedLocally("suspended", False)

        return wrapper

    gov = MagicMock()
    gov.check_tool = MagicMock(return_value=ToolDecision(decision="ALLOW"))
    gov.check_and_wait = MagicMock(return_value=ToolDecision(decision="ALLOW"))
    gov.guard.side_effect = guard_impl

    governed = govern(add, gov, mode="govern")
    with pytest.raises(AgentSuspendedLocally):
        governed(1, 1)


@pytest.mark.asyncio
async def test_govern_async_callable_with_sync_governor_is_bridged() -> None:
    from matimo_agdk.adapters.generic import govern

    gov = MagicMock()
    gov.check_and_wait = MagicMock(return_value=ToolDecision(decision="ALLOW"))
    gov.raise_if_suspended = MagicMock()
    gov.tool_span = MagicMock()

    governed = govern(async_add, gov, mode="govern")
    result = await governed(2, 3)
    assert result == 5
    gov.check_and_wait.assert_called_once()


@pytest.mark.asyncio
async def test_govern_async_callable_with_async_governor() -> None:
    from matimo_agdk.adapters.generic import govern

    gov = MagicMock()

    async def acheck(*a, **k):
        return ToolDecision(decision="ALLOW")

    def guard_sync_wrapper(fn, name=None, category=None):
        # AsyncGovernor.guard() is a plain (non-async) method that
        # *returns* an async wrapper -- emulate that shape here.
        async def wrapper(*a, **kw):
            return await fn(*a, **kw)

        return wrapper

    gov.check_tool = acheck
    gov.guard = MagicMock(side_effect=guard_sync_wrapper)

    governed = govern(async_add, gov, mode="govern")
    result = await governed(4, 5)
    assert result == 9
    gov.guard.assert_called_once()


def test_mismatched_sync_callable_with_async_governor_raises_type_error() -> None:
    from matimo_agdk.adapters.generic import govern

    gov = MagicMock()

    async def acheck(*a, **k):
        return ToolDecision(decision="ALLOW")

    gov.check_tool = acheck

    with pytest.raises(TypeError):
        govern(add, gov, mode="govern")


def test_govern_rejects_unknown_shape() -> None:
    from matimo_agdk.adapters.generic import govern

    gov = MagicMock()
    with pytest.raises(TypeError):
        govern(42, gov)  # type: ignore[arg-type]
