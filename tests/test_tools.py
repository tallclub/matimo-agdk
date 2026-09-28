from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest
import respx

from matimo_agdk.config import GatewayConfig
from matimo_agdk.exceptions import ToolCheckTimeout
from matimo_agdk.governor import RUN_ID_HEADER, AsyncGovernor, Governor
from matimo_agdk.identity import IdentityCredentials
from matimo_agdk.tools import (
    NO_RESUME_TOKEN_DENY_REASON,
    AsyncToolGovernor,
    ToolGovernor,
    hash_args,
    redact_args,
)
from matimo_agdk.transport import AsyncGatewayHTTP, GatewayHTTP

from .conftest import BASE_URL


def _config(identity: IdentityCredentials) -> GatewayConfig:
    """A config sufficient to construct a bound Governor purely to control
    the ambient run ContextVar (`Governor.run()`/`bind_run_id()`) -- no
    network I/O happens during construction. Mirrors test_run_context.py's
    identically-named helper."""
    return GatewayConfig(
        base_url=BASE_URL,
        api_key="org-key",
        identity_token=identity.identity_token,
        identity_id=identity.identity_id,
        tenant_id=identity.tenant_id,
        private_key_pem=identity.private_key_pem,
        agent_name=identity.display_name,
    )


def make_governor(identity: IdentityCredentials) -> ToolGovernor:
    http = GatewayHTTP(BASE_URL, "org-key")
    return ToolGovernor(
        http,
        identity_token=identity.identity_token,
        identity_id=identity.identity_id,
        tenant_id=identity.tenant_id,
        external_framework=identity.external_framework,
        poll_interval=0.01,
        poll_max_interval=0.02,
        max_wait_seconds=1.0,
        recheck_delays=(0.0, 0.0, 0.0),
    )


def make_async_governor(identity: IdentityCredentials) -> AsyncToolGovernor:
    http = AsyncGatewayHTTP(BASE_URL, "org-key")
    return AsyncToolGovernor(
        http,
        identity_token=identity.identity_token,
        identity_id=identity.identity_id,
        tenant_id=identity.tenant_id,
        external_framework=identity.external_framework,
        poll_interval=0.01,
        poll_max_interval=0.02,
        max_wait_seconds=1.0,
        recheck_delays=(0.0, 0.0, 0.0),
    )


def test_hash_args_is_stable_regardless_of_key_order() -> None:
    assert hash_args({"a": 1, "b": 2}) == hash_args({"b": 2, "a": 1})


def test_hash_args_differs_for_different_args() -> None:
    assert hash_args({"a": 1}) != hash_args({"a": 2})


def test_redact_args_masks_secret_like_keys() -> None:
    out = redact_args({"password": "hunter2", "q": "search term"})
    assert out["password"] == "[REDACTED]"
    assert out["q"] == "search term"


@respx.mock
def test_check_allow(identity: IdentityCredentials) -> None:
    respx.post(f"{BASE_URL}/tools/check").mock(
        return_value=httpx.Response(200, json={"data": {"decision": "ALLOW"}})
    )
    gov = make_governor(identity)
    decision = gov.check("search", {"q": "x"})
    assert decision.allowed
    assert not decision.pending
    assert not decision.denied


@respx.mock
def test_check_deny_with_reason(identity: IdentityCredentials) -> None:
    respx.post(f"{BASE_URL}/tools/check").mock(
        return_value=httpx.Response(
            200, json={"data": {"decision": "DENY", "reason": "tool_category_not_allowed"}}
        )
    )
    gov = make_governor(identity)
    decision = gov.check("delete_database", {})
    assert decision.denied
    assert decision.reason == "tool_category_not_allowed"


@respx.mock
def test_check_pending_then_status_resolves_to_allow(identity: IdentityCredentials) -> None:
    respx.post(f"{BASE_URL}/tools/check").mock(
        return_value=httpx.Response(
            200, json={"data": {"decision": "PENDING", "resumeToken": "rt-1", "requestId": "req-1"}}
        )
    )
    status_route = respx.post(f"{BASE_URL}/tools/check/status")
    status_route.side_effect = [
        httpx.Response(200, json={"data": {"decision": "PENDING"}}),
        httpx.Response(200, json={"data": {"decision": "PENDING"}}),
        httpx.Response(200, json={"data": {"decision": "ALLOW"}}),
    ]
    gov = make_governor(identity)
    decision = gov.check("send_email", {"to": "x@example.com"})
    assert decision.pending
    assert decision.resume_token == "rt-1"

    final = gov.await_decision(decision.resume_token)
    assert final.allowed
    assert status_route.call_count == 3


@respx.mock
def test_await_decision_resolves_to_deny(identity: IdentityCredentials) -> None:
    respx.post(f"{BASE_URL}/tools/check/status").mock(
        return_value=httpx.Response(200, json={"data": {"decision": "DENY", "reason": "rejected"}})
    )
    gov = make_governor(identity)
    decision = gov.await_decision("rt-1")
    assert decision.denied
    assert decision.reason == "rejected"


@respx.mock
def test_await_decision_times_out(identity: IdentityCredentials) -> None:
    respx.post(f"{BASE_URL}/tools/check/status").mock(
        return_value=httpx.Response(200, json={"data": {"decision": "PENDING"}})
    )
    gov = make_governor(identity)
    with pytest.raises(ToolCheckTimeout):
        gov.await_decision("rt-1", poll_interval=0.01, max_wait_seconds=0.05)


@respx.mock
def test_report_result_swallows_errors(identity: IdentityCredentials) -> None:
    respx.post(f"{BASE_URL}/tools/result").mock(
        return_value=httpx.Response(500, json={"error": "boom"})
    )
    gov = make_governor(identity)
    # Should not raise.
    gov.report_result("rt-1", status="completed", duration_ms=10)


@respx.mock
def test_report_result_success(identity: IdentityCredentials) -> None:
    route = respx.post(f"{BASE_URL}/tools/result").mock(
        return_value=httpx.Response(202, json={"data": {"accepted": True}})
    )
    gov = make_governor(identity)
    gov.report_result("rt-1", status="completed", duration_ms=10)
    assert route.call_count == 1


@respx.mock
def test_check_signs_the_request(identity: IdentityCredentials, keypair: tuple[str, str]) -> None:
    from matimo_agdk.identity import JWSSigner

    private_pem, public_pem = keypair
    captured = {}

    def responder(request: httpx.Request) -> httpx.Response:
        captured["sig"] = request.headers.get("Matimo-Agent-Signature", "")
        captured["identity_header"] = request.headers.get("X-Matimo-Agent-Identity-Token", "")
        return httpx.Response(200, json={"data": {"decision": "ALLOW"}})

    respx.post(f"{BASE_URL}/tools/check").mock(side_effect=responder)

    http = GatewayHTTP(BASE_URL, "org-key")
    signer = JWSSigner(
        private_pem,
        identity_token=identity.identity_token,
        identity_id=identity.identity_id,
        tenant_id=identity.tenant_id,
    )
    http.signer = signer
    gov = ToolGovernor(
        http,
        identity_token=identity.identity_token,
        identity_id=identity.identity_id,
        tenant_id=identity.tenant_id,
    )
    gov.check("search", {"q": "x"})
    assert captured["sig"] != ""
    assert captured["identity_header"] == identity.identity_token


@respx.mock
def test_set_category_is_unsigned(identity: IdentityCredentials) -> None:
    captured = {}

    def responder(request: httpx.Request) -> httpx.Response:
        captured["sig"] = request.headers.get("Matimo-Agent-Signature")
        return httpx.Response(200, json={"data": {"category": "web"}})

    respx.put(f"{BASE_URL}/tools/search/category").mock(side_effect=responder)
    gov = make_governor(identity)
    gov.set_category("search", "web")
    assert captured["sig"] is None


# Gateway answers PENDING with no resumeToken when an identical check is
# already in flight (GatewayToolCheckService, "duplicate_check_in_flight").
# There is nothing to poll, and it must never be mistaken for "not denied".
_TOKENLESS_PENDING = {"data": {"decision": "PENDING", "reason": "duplicate_check_in_flight"}}


@respx.mock
def test_check_and_wait_rechecks_tokenless_pending_then_polls(
    identity: IdentityCredentials,
) -> None:
    check_route = respx.post(f"{BASE_URL}/tools/check")
    check_route.side_effect = [
        httpx.Response(200, json=_TOKENLESS_PENDING),
        httpx.Response(200, json={"data": {"decision": "PENDING", "resumeToken": "rt-1"}}),
    ]
    respx.post(f"{BASE_URL}/tools/check/status").mock(
        return_value=httpx.Response(200, json={"data": {"decision": "ALLOW"}})
    )
    decision = make_governor(identity).check_and_wait("send_email", {"to": "x@example.com"})
    assert decision.allowed
    assert check_route.call_count == 2


@respx.mock
def test_check_and_wait_denies_when_pending_never_gets_a_token(
    identity: IdentityCredentials,
) -> None:
    check_route = respx.post(f"{BASE_URL}/tools/check").mock(
        return_value=httpx.Response(200, json=_TOKENLESS_PENDING)
    )
    status_route = respx.post(f"{BASE_URL}/tools/check/status")
    decision = make_governor(identity).check_and_wait("send_email", {"to": "x@example.com"})
    assert decision.denied
    assert decision.reason == NO_RESUME_TOKEN_DENY_REASON
    assert check_route.call_count == 1 + 3  # first check + one per recheck delay
    assert not status_route.called


@respx.mock
def test_check_and_wait_passes_allow_and_deny_straight_through(
    identity: IdentityCredentials,
) -> None:
    check_route = respx.post(f"{BASE_URL}/tools/check")
    check_route.side_effect = [
        httpx.Response(200, json={"data": {"decision": "ALLOW"}}),
        httpx.Response(200, json={"data": {"decision": "DENY", "reason": "nope"}}),
    ]
    gov = make_governor(identity)
    assert gov.check_and_wait("a", {}).allowed
    denied = gov.check_and_wait("b", {})
    assert denied.denied
    assert denied.reason == "nope"
    assert check_route.call_count == 2  # no rechecks for a final answer


@respx.mock
async def test_async_check_and_wait_rechecks_tokenless_pending_then_polls(
    identity: IdentityCredentials,
) -> None:
    check_route = respx.post(f"{BASE_URL}/tools/check")
    check_route.side_effect = [
        httpx.Response(200, json=_TOKENLESS_PENDING),
        httpx.Response(200, json={"data": {"decision": "PENDING", "resumeToken": "rt-1"}}),
    ]
    respx.post(f"{BASE_URL}/tools/check/status").mock(
        return_value=httpx.Response(200, json={"data": {"decision": "ALLOW"}})
    )
    decision = await make_async_governor(identity).check_and_wait("send_email", {"to": "x"})
    assert decision.allowed
    assert check_route.call_count == 2


@respx.mock
async def test_async_check_and_wait_denies_when_pending_never_gets_a_token(
    identity: IdentityCredentials,
) -> None:
    check_route = respx.post(f"{BASE_URL}/tools/check").mock(
        return_value=httpx.Response(200, json=_TOKENLESS_PENDING)
    )
    decision = await make_async_governor(identity).check_and_wait("send_email", {"to": "x"})
    assert decision.denied
    assert decision.reason == NO_RESUME_TOKEN_DENY_REASON
    assert check_route.call_count == 1 + 3


# -- X-Matimo-Run-Id correlation (2026-09-22) -------------------------------
#
# Tool checks were never correlated to the run that triggered them (only the
# LLM-call path attached this header, via `_current_run`/`bind_run_id()`) --
# Gateway can now use it to place a tool-check decision exactly in the
# Observability Hub run waterfall instead of guessing from a time window.
# `ToolGovernor`/`AsyncToolGovernor._headers()` now attach it whenever
# `current_run_id()` is non-empty, and omit it entirely otherwise -- these
# tests exercise both sides for check()/status()/report_result(), plus the
# regression guard that set_category() (tenant-wide, unsigned) never gains it.


def _captured_run_id_header(
    store: dict[str, str | None],
) -> Callable[[httpx.Request], httpx.Response]:
    def responder(request: httpx.Request) -> httpx.Response:
        store["run_id"] = request.headers.get(RUN_ID_HEADER)
        store["header_present"] = str(RUN_ID_HEADER in request.headers)
        return httpx.Response(200, json={"data": {"decision": "ALLOW"}})

    return responder


@respx.mock
def test_check_sends_run_id_header_inside_an_active_run(identity: IdentityCredentials) -> None:
    captured: dict[str, str | None] = {}
    respx.post(f"{BASE_URL}/tools/check").mock(side_effect=_captured_run_id_header(captured))
    gov = make_governor(identity)
    control = Governor(_config(identity))
    with control.run("my-run") as run_id:
        gov.check("search", {"q": "x"})
    assert captured["run_id"] == run_id


@respx.mock
def test_check_omits_run_id_header_with_no_active_run(identity: IdentityCredentials) -> None:
    captured: dict[str, str | None] = {}
    respx.post(f"{BASE_URL}/tools/check").mock(side_effect=_captured_run_id_header(captured))
    gov = make_governor(identity)
    gov.check("search", {"q": "x"})
    assert captured["run_id"] is None
    # Not just falsy -- genuinely absent from the request, never sent as "".
    assert captured["header_present"] == "False"


@respx.mock
async def test_async_check_sends_run_id_header_inside_an_active_run(
    identity: IdentityCredentials,
) -> None:
    captured: dict[str, str | None] = {}
    respx.post(f"{BASE_URL}/tools/check").mock(side_effect=_captured_run_id_header(captured))
    gov = make_async_governor(identity)
    control = AsyncGovernor(_config(identity))
    async with control.run("my-async-run") as run_id:
        await gov.check("search", {"q": "x"})
    assert captured["run_id"] == run_id


@respx.mock
async def test_async_check_omits_run_id_header_with_no_active_run(
    identity: IdentityCredentials,
) -> None:
    captured: dict[str, str | None] = {}
    respx.post(f"{BASE_URL}/tools/check").mock(side_effect=_captured_run_id_header(captured))
    gov = make_async_governor(identity)
    await gov.check("search", {"q": "x"})
    assert captured["run_id"] is None


@respx.mock
def test_status_sends_run_id_header_inside_an_active_run(identity: IdentityCredentials) -> None:
    """status() is reached from await_decision()'s poll loop, which for a
    real caller (check_and_wait()) runs in the same call stack -- and thus
    the same ambient run -- as the original check(). Carrying the header
    here too keeps a polled decision correlated the same way its initial
    check was."""
    captured: dict[str, str | None] = {}

    def responder(request: httpx.Request) -> httpx.Response:
        captured["run_id"] = request.headers.get(RUN_ID_HEADER)
        return httpx.Response(200, json={"data": {"decision": "ALLOW"}})

    respx.post(f"{BASE_URL}/tools/check/status").mock(side_effect=responder)
    gov = make_governor(identity)
    control = Governor(_config(identity))
    with control.run("my-run") as run_id:
        gov.status("rt-1")
    assert captured["run_id"] == run_id


@respx.mock
def test_status_omits_run_id_header_with_no_active_run(identity: IdentityCredentials) -> None:
    captured: dict[str, str | None] = {}

    def responder(request: httpx.Request) -> httpx.Response:
        captured["run_id"] = request.headers.get(RUN_ID_HEADER)
        return httpx.Response(200, json={"data": {"decision": "ALLOW"}})

    respx.post(f"{BASE_URL}/tools/check/status").mock(side_effect=responder)
    gov = make_governor(identity)
    gov.status("rt-1")
    assert captured["run_id"] is None


@respx.mock
def test_report_result_sends_run_id_header_inside_an_active_run(
    identity: IdentityCredentials,
) -> None:
    captured: dict[str, str | None] = {}

    def responder(request: httpx.Request) -> httpx.Response:
        captured["run_id"] = request.headers.get(RUN_ID_HEADER)
        return httpx.Response(202, json={"data": {"accepted": True}})

    respx.post(f"{BASE_URL}/tools/result").mock(side_effect=responder)
    gov = make_governor(identity)
    control = Governor(_config(identity))
    with control.run("my-run") as run_id:
        gov.report_result("rt-1", status="completed", duration_ms=10)
    assert captured["run_id"] == run_id


@respx.mock
def test_report_result_omits_run_id_header_with_no_active_run(
    identity: IdentityCredentials,
) -> None:
    captured: dict[str, str | None] = {}

    def responder(request: httpx.Request) -> httpx.Response:
        captured["run_id"] = request.headers.get(RUN_ID_HEADER)
        return httpx.Response(202, json={"data": {"accepted": True}})

    respx.post(f"{BASE_URL}/tools/result").mock(side_effect=responder)
    gov = make_governor(identity)
    gov.report_result("rt-1", status="completed", duration_ms=10)
    assert captured["run_id"] is None


@respx.mock
def test_set_category_never_sends_run_id_header(identity: IdentityCredentials) -> None:
    """set_category() is a tenant-wide admin action (unsigned, no identity
    header either -- see test_set_category_is_unsigned) and doesn't call
    _headers() at all. Even inside an active run, it must never gain a
    run-id header it was never designed to carry."""
    captured: dict[str, str | None] = {}

    def responder(request: httpx.Request) -> httpx.Response:
        captured["run_id"] = request.headers.get(RUN_ID_HEADER)
        return httpx.Response(200, json={"data": {"category": "web"}})

    respx.put(f"{BASE_URL}/tools/search/category").mock(side_effect=responder)
    gov = make_governor(identity)
    control = Governor(_config(identity))
    with control.run("my-run"):
        gov.set_category("search", "web")
    assert captured["run_id"] is None
