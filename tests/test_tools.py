from __future__ import annotations

import httpx
import pytest
import respx

from matimo_agdk.exceptions import ToolCheckTimeout
from matimo_agdk.identity import IdentityCredentials
from matimo_agdk.tools import ToolGovernor, hash_args, redact_args
from matimo_agdk.transport import GatewayHTTP

from .conftest import BASE_URL


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
