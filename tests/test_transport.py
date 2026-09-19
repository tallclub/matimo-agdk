from __future__ import annotations

import httpx
import pytest
import respx

from matimo_agdk.exceptions import (
    AgentSuspended,
    GatewayError,
    GatewayUnavailable,
    PolicyDenied,
    RateLimited,
    SessionExpired,
    SignatureRejected,
    TelemetryStale,
)
from matimo_agdk.identity import JWSSigner
from matimo_agdk.transport import GatewayHTTP, RetryPolicy

BASE_URL = "http://testserver/v1"


def make_http(
    signer: JWSSigner | None = None, retry_policy: RetryPolicy | None = None
) -> GatewayHTTP:
    return GatewayHTTP(BASE_URL, "me-live-testkey", signer=signer, retry_policy=retry_policy)


@respx.mock
def test_successful_request_unwraps_data() -> None:
    respx.post(f"{BASE_URL}/identities").mock(
        return_value=httpx.Response(201, json={"data": {"id": "abc"}})
    )
    http = make_http()
    resp = http.request("POST", "/identities", json_body={"displayName": "x"})
    assert resp.data == {"id": "abc"}


@respx.mock
def test_bare_body_without_data_wrapper_passthrough() -> None:
    respx.get(f"{BASE_URL}/identities/x/jwks").mock(
        return_value=httpx.Response(200, json={"keys": [{"kty": "EC"}]})
    )
    http = make_http()
    resp = http.request("GET", "/identities/x/jwks")
    assert resp.data == {"keys": [{"kty": "EC"}]}


@respx.mock
def test_request_body_bytes_are_exact_and_signed(keypair: tuple[str, str]) -> None:
    private_pem, public_pem = keypair
    signer = JWSSigner(private_pem, identity_token="kid1", identity_id="a1", tenant_id="t1")
    captured: dict[str, bytes] = {}

    def responder(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        captured["sig"] = request.headers.get("Matimo-Agent-Signature", "")
        return httpx.Response(201, json={"data": {"sessionToken": "s", "expiresAt": "x"}})

    respx.post(f"{BASE_URL}/sessions").mock(side_effect=responder)
    http = make_http(signer=signer)
    http.request("POST", "/sessions", json_body={}, sign=True, identity_id="a1", tenant_id="t1")

    assert captured["body"] == b"{}"
    from matimo_agdk.identity import verify_jws

    claims = verify_jws(captured["sig"], public_pem.encode("utf-8"))
    from matimo_agdk.identity import sha256_hex

    assert claims["body_hash"] == sha256_hex(b"{}")


@respx.mock
def test_session_expired_maps_to_typed_exception() -> None:
    respx.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(401, json={"error": "session_expired"})
    )
    http = make_http()
    with pytest.raises(SessionExpired):
        http.request("POST", "/chat/completions", json_body={})


@respx.mock
def test_signature_required_maps_to_typed_exception() -> None:
    respx.post(f"{BASE_URL}/sessions").mock(
        return_value=httpx.Response(
            403,
            json={
                "error": "signature_required",
                "message": "Matimo-Agent-Signature is missing or invalid",
            },
        )
    )
    http = make_http()
    with pytest.raises(SignatureRejected):
        http.request("POST", "/sessions", json_body={})


@respx.mock
def test_policy_denied_carries_reason_in_message() -> None:
    respx.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(
            403, json={"error": "policy_denied", "message": "monthly_budget_exceeded"}
        )
    )
    http = make_http()
    with pytest.raises(PolicyDenied) as excinfo:
        http.request("POST", "/chat/completions", json_body={})
    assert excinfo.value.reason == "monthly_budget_exceeded"


@respx.mock
def test_telemetry_stale_is_a_policy_denied_subtype() -> None:
    respx.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(
            403, json={"error": "policy_denied", "message": "telemetry_stale"}
        )
    )
    http = make_http()
    with pytest.raises(TelemetryStale):
        http.request("POST", "/chat/completions", json_body={})


@respx.mock
def test_agent_suspended_reasons() -> None:
    for reason in ("agent_suspended", "agent_revoked", "emergency_stop_active"):
        respx.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(403, json={"error": "policy_denied", "message": reason})
        )
        http = make_http()
        with pytest.raises(AgentSuspended):
            http.request("POST", "/chat/completions", json_body={})


@respx.mock
def test_upstream_error_502_maps_to_gateway_unavailable() -> None:
    respx.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(
            502, json={"error": "upstream_error", "message": "All configured LLM targets failed"}
        )
    )
    http = make_http()
    with pytest.raises(GatewayUnavailable):
        http.request("POST", "/chat/completions", json_body={})


@respx.mock
def test_generic_4xx_maps_to_base_gateway_error() -> None:
    respx.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(400, json={"error": "invalid_request", "message": "bad body"})
    )
    http = make_http()
    with pytest.raises(GatewayError) as excinfo:
        http.request("POST", "/chat/completions", json_body={})
    assert not isinstance(
        excinfo.value, (PolicyDenied, SessionExpired, SignatureRejected, RateLimited)
    )


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------


@respx.mock
def test_429_is_retried_and_eventually_succeeds(monkeypatch) -> None:
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    route = respx.post(f"{BASE_URL}/tools/check")
    route.side_effect = [
        httpx.Response(429, json={"error": "rate_limit_exceeded"}),
        httpx.Response(429, json={"error": "rate_limit_exceeded"}),
        httpx.Response(200, json={"data": {"decision": "ALLOW"}}),
    ]
    http = make_http(retry_policy=RetryPolicy(max_retries=3, base_delay=0.001, max_delay=0.01))
    resp = http.request("POST", "/tools/check", json_body={"toolName": "x", "argHash": "y"})
    assert resp.data == {"decision": "ALLOW"}
    assert route.call_count == 3


@respx.mock
def test_429_exhausting_retries_raises_rate_limited(monkeypatch) -> None:
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    respx.post(f"{BASE_URL}/tools/check").mock(
        return_value=httpx.Response(429, json={"error": "rate_limit_exceeded"})
    )
    http = make_http(retry_policy=RetryPolicy(max_retries=2, base_delay=0.001, max_delay=0.01))
    with pytest.raises(RateLimited):
        http.request("POST", "/tools/check", json_body={"toolName": "x", "argHash": "y"})


@respx.mock
def test_429_honors_retry_after_header(monkeypatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr("time.sleep", lambda s: slept.append(s))
    route = respx.post(f"{BASE_URL}/tools/check")
    route.side_effect = [
        httpx.Response(429, headers={"Retry-After": "2"}, json={"error": "rate_limit_exceeded"}),
        httpx.Response(200, json={"data": {"decision": "ALLOW"}}),
    ]
    http = make_http(retry_policy=RetryPolicy(max_retries=3, base_delay=0.001, max_delay=0.01))
    http.request("POST", "/tools/check", json_body={"toolName": "x", "argHash": "y"})
    assert slept[0] == 2.0


@respx.mock
def test_5xx_retried_only_when_idempotent(monkeypatch) -> None:
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    respx.post(f"{BASE_URL}/telemetry/batch").mock(
        return_value=httpx.Response(500, json={"error": "internal"})
    )
    http = make_http(retry_policy=RetryPolicy(max_retries=2, base_delay=0.001, max_delay=0.01))

    # Not idempotent: no retry, raises immediately after the first 500.
    route = respx.routes[0]
    with pytest.raises(GatewayError):
        http.request("POST", "/telemetry/batch", json_body={"events": []}, idempotent=False)
    assert route.call_count == 1


@respx.mock
def test_5xx_retried_when_idempotent(monkeypatch) -> None:
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    route = respx.post(f"{BASE_URL}/telemetry/batch")
    route.side_effect = [
        httpx.Response(500, json={"error": "internal"}),
        httpx.Response(200, json={"data": {"accepted": 0, "failed": []}}),
    ]
    http = make_http(retry_policy=RetryPolicy(max_retries=2, base_delay=0.001, max_delay=0.01))
    resp = http.request("POST", "/telemetry/batch", json_body={"events": []}, idempotent=True)
    assert resp.data == {"accepted": 0, "failed": []}
    assert route.call_count == 2


@respx.mock
def test_403_policy_denied_is_never_retried(monkeypatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr("time.sleep", lambda s: slept.append(s))
    route = respx.post(f"{BASE_URL}/chat/completions")
    route.mock(
        return_value=httpx.Response(
            403, json={"error": "policy_denied", "message": "agent_suspended"}
        )
    )
    http = make_http(retry_policy=RetryPolicy(max_retries=3, base_delay=0.01, max_delay=0.1))
    with pytest.raises(AgentSuspended):
        http.request("POST", "/chat/completions", json_body={}, idempotent=True)
    assert route.call_count == 1
    assert slept == []


def test_connection_error_retried_only_when_idempotent(monkeypatch) -> None:
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    calls = {"n": 0}

    class FlakyTransport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] < 2:
                raise httpx.ConnectError("boom")
            return httpx.Response(200, json={"data": {"ok": True}})

    client = httpx.Client(base_url=BASE_URL, transport=FlakyTransport())
    http = GatewayHTTP(
        BASE_URL,
        "key",
        client=client,
        retry_policy=RetryPolicy(max_retries=3, base_delay=0.001, max_delay=0.01),
    )
    resp = http.request("GET", "/identities/x/jwks", idempotent=True)
    assert resp.data == {"ok": True}
    assert calls["n"] == 2


def test_connection_error_not_retried_when_not_idempotent() -> None:
    class AlwaysFailTransport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("boom")

    client = httpx.Client(base_url=BASE_URL, transport=AlwaysFailTransport())
    http = GatewayHTTP(BASE_URL, "key", client=client)
    with pytest.raises(GatewayUnavailable):
        http.request("POST", "/identities", json_body={}, idempotent=False)
