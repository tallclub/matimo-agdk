from __future__ import annotations

import httpx
import respx

from matimo_agdk.exceptions import SessionExpired
from matimo_agdk.identity import IdentityCredentials, JWSSigner
from matimo_agdk.session import SESSION_RENEWAL_FRACTION, SessionManager
from matimo_agdk.transport import GatewayHTTP

from .conftest import BASE_URL, future_iso


def make_manager(identity: IdentityCredentials) -> SessionManager:
    http = GatewayHTTP(BASE_URL, "org-key")
    signer = JWSSigner.from_credentials(identity)
    http.signer = signer
    return SessionManager(http, signer, identity)


@respx.mock
def test_get_token_handshakes_once_and_caches(identity: IdentityCredentials) -> None:
    route = respx.post(f"{BASE_URL}/sessions").mock(
        return_value=httpx.Response(
            201,
            json={
                "data": {"sessionToken": "tok-1", "expiresAt": future_iso(3600), "identityId": "x"}
            },
        )
    )
    mgr = make_manager(identity)
    token1 = mgr.get_token()
    token2 = mgr.get_token()
    assert token1 == "tok-1"
    assert token2 == "tok-1"
    assert route.call_count == 1


@respx.mock
def test_get_token_renews_past_renewal_fraction(identity: IdentityCredentials) -> None:
    route = respx.post(f"{BASE_URL}/sessions")
    route.side_effect = [
        httpx.Response(
            201,
            json={
                "data": {"sessionToken": "tok-1", "expiresAt": future_iso(1.0), "identityId": "x"}
            },
        ),
        httpx.Response(
            201,
            json={
                "data": {"sessionToken": "tok-2", "expiresAt": future_iso(3600), "identityId": "x"}
            },
        ),
    ]
    mgr = make_manager(identity)
    token1 = mgr.get_token()
    assert token1 == "tok-1"

    import time

    time.sleep(1.0 * SESSION_RENEWAL_FRACTION + 0.05)
    token2 = mgr.get_token()
    assert token2 == "tok-2"
    assert route.call_count == 2


@respx.mock
def test_invalidate_forces_rehandshake(identity: IdentityCredentials) -> None:
    route = respx.post(f"{BASE_URL}/sessions")
    route.side_effect = [
        httpx.Response(
            201,
            json={
                "data": {"sessionToken": "tok-1", "expiresAt": future_iso(3600), "identityId": "x"}
            },
        ),
        httpx.Response(
            201,
            json={
                "data": {"sessionToken": "tok-2", "expiresAt": future_iso(3600), "identityId": "x"}
            },
        ),
    ]
    mgr = make_manager(identity)
    assert mgr.get_token() == "tok-1"
    mgr.invalidate()
    assert mgr.get_token() == "tok-2"
    assert route.call_count == 2


@respx.mock
def test_call_with_retry_reacts_to_session_expired(identity: IdentityCredentials) -> None:
    respx.post(f"{BASE_URL}/sessions").mock(
        return_value=httpx.Response(
            201,
            json={
                "data": {"sessionToken": "tok-1", "expiresAt": future_iso(3600), "identityId": "x"}
            },
        )
    )
    mgr = make_manager(identity)

    calls = {"n": 0}

    def fn(token: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise SessionExpired("session expired")
        return token

    result = mgr.call_with_retry(fn)
    assert result == "tok-1"
    assert calls["n"] == 2


@respx.mock
def test_handshake_sends_exact_empty_body(identity: IdentityCredentials) -> None:
    captured = {}

    def responder(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        return httpx.Response(
            201,
            json={"data": {"sessionToken": "t", "expiresAt": future_iso(3600), "identityId": "x"}},
        )

    respx.post(f"{BASE_URL}/sessions").mock(side_effect=responder)
    mgr = make_manager(identity)
    mgr.get_token()
    assert captured["body"] == b"{}"
