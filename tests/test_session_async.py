"""Async twin of test_session.py (review finding 2026-09-18)."""

from __future__ import annotations

import httpx
import respx

from matimo_agdk.exceptions import SessionExpired
from matimo_agdk.identity import IdentityCredentials, JWSSigner
from matimo_agdk.session import AsyncSessionManager
from matimo_agdk.transport import AsyncGatewayHTTP

from .conftest import BASE_URL, future_iso


def make_manager(identity: IdentityCredentials) -> AsyncSessionManager:
    http = AsyncGatewayHTTP(BASE_URL, "org-key")
    signer = JWSSigner.from_credentials(identity)
    http.signer = signer
    return AsyncSessionManager(http, signer, identity)


def _session_response(token: str) -> httpx.Response:
    return httpx.Response(
        201,
        json={"data": {"sessionToken": token, "expiresAt": future_iso(3600), "identityId": "x"}},
    )


@respx.mock
async def test_get_token_handshakes_once_and_caches(identity: IdentityCredentials) -> None:
    route = respx.post(f"{BASE_URL}/sessions").mock(return_value=_session_response("tok-1"))
    mgr = make_manager(identity)
    assert await mgr.get_token() == "tok-1"
    assert await mgr.get_token() == "tok-1"
    assert route.call_count == 1
    sent = route.calls[0].request
    assert sent.headers.get("Matimo-Agent-Signature")
    assert sent.headers.get("X-Matimo-Agent-Identity-Token") == identity.identity_token


@respx.mock
async def test_call_with_retry_rehandshakes_once_on_session_expired(
    identity: IdentityCredentials,
) -> None:
    route = respx.post(f"{BASE_URL}/sessions")
    route.side_effect = [_session_response("tok-1"), _session_response("tok-2")]
    mgr = make_manager(identity)
    tokens_seen: list[str] = []

    async def fn(token: str) -> str:
        tokens_seen.append(token)
        if len(tokens_seen) == 1:
            raise SessionExpired("expired", status_code=401, code="session_expired")
        return "ok"

    assert await mgr.call_with_retry(fn) == "ok"
    assert tokens_seen == ["tok-1", "tok-2"]
    assert route.call_count == 2
