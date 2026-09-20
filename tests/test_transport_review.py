"""Transport and credential-file fixes from the 2026-09-18 review: a fresh
nonce per retry, the async session-retry transport, 0600 from the first
byte, and connect_timeout actually applied."""

from __future__ import annotations

import base64
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from matimo_agdk._retry_transport import AsyncSessionRetryTransport
from matimo_agdk.config import GatewayConfig
from matimo_agdk.identity import IdentityCredentials, JWSSigner, save_credentials
from matimo_agdk.transport import GatewayHTTP, RetryPolicy

from .conftest import BASE_URL


def _jws_payload(jws: str) -> dict[str, Any]:
    part = jws.split(".")[1]
    part += "=" * (-len(part) % 4)
    return json.loads(base64.urlsafe_b64decode(part))


@respx.mock
def test_retry_after_429_carries_a_fresh_nonce(identity: IdentityCredentials, monkeypatch) -> None:
    monkeypatch.setattr("time.sleep", lambda *_: None)
    signatures: list[str] = []

    def responder(request: httpx.Request) -> httpx.Response:
        signatures.append(request.headers["Matimo-Agent-Signature"])
        if len(signatures) == 1:
            return httpx.Response(429, json={"error": "rate_limited"})
        return httpx.Response(200, json={"data": {"ok": True}})

    respx.post(f"{BASE_URL}/tools/check").mock(side_effect=responder)
    http = GatewayHTTP(BASE_URL, "org-key", retry_policy=RetryPolicy(base_delay=0, max_delay=0))
    http.signer = JWSSigner.from_credentials(identity)
    resp = http.request("POST", "/tools/check", json_body={"toolName": "t"}, sign=True)
    assert resp.data == {"ok": True}
    assert len(signatures) == 2
    first, second = (_jws_payload(s) for s in signatures)
    assert first["nonce"] != second["nonce"]
    assert first["body_hash"] == second["body_hash"]


class _FakeAsyncSession:
    def __init__(self) -> None:
        self.tokens = iter(["tok-1", "tok-2"])
        self.invalidations = 0

    async def invalidate(self) -> None:
        self.invalidations += 1

    async def get_token(self) -> str:
        return next(self.tokens)


class _FakeInner(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.seen: list[str | None] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.seen.append(request.headers.get("X-Matimo-Session-Token"))
        if len(self.seen) == 1:
            return httpx.Response(401, json={"error": "session_expired"})
        return httpx.Response(200, json={"id": "chatcmpl-1"})


async def test_async_session_retry_transport_resends_once(identity: IdentityCredentials) -> None:
    inner = _FakeInner()
    session = _FakeAsyncSession()
    transport = AsyncSessionRetryTransport(
        inner,
        session,
        JWSSigner.from_credentials(identity),
        signing_enabled=True,  # type: ignore[arg-type]
    )
    async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
        resp = await client.post(
            "/chat/completions", json={"model": "m"}, headers={"X-Matimo-Session-Token": "stale"}
        )
    assert resp.status_code == 200
    assert inner.seen == ["stale", "tok-1"]
    assert session.invalidations == 1


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes only")
def test_credentials_written_with_0600_from_the_start(
    identity: IdentityCredentials, credentials_dir: Path, monkeypatch
) -> None:
    monkeypatch.setattr(os, "umask", lambda *_: 0)  # a permissive umask must not matter
    _, key_path = save_credentials(identity, credentials_dir)
    assert stat.S_IMODE(key_path.stat().st_mode) == 0o600


def test_connect_timeout_reaches_the_http_client(identity: IdentityCredentials) -> None:
    config = GatewayConfig(base_url=BASE_URL, api_key="k", connect_timeout=3.5, read_timeout=42.0)
    timeout = config.http_timeout()
    assert timeout.connect == 3.5
    assert timeout.read == 42.0
    from matimo_agdk.governor import Governor

    governor = Governor(config)
    assert governor._http._client.timeout.connect == 3.5  # noqa: SLF001
