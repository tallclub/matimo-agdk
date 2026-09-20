"""Transport-level, transparent re-handshake on a live 401 session_expired.

Found live-verifying against a real Gateway (2026-09-18 live verification, see CHANGELOG.md): the
request event hook `Governor.httpx_client()`/`httpx_async_client()` installs
can only set headers *before* a request goes out -- it has no way to inspect
the response and resend. So if a session is invalidated behind the SDK's
back (another process calling DELETE /v1/sessions, or Redis simply evicting
the key), every subsequent LLM-proxy call via that httpx client failed with
a raw 401 forever, even though `SessionManager.call_with_retry()` already
existed and is exactly the recovery logic contract section 11 point 4
describes ("renew reactively on any 401 whose body is session_expired").

This module wraps the client's actual transport (the layer that owns
sending the request and getting a response back) so it CAN inspect the
response and, on exactly this one error shape, invalidate the cached
session, mint a fresh one, re-sign, and resend once -- transparently, with
no change needed at any LLM-SDK call site.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from .identity import JWSSigner
    from .session import AsyncSessionManager, SessionManager

SESSION_TOKEN_HEADER = "X-Matimo-Session-Token"
SIGNATURE_HEADER = "Matimo-Agent-Signature"


def _is_session_expired_body(response: Any) -> bool:
    """`response` is an httpx or httpx2 Response (duck-typed: this module and
    _retry_transport_httpx2 share it). Callers must have already called
    response.read()/aread() -- reading
    the body is only safe to do unconditionally on a 401 (status/headers are
    available without reading); a 200 streaming response must never be
    read here, or streaming would break."""
    try:
        body = response.json()
    except ValueError:
        return False
    return isinstance(body, dict) and body.get("error") == "session_expired"


def _resign(request: Any, signer: JWSSigner, *, signing_enabled: bool) -> None:
    if not signing_enabled:
        return
    jws = signer.sign_request(body_bytes=request.content or b"")
    request.headers[SIGNATURE_HEADER] = jws


class SessionRetryTransport(httpx.BaseTransport):
    """Wraps a sync httpx transport: on a 401 session_expired, invalidates
    the cached session, re-handshakes, updates the request's session header
    (and signature, if signing is enabled), and resends exactly once."""

    def __init__(
        self,
        inner: httpx.BaseTransport,
        session: SessionManager,
        signer: JWSSigner,
        *,
        signing_enabled: bool,
    ) -> None:
        self._inner = inner
        self._session = session
        self._signer = signer
        self._signing_enabled = signing_enabled

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        response = self._inner.handle_request(request)
        if response.status_code != 401:
            return response
        response.read()  # safe: only ever done for a (small) 401 error body
        response.close()
        if not _is_session_expired_body(response):
            return response

        self._session.invalidate()
        request.headers[SESSION_TOKEN_HEADER] = self._session.get_token()
        _resign(request, self._signer, signing_enabled=self._signing_enabled)
        return self._inner.handle_request(request)

    def close(self) -> None:
        self._inner.close()


class AsyncSessionRetryTransport(httpx.AsyncBaseTransport):
    """Async twin of SessionRetryTransport."""

    def __init__(
        self,
        inner: httpx.AsyncBaseTransport,
        session: AsyncSessionManager,
        signer: JWSSigner,
        *,
        signing_enabled: bool,
    ) -> None:
        self._inner = inner
        self._session = session
        self._signer = signer
        self._signing_enabled = signing_enabled

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self._inner.handle_async_request(request)
        if response.status_code != 401:
            return response
        await response.aread()  # safe: only ever done for a (small) 401 error body
        await response.aclose()
        if not _is_session_expired_body(response):
            return response

        await self._session.invalidate()
        request.headers[SESSION_TOKEN_HEADER] = await self._session.get_token()
        _resign(request, self._signer, signing_enabled=self._signing_enabled)
        return await self._inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self._inner.aclose()
