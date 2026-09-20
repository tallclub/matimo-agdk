"""`httpx2` twins of the transports in `_retry_transport`.

Newer LLM SDKs (`anthropic` >= 1.6) build on `httpx2`, a separate library whose
classes are unrelated to `httpx`'s, and they reject an `httpx.Client` outright
(`TypeError: ... uses httpx2. Use httpx2.Client instead`). So a client that must
sign and re-handshake for such an SDK has to be an `httpx2` client with an
`httpx2` transport. The logic is identical to `_retry_transport`; only the base
classes differ, and the helpers that never touch the library are shared.

Importing this module needs `httpx2`, which the SDKs that require it already
depend on. `Governor.httpx2_client()` imports it lazily and explains what to
install if it is missing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx2

from ._retry_transport import (
    SESSION_TOKEN_HEADER,
    _is_session_expired_body,
    _resign,
)

if TYPE_CHECKING:
    from .identity import JWSSigner
    from .session import AsyncSessionManager, SessionManager

__all__ = [
    "Httpx2AsyncSessionRetryTransport",
    "Httpx2SessionRetryTransport",
    "httpx2",
]


class Httpx2SessionRetryTransport(httpx2.BaseTransport):
    """See `SessionRetryTransport`: on a 401 session_expired, invalidate the
    cached session, re-handshake, re-sign, and resend exactly once."""

    def __init__(
        self,
        inner: httpx2.BaseTransport,
        session: SessionManager,
        signer: JWSSigner,
        *,
        signing_enabled: bool,
    ) -> None:
        self._inner = inner
        self._session = session
        self._signer = signer
        self._signing_enabled = signing_enabled

    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
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


class Httpx2AsyncSessionRetryTransport(httpx2.AsyncBaseTransport):
    """Async twin of `Httpx2SessionRetryTransport`."""

    def __init__(
        self,
        inner: httpx2.AsyncBaseTransport,
        session: AsyncSessionManager,
        signer: JWSSigner,
        *,
        signing_enabled: bool,
    ) -> None:
        self._inner = inner
        self._session = session
        self._signer = signer
        self._signing_enabled = signing_enabled

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
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
