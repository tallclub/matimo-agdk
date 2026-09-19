"""Session handshake management (docs/SERVER-CONTRACT.md section 4).

POST /v1/sessions is mandatory before the first /v1/chat/completions,
/v1/messages, or /v1/telemetry/batch call of a process's lifetime -- there
is no fallback path, and a bare identity-token bearer no longer works on
those three routes. Renewal is always a fresh signed handshake using the
same long-lived private key; there is deliberately no second, longer-lived
refresh-token secret.

SessionManager (sync, threading.Lock) and AsyncSessionManager (asyncio,
asyncio.Lock) are NOT interchangeable -- pick the one matching whether the
calling code is sync or asyncio-based.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, TypeVar

from .exceptions import SessionExpired
from .identity import IdentityCredentials, JWSSigner
from .transport import AsyncGatewayHTTP, GatewayHTTP

SESSION_TOKEN_HEADER = "X-Matimo-Session-Token"
IDENTITY_TOKEN_HEADER = "X-Matimo-Agent-Identity-Token"

# Matches the UAF reference client (tests/external-agents/langchain-agent/agent.py):
# re-handshake once a session has consumed this fraction of its own TTL,
# well ahead of the hard expiry.
SESSION_RENEWAL_FRACTION = 0.8

R = TypeVar("R")


def _parse_expires_at(value: str) -> datetime:
    # Python's fromisoformat() does not accept a trailing 'Z' before 3.11;
    # normalize defensively regardless of the running interpreter.
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@dataclass
class _SessionState:
    token: str
    expires_at: datetime
    issued_monotonic: float
    ttl_seconds: float


def _build_session_state(data: dict[str, Any]) -> _SessionState:
    expires_at = _parse_expires_at(data["expiresAt"])
    ttl = max((expires_at - datetime.now(timezone.utc)).total_seconds(), 1.0)
    return _SessionState(
        token=data["sessionToken"],
        expires_at=expires_at,
        issued_monotonic=time.monotonic(),
        ttl_seconds=ttl,
    )


def _needs_renewal(state: _SessionState, renewal_fraction: float) -> bool:
    age = time.monotonic() - state.issued_monotonic
    return age >= state.ttl_seconds * renewal_fraction


class SessionManager:
    """Owns the current Gateway session token for a synchronous Governor."""

    def __init__(
        self,
        http: GatewayHTTP,
        signer: JWSSigner,
        identity: IdentityCredentials,
        renewal_fraction: float = SESSION_RENEWAL_FRACTION,
    ) -> None:
        self._http = http
        self._signer = signer
        self._identity = identity
        self._renewal_fraction = renewal_fraction
        self._lock = threading.Lock()
        self._state: _SessionState | None = None

    def _handshake(self) -> _SessionState:
        # The handshake body MUST be the exact bytes b"{}" -- serialize_json({})
        # produces exactly that, and GatewayHTTP.request() signs and sends
        # those same bytes without any intervening re-serialization
        # (docs/SERVER-CONTRACT.md section 4.1, section 11 point 3).
        resp = self._http.request(
            "POST",
            "/sessions",
            json_body={},
            headers={IDENTITY_TOKEN_HEADER: self._identity.identity_token},
            sign=True,
            identity_id=self._identity.identity_id,
            tenant_id=self._identity.tenant_id,
            external_framework=self._identity.external_framework,
        )
        return _build_session_state(resp.data)

    def get_token(self) -> str:
        with self._lock:
            if self._state is None or _needs_renewal(self._state, self._renewal_fraction):
                self._state = self._handshake()
            return self._state.token

    def invalidate(self) -> None:
        """Forces the next get_token() call to re-handshake -- called
        reactively after a live SessionExpired response, and after a key
        rotation."""
        with self._lock:
            self._state = None

    def call_with_retry(self, fn: Callable[[str], R]) -> R:
        """Runs fn(session_token), catching exactly one SessionExpired and
        retrying once after a fresh handshake."""
        token = self.get_token()
        try:
            return fn(token)
        except SessionExpired:
            self.invalidate()
            token = self.get_token()
            return fn(token)


class AsyncSessionManager:
    """Async twin of SessionManager. The asyncio.Lock is created lazily so
    this object can be constructed before an event loop is running."""

    def __init__(
        self,
        http: AsyncGatewayHTTP,
        signer: JWSSigner,
        identity: IdentityCredentials,
        renewal_fraction: float = SESSION_RENEWAL_FRACTION,
    ) -> None:
        self._http = http
        self._signer = signer
        self._identity = identity
        self._renewal_fraction = renewal_fraction
        self._lock: asyncio.Lock | None = None
        self._state: _SessionState | None = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def _handshake(self) -> _SessionState:
        resp = await self._http.request(
            "POST",
            "/sessions",
            json_body={},
            headers={IDENTITY_TOKEN_HEADER: self._identity.identity_token},
            sign=True,
            identity_id=self._identity.identity_id,
            tenant_id=self._identity.tenant_id,
            external_framework=self._identity.external_framework,
        )
        return _build_session_state(resp.data)

    async def get_token(self) -> str:
        async with self._get_lock():
            if self._state is None or _needs_renewal(self._state, self._renewal_fraction):
                self._state = await self._handshake()
            return self._state.token

    async def invalidate(self) -> None:
        async with self._get_lock():
            self._state = None

    async def call_with_retry(self, fn: Callable[[str], Awaitable[R]]) -> R:
        token = await self.get_token()
        try:
            return await fn(token)
        except SessionExpired:
            await self.invalidate()
            token = await self.get_token()
            return await fn(token)
