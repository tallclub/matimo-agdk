"""Sync and async HTTP transport for the Matimo Gateway /v1 API.

Both classes below share the same request-building and error-mapping
logic (the free functions at module scope); only the actual I/O call
differs (blocking httpx.Client vs httpx.AsyncClient + asyncio.sleep).

Contract details this module is responsible for getting right
(docs/SERVER-CONTRACT.md section 0, section 5, section 11):
- The request body is serialized exactly once; those exact bytes are both
  hashed into the signature's body_hash claim and sent over the wire via
  `content=`, never re-serialized independently by the HTTP client.
- Success envelope is `{data: ...}` for almost every route; two routes
  (`GET /v1/identities/:id/jwks`, and the LLM proxy routes) return a bare
  body -- callers of `.request()` that hit those routes read `.data` all
  the same, since `.data` falls back to the whole parsed body when there
  is no `data` key.
- Error envelope is flat: `{error, message?}`. Mapped to typed exceptions
  in `raise_for_error()`.
- Retry policy: 429 always retried (honors Retry-After if ever sent, else
  jittered exponential backoff); 5xx and connection errors retried only
  when the caller marks the call `idempotent=True`; a 403 policy_denied
  (or any other non-429 4xx) is never retried.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from .exceptions import (
    AgentSuspended,
    GatewayError,
    GatewayUnavailable,
    PolicyDenied,
    RateLimited,
    SessionExpired,
    SignatureRejected,
    TelemetryStale,
)
from .identity import JWSSigner

_AGENT_SUSPENDED_REASONS = ("agent_suspended", "agent_revoked", "emergency_stop_active")


def serialize_json(obj: Any) -> bytes:
    """The one place a request body is turned into bytes. Deliberately
    compact (no extra whitespace) and called exactly once per request so
    the same bytes get hashed and sent."""
    return json.dumps(obj, separators=(",", ":")).encode("utf-8")


@dataclass
class GatewayResponse:
    status_code: int
    data: Any
    headers: httpx.Headers


class RetryPolicy:
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 0.5,
        max_delay: float = 20.0,
        max_retry_after: float = 60.0,
    ) -> None:
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.max_retry_after = max_retry_after

    def delay_for(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            # A server-supplied Retry-After is honored but never trusted past
            # max_retry_after: one hostile or buggy header must not park a thread for hours.
            return min(max(retry_after, 0.0), self.max_retry_after)
        raw = min(self.base_delay * (2**attempt), self.max_delay)
        # Decorrelated-ish jitter: half to full of the computed backoff.
        return random.uniform(raw / 2, raw)


def parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None  # an HTTP-date Retry-After is not parsed in v1
    # float() accepts "nan" and "inf"; neither is a usable delay.
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def raise_for_error(
    status_code: int, body: Mapping[str, Any] | None, retry_after: float | None = None
) -> None:
    """Maps Gateway's flat {error, message?} envelope to a typed exception.
    Always raises when status_code >= 400 -- callers rely on this."""
    body = body or {}
    code = body.get("error")
    message = body.get("message")

    if status_code == 401 and code == "session_expired":
        raise SessionExpired(message or "session expired", status_code=status_code, code=code)
    if status_code == 403 and code == "signature_required":
        raise SignatureRejected(
            message or "Matimo-Agent-Signature is missing or invalid",
            status_code=status_code,
            code=code,
        )
    if status_code == 403 and code == "policy_denied":
        reason = message or "policy_denied"
        if reason == "telemetry_stale":
            raise TelemetryStale(reason, status_code=status_code, code=code)
        if reason in _AGENT_SUSPENDED_REASONS:
            raise AgentSuspended(reason, status_code=status_code, code=code)
        raise PolicyDenied(reason, status_code=status_code, code=code)
    if status_code == 429:
        raise RateLimited(
            message or "rate_limit_exceeded",
            status_code=status_code,
            code=code,
            retry_after=retry_after,
        )
    if status_code == 502:
        raise GatewayUnavailable(message or "upstream error", status_code=status_code, code=code)
    if status_code >= 400:
        raise GatewayError(
            message or code or f"gateway error ({status_code})", status_code=status_code, code=code
        )


def _finish_response(resp: httpx.Response) -> GatewayResponse:
    body: Any = None
    if resp.content:
        try:
            body = resp.json()
        except ValueError:
            body = None
    if resp.status_code >= 400:
        raise_for_error(
            resp.status_code,
            body if isinstance(body, dict) else None,
            parse_retry_after(resp.headers.get("Retry-After")),
        )
        # raise_for_error() always raises for >= 400; this is unreachable
        # in practice and exists only so type checkers see a return.
        raise GatewayError(f"gateway error ({resp.status_code})", status_code=resp.status_code)
    data = body.get("data") if isinstance(body, dict) and "data" in body else body
    return GatewayResponse(status_code=resp.status_code, data=data, headers=resp.headers)


class _HeaderMixin:
    api_key: str
    signer: JWSSigner | None

    def _base_headers(self, extra: Mapping[str, str] | None) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        if extra:
            headers.update(extra)
        return headers

    def _maybe_sign(
        self,
        headers: dict[str, str],
        body_bytes: bytes,
        *,
        sign: bool,
        identity_id: str | None,
        tenant_id: str | None,
        external_framework: str | None,
    ) -> None:
        if not sign or self.signer is None:
            return
        jws = self.signer.sign_request(
            body_bytes=body_bytes,
            identity_id=identity_id,
            tenant_id=tenant_id,
            external_framework=external_framework,
        )
        headers["Matimo-Agent-Signature"] = jws


class GatewayHTTP(_HeaderMixin):
    """Synchronous transport. Owns one pooled httpx.Client."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        signer: JWSSigner | None = None,
        timeout: float | httpx.Timeout = 30.0,
        retry_policy: RetryPolicy | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.signer = signer
        self.retry_policy = retry_policy or RetryPolicy()
        self._client = client or httpx.Client(base_url=self.base_url, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> GatewayHTTP:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        headers: Mapping[str, str] | None = None,
        sign: bool = False,
        idempotent: bool = False,
        identity_id: str | None = None,
        tenant_id: str | None = None,
        external_framework: str | None = None,
    ) -> GatewayResponse:
        has_body = json_body is not None
        body_bytes = serialize_json(json_body) if has_body else b""
        req_headers = self._base_headers(headers)
        attempt = 0
        while True:
            # Sign on every attempt: a retry after a 429 or 5xx must carry a
            # fresh nonce and iat, or Gateway's replay protection rejects it.
            self._maybe_sign(
                req_headers,
                body_bytes,
                sign=sign,
                identity_id=identity_id,
                tenant_id=tenant_id,
                external_framework=external_framework,
            )
            try:
                resp = self._client.request(
                    method,
                    path,
                    content=body_bytes if has_body else None,
                    headers=req_headers,
                )
            except httpx.TransportError as exc:
                if idempotent and attempt < self.retry_policy.max_retries:
                    time.sleep(self.retry_policy.delay_for(attempt, None))
                    attempt += 1
                    continue
                raise GatewayUnavailable(str(exc)) from exc

            if resp.status_code == 429 and attempt < self.retry_policy.max_retries:
                retry_after = parse_retry_after(resp.headers.get("Retry-After"))
                time.sleep(self.retry_policy.delay_for(attempt, retry_after))
                attempt += 1
                continue

            if resp.status_code >= 500 and idempotent and attempt < self.retry_policy.max_retries:
                time.sleep(self.retry_policy.delay_for(attempt, None))
                attempt += 1
                continue

            return _finish_response(resp)


class AsyncGatewayHTTP(_HeaderMixin):
    """Async transport. Owns one pooled httpx.AsyncClient."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        signer: JWSSigner | None = None,
        timeout: float | httpx.Timeout = 30.0,
        retry_policy: RetryPolicy | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.signer = signer
        self.retry_policy = retry_policy or RetryPolicy()
        self._client = client or httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> AsyncGatewayHTTP:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        headers: Mapping[str, str] | None = None,
        sign: bool = False,
        idempotent: bool = False,
        identity_id: str | None = None,
        tenant_id: str | None = None,
        external_framework: str | None = None,
    ) -> GatewayResponse:
        has_body = json_body is not None
        body_bytes = serialize_json(json_body) if has_body else b""
        req_headers = self._base_headers(headers)
        attempt = 0
        while True:
            # Sign on every attempt: a retry after a 429 or 5xx must carry a
            # fresh nonce and iat, or Gateway's replay protection rejects it.
            self._maybe_sign(
                req_headers,
                body_bytes,
                sign=sign,
                identity_id=identity_id,
                tenant_id=tenant_id,
                external_framework=external_framework,
            )
            try:
                resp = await self._client.request(
                    method,
                    path,
                    content=body_bytes if has_body else None,
                    headers=req_headers,
                )
            except httpx.TransportError as exc:
                if idempotent and attempt < self.retry_policy.max_retries:
                    await asyncio.sleep(self.retry_policy.delay_for(attempt, None))
                    attempt += 1
                    continue
                raise GatewayUnavailable(str(exc)) from exc

            if resp.status_code == 429 and attempt < self.retry_policy.max_retries:
                retry_after = parse_retry_after(resp.headers.get("Retry-After"))
                await asyncio.sleep(self.retry_policy.delay_for(attempt, retry_after))
                attempt += 1
                continue

            if resp.status_code >= 500 and idempotent and attempt < self.retry_policy.max_retries:
                await asyncio.sleep(self.retry_policy.delay_for(attempt, None))
                attempt += 1
                continue

            return _finish_response(resp)
