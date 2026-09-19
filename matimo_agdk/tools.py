"""Tool governance: /v1/tools/check, /check/status, /result
(docs/SERVER-CONTRACT.md section 8).

All three routes always require the Matimo-Agent-Signature JWS and use the
bare identity token (X-Matimo-Agent-Identity-Token), never the session
token -- tool checks are identity-scoped, not session-scoped.

The idempotency/dedup key is entirely server-derived from
(identityToken, toolName, argHash) -- AGDK only ever computes argHash (a
hash of its own choosing over the tool's actual arguments, consistent for
the same logical call) and never invents or trusts a client-side dedup
key (docs/SERVER-CONTRACT.md section 11 point 11).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from dataclasses import dataclass
from typing import Any, Literal

from ._redact import redact
from .exceptions import GatewayError, ToolCheckTimeout
from .transport import AsyncGatewayHTTP, GatewayHTTP

Decision = Literal["ALLOW", "DENY", "PENDING"]

IDENTITY_TOKEN_HEADER = "X-Matimo-Agent-Identity-Token"

_MAX_ARG_VALUE_LEN = 2000

# TRD section 4.3's decided polling posture: 3s initial, doubling to a 60s
# ceiling, with +/-10% jitter, bounded by the server's own 4-hour approval TTL
# (docs/SERVER-CONTRACT.md section 8.1).
DEFAULT_POLL_INTERVAL_SECONDS = 3.0
DEFAULT_POLL_MAX_INTERVAL_SECONDS = 60.0
DEFAULT_MAX_WAIT_SECONDS = 4 * 3600.0

# Gateway answers PENDING with *no* resumeToken when an identical check is
# already in flight (reason "duplicate_check_in_flight"): the caller has
# nothing to poll. The in-flight check's answer is cached server-side for
# 15 minutes, so re-sending the same check shortly after returns it (with
# the token). Re-check after each of these delays; if it is still tokenless
# after the last one, fail closed rather than run an unapproved tool.
DEFAULT_RECHECK_DELAYS = (0.5, 1.0, 2.0, 4.0)
NO_RESUME_TOKEN_DENY_REASON = "tool_check_pending_without_resume_token"


def redact_args(args: dict[str, Any]) -> dict[str, Any]:
    """Recursive key-based redaction of a tool's arguments before they are
    sent to POST /v1/tools/check (see matimo_agdk._redact)."""
    return redact(args, max_string=_MAX_ARG_VALUE_LEN)


def hash_args(args: dict[str, Any]) -> str:
    """A stable fingerprint of these exact arguments. Gateway's own dedup
    key is derived server-side from (identityToken, toolName, argHash) --
    this function only needs to be a consistent hash of its choosing for
    the same logical call, never a value AGDK trusts as an idempotency key
    on its own."""
    canonical = json.dumps(args, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class ToolDecision:
    decision: Decision
    reason: str | None = None
    resume_token: str | None = None
    request_id: str | None = None

    @property
    def allowed(self) -> bool:
        return self.decision == "ALLOW"

    @property
    def denied(self) -> bool:
        return self.decision == "DENY"

    @property
    def pending(self) -> bool:
        return self.decision == "PENDING"

    @property
    def pending_without_token(self) -> bool:
        """PENDING but nothing to poll -- see DEFAULT_RECHECK_DELAYS."""
        return self.pending and not self.resume_token


def _check_body(
    tool_name: str, args: dict[str, Any] | None, category_hint: str | None, include_args: bool
) -> dict[str, Any]:
    body: dict[str, Any] = {"toolName": tool_name, "argHash": hash_args(args or {})}
    if category_hint:
        body["toolCategory"] = category_hint
    if include_args and args:
        body["args"] = redact_args(args)
    return body


def _decision_from(data: dict[str, Any]) -> ToolDecision:
    return ToolDecision(
        decision=data["decision"],
        reason=data.get("reason"),
        resume_token=data.get("resumeToken"),
        request_id=data.get("requestId"),
    )


class ToolGovernor:
    """Synchronous tool-check client."""

    def __init__(
        self,
        http: GatewayHTTP,
        *,
        identity_token: str,
        identity_id: str,
        tenant_id: str,
        external_framework: str | None = None,
        poll_interval: float = DEFAULT_POLL_INTERVAL_SECONDS,
        poll_max_interval: float = DEFAULT_POLL_MAX_INTERVAL_SECONDS,
        max_wait_seconds: float = DEFAULT_MAX_WAIT_SECONDS,
        recheck_delays: tuple[float, ...] = DEFAULT_RECHECK_DELAYS,
    ) -> None:
        self._http = http
        self._identity_token = identity_token
        self._identity_id = identity_id
        self._tenant_id = tenant_id
        self._external_framework = external_framework
        self.poll_interval = poll_interval
        self.poll_max_interval = poll_max_interval
        self.max_wait_seconds = max_wait_seconds
        self.recheck_delays = recheck_delays

    def _headers(self) -> dict[str, str]:
        return {IDENTITY_TOKEN_HEADER: self._identity_token}

    def _sign_kwargs(self) -> dict[str, Any]:
        return dict(
            sign=True,
            identity_id=self._identity_id,
            tenant_id=self._tenant_id,
            external_framework=self._external_framework,
        )

    def check_and_wait(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
        *,
        category_hint: str | None = None,
    ) -> ToolDecision:
        """check(), then whatever it takes to reach a final ALLOW or DENY.

        PENDING with a resume token is polled to a decision. PENDING with no
        token is re-checked (see DEFAULT_RECHECK_DELAYS) and, if it never
        yields one, becomes a DENY -- never a PENDING the caller might treat
        as "not denied" and run the tool on."""
        decision = self.check(tool_name, args, category_hint=category_hint)
        for delay in self.recheck_delays:
            if not decision.pending_without_token:
                break
            time.sleep(delay * random.uniform(0.9, 1.1))
            decision = self.check(tool_name, args, category_hint=category_hint)
        if decision.pending_without_token:
            return ToolDecision(decision="DENY", reason=NO_RESUME_TOKEN_DENY_REASON)
        if decision.pending and decision.resume_token:
            return self.await_decision(decision.resume_token)
        return decision

    def check(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
        *,
        category_hint: str | None = None,
        include_args: bool = True,
    ) -> ToolDecision:
        resp = self._http.request(
            "POST",
            "/tools/check",
            json_body=_check_body(tool_name, args, category_hint, include_args),
            headers=self._headers(),
            **self._sign_kwargs(),
        )
        return _decision_from(resp.data or {})

    def status(self, resume_token: str) -> ToolDecision:
        resp = self._http.request(
            "POST",
            "/tools/check/status",
            json_body={"resumeToken": resume_token},
            headers=self._headers(),
            **self._sign_kwargs(),
        )
        return _decision_from(resp.data or {})

    def await_decision(
        self,
        resume_token: str,
        *,
        poll_interval: float | None = None,
        max_wait_seconds: float | None = None,
    ) -> ToolDecision:
        """Blocking poll with exponential backoff and +/-10% jitter."""
        interval = poll_interval if poll_interval is not None else self.poll_interval
        budget = max_wait_seconds if max_wait_seconds is not None else self.max_wait_seconds
        deadline = time.monotonic() + budget
        while True:
            decision = self.status(resume_token)
            if not decision.pending:
                return decision
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ToolCheckTimeout(
                    f"tool check {resume_token} did not resolve within {budget:.0f}s"
                )
            time.sleep(min(interval * random.uniform(0.9, 1.1), remaining))
            interval = min(interval * 2, self.poll_max_interval)

    def report_result(
        self,
        resume_token: str,
        *,
        status: str,
        duration_ms: int | None = None,
        error: str | None = None,
    ) -> None:
        """Fire-and-forget: persists nothing queryable server-side today
        (docs/SERVER-CONTRACT.md section 8.3). Swallows GatewayError."""
        body: dict[str, Any] = {"resumeToken": resume_token, "status": status}
        if duration_ms is not None:
            body["durationMs"] = duration_ms
        if error is not None:
            body["error"] = error[:2000]
        try:
            self._http.request(
                "POST",
                "/tools/result",
                json_body=body,
                headers=self._headers(),
                **self._sign_kwargs(),
            )
        except GatewayError:
            pass

    def set_category(self, tool_name: str, category: str) -> None:
        """Tenant-wide admin action, identity:manage scoped, unsigned."""
        self._http.request(
            "PUT",
            f"/tools/{tool_name}/category",
            json_body={"category": category},
            sign=False,
        )


class AsyncToolGovernor:
    """Async twin of ToolGovernor."""

    def __init__(
        self,
        http: AsyncGatewayHTTP,
        *,
        identity_token: str,
        identity_id: str,
        tenant_id: str,
        external_framework: str | None = None,
        poll_interval: float = DEFAULT_POLL_INTERVAL_SECONDS,
        poll_max_interval: float = DEFAULT_POLL_MAX_INTERVAL_SECONDS,
        max_wait_seconds: float = DEFAULT_MAX_WAIT_SECONDS,
        recheck_delays: tuple[float, ...] = DEFAULT_RECHECK_DELAYS,
    ) -> None:
        self._http = http
        self._identity_token = identity_token
        self._identity_id = identity_id
        self._tenant_id = tenant_id
        self._external_framework = external_framework
        self.poll_interval = poll_interval
        self.poll_max_interval = poll_max_interval
        self.max_wait_seconds = max_wait_seconds
        self.recheck_delays = recheck_delays

    def _headers(self) -> dict[str, str]:
        return {IDENTITY_TOKEN_HEADER: self._identity_token}

    def _sign_kwargs(self) -> dict[str, Any]:
        return dict(
            sign=True,
            identity_id=self._identity_id,
            tenant_id=self._tenant_id,
            external_framework=self._external_framework,
        )

    async def check_and_wait(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
        *,
        category_hint: str | None = None,
    ) -> ToolDecision:
        """Async twin of ToolGovernor.check_and_wait()."""
        decision = await self.check(tool_name, args, category_hint=category_hint)
        for delay in self.recheck_delays:
            if not decision.pending_without_token:
                break
            await asyncio.sleep(delay * random.uniform(0.9, 1.1))
            decision = await self.check(tool_name, args, category_hint=category_hint)
        if decision.pending_without_token:
            return ToolDecision(decision="DENY", reason=NO_RESUME_TOKEN_DENY_REASON)
        if decision.pending and decision.resume_token:
            return await self.await_decision(decision.resume_token)
        return decision

    async def check(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
        *,
        category_hint: str | None = None,
        include_args: bool = True,
    ) -> ToolDecision:
        resp = await self._http.request(
            "POST",
            "/tools/check",
            json_body=_check_body(tool_name, args, category_hint, include_args),
            headers=self._headers(),
            **self._sign_kwargs(),
        )
        return _decision_from(resp.data or {})

    async def status(self, resume_token: str) -> ToolDecision:
        resp = await self._http.request(
            "POST",
            "/tools/check/status",
            json_body={"resumeToken": resume_token},
            headers=self._headers(),
            **self._sign_kwargs(),
        )
        return _decision_from(resp.data or {})

    async def await_decision(
        self,
        resume_token: str,
        *,
        poll_interval: float | None = None,
        max_wait_seconds: float | None = None,
    ) -> ToolDecision:
        interval = poll_interval if poll_interval is not None else self.poll_interval
        budget = max_wait_seconds if max_wait_seconds is not None else self.max_wait_seconds
        deadline = time.monotonic() + budget
        while True:
            decision = await self.status(resume_token)
            if not decision.pending:
                return decision
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ToolCheckTimeout(
                    f"tool check {resume_token} did not resolve within {budget:.0f}s"
                )
            await asyncio.sleep(min(interval * random.uniform(0.9, 1.1), remaining))
            interval = min(interval * 2, self.poll_max_interval)

    async def report_result(
        self,
        resume_token: str,
        *,
        status: str,
        duration_ms: int | None = None,
        error: str | None = None,
    ) -> None:
        body: dict[str, Any] = {"resumeToken": resume_token, "status": status}
        if duration_ms is not None:
            body["durationMs"] = duration_ms
        if error is not None:
            body["error"] = error[:2000]
        try:
            await self._http.request(
                "POST",
                "/tools/result",
                json_body=body,
                headers=self._headers(),
                **self._sign_kwargs(),
            )
        except GatewayError:
            pass

    async def set_category(self, tool_name: str, category: str) -> None:
        await self._http.request(
            "PUT",
            f"/tools/{tool_name}/category",
            json_body={"category": category},
            sign=False,
        )
