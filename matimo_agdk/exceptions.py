"""Typed exceptions raised by matimo_agdk.

Matimo Gateway's /v1 router uses a flat error envelope, {error, message?},
not the nested {success, error: {code, message}} shape the rest of
Universal-AgentForge uses (docs/SERVER-CONTRACT.md section 0). Every
exception below is built from that flat envelope: .code is the machine
string from the "error" field, .message is the optional human string from
"message".
"""

from __future__ import annotations

# GatewayError also inherits from openai.OpenAIError / anthropic.AnthropicError
# (whichever is installed) -- both SDKs' own request() methods special-case
# "is this already one of our own exception types?" and re-raise it completely
# untouched; anything else gets wrapped into a generic APIConnectionError,
# destroying the original PolicyDenied/AgentSuspended/etc. type and message
# (openai._base_client.request(): "except OpenAIError as err: raise err" vs.
# "except Exception as err: raise APIConnectionError(...) from err";
# anthropic._base_client.py has the identical "isinstance(err, AnthropicError)"
# check). Without this, a 403 raised from matimo_agdk's own httpx transport
# (see _retry_transport.py) still reaches the caller as a bare
# openai.APIConnectionError / anthropic.APIConnectionError, one level below
# where the transport raised it -- exactly the bug this base class exists to
# close. Deliberately the *base* OpenAIError/AnthropicError, never the more
# specific APIError (which requires request/body/response constructor args
# these exceptions don't carry). Both imports are optional and best-effort:
# matimo_agdk itself has no hard dependency on either provider SDK.
_OpenAIError: type[Exception] | None
_AnthropicError: type[Exception] | None

try:
    from openai import OpenAIError as _OpenAIError
except ImportError:  # pragma: no cover - openai not installed
    _OpenAIError = None

try:
    from anthropic import AnthropicError as _AnthropicError
except ImportError:  # pragma: no cover - anthropic not installed
    _AnthropicError = None

_LLM_SDK_BASES: tuple[type[Exception], ...] = tuple(
    base for base in (_OpenAIError, _AnthropicError) if base is not None
)


class GatewayError(*(_LLM_SDK_BASES or (Exception,))):  # type: ignore[misc]
    """Base class for every error matimo_agdk's transport layer raises."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(message={self.message!r}, "
            f"status_code={self.status_code!r}, code={self.code!r})"
        )


class PolicyDenied(GatewayError):
    """403 policy_denied. `reason` is the machine-readable denial reason
    that travels in the response's `message` field, never decorative text
    (docs/SERVER-CONTRACT.md section 6.6). Retrying a policy_denied call
    unchanged will simply recur -- surface the reason, do not loop on it.
    """

    def __init__(
        self,
        reason: str,
        *,
        status_code: int | None = 403,
        code: str | None = "policy_denied",
    ) -> None:
        super().__init__(reason, status_code=status_code, code=code)
        self.reason = reason


class TelemetryStale(PolicyDenied):
    """policy_denied with reason=telemetry_stale. The fix is resuming
    telemetry pushes, not retrying the LLM call (docs/SERVER-CONTRACT.md
    section 11 point 9)."""


class AgentSuspended(PolicyDenied):
    """policy_denied with reason in {agent_suspended, agent_revoked,
    emergency_stop_active}."""


class SessionExpired(GatewayError):
    """401 session_expired. Covers five collapsed server-side failure modes
    (docs/SERVER-CONTRACT.md section 4.3) -- the only correct client
    reaction to any of them is a fresh handshake, never a distinction
    between which one occurred."""


class SignatureRejected(GatewayError):
    """403 signature_required. The Matimo-Agent-Signature header was
    missing, malformed, or failed verification. The server never says
    which of the many possible reasons applied (docs/SERVER-CONTRACT.md
    section 5.1)."""


class SpendCapExceeded(GatewayError):
    """403 spend_cap_exceeded. A tenant/team/user/virtual-key/agent-identity
    LLM budget (BUILD-PLAN 2026-09-25/26 unified budget ledger) was already
    at or over its cap when the call was reserved -- the call was never
    dispatched to the LLM provider, no cost was incurred. Unlike
    `PolicyDenied`, the server names no machine-readable scope in the
    response body; `.message` is a human sentence naming which scope(s)
    blocked the call (`GatewayProxyService.describeBudgetBlock()`) and
    points at `POST /api/v1/budgets/requests` for a human to ask for more.
    Retrying immediately will simply recur until an admin raises the cap or
    the scope's reset period rolls over -- do not loop on it.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = 403,
        code: str | None = "spend_cap_exceeded",
    ) -> None:
        super().__init__(message, status_code=status_code, code=code)


class RateLimited(GatewayError):
    """429 rate_limit_exceeded. Gateway sends no Retry-After header on this
    route today (docs/SERVER-CONTRACT.md section 0) -- `retry_after` is
    populated only if a future response ever includes one."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = 429,
        code: str | None = "rate_limit_exceeded",
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message, status_code=status_code, code=code)
        self.retry_after = retry_after


class GatewayUnavailable(GatewayError):
    """502 upstream_error, or a connection-level failure / exhausted retry
    budget talking to Gateway at all."""


class ToolCheckUnavailable(GatewayUnavailable):
    """A tool check could not be answered because Gateway is unreachable or
    failing (a connection error, a timeout, a 5xx), and the configured
    `tool_check_failure_mode` did not allow the call to proceed.

    `circuit_open` is True when the SDK did not even try: its circuit breaker
    is open after repeated transport failures, so it failed fast instead of
    waiting out the transport's retries again. Adapters surface this as a
    normal recoverable tool error, the same way they surface `ToolDenied`.
    It subclasses `GatewayUnavailable`, so existing `except` clauses still
    catch it."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: str | None = "tool_check_unavailable",
        circuit_open: bool = False,
    ) -> None:
        super().__init__(message, status_code=status_code, code=code)
        self.circuit_open = circuit_open


class ToolDenied(GatewayError):
    """Raised by Governor.guard() when a tool check resolves to DENY (either
    immediately, or after a PENDING resolves to DENY)."""

    def __init__(self, reason: str | None) -> None:
        super().__init__(reason or "tool call denied", code="tool_denied")
        self.reason = reason


class ToolCheckTimeout(GatewayError):
    """A PENDING tool check never resolved within the configured max wait
    (default 4 hours, matching the server's own approval TTL --
    docs/SERVER-CONTRACT.md section 8.1)."""


class AgentSuspendedLocally(Exception):
    """Raised by Governor.raise_if_suspended() when the SDK's own polled
    GovernanceState (refreshed from the telemetry heartbeat) says the agent
    is suspended, revoked, or under an active emergency stop.

    This is a LOCAL, heartbeat-derived judgement, not a live server call --
    it can lag the true server state by up to one heartbeat interval. This
    is why the product calls it "rapid suspend," not "instant kill": see
    README.md's honesty section and docs/SERVER-CONTRACT.md section 10.
    """

    def __init__(self, lifecycle_status: str, emergency_stop: bool) -> None:
        super().__init__(
            f"agent is locally marked '{lifecycle_status}' (emergency_stop={emergency_stop})"
        )
        self.lifecycle_status = lifecycle_status
        self.emergency_stop = emergency_stop

    def __reduce__(self) -> tuple[type[AgentSuspendedLocally], tuple[str, bool]]:
        # The default reduce replays `args` (the formatted message) into
        # __init__, which takes two other parameters; without this the
        # exception cannot cross a process boundary (multiprocessing, celery).
        return (type(self), (self.lifecycle_status, self.emergency_stop))


class SigningError(Exception):
    """Raised for a malformed private/public key, or an unverifiable or
    malformed compact JWS."""
