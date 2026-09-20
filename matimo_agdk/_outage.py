"""What a tool check does when Gateway cannot answer it at all.

TRD section 4.1 gives each customer a local choice per agent (block, or carry
on) and pairs it with a circuit breaker so an outage does not hang the agent on
every call. BUILD-PLAN D19 caps how long any fail-open posture may rest on
stale state at 5 minutes. This module is that logic, shared by the sync and
async tool governors (tools.py); it does no I/O.

Only *transport-level* failures count: a connection error, a timeout, a 5xx.
Those say "Gateway did not render a decision". Everything else keeps its
existing behaviour and is never softened:

- an explicit DENY, and an unrecognized decision (already turned into a DENY),
- any 4xx (bad key, missing scope, signature rejected, suspended, rate limited),
- a locally-known suspended state (the heartbeat said suspended or stopped),
- PENDING polling (status() failures, and the re-checks that follow a PENDING
  with no resume token): a tool awaiting a human's approval never runs on a
  guess.

`fail_open_bounded` adds two conservative conditions on top of the brief's
freshness rule, because the SDK cannot see the policy: the tool's most recent
decision must not have been DENY or PENDING (a tool that needs approval, or is
blocked, is not waved through by an outage), and Gateway must have been heard
from within `fail_open_max_stale_seconds`.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .exceptions import GatewayError, GatewayUnavailable, ToolCheckUnavailable

if TYPE_CHECKING:
    from .telemetry import GovernanceState
    from .tools import ToolDecision

_log = logging.getLogger("matimo_agdk.outage")

FAIL_CLOSED = "fail_closed"
FAIL_OPEN_BOUNDED = "fail_open_bounded"

DEGRADED_ALLOW_REASON = "gateway_unavailable_fail_open"

# The last decision is remembered per tool name; a process rarely has more than
# a few dozen, so this only bounds a pathological caller inventing names.
_MAX_TRACKED_TOOLS = 1024


def is_transport_failure(exc: GatewayError) -> bool:
    """True for a failure to reach Gateway or to get any real answer from it:
    a connection error or timeout (already mapped to GatewayUnavailable by the
    transport), a 502, or any other 5xx. False for every 4xx."""
    if isinstance(exc, GatewayUnavailable):
        return True
    status = exc.status_code
    return status is not None and status >= 500


class CircuitBreaker:
    """Closed -> open after `threshold` consecutive failures -> half-open after
    `cooldown` seconds, where exactly one probe call is allowed through. A
    probe that gets any reachable answer closes the circuit; a probe that fails
    re-opens it for another cooldown. Thread-safe, and safe to share with async
    code: nothing awaits while the lock is held."""

    def __init__(
        self,
        threshold: int = 3,
        cooldown: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.threshold = threshold
        self.cooldown = cooldown
        self._clock = clock
        self._lock = threading.Lock()
        self._failures = 0
        self._opened_at: float | None = None
        self._probing = False

    @property
    def state(self) -> str:
        with self._lock:
            if self._opened_at is None:
                return "closed"
            return "half_open" if self._probing else "open"

    def acquire(self) -> bool:
        """True if a call may go out now. While open this is False until the
        cooldown elapses, then True for one caller only (the probe)."""
        with self._lock:
            if self._opened_at is None:
                return True
            if self._probing:
                return False
            if self._clock() - self._opened_at >= self.cooldown:
                self._probing = True
                return True
            return False

    def retry_in(self) -> float:
        with self._lock:
            if self._opened_at is None:
                return 0.0
            return max(0.0, self.cooldown - (self._clock() - self._opened_at))

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None
            self._probing = False

    def record_failure(self) -> None:
        with self._lock:
            if self._probing:
                self._probing = False
                self._opened_at = self._clock()
                return
            self._failures += 1
            if self._opened_at is None and self._failures >= self.threshold:
                self._opened_at = self._clock()
                _log.warning(
                    "tool-check circuit opened after %d consecutive Gateway failures; "
                    "failing fast for %.0fs",
                    self._failures,
                    self.cooldown,
                )

    def release_probe(self) -> None:
        """A probe ended without a verdict (an unexpected exception, a
        cancelled task): let the next caller probe instead of wedging."""
        with self._lock:
            self._probing = False


class OutageGuard:
    """Per-ToolGovernor outage state: the breaker, when Gateway was last heard
    from, and the last decision seen per tool."""

    def __init__(
        self,
        *,
        failure_mode: str = FAIL_CLOSED,
        max_stale_seconds: float = 300.0,
        breaker_threshold: int = 3,
        breaker_cooldown: float = 30.0,
        state_provider: Callable[[], GovernanceState] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.failure_mode = failure_mode
        self.max_stale_seconds = max_stale_seconds
        self.breaker = CircuitBreaker(breaker_threshold, breaker_cooldown, clock)
        self._state_provider = state_provider
        self._clock = clock
        self._lock = threading.Lock()
        self._last_ok: float | None = None
        self._last_decision: OrderedDict[str, str] = OrderedDict()

    # -- observations, called by the tool governors ------------------------

    def before_call(self) -> None:
        """Fail fast while the circuit is open."""
        if not self.breaker.acquire():
            raise ToolCheckUnavailable(
                "Gateway tool checks are unavailable: circuit open after repeated "
                f"failures, next attempt in {self.breaker.retry_in():.0f}s",
                circuit_open=True,
            )

    def transport_failed(self, exc: GatewayError) -> ToolCheckUnavailable:
        self.breaker.record_failure()
        return ToolCheckUnavailable(
            f"Gateway tool check unavailable: {exc.message}",
            status_code=exc.status_code,
            code=exc.code or "tool_check_unavailable",
        )

    def reachable(self) -> None:
        """Gateway answered (with anything, a 4xx included)."""
        self.breaker.record_success()

    def aborted(self) -> None:
        self.breaker.release_probe()

    def record_decision(self, tool_name: str, decision: str, *, recognized: bool = True) -> None:
        """The immediate decision `/tools/check` returned for this tool."""
        with self._lock:
            if recognized:
                self._last_ok = self._clock()
            self._last_decision[tool_name] = decision
            self._last_decision.move_to_end(tool_name)
            while len(self._last_decision) > _MAX_TRACKED_TOOLS:
                self._last_decision.popitem(last=False)

    def record_contact(self) -> None:
        """Gateway answered a status poll: proof it was reachable just now."""
        with self._lock:
            self._last_ok = self._clock()

    # -- the fail-open decision ---------------------------------------------

    def _state(self) -> GovernanceState | None:
        if self._state_provider is None:
            return None
        try:
            return self._state_provider()
        except Exception:  # noqa: BLE001 -- never let a probe of local state break a check
            return None

    def contact_age(self) -> float | None:
        """Seconds since Gateway was last heard from (a successful check or a
        telemetry heartbeat), or None if it never was."""
        with self._lock:
            latest = self._last_ok
        state = self._state()
        heartbeat = getattr(state, "last_heartbeat_monotonic", None) if state else None
        if heartbeat is not None and (latest is None or heartbeat > latest):
            latest = heartbeat
        if latest is None:
            return None
        return max(0.0, self._clock() - latest)

    def degrade(self, tool_name: str, exc: ToolCheckUnavailable) -> ToolDecision:
        """A tool check just failed at the transport level. Returns a degraded
        ALLOW if the failure mode and every condition permit it, else raises
        `exc` (fail closed)."""
        from .tools import ToolDecision

        if self.failure_mode != FAIL_OPEN_BOUNDED:
            raise exc
        state = self._state()
        if state is not None and state.is_suspended:
            raise exc
        with self._lock:
            last = self._last_decision.get(tool_name)
        if last is not None and last != "ALLOW":
            raise exc
        age = self.contact_age()
        if age is None or age > self.max_stale_seconds:
            raise exc
        _log.warning(
            "Gateway tool check unavailable; failing open for %r (last contact %.0fs ago, "
            "limit %.0fs): %s",
            tool_name,
            age,
            self.max_stale_seconds,
            exc.message,
        )
        return ToolDecision(
            decision="ALLOW",
            reason=DEGRADED_ALLOW_REASON,
            degraded=True,
            degraded_age_seconds=round(age, 1),
        )


def degraded_attributes(decision: Any) -> dict[str, Any] | None:
    """Span attributes marking a tool call that ran in degraded mode, else
    None. Duck-typed on `is True` so a test double's auto-attributes never
    count as degraded."""
    if getattr(decision, "degraded", False) is not True:
        return None
    attrs: dict[str, Any] = {"matimo.degraded_mode": True}
    age = getattr(decision, "degraded_age_seconds", None)
    if isinstance(age, (int, float)):
        attrs["matimo.degraded_cache_age_seconds"] = age
    return attrs
