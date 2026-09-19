"""Matimo AGDK: govern any agent in three lines.

    governor = Governor.from_env()
    governor.start()
    with governor.run("my-run"):
        result = governor.guard(my_tool_fn, name="search")(query="...")

See README.md for the quickstart, the register CLI, and framework
adapters. See docs/SERVER-CONTRACT.md for the exact wire contract this
package implements.
"""

from .config import GatewayConfig
from .exceptions import (
    AgentSuspended,
    AgentSuspendedLocally,
    GatewayError,
    GatewayUnavailable,
    PolicyDenied,
    RateLimited,
    SessionExpired,
    SignatureRejected,
    SigningError,
    TelemetryStale,
    ToolCheckTimeout,
    ToolDenied,
)
from .governor import AsyncGovernor, Governor
from .telemetry import GovernanceState
from .tools import ToolDecision

__version__ = "0.1.0"

__all__ = [
    "Governor",
    "AsyncGovernor",
    "GatewayConfig",
    "GovernanceState",
    "ToolDecision",
    "GatewayError",
    "PolicyDenied",
    "TelemetryStale",
    "AgentSuspended",
    "AgentSuspendedLocally",
    "SessionExpired",
    "SignatureRejected",
    "RateLimited",
    "GatewayUnavailable",
    "ToolDenied",
    "ToolCheckTimeout",
    "SigningError",
    "__version__",
]
