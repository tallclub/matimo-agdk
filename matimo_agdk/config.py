"""GatewayConfig: the one settings object a Governor is built from.

Load precedence, highest wins: explicit kwargs > environment variables > a
persisted credentials file written by `matimo-agdk register` (see
matimo_agdk.identity.save_credentials). Building a GatewayConfig never
talks to the network -- it is pure local state assembly.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .identity import load_credentials

Framework = Literal["langchain", "google-adk", "crewai", "autogen", "custom"]
ToolCheckFailureMode = Literal["fail_closed", "fail_open_bounded"]

# BUILD-PLAN D19: a fail-open posture never rides on state more than 5 minutes
# old, however the customer configures it.
MAX_FAIL_OPEN_STALE_SECONDS = 300.0

DEFAULT_BASE_URL = "http://localhost:8000/v1"

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


class GatewayConfig(BaseModel):
    """Configuration for one Governor instance."""

    # extra="forbid": a misspelled option (`telemetry_batchsize=10`) must fail
    # loudly, not be silently ignored while the default stays in force.
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    base_url: str = DEFAULT_BASE_URL
    # repr=False on both secrets: a config object is routinely logged.
    api_key: str = Field(default="", repr=False)
    identity_token: str | None = Field(default=None, repr=False)
    identity_id: str | None = None
    tenant_id: str | None = None
    private_key_pem: str | None = Field(default=None, repr=False)
    agent_name: str = "matimo-agent"
    framework: Framework = "custom"

    connect_timeout: float = Field(default=10.0, gt=0)
    read_timeout: float = Field(default=30.0, gt=0)

    telemetry_flush_interval: float = Field(default=5.0, gt=0)
    telemetry_batch_size: int = Field(default=50, ge=1)
    telemetry_queue_max: int = Field(default=2000, ge=1)
    # None means "compute from the server-reported staleness window at
    # runtime" -- see resolved_heartbeat_interval().
    heartbeat_interval: float | None = Field(default=None, gt=0)

    fail_open_telemetry: bool = True
    signing_enabled: bool = True

    # What a tool call does when Gateway cannot answer its check at all (a
    # connection error, timeout or 5xx; never a 4xx and never a DENY). See
    # matimo_agdk._outage. fail_closed is the safe default.
    tool_check_failure_mode: ToolCheckFailureMode = "fail_closed"
    # fail_open_bounded only lets a call through if Gateway was last heard
    # from (a successful check or a telemetry heartbeat) within this many
    # seconds. Hard-capped at BUILD-PLAN D19's 5 minutes.
    fail_open_max_stale_seconds: float = Field(default=MAX_FAIL_OPEN_STALE_SECONDS, gt=0)
    # Circuit breaker: after this many consecutive transport-level check
    # failures the SDK stops trying for `tool_check_breaker_cooldown` seconds.
    tool_check_breaker_threshold: int = Field(default=3, ge=1)
    tool_check_breaker_cooldown: float = Field(default=30.0, gt=0)

    # Send each tool's result (redacted, truncated to 500 characters) on its
    # tool span. Off by default: a result can hold anything the tool read.
    capture_tool_results: bool = False

    credentials_dir: Path | None = None

    @field_validator("base_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/") or v

    @field_validator("fail_open_max_stale_seconds")
    @classmethod
    def _cap_fail_open_staleness(cls, v: float) -> float:
        if v > MAX_FAIL_OPEN_STALE_SECONDS:
            raise ValueError(
                f"fail_open_max_stale_seconds is hard-capped at {MAX_FAIL_OPEN_STALE_SECONDS:g} "
                f"(5 minutes), got {v:g}: a fail-open tool check must never rest on "
                "older Gateway state than that"
            )
        return v

    @model_validator(mode="after")
    def _warn_on_cleartext_credentials(self) -> GatewayConfig:
        parts = urlsplit(self.base_url)
        if self.api_key and parts.scheme == "http" and parts.hostname not in _LOOPBACK_HOSTS:
            warnings.warn(
                f"base_url {self.base_url!r} is plain http: the org API key and session "
                "tokens would cross the network unencrypted. Use https outside a trusted "
                "network.",
                stacklevel=2,
            )
        return self

    def http_timeout(self, library: Any = httpx) -> Any:
        """connect_timeout for connection setup and pool waits,
        read_timeout for reads and writes. `library` is the HTTP module the
        timeout is for (`httpx` by default, or `httpx2`): the two libraries'
        `Timeout` classes are unrelated."""
        return library.Timeout(
            self.read_timeout, connect=self.connect_timeout, pool=self.connect_timeout
        )

    def has_identity(self) -> bool:
        return bool(
            self.identity_token and self.private_key_pem and self.identity_id and self.tenant_id
        )

    def resolved_heartbeat_interval(self, telemetry_staleness_minutes: float = 30.0) -> float:
        """Sizes the heartbeat interval to a fraction of the server-reported
        staleness window when not explicitly configured: staleness / 3,
        clamped to [15s, 300s]."""
        if self.heartbeat_interval is not None:
            return self.heartbeat_interval
        seconds = (telemetry_staleness_minutes * 60.0) / 3.0
        return max(15.0, min(300.0, seconds))

    @classmethod
    def load(
        cls,
        *,
        agent_name: str | None = None,
        credentials_dir: Path | None = None,
        **overrides: Any,
    ) -> GatewayConfig:
        values: dict[str, Any] = {}

        name_hint = (
            agent_name
            or overrides.get("agent_name")
            or os.environ.get("MATIMO_AGENT_NAME")
            or "matimo-agent"
        )

        # 1. Credentials file (lowest precedence).
        creds = load_credentials(str(name_hint), credentials_dir)
        if creds is not None:
            values.update(
                identity_token=creds.identity_token,
                identity_id=creds.identity_id,
                tenant_id=creds.tenant_id,
                agent_name=creds.display_name,
                private_key_pem=creds.private_key_pem,
            )
            if creds.external_framework:
                values["framework"] = creds.external_framework
            if creds.base_url:
                values["base_url"] = creds.base_url

        # 2. Environment variables (override the file).
        env = os.environ
        if env.get("MATIMO_GATEWAY_URL"):
            values["base_url"] = env["MATIMO_GATEWAY_URL"]
        if env.get("MATIMO_API_KEY"):
            values["api_key"] = env["MATIMO_API_KEY"]
        if env.get("MATIMO_IDENTITY_TOKEN"):
            values["identity_token"] = env["MATIMO_IDENTITY_TOKEN"]
        if env.get("MATIMO_IDENTITY_ID"):
            values["identity_id"] = env["MATIMO_IDENTITY_ID"]
        if env.get("MATIMO_TENANT_ID"):
            values["tenant_id"] = env["MATIMO_TENANT_ID"]
        if env.get("MATIMO_PRIVATE_KEY"):
            values["private_key_pem"] = env["MATIMO_PRIVATE_KEY"]
        elif env.get("MATIMO_PRIVATE_KEY_FILE"):
            key_file = env["MATIMO_PRIVATE_KEY_FILE"]
            try:
                values["private_key_pem"] = Path(key_file).read_text(encoding="utf-8")
            except OSError as exc:
                # Fail here, with the path, rather than later as an opaque
                # "Governor has no identity".
                raise ValueError(
                    f"MATIMO_PRIVATE_KEY_FILE={key_file!r} is unreadable: {exc}"
                ) from exc
        if env.get("MATIMO_AGENT_NAME"):
            values["agent_name"] = env["MATIMO_AGENT_NAME"]
        if env.get("MATIMO_FRAMEWORK"):
            values["framework"] = env["MATIMO_FRAMEWORK"]
        if env.get("MATIMO_TOOL_CHECK_FAILURE_MODE"):
            values["tool_check_failure_mode"] = env["MATIMO_TOOL_CHECK_FAILURE_MODE"]
        if env.get("MATIMO_FAIL_OPEN_MAX_STALE_SECONDS"):
            values["fail_open_max_stale_seconds"] = env["MATIMO_FAIL_OPEN_MAX_STALE_SECONDS"]

        # 3. Explicit kwargs (highest precedence).
        if agent_name is not None:
            values["agent_name"] = agent_name
        if credentials_dir is not None:
            values["credentials_dir"] = credentials_dir
        for key, value in overrides.items():
            if value is not None:
                values[key] = value

        values = {k: v for k, v in values.items() if v is not None}
        return cls(**values)
