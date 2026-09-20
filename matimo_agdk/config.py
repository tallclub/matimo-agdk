"""GatewayConfig: the one settings object a Governor is built from.

Load precedence, highest wins: explicit kwargs > environment variables > a
persisted credentials file written by `matimo-agdk register` (see
matimo_agdk.identity.save_credentials). Building a GatewayConfig never
talks to the network -- it is pure local state assembly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, field_validator

from .identity import load_credentials

Framework = Literal["langchain", "google-adk", "crewai", "autogen", "custom"]

DEFAULT_BASE_URL = "http://localhost:8000/v1"


class GatewayConfig(BaseModel):
    """Configuration for one Governor instance."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_url: str = DEFAULT_BASE_URL
    api_key: str = ""
    identity_token: str | None = None
    identity_id: str | None = None
    tenant_id: str | None = None
    private_key_pem: str | None = None
    agent_name: str = "matimo-agent"
    framework: Framework = "custom"

    connect_timeout: float = 10.0
    read_timeout: float = 30.0

    telemetry_flush_interval: float = 5.0
    telemetry_batch_size: int = 50
    telemetry_queue_max: int = 2000
    # None means "compute from the server-reported staleness window at
    # runtime" -- see resolved_heartbeat_interval().
    heartbeat_interval: float | None = None

    fail_open_telemetry: bool = True
    signing_enabled: bool = True

    credentials_dir: Path | None = None

    @field_validator("base_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/") or v

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
            try:
                values["private_key_pem"] = Path(env["MATIMO_PRIVATE_KEY_FILE"]).read_text()
            except OSError:
                pass
        if env.get("MATIMO_AGENT_NAME"):
            values["agent_name"] = env["MATIMO_AGENT_NAME"]
        if env.get("MATIMO_FRAMEWORK"):
            values["framework"] = env["MATIMO_FRAMEWORK"]

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
