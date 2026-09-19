from __future__ import annotations

from pathlib import Path

from matimo_agdk.config import GatewayConfig
from matimo_agdk.governor import AsyncGovernor, Governor
from matimo_agdk.identity import IdentityCredentials

from ..conftest import BASE_URL


def bound_config(identity: IdentityCredentials, credentials_dir: Path) -> GatewayConfig:
    return GatewayConfig(
        base_url=BASE_URL,
        api_key="org-key",
        identity_token=identity.identity_token,
        identity_id=identity.identity_id,
        tenant_id=identity.tenant_id,
        private_key_pem=identity.private_key_pem,
        agent_name=identity.display_name,
        framework="custom",
        credentials_dir=credentials_dir,
        telemetry_flush_interval=9999,
        heartbeat_interval=9999,
    )


def bound_governor(identity: IdentityCredentials, credentials_dir: Path) -> Governor:
    return Governor(bound_config(identity, credentials_dir))


def bound_async_governor(identity: IdentityCredentials, credentials_dir: Path) -> AsyncGovernor:
    return AsyncGovernor(bound_config(identity, credentials_dir))
