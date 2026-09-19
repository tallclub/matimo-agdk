from __future__ import annotations

from matimo_agdk.config import GatewayConfig
from matimo_agdk.identity import IdentityCredentials, save_credentials


def test_load_defaults_when_nothing_set(monkeypatch, credentials_dir) -> None:
    for var in (
        "MATIMO_GATEWAY_URL",
        "MATIMO_API_KEY",
        "MATIMO_IDENTITY_TOKEN",
        "MATIMO_IDENTITY_ID",
        "MATIMO_TENANT_ID",
        "MATIMO_PRIVATE_KEY",
        "MATIMO_PRIVATE_KEY_FILE",
        "MATIMO_AGENT_NAME",
        "MATIMO_FRAMEWORK",
    ):
        monkeypatch.delenv(var, raising=False)
    config = GatewayConfig.load(credentials_dir=credentials_dir)
    assert config.base_url == "http://localhost:8000/v1"
    assert config.api_key == ""
    assert config.agent_name == "matimo-agent"
    assert config.framework == "custom"


def test_env_overrides_file(monkeypatch, identity: IdentityCredentials, credentials_dir) -> None:
    save_credentials(identity, credentials_dir)
    monkeypatch.setenv("MATIMO_API_KEY", "env-key")
    monkeypatch.setenv("MATIMO_GATEWAY_URL", "http://env-gateway/v1")

    config = GatewayConfig.load(agent_name=identity.display_name, credentials_dir=credentials_dir)
    # File-sourced fields survive.
    assert config.identity_token == identity.identity_token
    assert config.private_key_pem == identity.private_key_pem
    # Env overrides the file's base_url and sets api_key (file never carries one).
    assert config.base_url == "http://env-gateway/v1"
    assert config.api_key == "env-key"


def test_kwargs_override_env(monkeypatch, identity: IdentityCredentials, credentials_dir) -> None:
    save_credentials(identity, credentials_dir)
    monkeypatch.setenv("MATIMO_API_KEY", "env-key")

    config = GatewayConfig.load(
        agent_name=identity.display_name, credentials_dir=credentials_dir, api_key="kwarg-key"
    )
    assert config.api_key == "kwarg-key"


def test_kwargs_override_file(identity: IdentityCredentials, credentials_dir) -> None:
    save_credentials(identity, credentials_dir)
    config = GatewayConfig.load(
        agent_name=identity.display_name,
        credentials_dir=credentials_dir,
        identity_token="explicit-token-override",
    )
    assert config.identity_token == "explicit-token-override"


def test_has_identity(identity: IdentityCredentials, credentials_dir) -> None:
    save_credentials(identity, credentials_dir)
    config = GatewayConfig.load(agent_name=identity.display_name, credentials_dir=credentials_dir)
    assert config.has_identity() is True

    empty_config = GatewayConfig()
    assert empty_config.has_identity() is False


def test_resolved_heartbeat_interval_clamped() -> None:
    config = GatewayConfig()
    # staleness/3 clamped to [15, 300]
    assert config.resolved_heartbeat_interval(telemetry_staleness_minutes=30.0) == 300.0
    assert config.resolved_heartbeat_interval(telemetry_staleness_minutes=1.0) == 20.0
    assert config.resolved_heartbeat_interval(telemetry_staleness_minutes=0.1) == 15.0


def test_resolved_heartbeat_interval_explicit_override() -> None:
    config = GatewayConfig(heartbeat_interval=42.0)
    assert config.resolved_heartbeat_interval(telemetry_staleness_minutes=30.0) == 42.0


def test_base_url_trailing_slash_stripped() -> None:
    config = GatewayConfig(base_url="http://localhost:8000/v1/")
    assert config.base_url == "http://localhost:8000/v1"
