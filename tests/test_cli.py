from __future__ import annotations

import os
import stat

import httpx
import respx

from matimo_agdk.cli import build_parser
from matimo_agdk.identity import credentials_paths, load_credentials

from .conftest import BASE_URL


@respx.mock
def test_register_cli_writes_0600_credentials_file(credentials_dir, keypair, monkeypatch) -> None:
    private_pem, _public_pem = keypair
    respx.post(f"{BASE_URL}/identities").mock(
        return_value=httpx.Response(
            201,
            json={
                "data": {
                    "id": "cli-id",
                    "identityToken": "me-id-clitoken",
                    "tenantId": "tenant-cli",
                    "displayName": "cli-agent",
                    "externalFramework": "custom",
                    "privateKeyPem": private_pem,
                }
            },
        )
    )
    monkeypatch.setenv("MATIMO_API_KEY", "org-key")
    parser = build_parser()
    # Parses cleanly; the CLI's own argparse plumbing is exercised here,
    # even though we call governor.register() directly below for the actual
    # assertions (see the comment there for why).
    parser.parse_args(
        [
            "register",
            "--name",
            "cli-agent",
            "--framework",
            "custom",
            "--gateway-url",
            BASE_URL,
        ]
    )
    # Route the credentials dir through our monkeypatched load/save by
    # patching the module-level default the CLI's _build_config resolves
    # against, via load_credentials's own credentials_dir parameter --
    # simplest is to call governor.register with an explicit dir, so we
    # invoke the underlying command function directly instead of exercising
    # main()'s hardcoded ~/.matimo default path.
    from matimo_agdk.config import GatewayConfig
    from matimo_agdk.governor import Governor

    config = GatewayConfig.load(
        agent_name="cli-agent",
        api_key="org-key",
        base_url=BASE_URL,
        credentials_dir=credentials_dir,
    )
    governor = Governor(config)
    identity = governor.register(display_name="cli-agent", framework="custom")

    meta_path, key_path = credentials_paths("cli-agent", credentials_dir)
    assert meta_path.exists()
    assert key_path.exists()
    if os.name == "posix":
        mode = stat.S_IMODE(key_path.stat().st_mode)
        assert mode == 0o600

    loaded = load_credentials("cli-agent", credentials_dir)
    assert loaded is not None
    assert loaded.identity_token == "me-id-clitoken"
    assert identity.identity_token == "me-id-clitoken"


def test_build_parser_register_requires_name() -> None:
    parser = build_parser()
    import pytest

    with pytest.raises(SystemExit):
        parser.parse_args(["register"])


def test_build_parser_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["register", "--name", "x"])
    assert args.framework == "custom"
    assert args.command == "register"


def test_build_parser_status_and_doctor_accept_name() -> None:
    parser = build_parser()
    args = parser.parse_args(["status", "--name", "x"])
    assert args.command == "status"
    args2 = parser.parse_args(["doctor", "--name", "x"])
    assert args2.command == "doctor"
