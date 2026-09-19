"""The CLI command functions themselves (review finding 2026-09-18: only
the argument parser was tested)."""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx
import respx

from matimo_agdk import cli, identity as identity_mod
from matimo_agdk.identity import IdentityCredentials, load_credentials, save_credentials

from .conftest import BASE_URL, future_iso


def _args(name: str, **extra: object) -> argparse.Namespace:
    return argparse.Namespace(name=name, gateway_url=BASE_URL, api_key="org-key", **extra)


def _mock_session_and_heartbeat(lifecycle: str = "active") -> None:
    respx.post(f"{BASE_URL}/sessions").mock(
        return_value=httpx.Response(
            201,
            json={"data": {"sessionToken": "tok", "expiresAt": future_iso(3600), "identityId": "x"}},
        )
    )
    respx.post(f"{BASE_URL}/telemetry/batch").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "accepted": 0,
                    "failed": [],
                    "heartbeat": {
                        "lifecycleStatus": lifecycle,
                        "emergencyStop": False,
                        "telemetryMode": "advisory",
                        "telemetryStalenessMinutes": 30,
                        "serverTime": "2026-09-18T00:00:00Z",
                    },
                }
            },
        )
    )


@respx.mock
def test_cmd_status_prints_live_state(
    identity: IdentityCredentials, credentials_dir: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(identity_mod, "DEFAULT_CREDENTIALS_DIR", credentials_dir)
    save_credentials(identity, credentials_dir)
    _mock_session_and_heartbeat(lifecycle="suspended")
    assert cli.cmd_status(_args(identity.display_name)) == 0
    out = capsys.readouterr().out
    assert "lifecycle_status:    suspended" in out
    assert "suspended (locally): True" in out


@respx.mock
def test_cmd_doctor_passes_end_to_end(
    identity: IdentityCredentials, credentials_dir: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(identity_mod, "DEFAULT_CREDENTIALS_DIR", credentials_dir)
    save_credentials(identity, credentials_dir)
    _mock_session_and_heartbeat()
    assert cli.cmd_doctor(_args(identity.display_name)) == 0
    out = capsys.readouterr().out
    assert "[ok] session handshake succeeded" in out
    assert "doctor: all checks passed" in out
    assert identity.private_key_pem not in out


def test_cmd_status_without_identity_fails_cleanly(credentials_dir: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(identity_mod, "DEFAULT_CREDENTIALS_DIR", credentials_dir)
    assert cli.cmd_status(_args("nobody")) == 1
    assert "no identity found" in capsys.readouterr().err


@respx.mock
def test_cmd_rotate_key_rewrites_the_credentials_file(
    identity: IdentityCredentials, credentials_dir: Path, keypair, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(identity_mod, "DEFAULT_CREDENTIALS_DIR", credentials_dir)
    save_credentials(identity, credentials_dir)
    new_private, _ = keypair
    from matimo_agdk.identity import generate_ecdsa_keypair_pem

    new_private, _ = generate_ecdsa_keypair_pem()
    respx.post(f"{BASE_URL}/identities/{identity.identity_id}/rotate-key").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "id": identity.identity_id,
                    "identityToken": identity.identity_token,
                    "tenantId": identity.tenant_id,
                    "displayName": identity.display_name,
                    "externalFramework": "custom",
                    "privateKeyPem": new_private,
                }
            },
        )
    )
    assert cli.cmd_rotate_key(_args(identity.display_name)) == 0
    reloaded = load_credentials(identity.display_name, credentials_dir)
    assert reloaded is not None
    assert reloaded.private_key_pem == new_private
    assert new_private not in capsys.readouterr().out
