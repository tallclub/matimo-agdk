from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from matimo_agdk.identity import IdentityCredentials, generate_ecdsa_keypair_pem

BASE_URL = "http://testserver/v1"


@pytest.fixture()
def keypair() -> tuple[str, str]:
    """(private_key_pem, public_key_pem)."""
    return generate_ecdsa_keypair_pem()


@pytest.fixture()
def identity(keypair: tuple[str, str]) -> IdentityCredentials:
    private_pem, _public_pem = keypair
    return IdentityCredentials(
        identity_id="11111111-1111-1111-1111-111111111111",
        identity_token="me-id-testtoken1234567890ab",
        tenant_id="22222222-2222-2222-2222-222222222222",
        display_name="test-agent",
        external_framework="custom",
        private_key_pem=private_pem,
        base_url=BASE_URL,
    )


@pytest.fixture()
def credentials_dir(tmp_path: Path) -> Path:
    d = tmp_path / "creds"
    d.mkdir()
    return d


def future_iso(seconds: float) -> str:
    import datetime as dt

    return (dt.datetime.now(dt.UTC) + dt.timedelta(seconds=seconds)).isoformat()


def make_session_response(
    token: str = "me-sess-abc", ttl_seconds: float = 3600.0
) -> dict[str, Any]:
    return {
        "data": {"sessionToken": token, "expiresAt": future_iso(ttl_seconds), "identityId": "id-1"}
    }


def wait_until(predicate: Any, timeout: float = 2.0, interval: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()
