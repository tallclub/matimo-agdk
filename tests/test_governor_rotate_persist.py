"""rotate_key() must not write credentials for an identity that was never
persisted (found 2026-09-18: live_check's persist=False identities leaked
into the real ~/.matimo/agents on rotation)."""

from __future__ import annotations

from pathlib import Path

import httpx
import respx

from matimo_agdk.config import GatewayConfig
from matimo_agdk.governor import Governor
from matimo_agdk.identity import generate_ecdsa_keypair_pem
from tests.conftest import BASE_URL


def _identity_json(name: str, private_pem: str) -> dict:
    return {
        "data": {
            "id": "33333333-3333-3333-3333-333333333333",
            "identityToken": "me-id-rotatetoken",
            "tenantId": "22222222-2222-2222-2222-222222222222",
            "displayName": name,
            "externalFramework": "custom",
            "privateKeyPem": private_pem,
        }
    }


def _mock(name: str) -> None:
    first_pem, _ = generate_ecdsa_keypair_pem()
    second_pem, _ = generate_ecdsa_keypair_pem()
    respx.post(f"{BASE_URL}/identities").mock(
        return_value=httpx.Response(201, json=_identity_json(name, first_pem))
    )
    respx.post(url__regex=rf"{BASE_URL}/identities/.*/rotate-key").mock(
        return_value=httpx.Response(200, json=_identity_json(name, second_pem))
    )


@respx.mock
def test_rotate_key_does_not_persist_unpersisted_identity(credentials_dir: Path) -> None:
    _mock("ephemeral")
    gov = Governor(
        GatewayConfig(
            base_url=BASE_URL, api_key="k", agent_name="ephemeral", credentials_dir=credentials_dir
        )
    )
    gov.register(persist=False)
    gov.rotate_key()
    assert list(credentials_dir.iterdir()) == []


@respx.mock
def test_rotate_key_persists_when_registered_with_persist(credentials_dir: Path) -> None:
    _mock("durable")
    gov = Governor(
        GatewayConfig(
            base_url=BASE_URL, api_key="k", agent_name="durable", credentials_dir=credentials_dir
        )
    )
    gov.register(persist=True)
    before = {p.name: p.read_bytes() for p in credentials_dir.iterdir()}
    gov.rotate_key()
    after = {p.name: p.read_bytes() for p in credentials_dir.iterdir()}
    assert set(after) == set(before)
    assert after != before
