from __future__ import annotations

import json
import time

import pytest

from matimo_agdk.identity import (
    IdentityCredentials,
    JWSSigner,
    SigningError,
    generate_ecdsa_keypair_pem,
    load_credentials,
    save_credentials,
    sha256_hex,
    verify_jws,
)


def test_sign_and_verify_round_trip(keypair: tuple[str, str]) -> None:
    private_pem, public_pem = keypair
    signer = JWSSigner(
        private_pem,
        identity_token="me-id-abc123",
        identity_id="agent-1",
        tenant_id="tenant-1",
        external_framework="langchain",
    )
    body = b'{"hello":"world"}'
    jws = signer.sign_request(body_bytes=body)

    claims = verify_jws(jws, public_pem.encode("utf-8"))
    assert claims["iss"] == "matimo-agdk"
    assert claims["sub"] == "agent-1"
    assert claims["tenant_id"] == "tenant-1"
    assert claims["external_framework"] == "langchain"
    assert claims["body_hash"] == sha256_hex(body)
    assert "nonce" in claims
    assert claims["exp"] > claims["iat"]


def test_body_hash_equals_sha256_of_exact_bytes(keypair: tuple[str, str]) -> None:
    private_pem, public_pem = keypair
    signer = JWSSigner(
        private_pem, identity_token="kid-1", identity_id="agent-1", tenant_id="tenant-1"
    )
    body = b"{}"
    jws = signer.sign_request(body_bytes=body)
    claims = verify_jws(jws, public_pem.encode("utf-8"))
    assert claims["body_hash"] == sha256_hex(b"{}")
    # A different body must produce a different hash and therefore a
    # signature that would fail verification if that hash were reused.
    assert sha256_hex(b"{}") != sha256_hex(b'{"x":1}')


def test_header_contains_kid_and_alg(keypair: tuple[str, str]) -> None:
    private_pem, _ = keypair
    signer = JWSSigner(
        private_pem, identity_token="my-kid", identity_id="agent-1", tenant_id="tenant-1"
    )
    jws = signer.sign_request(body_bytes=b"{}")
    header_b64 = jws.split(".")[0]
    import base64

    padded = header_b64 + "=" * (-len(header_b64) % 4)
    header = json.loads(base64.urlsafe_b64decode(padded))
    assert header == {"alg": "ES256", "kid": "my-kid"}


def test_no_external_framework_claim_when_none(keypair: tuple[str, str]) -> None:
    private_pem, public_pem = keypair
    signer = JWSSigner(
        private_pem, identity_token="kid-1", identity_id="agent-1", tenant_id="tenant-1"
    )
    jws = signer.sign_request(body_bytes=b"{}")
    claims = verify_jws(jws, public_pem.encode("utf-8"))
    assert "external_framework" not in claims


def test_verify_fails_on_tampered_payload(keypair: tuple[str, str]) -> None:
    private_pem, public_pem = keypair
    signer = JWSSigner(
        private_pem, identity_token="kid-1", identity_id="agent-1", tenant_id="tenant-1"
    )
    jws = signer.sign_request(body_bytes=b"{}")
    header_b64, payload_b64, sig_b64 = jws.split(".")
    tampered = f"{header_b64}.{payload_b64}x.{sig_b64}"
    with pytest.raises(Exception):  # noqa: B017 - either SigningError or a crypto InvalidSignature
        verify_jws(tampered, public_pem.encode("utf-8"))


def test_verify_fails_with_wrong_public_key(keypair: tuple[str, str]) -> None:
    private_pem, _public_pem = keypair
    other_private, other_public = generate_ecdsa_keypair_pem()
    signer = JWSSigner(
        private_pem, identity_token="kid-1", identity_id="agent-1", tenant_id="tenant-1"
    )
    jws = signer.sign_request(body_bytes=b"{}")
    with pytest.raises(Exception):  # noqa: B017
        verify_jws(jws, other_public.encode("utf-8"))


def test_sign_request_default_ttl_is_60_seconds(keypair: tuple[str, str]) -> None:
    private_pem, public_pem = keypair
    signer = JWSSigner(
        private_pem, identity_token="kid-1", identity_id="agent-1", tenant_id="tenant-1"
    )
    before = int(time.time())
    jws = signer.sign_request(body_bytes=b"{}")
    claims = verify_jws(jws, public_pem.encode("utf-8"))
    assert claims["exp"] - claims["iat"] == 60
    assert claims["iat"] >= before


def test_malformed_private_key_rejected() -> None:
    with pytest.raises(SigningError):
        JWSSigner("not a real pem", identity_token="k", identity_id="a", tenant_id="t")


def test_save_and_load_credentials_round_trip(
    identity: IdentityCredentials, credentials_dir
) -> None:
    meta_path, key_path = save_credentials(identity, credentials_dir)
    assert meta_path.exists()
    assert key_path.exists()

    loaded = load_credentials(identity.display_name, credentials_dir)
    assert loaded is not None
    assert loaded.identity_id == identity.identity_id
    assert loaded.identity_token == identity.identity_token
    assert loaded.tenant_id == identity.tenant_id
    assert loaded.private_key_pem == identity.private_key_pem

    # The metadata file must never contain the private key.
    meta_contents = meta_path.read_text()
    assert "PRIVATE KEY" not in meta_contents


def test_save_credentials_restricts_permissions(
    identity: IdentityCredentials, credentials_dir
) -> None:
    import os
    import stat

    _meta_path, key_path = save_credentials(identity, credentials_dir)
    if os.name == "posix":
        mode = stat.S_IMODE(key_path.stat().st_mode)
        assert mode == 0o600


def test_load_credentials_missing_returns_none(credentials_dir) -> None:
    assert load_credentials("does-not-exist", credentials_dir) is None
