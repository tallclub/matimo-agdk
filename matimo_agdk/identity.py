"""Identity persistence and the ES256 JWS envelope every Gateway check-in
call must carry.

See docs/SERVER-CONTRACT.md section 5.1 for the exact wire envelope this
module implements: header {alg: ES256, kid: identityToken}, claims
{iss, sub, tenant_id, external_framework?, nonce, iat, exp, body_hash}.

Two artifacts are kept deliberately distinct, per the contract's own
framing (section 3.1, section 5.1):

- `identity_token`: an opaque bearer string (the `kid` claim, and the
  X-Matimo-Agent-Identity-Token header value).
- `identity_id`: the identity's UUID (the `sub` claim,
  matimo_agent_identities.id).

The server generates the ECDSA keypair at registration and at key
rotation -- this module never mints a keypair as part of those flows. A
key-generation helper is provided anyway (for local testing / anything
bootstrapping a keypair outside the register flow), clearly separated from
the register/rotate paths.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import (
    decode_dss_signature,
    encode_dss_signature,
)

from .exceptions import SigningError

ISSUER = "matimo-agdk"
DEFAULT_JWS_TTL_SECONDS = 60  # matches the +-60s clock-skew window Gateway enforces
_EC_COORDINATE_SIZE = 32  # P-256

DEFAULT_CREDENTIALS_DIR = Path.home() / ".matimo" / "agents"


# ---------------------------------------------------------------------------
# base64url + DER<->raw signature helpers
# ---------------------------------------------------------------------------


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def _der_to_raw(der_sig: bytes, size: int = _EC_COORDINATE_SIZE) -> bytes:
    r, s = decode_dss_signature(der_sig)
    return r.to_bytes(size, "big") + s.to_bytes(size, "big")


def _raw_to_der(raw_sig: bytes, size: int = _EC_COORDINATE_SIZE) -> bytes:
    if len(raw_sig) != size * 2:
        raise SigningError(f"expected a {size * 2}-byte raw ES256 signature, got {len(raw_sig)}")
    r = int.from_bytes(raw_sig[:size], "big")
    s = int.from_bytes(raw_sig[size:], "big")
    return encode_dss_signature(r, s)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Identity credentials
# ---------------------------------------------------------------------------


@dataclass
class IdentityCredentials:
    """Everything AGDK needs to act as one registered Gateway identity."""

    identity_id: str
    identity_token: str
    tenant_id: str
    display_name: str
    external_framework: str | None
    private_key_pem: str
    base_url: str | None = None
    public_key_fingerprint: str | None = None
    created_at: str | None = None

    def to_metadata_dict(self) -> dict[str, Any]:
        """Non-secret fields only -- never includes private_key_pem. This
        is what gets written to the plain-JSON half of the credentials
        file; the PEM goes in its own file, see save_credentials()."""
        return {
            "identity_id": self.identity_id,
            "identity_token": self.identity_token,
            "tenant_id": self.tenant_id,
            "display_name": self.display_name,
            "external_framework": self.external_framework,
            "base_url": self.base_url,
            "public_key_fingerprint": self.public_key_fingerprint,
            "created_at": self.created_at,
        }


def _safe_filename(name: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in "-_." else "_" for c in name.strip())
    return cleaned or "agent"


def credentials_paths(agent_name: str, credentials_dir: Path | None = None) -> tuple[Path, Path]:
    base = credentials_dir or DEFAULT_CREDENTIALS_DIR
    safe_name = _safe_filename(agent_name)
    return base / f"{safe_name}.json", base / f"{safe_name}.pem"


def _write_private(path: Path, text: str) -> None:
    """Creates (or truncates) `path` with mode 0600 from the first byte, so
    there is no window at the process umask between write and chmod."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)


def _restrict_permissions(path: Path) -> None:
    """Best-effort 0600. Real on POSIX; on Windows os.chmod does not enforce
    genuine ACL restriction -- the same documented gap as the UAF LangChain
    reference client (tests/external-agents/langchain-agent/agent.py)."""
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass


def save_credentials(
    creds: IdentityCredentials, credentials_dir: Path | None = None
) -> tuple[Path, Path]:
    """Persists identity metadata and the private key to two separate
    files under ~/.matimo/agents/ by default (or credentials_dir). The
    private key is genuine key material, so it is kept apart from the
    rest of the (non-secret) identity metadata, with restrictive
    permissions applied to both files.

    There is no server-side retrieval of a lost private key -- losing this
    file means registering fresh or, if the old identity + org API key are
    both still known, rotating the key (Governor.rotate_key()).
    """
    meta_path, key_path = credentials_paths(creds.display_name, credentials_dir)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    _write_private(meta_path, json.dumps(creds.to_metadata_dict(), indent=2))
    _write_private(key_path, creds.private_key_pem)
    _restrict_permissions(meta_path)
    _restrict_permissions(key_path)
    return meta_path, key_path


def load_credentials(
    agent_name: str, credentials_dir: Path | None = None
) -> IdentityCredentials | None:
    meta_path, key_path = credentials_paths(agent_name, credentials_dir)
    if not meta_path.exists() or not key_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
        private_key_pem = key_path.read_text()
    except (OSError, json.JSONDecodeError):
        return None
    required = ("identity_id", "identity_token", "tenant_id", "display_name")
    if not all(meta.get(field) for field in required):
        return None
    return IdentityCredentials(
        identity_id=meta["identity_id"],
        identity_token=meta["identity_token"],
        tenant_id=meta["tenant_id"],
        display_name=meta["display_name"],
        external_framework=meta.get("external_framework"),
        private_key_pem=private_key_pem,
        base_url=meta.get("base_url"),
        public_key_fingerprint=meta.get("public_key_fingerprint"),
        created_at=meta.get("created_at"),
    )


# ---------------------------------------------------------------------------
# ES256 JWS signer/verifier
# ---------------------------------------------------------------------------


class JWSSigner:
    """Signs the compact ES256 JWS envelope every Gateway check-in call
    needs: the mandatory session handshake, /v1/tools/*, and (whenever
    signing is enabled) /v1/chat/completions and /v1/messages.
    """

    def __init__(
        self,
        private_key_pem: str,
        *,
        identity_token: str,
        identity_id: str,
        tenant_id: str,
        external_framework: str | None = None,
    ) -> None:
        try:
            key = serialization.load_pem_private_key(private_key_pem.encode("utf-8"), password=None)
        except (ValueError, TypeError) as exc:
            raise SigningError(f"malformed private key PEM: {exc}") from exc
        if not isinstance(key, ec.EllipticCurvePrivateKey):
            raise SigningError("private key is not an EC private key")
        self._private_key = key
        self.identity_token = identity_token
        self.default_identity_id = identity_id
        self.default_tenant_id = tenant_id
        self.default_external_framework = external_framework

    @classmethod
    def from_credentials(cls, creds: IdentityCredentials) -> JWSSigner:
        return cls(
            creds.private_key_pem,
            identity_token=creds.identity_token,
            identity_id=creds.identity_id,
            tenant_id=creds.tenant_id,
            external_framework=creds.external_framework,
        )

    def public_key_pem(self) -> bytes:
        return self._private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

    def sign_request(
        self,
        *,
        body_bytes: bytes,
        identity_id: str | None = None,
        tenant_id: str | None = None,
        external_framework: str | None = None,
        ttl_seconds: int = DEFAULT_JWS_TTL_SECONDS,
        nonce: str | None = None,
    ) -> str:
        """Signs `body_bytes` (the EXACT bytes about to be sent over the
        wire -- never a re-serialization of them) and returns a compact
        JWS string for the Matimo-Agent-Signature header."""
        now = int(time.time())
        claims: dict[str, Any] = {
            "iss": ISSUER,
            "sub": identity_id or self.default_identity_id,
            "tenant_id": tenant_id or self.default_tenant_id,
            "nonce": nonce or uuid.uuid4().hex,
            "iat": now,
            "exp": now + ttl_seconds,
            "body_hash": sha256_hex(body_bytes),
        }
        framework = (
            external_framework
            if external_framework is not None
            else self.default_external_framework
        )
        if framework:
            claims["external_framework"] = framework
        return self._encode(claims)

    def _encode(self, claims: Mapping[str, Any]) -> str:
        header = {"alg": "ES256", "kid": self.identity_token}
        header_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
        payload_b64 = _b64url_encode(
            json.dumps(dict(claims), separators=(",", ":")).encode("utf-8")
        )
        signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
        der_sig = self._private_key.sign(signing_input, ec.ECDSA(hashes.SHA256()))
        raw_sig = _der_to_raw(der_sig)
        return f"{header_b64}.{payload_b64}.{_b64url_encode(raw_sig)}"


def verify_jws(compact_jws: str, public_key_pem: bytes) -> dict[str, Any]:
    """Verifies a compact ES256 JWS against a public key PEM and returns
    the decoded claims. Used by matimo_agdk's own test suite to prove
    signatures round-trip end to end -- Gateway performs its own
    independent server-side verification and never calls this."""
    try:
        header_b64, payload_b64, sig_b64 = compact_jws.split(".")
    except ValueError as exc:
        raise SigningError("malformed compact JWS: expected exactly 3 dot-separated parts") from exc
    header = json.loads(_b64url_decode(header_b64))
    if header.get("alg") != "ES256":
        raise SigningError(f"unexpected alg {header.get('alg')!r}, expected ES256")
    public_key = serialization.load_pem_public_key(public_key_pem)
    if not isinstance(public_key, ec.EllipticCurvePublicKey):
        raise SigningError("public key is not an EC public key")
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    der_sig = _raw_to_der(_b64url_decode(sig_b64))
    public_key.verify(der_sig, signing_input, ec.ECDSA(hashes.SHA256()))
    return json.loads(_b64url_decode(payload_b64))


def generate_ecdsa_keypair_pem() -> tuple[str, str]:
    """Generates a fresh P-256 keypair locally (PKCS8 private PEM, SPKI
    public PEM).

    NOT used by registration or rotation -- in both of those flows the
    server generates the keypair and returns the private key exactly once
    (docs/SERVER-CONTRACT.md section 3.1, section 3.4). This helper exists
    for tests and for anything that needs a standalone keypair outside
    those two flows.
    """
    key = ec.generate_private_key(ec.SECP256R1())
    private_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")
    public_pem = (
        key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("utf-8")
    )
    return private_pem, public_pem
