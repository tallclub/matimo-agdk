"""The `httpx2` client path (anthropic >= 1.6 rejects an `httpx.Client`).

Found by `scripts/live_check.py`: `anthropic.Anthropic(http_client=gov.httpx_client())`
raised `TypeError: ... uses httpx2. Use httpx2.Client instead`, so a live, signed
client was impossible for Anthropic. These tests skip when `httpx2` is not
installed (it arrives with anthropic >= 1.6).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from matimo_agdk import _compat
from matimo_agdk._compat import sdk_requires_httpx2
from matimo_agdk.identity import IdentityCredentials, verify_jws

from .adapters.conftest import bound_async_governor, bound_governor
from .conftest import BASE_URL, future_iso

httpx2 = pytest.importorskip("httpx2")


def _sessions(*tokens: str) -> None:
    route = respx.post(f"{BASE_URL}/sessions")
    route.side_effect = [
        httpx.Response(
            201,
            json={"data": {"sessionToken": t, "expiresAt": future_iso(3600), "identityId": "x"}},
        )
        for t in tokens
    ]


def _patch_transport(monkeypatch: pytest.MonkeyPatch, handler: Any) -> None:
    monkeypatch.setattr(httpx2, "HTTPTransport", lambda: httpx2.MockTransport(handler))
    monkeypatch.setattr(httpx2, "AsyncHTTPTransport", lambda: httpx2.MockTransport(handler))


# ---------------------------------------------------------------------------
# _compat.sdk_requires_httpx2
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("requires", "expected"),
    [
        (["httpx2<3,>=2.0.0", "pydantic<3,>=1.9.0"], True),
        (["pydantic>=2", "HTTPX2>=2"], True),
        (["httpx<1,>=0.23.0", "httpx2<3,>=2.7.0; extra == 'httpx2'"], False),  # optional extra only
        (["httpx<1,>=0.23.0"], False),
        (["httpx2-extras>=1"], False),  # a different distribution
        ([], False),
        (None, False),
    ],
)
def test_sdk_requires_httpx2_reads_declared_requirements(
    monkeypatch: pytest.MonkeyPatch, requires: list[str] | None, expected: bool
) -> None:
    monkeypatch.setattr(_compat.metadata, "requires", lambda _dist: requires)
    assert sdk_requires_httpx2("anything") is expected


def test_sdk_requires_httpx2_is_false_for_an_uninstalled_sdk() -> None:
    assert sdk_requires_httpx2("definitely-not-an-installed-distribution-xyz") is False


# ---------------------------------------------------------------------------
# httpx2_client(): live headers, signing, transparent re-handshake
# ---------------------------------------------------------------------------


@respx.mock
def test_httpx2_client_attaches_session_run_id_and_a_valid_signature(
    identity: IdentityCredentials,
    credentials_dir: Path,
    keypair: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _sessions("sess-1")
    seen: list[Any] = []

    def handler(request: Any) -> Any:
        seen.append(request)
        return httpx2.Response(200, json={"ok": True})

    _patch_transport(monkeypatch, handler)
    governor = bound_governor(identity, credentials_dir)
    client = governor.httpx2_client()
    assert isinstance(client, httpx2.Client)
    assert isinstance(client.timeout, httpx2.Timeout)

    body = b'{"model":"m"}'
    with governor.run("r") as run_id:
        assert client.post("/chat/completions", content=body).status_code == 200

    request = seen[0]
    assert request.headers["X-Matimo-Session-Token"] == "sess-1"
    assert request.headers["X-Matimo-Run-Id"] == run_id
    assert request.headers["Authorization"] == "Bearer org-key"
    claims = verify_jws(request.headers["Matimo-Agent-Signature"], keypair[1].encode())
    assert claims["body_hash"] == hashlib.sha256(body).hexdigest()
    assert claims["sub"] == identity.identity_id


@respx.mock
def test_httpx2_client_rehandshakes_once_on_session_expired_and_resigns(
    identity: IdentityCredentials,
    credentials_dir: Path,
    keypair: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _sessions("sess-old", "sess-new")
    seen: list[Any] = []

    def handler(request: Any) -> Any:
        seen.append(request.headers.copy())  # the retry re-sends (and mutates) this same Request
        if len(seen) == 1:
            return httpx2.Response(401, json={"error": "session_expired"})
        return httpx2.Response(200, json={"ok": True})

    _patch_transport(monkeypatch, handler)
    governor = bound_governor(identity, credentials_dir)
    response = governor.httpx2_client().post("/chat/completions", content=b'{"a":1}')

    assert response.status_code == 200
    assert [h["X-Matimo-Session-Token"] for h in seen] == ["sess-old", "sess-new"]
    # The retry carries its own fresh signature (new nonce), not the first request's.
    first, second = (verify_jws(h["Matimo-Agent-Signature"], keypair[1].encode()) for h in seen)
    assert first["nonce"] != second["nonce"]


@respx.mock
def test_httpx2_client_does_not_retry_other_401s(
    identity: IdentityCredentials, credentials_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _sessions("sess-1", "sess-2")
    calls: list[int] = []

    def handler(request: Any) -> Any:
        calls.append(1)
        return httpx2.Response(401, json={"error": "invalid_api_key"})

    _patch_transport(monkeypatch, handler)
    governor = bound_governor(identity, credentials_dir)
    assert governor.httpx2_client().post("/chat/completions", content=b"{}").status_code == 401
    assert len(calls) == 1


@respx.mock
async def test_httpx2_async_client_rehandshakes_once_on_session_expired(
    identity: IdentityCredentials,
    credentials_dir: Path,
    keypair: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _sessions("sess-old", "sess-new")
    seen: list[Any] = []

    def handler(request: Any) -> Any:
        seen.append(request.headers.copy())  # the retry re-sends (and mutates) this same Request
        if len(seen) == 1:
            return httpx2.Response(401, json={"error": "session_expired"})
        return httpx2.Response(200, json={"ok": True})

    _patch_transport(monkeypatch, handler)
    governor = bound_async_governor(identity, credentials_dir)
    client = governor.httpx2_async_client()
    assert isinstance(client, httpx2.AsyncClient)
    response = await client.post("/chat/completions", content=b'{"a":1}')

    assert response.status_code == 200
    assert [h["X-Matimo-Session-Token"] for h in seen] == ["sess-old", "sess-new"]
    verify_jws(seen[1]["Matimo-Agent-Signature"], keypair[1].encode())
    await client.aclose()


def test_httpx2_client_needs_a_bound_identity(credentials_dir: Path) -> None:
    from matimo_agdk.config import GatewayConfig
    from matimo_agdk.exceptions import GatewayError
    from matimo_agdk.governor import Governor

    governor = Governor(
        GatewayConfig(base_url=BASE_URL, api_key="k", credentials_dir=credentials_dir)
    )
    with pytest.raises(GatewayError, match="no bound identity"):
        governor.httpx2_client()


def test_httpx2_client_explains_what_to_install_when_httpx2_is_missing(
    identity: IdentityCredentials, credentials_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "matimo_agdk._retry_transport_httpx2", None)
    governor = bound_governor(identity, credentials_dir)
    with pytest.raises(ImportError, match="pip install httpx2"):
        governor.httpx2_client()


# ---------------------------------------------------------------------------
# anthropic_http_client(): picks the client the installed SDK will accept
# ---------------------------------------------------------------------------


def test_anthropic_http_client_follows_the_installed_sdk(
    identity: IdentityCredentials, credentials_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    governor = bound_governor(identity, credentials_dir)
    monkeypatch.setattr("matimo_agdk.governor.sdk_requires_httpx2", lambda _d: True)
    assert isinstance(governor.anthropic_http_client(), httpx2.Client)
    monkeypatch.setattr("matimo_agdk.governor.sdk_requires_httpx2", lambda _d: False)
    assert isinstance(governor.anthropic_http_client(), httpx.Client)


@respx.mock
def test_the_real_anthropic_sdk_accepts_the_helpers_client(
    identity: IdentityCredentials, credentials_dir: Path
) -> None:
    """The exact failure from the live run: anthropic 1.6.0 raised TypeError for
    the httpx.Client that httpx_client() returns."""
    anthropic = pytest.importorskip("anthropic")
    _sessions("sess-1")
    governor = bound_governor(identity, credentials_dir)
    client = anthropic.Anthropic(
        **governor.anthropic_client_kwargs(), http_client=governor.anthropic_http_client()
    )
    assert client is not None


@respx.mock
async def test_the_real_async_anthropic_sdk_accepts_the_helpers_client(
    identity: IdentityCredentials, credentials_dir: Path
) -> None:
    anthropic = pytest.importorskip("anthropic")
    _sessions("sess-1")
    governor = bound_async_governor(identity, credentials_dir)
    http_client = governor.anthropic_http_client()
    client = anthropic.AsyncAnthropic(
        **await governor.anthropic_client_kwargs(), http_client=http_client
    )
    assert client is not None
    await http_client.aclose()
