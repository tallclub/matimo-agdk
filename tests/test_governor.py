from __future__ import annotations

import httpx
import pytest
import respx

from matimo_agdk.config import GatewayConfig
from matimo_agdk.exceptions import GatewayError, ToolDenied
from matimo_agdk.governor import Governor
from matimo_agdk.identity import IdentityCredentials, load_credentials
from matimo_agdk.tools import NO_RESUME_TOKEN_DENY_REASON

from .conftest import BASE_URL, future_iso


def bound_config(identity: IdentityCredentials, credentials_dir) -> GatewayConfig:
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


@respx.mock
def test_register_persists_credentials_and_binds_identity(credentials_dir, keypair) -> None:
    private_pem, _public_pem = keypair
    respx.post(f"{BASE_URL}/identities").mock(
        return_value=httpx.Response(
            201,
            json={
                "data": {
                    "id": "new-id",
                    "identityToken": "me-id-newtoken",
                    "tenantId": "tenant-x",
                    "displayName": "fresh-agent",
                    "externalFramework": "custom",
                    "privateKeyPem": private_pem,
                    "publicKeyFingerprint": "fp",
                    "createdAt": "2026-09-18T00:00:00Z",
                }
            },
        )
    )
    config = GatewayConfig(
        base_url=BASE_URL,
        api_key="org-key",
        agent_name="fresh-agent",
        credentials_dir=credentials_dir,
    )
    governor = Governor(config)
    identity = governor.register()

    assert identity.identity_id == "new-id"
    assert governor.identity is not None
    assert governor.identity.identity_id == "new-id"

    loaded = load_credentials("fresh-agent", credentials_dir)
    assert loaded is not None
    assert loaded.identity_token == "me-id-newtoken"


@respx.mock
def test_rotate_key_updates_identity_and_invalidates_session(
    identity: IdentityCredentials, credentials_dir, keypair
) -> None:
    new_private_pem, _ = keypair
    respx.post(f"{BASE_URL}/identities/{identity.identity_id}/rotate-key").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "id": identity.identity_id,
                    "identityToken": "me-id-rotated",
                    "tenantId": identity.tenant_id,
                    "displayName": identity.display_name,
                    "externalFramework": identity.external_framework,
                    "privateKeyPem": new_private_pem,
                }
            },
        )
    )
    config = bound_config(identity, credentials_dir)
    governor = Governor(config)
    rotated = governor.rotate_key()
    assert rotated.identity_token == "me-id-rotated"
    assert governor.identity is not None
    assert governor.identity.identity_token == "me-id-rotated"


@respx.mock
def test_guard_allow_runs_the_function_and_records_span(
    identity: IdentityCredentials, credentials_dir
) -> None:
    respx.post(f"{BASE_URL}/sessions").mock(
        return_value=httpx.Response(
            201,
            json={
                "data": {"sessionToken": "tok", "expiresAt": future_iso(3600), "identityId": "x"}
            },
        )
    )
    respx.post(f"{BASE_URL}/tools/check").mock(
        return_value=httpx.Response(200, json={"data": {"decision": "ALLOW"}})
    )
    telemetry_calls = []

    def telemetry_responder(request: httpx.Request) -> httpx.Response:
        import json

        telemetry_calls.append(json.loads(request.content))
        return httpx.Response(200, json={"data": {"accepted": 1, "failed": []}})

    respx.post(f"{BASE_URL}/telemetry/batch").mock(side_effect=telemetry_responder)

    config = bound_config(identity, credentials_dir)
    governor = Governor(config)
    governor.start()
    try:
        calls = {"n": 0}

        def search(query: str) -> str:
            calls["n"] += 1
            return f"result for {query}"

        with governor.run("test-run"):
            result = governor.guard(search, name="search")(query="roaiq")
        assert result == "result for roaiq"
        assert calls["n"] == 1

        governor._telemetry.flush_now()  # noqa: SLF001
    finally:
        governor.stop()

    all_events = [e for batch in telemetry_calls for e in batch["events"]]
    tool_events = [e for e in all_events if e["kind"] == "tool" and e["name"] == "search"]
    assert len(tool_events) == 1
    assert tool_events[0]["status"] == "completed"


@respx.mock
def test_guard_deny_raises_tool_denied(identity: IdentityCredentials, credentials_dir) -> None:
    respx.post(f"{BASE_URL}/sessions").mock(
        return_value=httpx.Response(
            201,
            json={
                "data": {"sessionToken": "tok", "expiresAt": future_iso(3600), "identityId": "x"}
            },
        )
    )
    respx.post(f"{BASE_URL}/tools/check").mock(
        return_value=httpx.Response(
            200, json={"data": {"decision": "DENY", "reason": "tool_category_not_allowed"}}
        )
    )
    respx.post(f"{BASE_URL}/telemetry/batch").mock(
        return_value=httpx.Response(200, json={"data": {"accepted": 0, "failed": []}})
    )

    config = bound_config(identity, credentials_dir)
    governor = Governor(config)
    governor.start()
    try:
        called = {"n": 0}

        def dangerous() -> str:
            called["n"] += 1
            return "should not happen"

        with governor.run("test-run"):
            with pytest.raises(ToolDenied) as excinfo:
                governor.guard(dangerous, name="dangerous")()
        assert excinfo.value.reason == "tool_category_not_allowed"
        assert called["n"] == 0
    finally:
        governor.stop()


@respx.mock
def test_guard_pending_then_approved_runs_function(
    identity: IdentityCredentials, credentials_dir
) -> None:
    respx.post(f"{BASE_URL}/sessions").mock(
        return_value=httpx.Response(
            201,
            json={
                "data": {"sessionToken": "tok", "expiresAt": future_iso(3600), "identityId": "x"}
            },
        )
    )
    respx.post(f"{BASE_URL}/tools/check").mock(
        return_value=httpx.Response(
            200, json={"data": {"decision": "PENDING", "resumeToken": "rt-1"}}
        )
    )
    status_route = respx.post(f"{BASE_URL}/tools/check/status")
    status_route.side_effect = [
        httpx.Response(200, json={"data": {"decision": "PENDING"}}),
        httpx.Response(200, json={"data": {"decision": "ALLOW"}}),
    ]
    respx.post(f"{BASE_URL}/tools/result").mock(
        return_value=httpx.Response(202, json={"data": {"accepted": True}})
    )
    respx.post(f"{BASE_URL}/telemetry/batch").mock(
        return_value=httpx.Response(200, json={"data": {"accepted": 0, "failed": []}})
    )

    config = bound_config(identity, credentials_dir)
    config.telemetry_flush_interval = 9999
    governor = Governor(config)
    governor.start()
    try:
        gov_tools = governor._tools  # noqa: SLF001
        gov_tools.poll_interval = 0.01
        gov_tools.poll_max_interval = 0.02

        def send_email(to: str) -> str:
            return f"sent to {to}"

        with governor.run("test-run"):
            result = governor.guard(send_email, name="send_email")(to="a@example.com")
        assert result == "sent to a@example.com"
    finally:
        governor.stop()


@respx.mock
def test_guard_tokenless_pending_denies_and_never_runs_the_function(
    identity: IdentityCredentials, credentials_dir
) -> None:
    """Gateway can answer PENDING with no resumeToken (an identical check is
    already in flight). guard() used to treat that as "not denied" and run
    the tool without the approval a policy demanded."""
    respx.post(f"{BASE_URL}/sessions").mock(
        return_value=httpx.Response(
            201,
            json={
                "data": {"sessionToken": "tok", "expiresAt": future_iso(3600), "identityId": "x"}
            },
        )
    )
    check_route = respx.post(f"{BASE_URL}/tools/check").mock(
        return_value=httpx.Response(
            200, json={"data": {"decision": "PENDING", "reason": "duplicate_check_in_flight"}}
        )
    )
    respx.post(f"{BASE_URL}/telemetry/batch").mock(
        return_value=httpx.Response(200, json={"data": {"accepted": 0, "failed": []}})
    )

    governor = Governor(bound_config(identity, credentials_dir))
    governor.start()
    try:
        governor._tools.recheck_delays = (0.0, 0.0)  # noqa: SLF001
        ran: list[str] = []

        def send_email(to: str) -> str:
            ran.append(to)
            return f"sent to {to}"

        with governor.run("test-run"):
            with pytest.raises(ToolDenied) as excinfo:
                governor.guard(send_email, name="send_email")(to="a@example.com")
        assert excinfo.value.reason == NO_RESUME_TOKEN_DENY_REASON
        assert ran == []
        assert check_route.call_count == 3  # first check + two rechecks
    finally:
        governor.stop()


def test_guard_without_identity_raises() -> None:
    config = GatewayConfig(base_url=BASE_URL, api_key="key")
    governor = Governor(config)
    with pytest.raises(GatewayError):
        governor.guard(lambda: None, name="x")


def test_llm_span_and_tool_span_need_active_run(
    identity: IdentityCredentials, credentials_dir
) -> None:
    config = bound_config(identity, credentials_dir)
    governor = Governor(config)
    with pytest.raises(GatewayError):
        governor.llm_span()
    with pytest.raises(GatewayError):
        governor.tool_span("x")


@respx.mock
def test_httpx_client_injects_session_and_signature(
    identity: IdentityCredentials, credentials_dir
) -> None:
    respx.post(f"{BASE_URL}/sessions").mock(
        return_value=httpx.Response(
            201,
            json={
                "data": {
                    "sessionToken": "tok-abc",
                    "expiresAt": future_iso(3600),
                    "identityId": "x",
                }
            },
        )
    )
    captured = {}

    def responder(request: httpx.Request) -> httpx.Response:
        captured["session"] = request.headers.get("X-Matimo-Session-Token")
        captured["sig"] = request.headers.get("Matimo-Agent-Signature")
        return httpx.Response(200, json={"id": "chatcmpl-1"})

    respx.post(f"{BASE_URL}/chat/completions").mock(side_effect=responder)

    config = bound_config(identity, credentials_dir)
    governor = Governor(config)
    client = governor.httpx_client()
    try:
        client.post("/chat/completions", json={"model": "gpt-4o-mini", "messages": []})
    finally:
        client.close()

    assert captured["session"] == "tok-abc"
    assert captured["sig"]


@respx.mock
def test_httpx_client_transparently_rehandshakes_on_session_expired(
    identity: IdentityCredentials, credentials_dir
) -> None:
    """Regression test for a real bug found live-verifying against Gateway
    (2026-09-18 live verification, see CHANGELOG.md): the request event hook can only set headers
    before sending -- with no transport-level retry, a session invalidated
    behind the SDK's back (e.g. another process calling DELETE
    /v1/sessions) made every subsequent /v1/chat/completions call fail with
    a raw 401 forever. SessionRetryTransport fixes this by catching exactly
    a 401 {error: session_expired} response and resending once with a
    freshly handshaked session."""
    session_tokens = iter(["tok-first", "tok-second"])

    def session_responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            201,
            json={
                "data": {
                    "sessionToken": next(session_tokens),
                    "expiresAt": future_iso(3600),
                    "identityId": "x",
                }
            },
        )

    respx.post(f"{BASE_URL}/sessions").mock(side_effect=session_responder)

    seen_session_headers: list[str | None] = []

    def chat_responder(request: httpx.Request) -> httpx.Response:
        seen_session_headers.append(request.headers.get("X-Matimo-Session-Token"))
        if len(seen_session_headers) == 1:
            return httpx.Response(401, json={"error": "session_expired"})
        return httpx.Response(200, json={"id": "chatcmpl-1"})

    respx.post(f"{BASE_URL}/chat/completions").mock(side_effect=chat_responder)

    config = bound_config(identity, credentials_dir)
    governor = Governor(config)
    client = governor.httpx_client()
    try:
        resp = client.post("/chat/completions", json={"model": "gpt-4o-mini", "messages": []})
    finally:
        client.close()

    assert resp.status_code == 200
    assert resp.json() == {"id": "chatcmpl-1"}
    assert seen_session_headers == ["tok-first", "tok-second"]


@respx.mock
def test_run_context_manager_emits_start_and_completed_spans(
    identity: IdentityCredentials, credentials_dir
) -> None:
    respx.post(f"{BASE_URL}/sessions").mock(
        return_value=httpx.Response(
            201,
            json={
                "data": {"sessionToken": "tok", "expiresAt": future_iso(3600), "identityId": "x"}
            },
        )
    )
    batches = []

    def responder(request: httpx.Request) -> httpx.Response:
        import json

        batches.append(json.loads(request.content))
        return httpx.Response(200, json={"data": {"accepted": 1, "failed": []}})

    respx.post(f"{BASE_URL}/telemetry/batch").mock(side_effect=responder)

    config = bound_config(identity, credentials_dir)
    governor = Governor(config)
    governor.start()
    try:
        with governor.run("my-run"):
            pass
        governor._telemetry.flush_now()  # noqa: SLF001
    finally:
        governor.stop()

    all_events = [e for batch in batches for e in batch["events"]]
    run_events = [e for e in all_events if e["kind"] == "run"]
    assert any(e["status"] == "running" for e in run_events)
    assert any(e["status"] == "completed" for e in run_events)


@respx.mock
def test_run_context_manager_emits_failed_span_on_exception(
    identity: IdentityCredentials, credentials_dir
) -> None:
    respx.post(f"{BASE_URL}/sessions").mock(
        return_value=httpx.Response(
            201,
            json={
                "data": {"sessionToken": "tok", "expiresAt": future_iso(3600), "identityId": "x"}
            },
        )
    )
    batches = []

    def responder(request: httpx.Request) -> httpx.Response:
        import json

        batches.append(json.loads(request.content))
        return httpx.Response(200, json={"data": {"accepted": 1, "failed": []}})

    respx.post(f"{BASE_URL}/telemetry/batch").mock(side_effect=responder)

    config = bound_config(identity, credentials_dir)
    governor = Governor(config)
    governor.start()
    try:
        with pytest.raises(ValueError):
            with governor.run("my-run"):
                raise ValueError("boom")
        governor._telemetry.flush_now()  # noqa: SLF001
    finally:
        governor.stop()

    all_events = [e for batch in batches for e in batch["events"]]
    run_events = [e for e in all_events if e["kind"] == "run"]
    assert any(e["status"] == "failed" for e in run_events)
