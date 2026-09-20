"""Regression tests for the end-to-end review of 2026-09-20.

Each test pins a defect that was first reproduced with a script against the
code as reviewed, then fixed. They are grouped by module, not by severity.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import pickle
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from matimo_agdk import cli
from matimo_agdk._redact import redact, scrub_string
from matimo_agdk.adapters._shared import default_llm_headers
from matimo_agdk.adapters.generic import govern
from matimo_agdk.config import GatewayConfig
from matimo_agdk.exceptions import (
    AgentSuspendedLocally,
    GatewayError,
    RateLimited,
    ToolDenied,
)
from matimo_agdk.governor import Governor
from matimo_agdk.identity import (
    IdentityCredentials,
    generate_ecdsa_keypair_pem,
    load_credentials,
    save_credentials,
    verify_jws,
)
from matimo_agdk.session import _build_session_state
from matimo_agdk.tools import (
    UNRECOGNIZED_DECISION_DENY_REASON,
    ToolDecision,
    ToolGovernor,
    _decision_from,
)
from matimo_agdk.transport import (
    GatewayHTTP,
    RetryPolicy,
    parse_retry_after,
)

from .adapters.conftest import bound_async_governor, bound_config, bound_governor
from .conftest import BASE_URL, make_session_response, wait_until

# ---------------------------------------------------------------------------
# tools.py: a governance check that cannot be understood must fail closed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"decision": "ESCALATE"},
        {"decision": "allow"},  # wrong case is not ALLOW
        {"decision": None},
        {},
    ],
)
def test_unrecognized_decision_becomes_deny(payload: dict[str, Any]) -> None:
    decision = _decision_from(payload)
    assert decision.denied
    assert decision.reason == UNRECOGNIZED_DECISION_DENY_REASON


@respx.mock
def test_guard_does_not_run_tool_on_unrecognized_decision(
    identity: IdentityCredentials, credentials_dir: Path
) -> None:
    respx.post(f"{BASE_URL}/tools/check").mock(
        return_value=httpx.Response(200, json={"data": {"decision": "ESCALATE"}})
    )
    governor = bound_governor(identity, credentials_dir)
    ran: list[int] = []

    @governor.guard(name="danger")
    def danger(x: int) -> str:
        ran.append(x)
        return "done"

    with pytest.raises(ToolDenied) as excinfo:
        with governor.run("r"):
            danger(x=1)
    assert excinfo.value.reason == UNRECOGNIZED_DECISION_DENY_REASON
    assert ran == []


def test_valid_decisions_still_pass_through() -> None:
    assert _decision_from({"decision": "ALLOW"}).allowed
    pending = _decision_from({"decision": "PENDING", "resumeToken": "t", "reason": "r"})
    assert pending.pending and pending.resume_token == "t"


@respx.mock
def test_set_category_percent_encodes_the_tool_name(
    identity: IdentityCredentials, credentials_dir: Path
) -> None:
    route = respx.put(url__regex=rf"{BASE_URL}/tools/.*").mock(
        return_value=httpx.Response(200, json={"data": {}})
    )
    governor = bound_governor(identity, credentials_dir)
    governor.set_tool_category("../../identities/x/rotate-key?y=", "safe")
    sent = str(route.calls.last.request.url)
    assert sent.startswith(f"{BASE_URL}/tools/")
    assert sent.endswith("/category")
    assert "?" not in sent.removeprefix(BASE_URL)
    assert route.calls.last.request.url.path.startswith("/v1/tools/")


@respx.mock
def test_status_polling_retries_a_transient_5xx(monkeypatch: pytest.MonkeyPatch) -> None:
    """A human-approval wait can last hours; one 502 must not abort it."""
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    route = respx.post(f"{BASE_URL}/tools/check/status")
    route.side_effect = [
        httpx.Response(500, json={"error": "internal"}),
        httpx.Response(200, json={"data": {"decision": "ALLOW"}}),
    ]
    http = GatewayHTTP(BASE_URL, "k", retry_policy=RetryPolicy(base_delay=0.001, max_delay=0.01))
    tools = ToolGovernor(http, identity_token="t", identity_id="i", tenant_id="x")
    assert tools.status("resume").allowed
    assert route.call_count == 2


@respx.mock
def test_report_result_scrubs_secrets_from_the_error_text(
    identity: IdentityCredentials, credentials_dir: Path
) -> None:
    route = respx.post(f"{BASE_URL}/tools/result").mock(
        return_value=httpx.Response(200, json={"data": {}})
    )
    governor = bound_governor(identity, credentials_dir)
    assert governor._tools is not None
    governor._tools.report_result(
        "resume", status="error", error="401 for Authorization: Bearer abcdef1234567890 on db"
    )
    body = json.loads(route.calls.last.request.content)
    assert "abcdef1234567890" not in body["error"]
    assert "Bearer [REDACTED]" in body["error"]


# ---------------------------------------------------------------------------
# _redact.py: value-level backstop
# ---------------------------------------------------------------------------


def test_scrub_string_masks_unmistakable_secret_shapes() -> None:
    pem = "-----BEGIN PRIVATE KEY-----\nMIGHAgEAMBMG\n-----END PRIVATE KEY-----"
    assert scrub_string(f"key was {pem} ok") == "key was [REDACTED] ok"
    assert scrub_string("-----BEGIN PRIVATE KEY-----\nMIGHAg...") == "[REDACTED]"  # truncated PEM
    assert "sk-abcdefghijklmnop1234" not in scrub_string("sk-abcdefghijklmnop1234")
    assert "me-live-abcdef123456" not in scrub_string("key=me-live-abcdef123456")
    assert "ghp_" + "a" * 24 not in scrub_string("ghp_" + "a" * 24)
    assert scrub_string("monkey keyword tokenizer") == "monkey keyword tokenizer"


def test_redact_scrubs_string_values_under_innocuous_keys() -> None:
    out = redact({"note": "curl -H 'Authorization: Bearer abcdef1234567890'", "n": 3})
    assert "abcdef1234567890" not in out["note"]
    assert out["n"] == 3


# ---------------------------------------------------------------------------
# transport.py: Retry-After
# ---------------------------------------------------------------------------


def test_retry_after_is_capped_and_non_finite_values_ignored() -> None:
    policy = RetryPolicy()
    assert policy.delay_for(0, parse_retry_after("86400")) == policy.max_retry_after
    assert parse_retry_after("inf") is None
    assert parse_retry_after("nan") is None
    assert parse_retry_after("-5") is None
    assert parse_retry_after("2") == 2.0


@respx.mock
def test_exhausted_429_exposes_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    respx.post(f"{BASE_URL}/tools/check").mock(
        return_value=httpx.Response(
            429, headers={"Retry-After": "7"}, json={"error": "rate_limit_exceeded"}
        )
    )
    http = GatewayHTTP(BASE_URL, "k", retry_policy=RetryPolicy(max_retries=1))
    with pytest.raises(RateLimited) as excinfo:
        http.request("POST", "/tools/check", json_body={})
    assert excinfo.value.retry_after == 7.0


# ---------------------------------------------------------------------------
# exceptions.py
# ---------------------------------------------------------------------------


def test_agent_suspended_locally_survives_pickling() -> None:
    restored = pickle.loads(pickle.dumps(AgentSuspendedLocally("suspended", True)))
    assert isinstance(restored, AgentSuspendedLocally)
    assert (restored.lifecycle_status, restored.emergency_stop) == ("suspended", True)


# ---------------------------------------------------------------------------
# config.py
# ---------------------------------------------------------------------------


def test_config_repr_hides_secrets() -> None:
    private_pem, _ = generate_ecdsa_keypair_pem()
    config = GatewayConfig(
        base_url=BASE_URL,
        api_key="ORG-KEY-SECRET",
        identity_token="IDENTITY-TOKEN-SECRET",
        private_key_pem=private_pem,
    )
    text = repr(config) + str(config)
    assert "ORG-KEY-SECRET" not in text
    assert "IDENTITY-TOKEN-SECRET" not in text
    assert "BEGIN PRIVATE KEY" not in text


def test_identity_credentials_repr_hides_secrets(identity: IdentityCredentials) -> None:
    text = repr(identity)
    assert identity.private_key_pem not in text
    assert identity.identity_token not in text
    assert identity.identity_id in text


def test_config_rejects_misspelled_and_nonsensical_options() -> None:
    with pytest.raises(ValueError):
        GatewayConfig(base_url=BASE_URL, telemetry_batchsize=10)  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        GatewayConfig(base_url=BASE_URL, telemetry_batch_size=0)
    with pytest.raises(ValueError):
        GatewayConfig(base_url=BASE_URL, read_timeout=-1)


def test_cleartext_credentials_warning_only_off_loopback() -> None:
    with pytest.warns(UserWarning, match="plain http"):
        GatewayConfig(base_url="http://gateway.example.com/v1", api_key="k")
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        GatewayConfig(base_url="http://localhost:8000/v1", api_key="k")
        GatewayConfig(base_url="https://gateway.example.com/v1", api_key="k")
        GatewayConfig(base_url="http://gateway.example.com/v1")  # no key, nothing to leak


def test_unreadable_private_key_file_is_an_error_not_silence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MATIMO_PRIVATE_KEY_FILE", str(tmp_path / "missing.pem"))
    with pytest.raises(ValueError, match="MATIMO_PRIVATE_KEY_FILE"):
        GatewayConfig.load(agent_name="nobody", credentials_dir=tmp_path)


# ---------------------------------------------------------------------------
# session.py
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "data", [{}, {"sessionToken": "t"}, {"sessionToken": "t", "expiresAt": "x"}, None]
)
def test_malformed_session_response_is_a_gateway_error(data: Any) -> None:
    with pytest.raises(GatewayError):
        _build_session_state(data)


# ---------------------------------------------------------------------------
# identity.py: atomic credentials
# ---------------------------------------------------------------------------


def test_save_credentials_leaves_no_temp_files_and_round_trips(
    identity: IdentityCredentials, credentials_dir: Path
) -> None:
    save_credentials(identity, credentials_dir)
    save_credentials(identity, credentials_dir)  # replace in place
    assert sorted(p.name for p in credentials_dir.iterdir()) == [
        "test-agent.json",
        "test-agent.pem",
    ]
    loaded = load_credentials("test-agent", credentials_dir)
    assert loaded is not None and loaded.private_key_pem == identity.private_key_pem


def test_failed_credentials_write_keeps_the_existing_file_intact(
    identity: IdentityCredentials, credentials_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_credentials(identity, credentials_dir)
    original = (credentials_dir / "test-agent.pem").read_text(encoding="utf-8")

    def boom(*_a: Any, **_k: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        save_credentials(identity, credentials_dir)
    assert (credentials_dir / "test-agent.pem").read_text(encoding="utf-8") == original
    assert not [p for p in credentials_dir.iterdir() if p.suffix == ".tmp"]


# ---------------------------------------------------------------------------
# governor.py: register / rotate
# ---------------------------------------------------------------------------


def _register_payload(pem: str, n: int = 1) -> dict[str, Any]:
    return {
        "data": {
            "id": f"id-{n}",
            "identityToken": f"me-id-token{n}",
            "tenantId": "tenant",
            "displayName": "dup",
            "privateKeyPem": pem,
            "externalFramework": "custom",
        }
    }


@respx.mock
def test_register_refuses_to_overwrite_existing_credentials_before_any_network_call(
    credentials_dir: Path,
) -> None:
    pem1, _ = generate_ecdsa_keypair_pem()
    pem2, _ = generate_ecdsa_keypair_pem()
    route = respx.post(f"{BASE_URL}/identities")
    route.side_effect = [
        httpx.Response(201, json=_register_payload(pem1, 1)),
        httpx.Response(201, json=_register_payload(pem2, 2)),
    ]
    governor = Governor(
        GatewayConfig(base_url=BASE_URL, api_key="k", credentials_dir=credentials_dir)
    )
    governor.register(display_name="dup")

    with pytest.raises(GatewayError, match="already exist") as excinfo:
        governor.register(display_name="dup")
    assert excinfo.value.code == "credentials_exist"
    assert route.call_count == 1  # refused before the (non-idempotent) POST
    loaded = load_credentials("dup", credentials_dir)
    assert loaded is not None and loaded.identity_id == "id-1"

    governor.register(display_name="dup", overwrite=True)
    loaded = load_credentials("dup", credentials_dir)
    assert loaded is not None and loaded.identity_id == "id-2"


@respx.mock
def test_register_keeps_the_key_in_memory_when_the_disk_write_fails(
    credentials_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pem, _ = generate_ecdsa_keypair_pem()
    respx.post(f"{BASE_URL}/identities").mock(
        return_value=httpx.Response(201, json=_register_payload(pem))
    )

    def boom(*_a: Any, **_k: Any) -> None:
        raise OSError("read-only file system")

    monkeypatch.setattr("matimo_agdk.governor.save_credentials", boom)
    governor = Governor(
        GatewayConfig(base_url=BASE_URL, api_key="k", credentials_dir=credentials_dir)
    )
    with pytest.raises(GatewayError, match="private key") as excinfo:
        governor.register(display_name="dup")
    assert excinfo.value.code == "credentials_not_saved"
    assert governor.identity is not None
    assert governor.identity.private_key_pem == pem  # recoverable by the caller


def test_register_and_start_need_an_api_key(credentials_dir: Path) -> None:
    governor = Governor(GatewayConfig(base_url=BASE_URL, credentials_dir=credentials_dir))
    with pytest.raises(GatewayError, match="API key"):
        governor.register(display_name="x")


@respx.mock
def test_rotate_key_keeps_previously_created_httpx_client_signing_with_the_new_key(
    identity: IdentityCredentials, credentials_dir: Path
) -> None:
    """Reproduced before the fix: the client's hook had captured the old signer,
    so after rotation it signed with the revoked key (403 signature_required)."""
    new_pem, new_public = generate_ecdsa_keypair_pem()
    respx.post(f"{BASE_URL}/sessions").mock(
        return_value=httpx.Response(201, json=make_session_response())
    )
    respx.post(f"{BASE_URL}/identities/{identity.identity_id}/rotate-key").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "id": identity.identity_id,
                    "identityToken": identity.identity_token,
                    "tenantId": identity.tenant_id,
                    "displayName": identity.display_name,
                    "privateKeyPem": new_pem,
                    "externalFramework": "custom",
                }
            },
        )
    )
    governor = bound_governor(identity, credentials_dir)
    client = governor.httpx_client()  # created BEFORE the rotation
    signer_before = governor._signer

    governor.rotate_key()

    assert governor._signer is signer_before  # rekeyed in place, not replaced
    hook = client._event_hooks["request"][0]
    request = client.build_request("POST", "/chat/completions", content=b'{"a":1}')
    hook(request)
    claims = verify_jws(request.headers["Matimo-Agent-Signature"], new_public.encode())
    assert claims["sub"] == identity.identity_id


@respx.mock
def test_rotate_key_repoints_a_guard_wrapped_callable_at_the_new_identity(
    identity: IdentityCredentials, credentials_dir: Path
) -> None:
    new_pem, _ = generate_ecdsa_keypair_pem()
    respx.post(f"{BASE_URL}/identities/{identity.identity_id}/rotate-key").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "id": identity.identity_id,
                    "identityToken": "me-id-ROTATED-token-0001",
                    "tenantId": identity.tenant_id,
                    "displayName": identity.display_name,
                    "privateKeyPem": new_pem,
                }
            },
        )
    )
    check = respx.post(f"{BASE_URL}/tools/check").mock(
        return_value=httpx.Response(200, json={"data": {"decision": "ALLOW"}})
    )
    respx.post(f"{BASE_URL}/tools/result")
    governor = bound_governor(identity, credentials_dir)

    @governor.guard(name="t")
    def tool() -> str:
        return "ok"

    governor.rotate_key()
    with governor.run("r"):
        assert tool() == "ok"
    assert (
        check.calls.last.request.headers["X-Matimo-Agent-Identity-Token"]
        == "me-id-ROTATED-token-0001"
    )


# ---------------------------------------------------------------------------
# governor.py: runs
# ---------------------------------------------------------------------------


class _Capture:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def submit(self, event: dict[str, Any]) -> None:
        self.events.append(event)

    def statuses(self) -> list[tuple[str, str | None]]:
        return [(e["kind"], e.get("status")) for e in self.events]


async def test_async_run_closes_with_cancelled_when_the_task_is_cancelled(
    identity: IdentityCredentials, credentials_dir: Path
) -> None:
    governor = bound_async_governor(identity, credentials_dir)
    capture = _Capture()
    governor._telemetry = capture  # type: ignore[assignment]

    async def body() -> None:
        async with governor.run("job"):
            await asyncio.sleep(10)

    task = asyncio.create_task(body())
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert capture.statuses() == [("run", "running"), ("run", "cancelled")]


def test_sync_run_closes_with_cancelled_on_keyboard_interrupt(
    identity: IdentityCredentials, credentials_dir: Path
) -> None:
    governor = bound_governor(identity, credentials_dir)
    capture = _Capture()
    governor._telemetry = capture  # type: ignore[assignment]
    with pytest.raises(KeyboardInterrupt):
        with governor.run("job"):
            raise KeyboardInterrupt
    assert capture.statuses() == [("run", "running"), ("run", "cancelled")]


def test_sync_run_still_reports_failed_for_ordinary_exceptions(
    identity: IdentityCredentials, credentials_dir: Path
) -> None:
    governor = bound_governor(identity, credentials_dir)
    capture = _Capture()
    governor._telemetry = capture  # type: ignore[assignment]
    with pytest.raises(RuntimeError):
        with governor.run("job"):
            raise RuntimeError("boom")
    assert capture.statuses() == [("run", "running"), ("run", "failed")]


# ---------------------------------------------------------------------------
# telemetry.py
# ---------------------------------------------------------------------------


def _heartbeat(lifecycle: str = "active") -> dict[str, Any]:
    return {"lifecycleStatus": lifecycle, "telemetryStalenessMinutes": 30}


@respx.mock
def test_stop_delivers_the_whole_backlog_not_just_one_batch(
    identity: IdentityCredentials, credentials_dir: Path
) -> None:
    """Reproduced before the fix: 302 queued events, 50 delivered on stop()."""
    respx.post(f"{BASE_URL}/sessions").mock(
        return_value=httpx.Response(201, json=make_session_response())
    )
    batches: list[int] = []

    def batch(request: httpx.Request) -> httpx.Response:
        events = json.loads(request.content)["events"]
        batches.append(len(events))
        return httpx.Response(
            200, json={"data": {"accepted": len(events), "failed": [], "heartbeat": _heartbeat()}}
        )

    respx.post(f"{BASE_URL}/telemetry/batch").mock(side_effect=batch)
    governor = bound_governor(identity, credentials_dir)
    governor.start()
    assert wait_until(lambda: len(batches) >= 1)  # the forced first heartbeat
    batches.clear()
    with governor.run("burst"):
        for _ in range(300):
            governor.tool_span("t", status="completed", duration_ms=1)
    governor.stop()
    assert sum(batches) == 302
    assert governor._telemetry is not None and governor._telemetry.dropped_count == 0


@respx.mock
async def test_async_flush_now_delivers_the_whole_backlog(
    identity: IdentityCredentials, credentials_dir: Path
) -> None:
    respx.post(f"{BASE_URL}/sessions").mock(
        return_value=httpx.Response(201, json=make_session_response())
    )
    sent: list[int] = []

    def batch(request: httpx.Request) -> httpx.Response:
        events = json.loads(request.content)["events"]
        sent.append(len(events))
        return httpx.Response(
            200, json={"data": {"accepted": len(events), "heartbeat": _heartbeat()}}
        )

    respx.post(f"{BASE_URL}/telemetry/batch").mock(side_effect=batch)
    governor = bound_async_governor(identity, credentials_dir)
    await governor.start()
    async with governor.run("burst"):
        for _ in range(120):
            governor.tool_span("t", status="completed", duration_ms=1)
    sent.clear()
    await governor.flush()
    assert sum(sent) >= 120
    await governor.aclose()


@respx.mock
def test_exporter_drops_a_batch_the_server_rejects_instead_of_resending_it_forever(
    identity: IdentityCredentials, credentials_dir: Path
) -> None:
    respx.post(f"{BASE_URL}/sessions").mock(
        return_value=httpx.Response(201, json=make_session_response())
    )
    route = respx.post(f"{BASE_URL}/telemetry/batch").mock(
        return_value=httpx.Response(400, json={"error": "validation_error", "message": "bad"})
    )
    governor = bound_governor(identity, credentials_dir)
    governor.start()
    with governor.run("r"):
        pass
    exporter = governor._telemetry
    assert exporter is not None
    try:
        exporter.flush_now()
        calls_after_first = route.call_count
        exporter.flush_now()  # nothing left to resend: one more heartbeat request only
        assert exporter._queue.qsize() == 0
        assert exporter.dropped_count >= 1
        assert route.call_count == calls_after_first + 1
    finally:
        governor.stop()


@respx.mock
def test_exporter_requeues_a_batch_on_a_transient_failure(
    identity: IdentityCredentials, credentials_dir: Path
) -> None:
    respx.post(f"{BASE_URL}/sessions").mock(
        return_value=httpx.Response(201, json=make_session_response())
    )
    respx.post(f"{BASE_URL}/telemetry/batch").mock(
        return_value=httpx.Response(429, json={"error": "rate_limit_exceeded"})
    )
    config = bound_config(identity, credentials_dir)
    governor = Governor(config)
    governor._http.retry_policy = RetryPolicy(max_retries=0)
    governor.start()
    with governor.run("r"):
        pass
    exporter = governor._telemetry
    assert exporter is not None
    try:
        exporter.flush_now()  # fail_open (default): the 429 is swallowed and the batch kept
        assert exporter._queue.qsize() >= 2  # 429 is transient: nothing was dropped
        assert exporter.dropped_count == 0
    finally:
        governor.stop()


@respx.mock
def test_exporter_thread_survives_a_malformed_session_response(
    identity: IdentityCredentials, credentials_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Server drift used to raise KeyError inside the background thread and
    end it silently, so heartbeats (and rapid suspend) stopped."""
    respx.post(f"{BASE_URL}/sessions").mock(
        return_value=httpx.Response(201, json={"data": {"sessionToken": "t"}})  # no expiresAt
    )
    respx.post(f"{BASE_URL}/telemetry/batch").mock(
        return_value=httpx.Response(200, json={"data": {"heartbeat": _heartbeat()}})
    )
    config = bound_config(identity, credentials_dir).model_copy(
        update={"telemetry_flush_interval": 0.02}
    )
    governor = Governor(config)
    uncaught: list[Any] = []
    old_hook = threading.excepthook
    threading.excepthook = lambda args: uncaught.append(args)
    try:
        with caplog.at_level(logging.WARNING):
            governor.start()
            time.sleep(0.3)
        assert governor._telemetry is not None
        assert governor._telemetry._thread is not None and governor._telemetry._thread.is_alive()
        assert uncaught == []
    finally:
        threading.excepthook = old_hook
        governor.stop()


@respx.mock
def test_a_raising_on_suspend_callback_does_not_kill_the_exporter(
    identity: IdentityCredentials, credentials_dir: Path
) -> None:
    respx.post(f"{BASE_URL}/sessions").mock(
        return_value=httpx.Response(201, json=make_session_response())
    )
    respx.post(f"{BASE_URL}/telemetry/batch").mock(
        return_value=httpx.Response(200, json={"data": {"heartbeat": _heartbeat("suspended")}})
    )
    config = bound_config(identity, credentials_dir).model_copy(
        update={"telemetry_flush_interval": 0.02}
    )
    governor = Governor(config)
    seen: list[str] = []

    def cb(state: Any) -> None:
        seen.append(state.lifecycle_status)
        raise RuntimeError("user callback bug")

    governor.start()
    governor.on_suspend(cb)
    try:
        assert wait_until(lambda: seen != [], timeout=3)
        time.sleep(0.15)
        assert governor._telemetry is not None and governor._telemetry._thread is not None
        assert governor._telemetry._thread.is_alive()
        assert governor.is_suspended()
    finally:
        governor.stop()


@respx.mock
def test_on_suspend_callback_survives_stop_then_start(
    identity: IdentityCredentials, credentials_dir: Path
) -> None:
    respx.post(f"{BASE_URL}/sessions").mock(
        return_value=httpx.Response(201, json=make_session_response())
    )
    respx.post(f"{BASE_URL}/telemetry/batch").mock(
        return_value=httpx.Response(200, json={"data": {"heartbeat": _heartbeat("suspended")}})
    )
    governor = bound_governor(identity, credentials_dir)
    seen: list[str] = []
    governor.start()
    governor.on_suspend(lambda s: seen.append(s.lifecycle_status))
    governor.stop()
    governor.start()
    try:
        assert wait_until(lambda: seen != [], timeout=3)
    finally:
        governor.close()


def test_stop_unregisters_the_atexit_hook(
    identity: IdentityCredentials, credentials_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import atexit

    registered: list[Any] = []
    unregistered: list[Any] = []
    monkeypatch.setattr(atexit, "register", lambda fn, *a, **k: registered.append(fn))
    monkeypatch.setattr(atexit, "unregister", lambda fn: unregistered.append(fn))
    with respx.mock:
        respx.post(f"{BASE_URL}/sessions").mock(
            return_value=httpx.Response(201, json=make_session_response())
        )
        respx.post(f"{BASE_URL}/telemetry/batch").mock(
            return_value=httpx.Response(200, json={"data": {"heartbeat": _heartbeat()}})
        )
        governor = bound_governor(identity, credentials_dir)
        governor.start()
        governor.stop()
    assert len(registered) == 1 and unregistered == registered


# ---------------------------------------------------------------------------
# adapters/generic.py + _shared.py
# ---------------------------------------------------------------------------


class _RecordingGovernor:
    def __init__(self) -> None:
        self.checked: list[dict[str, Any]] = []

    def check_tool(self, *_a: Any, **_k: Any) -> None: ...

    def check_and_wait(
        self, name: str, args: dict[str, Any], category_hint: str | None = None
    ) -> Any:
        self.checked.append(dict(args))
        return ToolDecision("ALLOW")

    def raise_if_suspended(self) -> None: ...

    def tool_span(self, *_a: Any, **_k: Any) -> None: ...


def test_generic_bridge_checks_positional_and_keyword_arguments_together() -> None:
    """Reproduced before the fix: `kwargs or {...}` dropped arg0 whenever any
    keyword was present, so f(1, b=2) and f(99, b=2) shared one dedup key."""
    recorder = _RecordingGovernor()

    async def tool(a: int, b: int = 0) -> int:
        return a

    governed = govern(tool, recorder)
    asyncio.run(governed(1, b=2))
    asyncio.run(governed(99, b=2))
    assert recorder.checked == [{"arg0": 1, "b": 2}, {"arg0": 99, "b": 2}]


def test_generic_observe_mode_records_positional_and_keyword_arguments() -> None:
    spans: list[dict[str, Any]] = []

    class Rec(_RecordingGovernor):
        def tool_span(self, name: str, **kw: Any) -> None:
            spans.append(kw)

    def tool(a: int, b: int = 0) -> int:
        return a

    govern(tool, Rec(), mode="observe")(5, b=6)
    assert spans[0]["arguments"] == {"arg0": 5, "b": 6}


def test_default_llm_headers_rejects_an_async_governor_with_guidance(
    identity: IdentityCredentials, credentials_dir: Path
) -> None:
    governor = bound_async_governor(identity, credentials_dir)
    with pytest.raises(TypeError, match="sync Governor"):
        default_llm_headers(governor, "gateway_llm")


# ---------------------------------------------------------------------------
# cli.py
# ---------------------------------------------------------------------------


def test_cli_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--version"])
    assert excinfo.value.code == 0
    assert "matimo-agdk" in capsys.readouterr().out


def test_cli_doctor_reports_a_malformed_key_without_a_traceback(
    credentials_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (credentials_dir / "bad.json").write_text(
        json.dumps(
            {"identity_id": "i", "identity_token": "t", "tenant_id": "x", "display_name": "bad"}
        )
    )
    (credentials_dir / "bad.pem").write_text("not a pem")
    monkeypatch.setattr("matimo_agdk.identity.DEFAULT_CREDENTIALS_DIR", credentials_dir)
    monkeypatch.setenv("MATIMO_API_KEY", "k")
    monkeypatch.setenv("MATIMO_GATEWAY_URL", BASE_URL)
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["doctor", "--name", "bad"])
    assert excinfo.value.code == 1
    assert "malformed private key" in capsys.readouterr().err


@respx.mock
def test_cli_register_refuses_to_clobber_credentials_without_force(
    credentials_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("matimo_agdk.identity.DEFAULT_CREDENTIALS_DIR", credentials_dir)
    monkeypatch.setenv("MATIMO_API_KEY", "k")
    monkeypatch.setenv("MATIMO_GATEWAY_URL", BASE_URL)
    pem, _ = generate_ecdsa_keypair_pem()
    route = respx.post(f"{BASE_URL}/identities").mock(
        return_value=httpx.Response(201, json=_register_payload(pem))
    )
    with pytest.raises(SystemExit) as first:
        cli.main(["register", "--name", "dup"])
    assert first.value.code == 0

    with pytest.raises(SystemExit) as second:
        cli.main(["register", "--name", "dup"])
    assert second.value.code == 1
    assert "already exist" in capsys.readouterr().err
    assert route.call_count == 1

    with pytest.raises(SystemExit) as forced:
        cli.main(["register", "--name", "dup", "--force"])
    assert forced.value.code == 0
    assert route.call_count == 2
