"""Contract test (BUILD-PLAN D16): the SDK against a machine-readable /v1 spec.

`gateway-v1.openapi.json` describes the routes the SDK calls, with the limits
the server's Zod schemas enforce. The tests drive the real SDK (Governor,
AsyncGovernor, the tool and telemetry clients) against mocked routes and assert
that

- every request the SDK builds conforms to the spec: body, required headers,
  path parameters;
- every recorded response fixture conforms to the spec, and the SDK parses it
  into the right result or the right typed error;
- the SDK cannot be talked into building a request the server would reject
  (over-long names, a batch over 500 events, values that are not JSON).

The spec and the fixtures are derived by hand from the server source, not
generated or recorded (see the spec's `info.description`). UAF should publish
the canonical document; when it does, swap it in here.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import httpx
import pytest
import respx
from jsonschema import Draft202012Validator, FormatChecker
from pydantic import ValidationError

from matimo_agdk.config import GatewayConfig
from matimo_agdk.exceptions import (
    AgentSuspended,
    GatewayError,
    GatewayUnavailable,
    PolicyDenied,
    RateLimited,
    SessionExpired,
    SignatureRejected,
    TelemetryStale,
)
from matimo_agdk.governor import AsyncGovernor, Governor
from matimo_agdk.identity import IdentityCredentials, generate_ecdsa_keypair_pem
from matimo_agdk.transport import RetryPolicy

from ..conftest import BASE_URL

HERE = Path(__file__).parent
SPEC: dict[str, Any] = json.loads((HERE / "gateway-v1.openapi.json").read_text(encoding="utf-8"))
FIXTURES: dict[str, Any] = {
    k: v
    for k, v in json.loads(
        (HERE / "fixtures" / "responses.json").read_text(encoding="utf-8")
    ).items()
    if not k.startswith("_")
}
FORMATS = FormatChecker()


# ---------------------------------------------------------------------------
# Spec helpers
# ---------------------------------------------------------------------------


def schema_validator(name: str) -> Draft202012Validator:
    # Embedding `components` lets "#/components/schemas/..." refs resolve in-document.
    return Draft202012Validator(
        {"$ref": f"#/components/schemas/{name}", "components": SPEC["components"]},
        format_checker=FORMATS,
    )


def assert_valid(name: str, instance: Any) -> None:
    errors = sorted(schema_validator(name).iter_errors(instance), key=lambda e: list(e.path))
    assert not errors, f"{name}: " + "; ".join(
        f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in errors[:5]
    )


def _resolve(ref_or_obj: Any) -> Any:
    if isinstance(ref_or_obj, dict) and "$ref" in ref_or_obj:
        node: Any = SPEC
        for part in ref_or_obj["$ref"].removeprefix("#/").split("/"):
            node = node[part]
        return node
    return ref_or_obj


def _template_regex(template: str) -> re.Pattern[str]:
    return re.compile("^" + re.sub(r"\{[^/]+\}", r"([^/]+)", template) + "$")


def find_operation(method: str, path: str) -> tuple[str, dict[str, Any], list[str]]:
    for template, item in SPEC["paths"].items():
        match = _template_regex(template).match(path)
        if match and method.lower() in item:
            return template, item[method.lower()], list(match.groups())
    raise AssertionError(f"{method} {path} is not in the spec")


def assert_request_conforms(request: httpx.Request) -> None:
    """Body, required headers and path parameters of one captured request."""
    path = urlsplit(str(request.url)).path.removeprefix(urlsplit(BASE_URL).path)
    template, operation, raw_path_values = find_operation(request.method, path)

    body_spec = operation.get("requestBody", {}).get("content", {}).get("application/json")
    if body_spec is not None:
        body = json.loads(request.content or b"null")
        ref = body_spec["schema"]["$ref"].rsplit("/", 1)[1]
        assert_valid(ref, body)

    path_names = re.findall(r"\{([^/]+)\}", template)
    for parameter in map(_resolve, operation.get("parameters", [])):
        if parameter["in"] == "header" and parameter.get("required"):
            value = request.headers.get(parameter["name"])
            assert value, f"{operation['operationId']}: missing header {parameter['name']}"
            validator = Draft202012Validator(parameter["schema"])
            assert not list(validator.iter_errors(value)), (parameter["name"], value)
        if parameter["in"] == "path":
            raw = raw_path_values[path_names.index(parameter["name"])]
            validator = Draft202012Validator(parameter["schema"], format_checker=FORMATS)
            assert not list(validator.iter_errors(unquote(raw))), (parameter["name"], raw)


def fixture_response(name: str, **replace: str) -> httpx.Response:
    fixture = FIXTURES[name]
    text = json.dumps(fixture["body"])
    for old, new in replace.items():
        text = text.replace(old, new)
    return httpx.Response(
        fixture["status"], content=text.encode(), headers={"content-type": "application/json"}
    )


class Recorder:
    """Registers respx routes that serve fixtures and keep every request."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def serve(self, method: str, path: str, *fixtures: str, **replace: str) -> respx.Route:
        served = list(fixtures)

        def responder(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            name = served.pop(0) if len(served) > 1 else served[0]
            return fixture_response(name, **replace)

        return respx.route(method=method, url=f"{BASE_URL}{path}").mock(side_effect=responder)

    def to(self, path: str) -> list[httpx.Request]:
        return [r for r in self.requests if urlsplit(str(r.url)).path.endswith(path)]


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def make_config(
    identity: IdentityCredentials, credentials_dir: Path, **extra: Any
) -> GatewayConfig:
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
        **extra,
    )


def make_governor(identity: IdentityCredentials, credentials_dir: Path, **extra: Any) -> Governor:
    governor = Governor(make_config(identity, credentials_dir, **extra))
    governor._http.retry_policy = RetryPolicy(max_retries=0)  # noqa: SLF001
    return governor


# ---------------------------------------------------------------------------
# The spec and the fixtures themselves
# ---------------------------------------------------------------------------


def test_the_spec_is_well_formed() -> None:
    assert SPEC["openapi"].startswith("3.1")
    for name, schema in SPEC["components"]["schemas"].items():
        Draft202012Validator.check_schema(schema)
        assert name
    refs = re.findall(r'"\$ref":\s*"([^"]+)"', json.dumps(SPEC))
    assert refs
    for ref in refs:
        node: Any = SPEC
        for part in ref.removeprefix("#/").split("/"):
            node = node[part]  # KeyError if a $ref dangles
    for template, item in SPEC["paths"].items():
        for method, operation in item.items():
            assert operation["operationId"], (template, method)


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_every_response_fixture_conforms_to_its_schema(name: str) -> None:
    body = FIXTURES[name]["body"]
    if name == "identity_registered":
        pem = "-----BEGIN PRIVATE KEY-----\\nx"  # escaped: it is spliced into JSON text
        body = json.loads(json.dumps(body).replace("PLACEHOLDER_REPLACED_BY_TEST", pem))
    assert_valid(FIXTURES[name]["schema"], body)


def test_every_fixture_status_is_declared_by_the_spec() -> None:
    declared: dict[str, set[str]] = {}
    for item in SPEC["paths"].values():
        for operation in item.values():
            for status, response in operation["responses"].items():
                ref = _resolve(response)["content"]["application/json"]["schema"]["$ref"]
                declared.setdefault(ref.rsplit("/", 1)[1], set()).add(status)
    for name, fixture in FIXTURES.items():
        if fixture["schema"] != "Error":  # error codes are shared across operations
            assert str(fixture["status"]) in declared[fixture["schema"]], name


def test_the_spec_rejects_what_the_server_would_reject() -> None:
    """Guards the spec itself: each of these is a request the Zod schemas refuse."""
    bad = {
        "TelemetryBatchRequest": [
            {"events": [{"runId": "r", "kind": "run"}] * 501},
            {"events": [{"runId": "", "kind": "run"}]},
            {"events": [{"runId": "r", "kind": "metric"}]},
            {"events": [{"runId": "r", "kind": "tool", "name": "x" * 256}]},
            {"events": [{"runId": "r", "kind": "tool", "status": "x" * 21}]},
            {"events": [{"runId": "r", "kind": "tool", "durationMs": -1}]},
            {"events": [{"runId": "r", "kind": "tool", "durationMs": 1.5}]},
            {"events": [], "extra": 1},
        ],
        "ToolCheckRequest": [
            {"toolName": "", "argHash": "a"},
            {"toolName": "t"},
            {"toolName": "t", "argHash": "a", "extra": 1},
            {"toolName": "t", "argHash": "a" * 129},
            {"toolName": "t", "argHash": "a", "toolCategory": "c" * 51},
        ],
        "ToolResultRequest": [
            {"resumeToken": "r", "status": "x" * 21},
            {"resumeToken": "r", "status": "ok", "error": "e" * 2001},
        ],
        "RegisterIdentityRequest": [
            {"displayName": "a", "externalFramework": "openai"},
            {"displayName": "", "externalFramework": "custom"},
        ],
    }
    for schema, instances in bad.items():
        for instance in instances:
            assert list(schema_validator(schema).iter_errors(instance)), (schema, instance)


# ---------------------------------------------------------------------------
# Requests the SDK builds
# ---------------------------------------------------------------------------


@respx.mock
def test_registration_and_key_rotation_requests_and_responses(
    credentials_dir: Path, keypair: tuple[str, str]
) -> None:
    private_pem, _ = keypair
    rec = Recorder()
    rec.serve(
        "POST",
        "/identities",
        "identity_registered",
        PLACEHOLDER_REPLACED_BY_TEST=private_pem.replace("\n", "\\n"),
    )
    config = GatewayConfig(
        base_url=BASE_URL,
        api_key="org-key",
        agent_name="contract-agent",
        credentials_dir=credentials_dir,
    )
    governor = Governor(config)
    identity = governor.register(
        framework="langchain",
        allowed_tool_categories=["web"],
        allowed_llm_models=["gpt-4o-mini"],
        registration_metadata={"team": "x"},
    )
    for request in rec.requests:
        assert_request_conforms(request)
    assert identity.identity_token == "me-id-AbCdEfGhIjKlMnOpQrStUvWx"
    assert identity.tenant_id == "33333333-3333-3333-3333-333333333333"

    rotate = Recorder()
    rotate.serve(
        "POST",
        f"/identities/{identity.identity_id}/rotate-key",
        "identity_registered",
        PLACEHOLDER_REPLACED_BY_TEST=generate_ecdsa_keypair_pem()[0].replace("\n", "\\n"),
    )
    governor.rotate_key()
    for request in rotate.requests:
        assert_request_conforms(request)


@pytest.mark.parametrize("framework", ["langchain", "google-adk", "crewai", "autogen", "custom"])
def test_every_framework_the_sdk_can_register_is_in_the_servers_enum(framework: str) -> None:
    from matimo_agdk.governor import _register_body

    body = _register_body(GatewayConfig(framework="custom"), "agent", framework, None, None, None)  # noqa: SLF001
    assert_valid("RegisterIdentityRequest", body)


@respx.mock
def test_handshake_and_heartbeat_requests_conform_and_the_response_is_parsed(
    identity: IdentityCredentials, credentials_dir: Path
) -> None:
    rec = Recorder()
    rec.serve("POST", "/sessions", "session_created")
    rec.serve("POST", "/telemetry/batch", "telemetry_heartbeat_suspended")
    governor = make_governor(identity, credentials_dir)
    governor.start()
    try:
        governor.flush()
    finally:
        governor.stop()

    assert rec.to("/sessions") and rec.to("/telemetry/batch")
    for request in rec.requests:
        assert_request_conforms(request)
    # The heartbeat fixture is parsed into the local governance state.
    assert governor.state.lifecycle_status == "suspended"
    assert governor.is_suspended()
    assert governor.state.telemetry_staleness_minutes == 30


@respx.mock
@pytest.mark.parametrize(
    ("fixture", "suspended"),
    [
        ("telemetry_ok", False),
        ("telemetry_partial_failure", False),
        ("telemetry_heartbeat_emergency_stop", True),
    ],
)
def test_heartbeat_fixtures_parse(
    identity: IdentityCredentials, credentials_dir: Path, fixture: str, suspended: bool
) -> None:
    rec = Recorder()
    rec.serve("POST", "/sessions", "session_created")
    rec.serve("POST", "/telemetry/batch", fixture)
    governor = make_governor(identity, credentials_dir)
    governor.start()
    try:
        governor.flush()
    finally:
        governor.stop()
    assert governor.is_suspended() is suspended
    assert (
        governor.state.telemetry_mode
        == FIXTURES[fixture]["body"]["data"]["heartbeat"]["telemetryMode"]
    )


@respx.mock
def test_telemetry_events_from_every_span_builder_conform(
    identity: IdentityCredentials, credentials_dir: Path
) -> None:
    rec = Recorder()
    rec.serve("POST", "/sessions", "session_created")
    rec.serve("POST", "/telemetry/batch", "telemetry_ok")
    rec.serve("POST", "/tools/check", "tool_check_allow")
    governor = make_governor(identity, credentials_dir)
    governor.start()
    try:
        with governor.run("contract-run"):
            governor.llm_span(
                model="gpt-4o-mini",
                provider="openai",
                status="completed",
                duration_ms=12,
                finish_reasons=["stop"],
            )
            governor.tool_span(
                "search", status="completed", duration_ms=3, arguments={"q": "x", "password": "p"}
            )
            governor.guard(lambda q: q, name="echo")(q="hi")
        with pytest.raises(RuntimeError), governor.run("failing-run"):
            raise RuntimeError("boom")
        governor.run_span("adk-invocation-1", status="running", name="adk")
        governor.run_span("adk-invocation-1", status="completed", name="adk", duration_ms=5)
        governor.flush()
    finally:
        governor.stop()

    batches = [json.loads(r.content) for r in rec.to("/telemetry/batch")]
    events = [e for b in batches for e in b["events"]]
    assert {e["kind"] for e in events} >= {"run", "llm", "tool"}
    for request in rec.to("/telemetry/batch"):
        assert_request_conforms(request)


@respx.mock
def test_tool_check_status_result_and_category_requests_conform(
    identity: IdentityCredentials, credentials_dir: Path
) -> None:
    rec = Recorder()
    rec.serve("POST", "/tools/check", "tool_check_pending")
    rec.serve("POST", "/tools/check/status", "tool_check_status_pending", "tool_check_status_allow")
    rec.serve("POST", "/tools/result", "tool_result_accepted")
    rec.serve("PUT", "/tools/send%20email%2Fv2/category", "tool_category_updated")
    governor = make_governor(identity, credentials_dir)
    tools = governor._tools  # noqa: SLF001
    tools.poll_interval = tools.poll_max_interval = 0.001

    decision = governor.check_tool(
        "search", {"q": "x", "nested": {"a": [1, 2]}}, category_hint="web"
    )
    assert decision.pending and decision.resume_token and decision.request_id
    final = governor.await_decision(decision.resume_token)
    assert final.allowed
    tools.report_result(decision.resume_token, status="completed", duration_ms=5, error="x" * 3000)
    governor.set_tool_category("send email/v2", "email")

    assert len(rec.requests) == 5  # check, status x2, result, category
    for request in rec.requests:
        assert_request_conforms(request)


@respx.mock
@pytest.mark.parametrize(
    ("fixture", "decision", "reason", "has_token"),
    [
        ("tool_check_allow", "ALLOW", None, False),
        ("tool_check_deny", "DENY", "tool_category_not_allowed", False),
        ("tool_check_pending", "PENDING", None, True),
        ("tool_check_pending_no_token", "PENDING", "duplicate_check_in_flight", False),
    ],
)
def test_tool_check_fixtures_parse(
    identity: IdentityCredentials,
    credentials_dir: Path,
    fixture: str,
    decision: str,
    reason: str | None,
    has_token: bool,
) -> None:
    Recorder().serve("POST", "/tools/check", fixture)
    result = make_governor(identity, credentials_dir).check_tool("t", {})
    assert result.decision == decision
    assert result.reason == reason
    assert bool(result.resume_token) is has_token
    assert result.pending_without_token is (decision == "PENDING" and not has_token)


@respx.mock
def test_status_fixtures_parse(identity: IdentityCredentials, credentials_dir: Path) -> None:
    rec = Recorder()
    rec.serve("POST", "/tools/check/status", "tool_check_status_expired")
    result = make_governor(identity, credentials_dir).await_decision("rt")
    assert result.denied and result.reason == "expired"


ERROR_MAPPING: dict[str, type[Exception]] = {
    "error_validation_failed": GatewayError,
    "error_session_expired": SessionExpired,
    "error_signature_required": SignatureRejected,
    "error_policy_denied": TelemetryStale,
    "error_resume_token_not_found": GatewayError,
    "error_rate_limited": RateLimited,
    "error_upstream": GatewayUnavailable,
}


@respx.mock
@pytest.mark.parametrize("fixture", sorted(ERROR_MAPPING))
def test_error_fixtures_map_to_typed_exceptions(
    identity: IdentityCredentials, credentials_dir: Path, fixture: str
) -> None:
    Recorder().serve("POST", "/tools/check", fixture)
    with pytest.raises(ERROR_MAPPING[fixture]) as info:
        make_governor(identity, credentials_dir).check_tool("t", {})
    body = FIXTURES[fixture]["body"]
    assert info.value.code == body["error"]
    assert info.value.status_code == FIXTURES[fixture]["status"]


@respx.mock
def test_the_other_policy_denied_reasons_map_to_their_types(
    identity: IdentityCredentials, credentials_dir: Path
) -> None:
    for reason, expected in [
        ("agent_suspended", AgentSuspended),
        ("tool_category_not_allowed", PolicyDenied),
    ]:
        respx.post(f"{BASE_URL}/tools/check").mock(
            return_value=httpx.Response(403, json={"error": "policy_denied", "message": reason})
        )
        with pytest.raises(expected):
            make_governor(identity, credentials_dir).check_tool("t", {})


@pytest.mark.asyncio
@respx.mock
async def test_async_governor_requests_conform(
    identity: IdentityCredentials, credentials_dir: Path
) -> None:
    rec = Recorder()
    rec.serve("POST", "/sessions", "session_created")
    rec.serve("POST", "/telemetry/batch", "telemetry_ok")
    rec.serve("POST", "/tools/check", "tool_check_pending")
    rec.serve("POST", "/tools/check/status", "tool_check_status_allow")
    rec.serve("POST", "/tools/result", "tool_result_accepted")
    governor = AsyncGovernor(make_config(identity, credentials_dir))
    governor._http.retry_policy = RetryPolicy(max_retries=0)  # noqa: SLF001
    await governor.start()
    try:
        async with governor.run("async-run"):
            governor.tool_span("search", status="completed", duration_ms=1)
            decision = await governor.check_tool("search", {"q": "x"}, category_hint="web")
            assert (await governor.await_decision(decision.resume_token)).allowed
            await governor._tools.report_result(decision.resume_token, status="completed")  # noqa: SLF001
        await governor.flush()
    finally:
        await governor.aclose()
    assert len(rec.requests) >= 6
    for request in rec.requests:
        assert_request_conforms(request)


# ---------------------------------------------------------------------------
# Limits: the SDK must not be able to build a request the server rejects
# ---------------------------------------------------------------------------

HOSTILE_ATTRIBUTES: dict[str, Any] = {
    "when": dt.datetime(2026, 9, 20, 10, 0, tzinfo=dt.UTC),
    "day": dt.date(2026, 9, 20),
    "path": Path("/tmp/x"),
    "id": uuid.UUID(int=1),
    "tags": {"a", "b"},
    "raw": b"bytes",
    "obj": object(),
    "nested": {"deep": [{"when": dt.datetime(2026, 1, 1, tzinfo=dt.UTC)}]},
    "nan": float("nan"),
}


@respx.mock
def test_telemetry_survives_values_that_are_not_json(
    identity: IdentityCredentials, credentials_dir: Path
) -> None:
    """A tool argument or result that is a datetime, Path, UUID, set or arbitrary
    object must not stop the batch from being built or wedge the exporter."""
    rec = Recorder()
    rec.serve("POST", "/sessions", "session_created")
    rec.serve("POST", "/telemetry/batch", "telemetry_ok")
    governor = make_governor(identity, credentials_dir, capture_tool_results=True)
    governor.start()
    try:
        with governor.run("hostile"):
            governor.tool_span(
                "t", status="completed", arguments=HOSTILE_ATTRIBUTES, result=HOSTILE_ATTRIBUTES
            )
        governor.flush()
    finally:
        governor.stop()
    sent = [r for r in rec.to("/telemetry/batch") if json.loads(r.content)["events"]]
    assert sent, "the hostile event never reached Gateway"
    for request in sent:
        assert_request_conforms(request)


@respx.mock
def test_tool_check_survives_arguments_that_are_not_json(
    identity: IdentityCredentials, credentials_dir: Path
) -> None:
    rec = Recorder()
    rec.serve("POST", "/tools/check", "tool_check_allow")
    make_governor(identity, credentials_dir).check_tool("t", HOSTILE_ATTRIBUTES)
    assert len(rec.requests) == 1
    assert_request_conforms(rec.requests[0])


@respx.mock
def test_over_long_event_fields_are_clamped_not_sent(
    identity: IdentityCredentials, credentials_dir: Path
) -> None:
    """One over-long `name` makes the server reject the WHOLE batch with a 400,
    which the exporter then drops: 49 good events lost to one bad one."""
    rec = Recorder()
    rec.serve("POST", "/sessions", "session_created")
    rec.serve("POST", "/telemetry/batch", "telemetry_ok")
    governor = make_governor(identity, credentials_dir)
    governor.start()
    try:
        with governor.run("x" * 400):
            governor.tool_span("t" * 400, status="completed", duration_ms=1)
            governor.llm_span(name="n" * 400, status="completed", duration_ms=1)
        governor.flush()
    finally:
        governor.stop()
    sent = [r for r in rec.to("/telemetry/batch") if json.loads(r.content)["events"]]
    assert sent
    for request in sent:
        assert_request_conforms(request)


def test_batch_size_can_never_exceed_the_servers_cap() -> None:
    assert GatewayConfig(telemetry_batch_size=500).telemetry_batch_size == 500
    with pytest.raises(ValidationError):
        GatewayConfig(telemetry_batch_size=501)


@respx.mock
def test_a_large_backlog_is_sent_in_batches_within_the_cap(
    identity: IdentityCredentials, credentials_dir: Path
) -> None:
    rec = Recorder()
    rec.serve("POST", "/sessions", "session_created")
    rec.serve("POST", "/telemetry/batch", "telemetry_ok")
    governor = make_governor(
        identity, credentials_dir, telemetry_batch_size=500, telemetry_queue_max=5000
    )
    governor.start()
    try:
        with governor.run("burst"):
            for i in range(1200):
                governor.tool_span("t", status="completed", duration_ms=i)
        governor.flush()
    finally:
        governor.stop()
    events = [e for r in rec.to("/telemetry/batch") for e in json.loads(r.content)["events"]]
    assert len([e for e in events if e["kind"] == "tool"]) == 1200
    for request in rec.to("/telemetry/batch"):
        assert_request_conforms(request)
