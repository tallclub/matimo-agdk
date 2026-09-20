#!/usr/bin/env python
"""live_check.py -- Matimo AGDK live smoke test against a real Gateway.

Mirrors the role `gateway-manual-test.ts` plays for the TypeScript side of
Universal-AgentForge (UAF): every other test in this SDK (`tests/`) mocks
the network with `respx`, proving the client builds correct requests and
handles every scripted response shape correctly. This script proves the
whole stack actually works wired together -- real HTTP, real Postgres, real
Redis, a real running backend -- the class of bug a mock can never surface.

This script never provisions accounts, API keys, Matimo Enterprise licenses,
or LLM providers. It is a *pure consumer* of a tenant an operator has already
set up by hand, the same way a real AGDK integrator would be. Concretely, it
never logs in, signs up, resets a password, activates a license, or creates
a BYOK provider -- there is no password anywhere in this file.

Required environment variables (no defaults, the script fails fast with a
clear message if any is missing):
    MATIMO_API_KEY       An org API key with gateway:proxy, identity:manage,
                          and agdk:check scopes. Mint one yourself, e.g. via
                          POST /api/v1/enterprise/api-keys with a tenant-admin
                          session JWT and
                          {"name": "...", "scopes": ["gateway:proxy",
                          "identity:manage", "agdk:check"]} -- this script
                          never does that for you.
    MATIMO_TENANT_ID     The tenant UUID the key above belongs to.
    MATIMO_ADMIN_TOKEN   A tenant-admin (role admin or owner) session JWT.
                          Used ONLY for the handful of admin REST APIs this
                          script calls instead of touching SQL directly:
                          agent suspend/restore, the require-signed-requests
                          toggle, policy create/activate/deactivate/delete,
                          listing pending governance approvals and deciding
                          them, listing configured LLM providers, and
                          browsing the Gateway admin call log. Obtain it the
                          way a human admin would (log in once, copy the
                          token) -- this script never logs in itself.

Optional, with sensible local-dev defaults:
    MATIMO_GATEWAY_URL   Gateway /v1 base URL (default http://localhost:8000/v1)
    MATIMO_GATEWAY_MODEL Optional model for raw chat calls and the signed-LLM
                         scenario; without it a raw call needs a tenant default
                         LLM connection, else Gateway answers no_default_connection
    MATIMO_BACKEND_URL   non-Gateway REST API base (default http://localhost:8000)
    DATABASE_URL         a direct Postgres connection string, tried via
                          psycopg first, falling back to `docker exec
                          <container> psql` when psycopg's platform build is
                          unavailable (the common case on a machine whose
                          Application Control policy blocks psycopg's binary
                          DLL -- this repo's own dev machine is exactly that
                          case). Entirely optional: the admin REST APIs above
                          cover every piece of server state this script used
                          to reach via SQL, EXCEPT three narrow sub-assertions
                          that have no API surface at all --
                          last_telemetry_at staying untouched by an
                          empty-batch heartbeat, a telemetry event's stored
                          attributes (including server-side secret masking),
                          and a matimo_gateway_runs row's existence. Those
                          three sub-assertions SKIP with a printed reason when
                          SQL access isn't available; nothing else in this
                          script touches a database.
    MATIMO_PG_CONTAINER  docker container name for the psql fallback
                          (default agentforge-postgres)
    MATIMO_PG_USER       default agentforge
    MATIMO_PG_DB         default agentforge

Usage (from the matimo-agdk repo root):
    uv run --group dev python scripts/live_check.py
    uv run --group dev python scripts/live_check.py --filter "tool check"

Requires: the local Universal-AgentForge stack running (`docker compose up
-d` for Postgres/Redis, `npm run dev:backend` from UAF/src/backend), and a
tenant that already has an active Matimo Enterprise license. This script
detects a missing license from the very first real Gateway call's error
(POST /v1/identities, a required step regardless) and fails fast with a
clear message -- it never tries to activate one itself.

Extending: add a new `@scenario("name")` function near the bottom, in the
section it belongs to. Scenarios run in declaration order by default;
`--filter SUBSTRING` runs only matching ones (breaks ordering guarantees for
a scenario that depends on another's side effects -- same documented caveat
gateway-manual-test.ts carries). Each scenario gets a struct of already-
provisioned resources (`ctx`) and is responsible for creating/cleaning up
anything scenario-specific itself (a policy, a suspended identity, ...) --
see existing scenarios for the pattern.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from matimo_agdk.config import GatewayConfig  # noqa: E402
from matimo_agdk.exceptions import AgentSuspendedLocally, GatewayError, ToolDenied  # noqa: E402
from matimo_agdk.governor import Governor  # noqa: E402
from matimo_agdk.identity import credentials_paths  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GATEWAY_URL = os.environ.get("MATIMO_GATEWAY_URL", "http://localhost:8000/v1")
BACKEND_URL = os.environ.get("MATIMO_BACKEND_URL", "http://localhost:8000")
PG_CONTAINER = os.environ.get("MATIMO_PG_CONTAINER", "agentforge-postgres")
PG_USER = os.environ.get("MATIMO_PG_USER", "agentforge")
PG_DB = os.environ.get("MATIMO_PG_DB", "agentforge")
DATABASE_URL = os.environ.get("DATABASE_URL")

_REQUIRED_ENV_HELP = {
    "MATIMO_API_KEY": (
        "an org API key with gateway:proxy, identity:manage, and agdk:check scopes. "
        "This script never mints one -- see the module docstring for how to create one."
    ),
    "MATIMO_TENANT_ID": "the tenant UUID the API key above belongs to.",
    "MATIMO_ADMIN_TOKEN": (
        "a tenant-admin (role admin or owner) session JWT, used only for this script's "
        "admin REST API calls. This script never logs in -- see the module docstring."
    ),
}


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        print(
            f"FATAL: required environment variable {name} is not set.\n"
            f"  {name} is {_REQUIRED_ENV_HELP.get(name, '(no description)')}\n"
            'See this script\'s module docstring (`python -c "import scripts.live_check"` '
            "or just read the top of the file) for the full list of required/optional inputs.",
            file=sys.stderr,
        )
        sys.exit(2)
    return value


GATEWAY_MODEL = os.environ.get(
    "MATIMO_GATEWAY_MODEL"
)  # optional; a raw chat call without it needs a tenant default connection


def chat_body(text: str = "hi") -> dict[str, Any]:
    body: dict[str, Any] = {"messages": [{"role": "user", "content": text}]}
    if GATEWAY_MODEL:
        body["model"] = GATEWAY_MODEL
    return body


RUN_TAG = uuid.uuid4().hex[:8]  # disambiguates this run's identities/policies in shared DB state


def _short(name: str) -> str:
    return f"live-check-{name}-{RUN_TAG}"


# ---------------------------------------------------------------------------
# Tiny result-reporting framework
# ---------------------------------------------------------------------------


class Skip(Exception):
    """Raise inside a scenario to record it as SKIPPED rather than PASS/FAIL."""


@dataclass
class ScenarioResult:
    name: str
    outcome: str  # "PASS" | "FAIL" | "SKIP"
    detail: str = ""
    seconds: float = 0.0


_REGISTRY: list[tuple[str, Callable[[Ctx], None]]] = []


def scenario(name: str) -> Callable[[Callable[[Ctx], None]], Callable[[Ctx], None]]:
    def decorator(fn: Callable[[Ctx], None]) -> Callable[[Ctx], None]:
        _REGISTRY.append((name, fn))
        return fn

    return decorator


# ---------------------------------------------------------------------------
# Backend REST helper (non-Gateway routes: admin REST APIs only -- no auth
# routes are ever called from this script)
# ---------------------------------------------------------------------------


def api(
    method: str, path: str, *, token: str | None = None, json_body: Any = None
) -> tuple[int, Any]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = httpx.request(
        method, f"{BACKEND_URL}{path}", headers=headers, json=json_body, timeout=30.0
    )
    try:
        body = resp.json()
    except ValueError:
        body = None
    return resp.status_code, body


def must(condition: object, message: str) -> None:
    if not condition:
        raise AssertionError(message)


# ---------------------------------------------------------------------------
# Direct SQL -- OPTIONAL, and used only for the three sub-assertions with no
# admin API equivalent (see module docstring): last_telemetry_at staying
# untouched by an empty heartbeat, a telemetry event's stored/masked
# attributes, and a matimo_gateway_runs row's existence. psycopg against
# DATABASE_URL when it works, else `docker exec <container> psql`.
# ---------------------------------------------------------------------------

_FIELD_SEP = "\x1f"
_SQL_AVAILABLE: bool | None = None


def _docker_psql(sql: str, *, fetch: bool) -> list[tuple[str, ...]]:
    args = [
        "docker",
        "exec",
        PG_CONTAINER,
        "psql",
        "-U",
        PG_USER,
        "-d",
        PG_DB,
        "-v",
        "ON_ERROR_STOP=1",
    ]
    if fetch:
        args += ["-t", "-A", "-F", _FIELD_SEP]
    args += ["-c", sql]
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"psql failed (rc={proc.returncode}): {proc.stderr.strip()}\nSQL: {sql}")
    if not fetch:
        return []
    rows: list[tuple[str, ...]] = []
    for line in proc.stdout.splitlines():
        if line == "":
            continue
        rows.append(tuple(line.split(_FIELD_SEP)))
    return rows


def sql_query(sql: str) -> list[tuple[str, ...]]:
    if DATABASE_URL:
        try:
            import psycopg  # type: ignore[import-not-found]

            with psycopg.connect(DATABASE_URL) as conn, conn.cursor() as cur:
                cur.execute(sql)
                if cur.description is None:
                    return []
                return [tuple("" if v is None else str(v) for v in row) for row in cur.fetchall()]
        except Exception as exc:  # noqa: BLE001 -- deliberately broad, this is a best-effort path
            print(f"    (psycopg unavailable ({exc}); falling back to docker exec psql)")
    return _docker_psql(sql, fetch=True)


def _sql_str(value: str) -> str:
    """Escapes a value for embedding as a SQL string literal. Every value
    this script embeds is either a server-generated UUID/token or a literal
    this script itself wrote -- there is no untrusted input on this path --
    but literal single quotes are still escaped defensively."""
    return "'" + value.replace("'", "''") + "'"


def _sql_available() -> bool:
    """Probes SQL access exactly once per run and caches the result. Used to
    decide whether the three SQL-only sub-assertions run for real or print a
    SKIP notice -- see module docstring for exactly what those three are."""
    global _SQL_AVAILABLE
    if _SQL_AVAILABLE is None:
        try:
            sql_query("SELECT 1")
            _SQL_AVAILABLE = True
        except Exception as exc:  # noqa: BLE001 -- deliberately broad, this only decides SKIP vs. try
            print(
                f"    (SQL access unavailable ({exc}); the handful of SQL-only sub-assertions "
                "will print SKIP and be omitted -- see module docstring for DATABASE_URL / "
                "MATIMO_PG_* / docker exec)"
            )
            _SQL_AVAILABLE = False
    return _SQL_AVAILABLE


def get_last_telemetry_at(identity_id: str) -> str | None:
    """Raw last_telemetry_at column value ('' if NULL), or None if SQL access
    is unavailable -- callers must treat None as "skip this sub-assertion,"
    not as evidence about the actual column value."""
    if not _sql_available():
        return None
    rows = sql_query(
        f"SELECT last_telemetry_at FROM matimo_agent_identities WHERE id = {_sql_str(identity_id)}"
    )
    return rows[0][0] if rows else ""


# ---------------------------------------------------------------------------
# Provisioning -- reads already-minted credentials from the environment.
# Never logs in, signs up, resets a password, or activates a license.
# ---------------------------------------------------------------------------


@dataclass
class Ctx:
    tenant_id: str
    org_api_key: str
    admin_token: str
    cleanup: list[Callable[[], Any]] = field(default_factory=list)
    # Set by scenario_register(), consumed by scenario_doctor() -- the two
    # CLI-subprocess scenarios share one on-disk identity.
    cli_home: str | None = None
    cli_agent_name: str | None = None

    def defer(self, fn: Callable[[], Any]) -> None:
        self.cleanup.append(fn)


def _jwt_tenant(token: str) -> str | None:
    """Reads tenantId from a JWT payload without verifying it (the server
    verifies; this only catches pointing the script at the wrong tenant)."""
    import base64

    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload)).get("tenantId")
    except Exception:  # noqa: BLE001
        return None


def provision() -> Ctx:
    admin_token_env = os.environ.get("MATIMO_ADMIN_TOKEN", "")
    tenant_env = os.environ.get("MATIMO_TENANT_ID", "")
    if admin_token_env and tenant_env:
        jwt_tenant = _jwt_tenant(admin_token_env)
        must(
            jwt_tenant == tenant_env,
            f"MATIMO_ADMIN_TOKEN belongs to tenant {jwt_tenant}, not MATIMO_TENANT_ID={tenant_env}: "
            "every admin API call (suspend, policies, approvals) would silently act on the wrong tenant",
        )
    api_key = _require_env("MATIMO_API_KEY")
    tenant_id = _require_env("MATIMO_TENANT_ID")
    admin_token = _require_env("MATIMO_ADMIN_TOKEN")
    return Ctx(tenant_id=tenant_id, org_api_key=api_key, admin_token=admin_token)


# ---------------------------------------------------------------------------
# Small scenario-local helpers
# ---------------------------------------------------------------------------


def fresh_governor(ctx: Ctx, name: str, **overrides: Any) -> Governor:
    """A Governor bound to a freshly registered identity, not persisted to
    disk (scenario 1 covers persistence separately). Registered identities
    can't be deleted server-side (no such endpoint exists -- see
    docs/SERVER-CONTRACT.md section 10) so these accumulate as inert rows,
    same as every other script that has ever exercised this Gateway."""
    config = GatewayConfig(
        base_url=GATEWAY_URL,
        api_key=ctx.org_api_key,
        agent_name=_short(name),
        framework="custom",
        telemetry_flush_interval=overrides.pop("telemetry_flush_interval", 9999.0),
        heartbeat_interval=overrides.pop("heartbeat_interval", 9999.0),
        **overrides,
    )
    gov = Governor(config)
    identity = gov.register(display_name=_short(name), framework="custom", persist=False)
    print(f"    registered identity {identity.identity_id} ({identity.display_name})")
    return gov


# -- Admin REST API helpers (replace every SQL write this script used to do) --


def suspend_identity(ctx: Ctx, identity_id: str, reason: str) -> None:
    status, body = api(
        "PUT",
        f"/api/v1/enterprise/agents/{identity_id}/suspend",
        token=ctx.admin_token,
        json_body={"reason": reason},
    )
    must(status in (200, 204), f"suspend of identity {identity_id} failed ({status}): {body}")


def restore_identity(ctx: Ctx, identity_id: str) -> None:
    status, body = api(
        "PUT", f"/api/v1/enterprise/agents/{identity_id}/restore", token=ctx.admin_token
    )
    must(status in (200, 204), f"restore of identity {identity_id} failed ({status}): {body}")


def set_require_signed_requests(ctx: Ctx, identity_id: str, value: bool) -> None:
    status, body = api(
        "PUT",
        f"/api/v1/enterprise/agents/{identity_id}/require-signed-requests",
        token=ctx.admin_token,
        json_body={"requireSignedRequests": value},
    )
    must(
        status in (200, 204),
        f"setting requireSignedRequests={value} for {identity_id} failed ({status}): {body}",
    )


def create_and_activate_policy(
    ctx: Ctx, name: str, conditions: list[dict[str, Any]], action: str
) -> str:
    """Creates and activates a tenant policy via the real admin API
    (POST /policies, then POST /policies/:id/activate) rather than inserting
    directly into matimo_enterprise_policies. `policyYaml` is parsed as YAML
    server-side (PolicyEngineService.compileFromYaml), and a JSON array (as
    built here) is valid YAML -- so the same compiled-rule shape the old
    direct-SQL insert used can be sent as-is via json.dumps()."""
    rule_id = f"agdk-live-{uuid.uuid4().hex[:8]}"
    compiled = [
        {
            "id": rule_id,
            "name": name,
            "conditions": conditions,
            "action": action,
            "priority": 950,
            "tags": ["agdk-live-check"],
        }
    ]
    status, body = api(
        "POST",
        "/api/v1/enterprise/policies",
        token=ctx.admin_token,
        json_body={
            "name": name,
            "description": "AGDK live_check scenario policy",
            "policyYaml": json.dumps(compiled),
            "tags": ["agdk-live-check"],
        },
    )
    must(status in (200, 201), f"policy creation failed ({status}): {body}")
    policy_id = body["data"]["id"]

    status, body = api(
        "POST", f"/api/v1/enterprise/policies/{policy_id}/activate", token=ctx.admin_token
    )
    must(status in (200, 201), f"policy activation failed ({status}): {body}")
    return policy_id


def deactivate_and_delete_policy(ctx: Ctx, policy_id: str) -> None:
    """Best-effort cleanup: deactivate (a policy can't be deleted while
    active), then attempt a hard delete. A policy that already produced a
    decision can hit an audit-table trigger that blocks the delete cascade
    server-side (see CHANGELOG.md, 2026-09-18 live verification, if this warns during a run) -- that
    is tolerated here, same as the old direct-SQL cleanup's own documented
    fallback-to-deactivation behavior."""
    api("POST", f"/api/v1/enterprise/policies/{policy_id}/deactivate", token=ctx.admin_token)
    status, body = api("DELETE", f"/api/v1/enterprise/policies/{policy_id}", token=ctx.admin_token)
    if status not in (200, 204):
        print(
            f"    (warning: could not hard-delete policy {policy_id} ({status}): {body} "
            "-- left deactivated)"
        )


def latest_pending_approval(ctx: Ctx, identity_id: str, timeout: float = 10.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status, body = api(
            "GET",
            "/api/v1/enterprise/governance-approvals?status=pending&limit=100",
            token=ctx.admin_token,
        )
        must(status == 200, f"listing governance approvals failed ({status}): {body}")
        rows = body.get("data") or []
        matches = [
            r
            for r in rows
            if r.get("agentIdentityId") == identity_id and r.get("requestKind") == "tool_check"
        ]
        if matches:
            matches.sort(key=lambda r: r.get("createdAt") or "", reverse=True)
            return matches[0]["id"]
        time.sleep(0.3)
    raise AssertionError(
        f"no pending tool_check approval appeared for identity {identity_id} within {timeout}s"
    )


def decide_approval(ctx: Ctx, request_id: str, decision: str) -> None:
    status, body = api(
        "POST",
        f"/api/v1/enterprise/governance-approvals/{request_id}/{decision}",
        token=ctx.admin_token,
        json_body={"reason": f"AGDK live_check {decision}"},
    )
    must(status in (200, 204), f"{decision} of approval {request_id} failed ({status}): {body}")


def call_log_rows(ctx: Ctx, identity_id: str, limit: int = 5) -> list[dict[str, Any]]:
    status, body = api(
        "GET",
        f"/api/v1/gateway/call-log?agentIdentityId={identity_id}&limit={limit}",
        token=ctx.admin_token,
    )
    must(status == 200, f"call-log fetch failed ({status}): {body}")
    return (body.get("data") or {}).get("rows") or []


def run_in_thread(fn: Callable[[], Any]) -> tuple[threading.Thread, dict[str, Any]]:
    result: dict[str, Any] = {}

    def _target() -> None:
        try:
            result["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 -- re-surfaced to the joining thread below
            result["error"] = exc

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    return t, result


# ---------------------------------------------------------------------------
# Scenarios: 1-2, register / doctor (real CLI subprocess, isolated $HOME)
# ---------------------------------------------------------------------------


@scenario("register: matimo-agdk register creates an identity, 0600 perms, key never printed")
def scenario_register(ctx: Ctx) -> None:
    with tempfile.TemporaryDirectory(prefix="agdk-live-check-home-") as fake_home:
        agent_name = _short("register")
        env = {**os.environ, "USERPROFILE": fake_home, "HOME": fake_home}
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "matimo_agdk.cli",
                "register",
                "--name",
                agent_name,
                "--framework",
                "custom",
                "--api-key",
                ctx.org_api_key,
                "--gateway-url",
                GATEWAY_URL,
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        must(proc.returncode == 0, f"matimo-agdk register exited {proc.returncode}: {proc.stderr}")
        must("Registered identity" in proc.stdout, f"unexpected register output: {proc.stdout}")
        must(
            "BEGIN PRIVATE KEY" not in proc.stdout and "BEGIN PRIVATE KEY" not in proc.stderr,
            "the private key PEM must never be printed to stdout/stderr",
        )

        meta_path, key_path = credentials_paths(agent_name, Path(fake_home) / ".matimo" / "agents")
        must(meta_path.exists(), f"expected credentials metadata file at {meta_path}")
        must(key_path.exists(), f"expected private key file at {key_path}")
        meta = json.loads(meta_path.read_text())
        must(
            meta["identity_token"].startswith("me-id-"), f"unexpected identity_token shape: {meta}"
        )

        if os.name != "nt":
            import stat

            mode = stat.S_IMODE(key_path.stat().st_mode)
            must(mode == 0o600, f"expected 0600 on the private key file, got {oct(mode)}")
        else:
            print(
                "    (Windows: os.chmod does not enforce real ACL restriction -- documented gap, not asserted)"
            )

        # Stash the fake home for scenario_doctor, which needs the exact
        # same registered identity on disk to exercise `matimo-agdk doctor`
        # against a real, already-registered identity. TemporaryDirectory
        # would delete `fake_home` on context exit, so copy it somewhere
        # durable first; scenario_doctor's caller cleans that up.
        durable = Path(tempfile.mkdtemp(prefix="agdk-live-check-home-durable-"))
        for item in Path(fake_home).rglob("*"):
            if item.is_file():
                rel = item.relative_to(fake_home)
                target = durable / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(item.read_bytes())
        ctx.cli_home = str(durable)
        ctx.cli_agent_name = agent_name
        ctx.defer(lambda: _rmtree_ignore_errors(durable))


def _rmtree_ignore_errors(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)


@scenario("doctor: matimo-agdk doctor passes (handshake + heartbeat)")
def scenario_doctor(ctx: Ctx) -> None:
    fake_home = ctx.cli_home
    agent_name = ctx.cli_agent_name
    if not fake_home or not agent_name:
        raise Skip(
            "scenario_register did not run first (filtered out?) -- doctor needs its identity on disk"
        )
    env = {**os.environ, "USERPROFILE": fake_home, "HOME": fake_home}
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "matimo_agdk.cli",
            "doctor",
            "--name",
            agent_name,
            "--api-key",
            ctx.org_api_key,
            "--gateway-url",
            GATEWAY_URL,
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    must(
        proc.returncode == 0,
        f"matimo-agdk doctor exited {proc.returncode}: {proc.stdout}\n{proc.stderr}",
    )
    must("doctor: all checks passed" in proc.stdout, f"unexpected doctor output: {proc.stdout}")
    must(
        "lifecycle_status=active" in proc.stdout,
        f"expected an active lifecycle status: {proc.stdout}",
    )


# ---------------------------------------------------------------------------
# Scenario 3: heartbeat
# ---------------------------------------------------------------------------


@scenario("heartbeat: empty batch reports active lifecycle, does not touch last_telemetry_at")
def scenario_heartbeat(ctx: Ctx) -> None:
    gov = fresh_governor(ctx, "heartbeat")
    identity_id = gov.identity.identity_id  # type: ignore[union-attr]
    before = get_last_telemetry_at(identity_id)
    if before is None:
        print("    SKIP sub-assertion: last_telemetry_at baseline check (no SQL access)")
    else:
        must(before == "", f"expected a fresh identity to have no telemetry yet, got {before!r}")

    gov.start()
    try:
        gov._telemetry.flush_now()  # noqa: SLF001 -- forces one heartbeat now, same idiom the CLI itself uses
        state = gov.state
        must(state.lifecycle_status == "active", f"expected active, got {state.lifecycle_status!r}")
        must(state.server_time is not None, "expected a serverTime field on the heartbeat")
    finally:
        gov.stop()

    after = get_last_telemetry_at(identity_id)
    if before is None or after is None:
        print(
            "    SKIP sub-assertion: last_telemetry_at unchanged-after-heartbeat check (no SQL access)"
        )
    else:
        must(
            after == before == "",
            f"an empty-events heartbeat must never touch last_telemetry_at: before={before!r} after={after!r}",
        )


# ---------------------------------------------------------------------------
# Scenario 4: telemetry
# ---------------------------------------------------------------------------


@scenario(
    "telemetry: run+llm+tool spans land with gen_ai.* attrs, a secret gets server-masked, run row correlates"
)
def scenario_telemetry(ctx: Ctx) -> None:
    gov = fresh_governor(ctx, "telemetry")
    identity_id = gov.identity.identity_id  # type: ignore[union-attr]
    gov.start()
    try:
        with gov.run("live-check-telemetry-run") as run_id:
            gov.llm_span(model="gpt-4o-mini", provider="openai", status="completed", duration_ms=42)
            # "note" doesn't match any client-side key-name redaction marker
            # (password/token/secret/key/authorization) -- this specifically
            # proves Gateway's own server-side VALUE-pattern secret masking,
            # not the SDK's local key-NAME-based redaction (which would
            # already strip anything literally named "apiKey" before it
            # ever left this process).
            gov.tool_span(
                "live_check_tool",
                status="completed",
                duration_ms=7,
                attributes={"note": "sk-1234567890abcdefghijklmnop"},
            )
        gov._telemetry.flush_now()  # noqa: SLF001
    finally:
        gov.stop()

    if not _sql_available():
        print(
            "    SKIP sub-assertion: telemetry event attributes/masking + matimo_gateway_runs "
            "row checks (no SQL access -- see module docstring)"
        )
        return

    rows = sql_query(
        "SELECT kind, attributes FROM matimo_agent_telemetry_events "
        f"WHERE agent_identity_id = {_sql_str(identity_id)} AND run_id = {_sql_str(run_id)} "
        "ORDER BY created_at"
    )
    kinds = [r[0] for r in rows]
    must({"run", "llm", "tool"} <= set(kinds), f"expected run+llm+tool events, got kinds={kinds}")

    llm_attrs = json.loads(next(r[1] for r in rows if r[0] == "llm"))
    must(
        llm_attrs.get("gen_ai.request.model") == "gpt-4o-mini", f"unexpected llm attrs: {llm_attrs}"
    )
    must(llm_attrs.get("gen_ai.operation.name") == "chat", f"unexpected llm attrs: {llm_attrs}")

    tool_attrs = json.loads(next(r[1] for r in rows if r[0] == "tool"))
    must(
        tool_attrs.get("gen_ai.tool.name") == "live_check_tool",
        f"unexpected tool attrs: {tool_attrs}",
    )
    note = tool_attrs.get("note", "")
    must(
        "sk-1234567890abcdefghijklmnop" not in note and "REDACT" in note,
        f"expected Gateway's server-side secret masking to have redacted the secret-shaped value, got: {note!r}",
    )

    run_rows = sql_query(
        "SELECT status FROM matimo_gateway_runs "
        f"WHERE tenant_id = {_sql_str(ctx.tenant_id)} AND agent_identity_id = {_sql_str(identity_id)} "
        f"AND run_id = {_sql_str(run_id)}"
    )
    must(
        len(run_rows) == 1,
        f"expected exactly one matimo_gateway_runs row for run_id={run_id}, got {run_rows}",
    )


# ---------------------------------------------------------------------------
# Scenario 5 (extra, unconditional): signature enforcement actually verifies
# ---------------------------------------------------------------------------


@scenario(
    "signature enforcement: requireSignedRequests actually verifies the JWS (no LLM key needed)"
)
def scenario_signature_enforcement(ctx: Ctx) -> None:
    gov = fresh_governor(ctx, "sig-enforce")
    identity_id = gov.identity.identity_id  # type: ignore[union-attr]
    set_require_signed_requests(ctx, identity_id, True)
    try:
        client = gov.httpx_client()
        try:
            # A properly signed call must get PAST the signature gate --
            # whatever it fails on next (no default BYOK connection, a fake
            # key's dispatch failure, ...) is fine; a 403 signature_required
            # here would mean signing is broken.
            resp = client.post("/chat/completions", json=chat_body())
            must(
                resp.status_code != 403 or resp.json().get("error") != "signature_required",
                f"a validly signed call was rejected as unsigned: {resp.status_code} {resp.text}",
            )
        finally:
            client.close()

        # An unsigned call against the same now-signature-required identity
        # must be rejected -- proves enforcement isn't a no-op.
        unsigned = httpx.post(
            f"{GATEWAY_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {ctx.org_api_key}",
                "X-Matimo-Session-Token": gov._session.get_token(),  # noqa: SLF001
            },
            json=chat_body(),
            timeout=15.0,
        )
        must(
            unsigned.status_code == 403 and unsigned.json().get("error") == "signature_required",
            f"expected 403 signature_required for an unsigned call once requireSignedRequests=true, "
            f"got {unsigned.status_code}: {unsigned.text}",
        )
    finally:
        set_require_signed_requests(ctx, identity_id, False)


# ---------------------------------------------------------------------------
# Scenario 6: signed LLM call using the tenant's own configured provider
# (SKIP if the tenant has none -- this script never creates one)
# ---------------------------------------------------------------------------


@scenario(
    "signed LLM call: governor.httpx_client() + the OpenAI SDK completes a real chat completion"
)
def scenario_signed_llm_call(ctx: Ctx) -> None:
    try:
        import openai
    except ImportError as exc:
        raise Skip(f"openai SDK not installed in this venv: {exc}") from exc

    status, body = api("GET", "/api/llm-providers", token=ctx.admin_token)
    must(status == 200, f"listing LLM providers failed ({status}): {body}")
    providers = (body.get("data") or {}).get("providers") or []
    # Gateway is BYOK-only in v1 (docs/SERVER-CONTRACT.md section 6.1) -- a
    # nova_managed/nova_credits connection would 400 non_byok_connection.
    usable = [
        p
        for p in providers
        if p.get("is_active") and p.get("credential_source") == "byok" and p.get("models")
    ]
    if not usable:
        raise Skip("tenant has no configured LLM provider")
    provider = usable[0]
    # MATIMO_GATEWAY_MODEL wins when set: a tenant with many leftover test
    # providers cannot be disambiguated from the list alone.
    model = GATEWAY_MODEL or provider["models"][0]
    print(
        f"    using tenant's configured provider {provider.get('name')!r} ({provider.get('provider')}), model {model!r}"
    )

    gov = fresh_governor(ctx, "llm-call")
    gov.start()
    try:
        with gov.run("live-check-signed-llm"):
            client = openai.OpenAI(**gov.openai_client_kwargs(), http_client=gov.httpx_client())
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Reply with exactly one word: pong"}],
            )
            must(completion.choices, f"expected at least one choice, got {completion}")
            gov.llm_span(
                model=model, provider=provider.get("provider", "unknown"), status="completed"
            )
    finally:
        gov.stop()


# ---------------------------------------------------------------------------
# Scenario 7: rapid suspend
# ---------------------------------------------------------------------------


@scenario("rapid suspend: local state flips within one heartbeat, next tool check DENYs live")
def scenario_rapid_suspend(ctx: Ctx) -> None:
    gov = fresh_governor(ctx, "suspend", telemetry_flush_interval=1.0, heartbeat_interval=2.0)
    identity_id = gov.identity.identity_id  # type: ignore[union-attr]
    gov.start()
    try:
        gov._telemetry.flush_now()  # noqa: SLF001 -- baseline: prove it starts active
        must(not gov.is_suspended(), f"expected an active baseline, got {gov.state}")

        suspend_identity(ctx, identity_id, "AGDK live_check rapid_suspend scenario")
        try:
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline and not gov.is_suspended():
                time.sleep(0.25)
            must(
                gov.is_suspended(),
                f"expected the background heartbeat to observe suspension within 10s, state={gov.state}",
            )

            raised = False
            try:
                gov.raise_if_suspended()
            except AgentSuspendedLocally:
                raised = True
            must(raised, "expected raise_if_suspended() to raise once locally marked suspended")

            decision = gov.check_tool("live_check_any_tool", {"x": 1})
            must(decision.denied, f"expected a live DENY for a suspended identity, got {decision}")
            must(
                decision.reason == "agent_suspended",
                f"expected reason=agent_suspended, got {decision.reason!r}",
            )
        finally:
            restore_identity(ctx, identity_id)
    finally:
        gov.stop()


# ---------------------------------------------------------------------------
# Scenarios 8-11: tool governance
# ---------------------------------------------------------------------------


@scenario("tool check ALLOW: guard() runs the wrapped function under a permissive policy")
def scenario_tool_check_allow(ctx: Ctx) -> None:
    gov = fresh_governor(ctx, "tool-allow")
    calls: list[int] = []

    @gov.guard(name="live_check_allowed_tool")
    def add(a: int, b: int) -> int:
        calls.append(1)
        return a + b

    with gov.run("live-check-tool-allow"):
        result: int = add(a=2, b=3)
    must(result == 5, f"expected the tool to actually run and return 5, got {result}")
    must(len(calls) == 1, f"expected the tool body to run exactly once, ran {len(calls)} times")


@scenario("tool check DENY: guard() raises ToolDenied, the tool body never runs")
def scenario_tool_check_deny(ctx: Ctx) -> None:
    gov = fresh_governor(ctx, "tool-deny")
    tool_name = _short("deny-tool")
    policy_id = create_and_activate_policy(
        ctx,
        "AGDK live_check deny destructive tool",
        [
            {"field": "call.kind", "operator": "equals", "value": "tool"},
            {"field": "tool.name", "operator": "equals", "value": tool_name},
        ],
        "DENY",
    )
    try:
        calls: list[int] = []

        @gov.guard(name=tool_name)
        def delete_everything() -> str:
            calls.append(1)
            return "should never run"

        denied = False
        try:
            with gov.run("live-check-tool-deny"):
                delete_everything()
        except ToolDenied:
            denied = True
        must(denied, "expected guard() to raise ToolDenied against the matching DENY policy")
        must(len(calls) == 0, "the tool body must never have run")
    finally:
        deactivate_and_delete_policy(ctx, policy_id)


@scenario("tool check PENDING -> approved: guard() blocks, approving via the API lets it proceed")
def scenario_tool_check_pending_approved(ctx: Ctx) -> None:
    gov = fresh_governor(ctx, "tool-pending-ok")
    identity_id = gov.identity.identity_id  # type: ignore[union-attr]
    tool_name = _short("pending-ok-tool")
    gov.set_tool_category(tool_name, "financial")
    policy_id = create_and_activate_policy(
        ctx,
        "AGDK live_check HITL financial tools (approve path)",
        [
            {"field": "call.kind", "operator": "equals", "value": "tool"},
            {"field": "tool.category", "operator": "equals", "value": "financial"},
        ],
        "ALLOW_WITH_CONDITIONS",
    )
    try:
        calls: list[int] = []

        @gov.guard(name=tool_name, category="financial")
        def wire_transfer(amount: int) -> str:
            calls.append(1)
            return f"transferred {amount}"

        def _call() -> str:
            with gov.run("live-check-tool-pending-approved"):
                return wire_transfer(amount=100)

        thread, result = run_in_thread(_call)
        request_id = latest_pending_approval(ctx, identity_id)
        decide_approval(ctx, request_id, "approve")
        thread.join(timeout=30.0)
        must(not thread.is_alive(), "guard() thread did not finish within 30s of approval")
        if "error" in result:
            raise AssertionError(
                f"guard() raised unexpectedly on the approve path: {result['error']!r}"
            ) from result["error"]
        must(
            result.get("value") == "transferred 100",
            f"expected the tool to have run post-approval, got {result}",
        )
        must(len(calls) == 1, f"expected the tool body to run exactly once, ran {len(calls)} times")
    finally:
        deactivate_and_delete_policy(ctx, policy_id)


@scenario("tool check PENDING -> rejected: guard() raises ToolDenied once rejected")
def scenario_tool_check_pending_rejected(ctx: Ctx) -> None:
    gov = fresh_governor(ctx, "tool-pending-no")
    identity_id = gov.identity.identity_id  # type: ignore[union-attr]
    tool_name = _short("pending-no-tool")
    gov.set_tool_category(tool_name, "financial")
    policy_id = create_and_activate_policy(
        ctx,
        "AGDK live_check HITL financial tools (reject path)",
        [
            {"field": "call.kind", "operator": "equals", "value": "tool"},
            {"field": "tool.category", "operator": "equals", "value": "financial"},
        ],
        "ALLOW_WITH_CONDITIONS",
    )
    try:
        calls: list[int] = []

        @gov.guard(name=tool_name, category="financial")
        def wire_transfer(amount: int) -> str:
            calls.append(1)
            return f"transferred {amount}"

        def _call() -> str:
            with gov.run("live-check-tool-pending-rejected"):
                return wire_transfer(amount=999)

        thread, result = run_in_thread(_call)
        request_id = latest_pending_approval(ctx, identity_id)
        decide_approval(ctx, request_id, "reject")
        thread.join(timeout=30.0)
        must(not thread.is_alive(), "guard() thread did not finish within 30s of rejection")
        must("error" in result, "expected guard() to raise on the reject path")
        must(
            isinstance(result["error"], ToolDenied), f"expected ToolDenied, got {result['error']!r}"
        )
        must(len(calls) == 0, "the tool body must never have run once rejected")
    finally:
        deactivate_and_delete_policy(ctx, policy_id)


# ---------------------------------------------------------------------------
# Scenario 12: LangChain adapter live
# ---------------------------------------------------------------------------


@scenario(
    "LangChain adapter live: govern_tools() ALLOWs/DENYs against a real tool, no crash on DENY"
)
def scenario_langchain_adapter(ctx: Ctx) -> None:
    try:
        from langchain_core.tools import tool as lc_tool
    except ImportError as exc:
        raise Skip(f"langchain-core not installed in this venv: {exc}") from exc

    from matimo_agdk.adapters.langchain import govern_tools

    gov = fresh_governor(ctx, "langchain")
    tool_name = _short("lc-tool")

    # -- ALLOW path: no policy, real result comes back ----------------------
    calls: list[int] = []

    @lc_tool(tool_name)
    def multiply(a: int, b: int) -> int:
        """Multiplies two integers."""
        calls.append(1)
        return a * b

    govern_tools([multiply], gov, mode="govern")
    with gov.run("live-check-langchain-allow"):
        result = multiply.invoke({"a": 6, "b": 7})
    must(result == 42, f"expected the real tool result 42, got {result}")
    must(len(calls) == 1, f"expected exactly one real invocation, got {len(calls)}")

    # -- DENY path: a matching policy makes it a graceful tool error, not a
    #    crash -- handle_tool_error=True is the documented LangChain idiom
    #    that lets BaseTool.run()'s own error machinery convert a raised
    #    ToolDenied into a string observation instead of propagating it.
    deny_tool_name = _short("lc-deny-tool")
    deny_calls: list[int] = []

    @lc_tool(deny_tool_name)
    def dangerous(x: int) -> int:
        """A tool that should never actually run once denied."""
        deny_calls.append(1)
        return x

    dangerous.handle_tool_error = True
    govern_tools([dangerous], gov, mode="govern")

    policy_id = create_and_activate_policy(
        ctx,
        "AGDK live_check LangChain deny",
        [
            {"field": "call.kind", "operator": "equals", "value": "tool"},
            {"field": "tool.name", "operator": "equals", "value": deny_tool_name},
        ],
        "DENY",
    )
    try:
        with gov.run("live-check-langchain-deny"):
            outcome = dangerous.run({"x": 1})
        must(isinstance(outcome, str), f"expected a string tool-error observation, got {outcome!r}")
        must(len(deny_calls) == 0, "the tool body must never have run once denied")
    finally:
        deactivate_and_delete_policy(ctx, policy_id)


# ---------------------------------------------------------------------------
# Scenario 13: session expiry
# ---------------------------------------------------------------------------


@scenario(
    "session expiry: DELETE /v1/sessions behind the SDK's back, next call transparently re-handshakes"
)
def scenario_session_expiry(ctx: Ctx) -> None:
    gov = fresh_governor(ctx, "session-expiry")
    gov.start()
    try:
        gov._telemetry.flush_now()  # noqa: SLF001 -- establishes the first session
        first_token = gov._session._state.token  # noqa: SLF001
        must(first_token, "expected a session token after the first flush")

        deleted = httpx.request(
            "DELETE",
            f"{GATEWAY_URL}/sessions",
            headers={
                "Authorization": f"Bearer {ctx.org_api_key}",
                "X-Matimo-Session-Token": first_token,
            },
            timeout=15.0,
        )
        must(
            deleted.status_code == 204,
            f"expected 204 deleting the session, got {deleted.status_code}: {deleted.text}",
        )

        # Path 1: the telemetry exporter's own session-consuming call
        # (the bug this run found: _flush() used to call get_token()
        # directly, bypassing call_with_retry() -- see CHANGELOG.md (2026-09-18 live verification)).
        gov._telemetry.submit(
            {"runId": "live-check-session-expiry", "kind": "log", "name": "after-delete"}
        )  # noqa: SLF001
        gov._telemetry.flush_now()  # noqa: SLF001
        second_token = gov._session._state.token  # noqa: SLF001
        must(
            second_token != first_token,
            "expected the SDK to have transparently re-handshaked with a new session token",
        )
        must(
            gov.state.lifecycle_status == "active",
            f"expected the retried flush to have landed and updated state, got {gov.state}",
        )

        # Path 2: the httpx_client() transport-level retry (SessionRetryTransport).
        # Delete again, then make a /chat/completions call through the
        # signed client -- this must reach past the session gate (any
        # non-401 outcome proves it), not fail with a raw 401.
        deleted2 = httpx.request(
            "DELETE",
            f"{GATEWAY_URL}/sessions",
            headers={
                "Authorization": f"Bearer {ctx.org_api_key}",
                "X-Matimo-Session-Token": second_token,
            },
            timeout=15.0,
        )
        must(
            deleted2.status_code == 204,
            f"expected 204 deleting the session again, got {deleted2.status_code}",
        )

        client = gov.httpx_client()
        try:
            resp = client.post("/chat/completions", json=chat_body())
        finally:
            client.close()
        must(
            not (resp.status_code == 401 and resp.json().get("error") == "session_expired"),
            f"expected the httpx client to have transparently re-handshaked, got a raw 401 session_expired: {resp.text}",
        )
    finally:
        gov.stop()


# ---------------------------------------------------------------------------
# Scenario 14: key rotation
# ---------------------------------------------------------------------------


@scenario(
    "key rotation: rotate-key produces a genuinely new key; JWKS updates; a signed call still verifies"
)
def scenario_key_rotation(ctx: Ctx) -> None:
    gov = fresh_governor(ctx, "key-rotation")
    identity_id = gov.identity.identity_id  # type: ignore[union-attr]
    old_token = gov.identity.identity_token  # type: ignore[union-attr]

    # /v1/identities/:id/jwks lives on the Gateway router (identity:manage
    # scope, Bearer org key), not the backend REST API -- call it against
    # GATEWAY_URL directly rather than api()'s BACKEND_URL.
    jwks_before_resp = httpx.get(
        f"{GATEWAY_URL}/identities/{identity_id}/jwks",
        headers={"Authorization": f"Bearer {ctx.org_api_key}"},
        timeout=15.0,
    )
    must(
        jwks_before_resp.status_code == 200,
        f"jwks fetch failed: {jwks_before_resp.status_code} {jwks_before_resp.text}",
    )
    jwk_before = jwks_before_resp.json()["keys"][0]
    must(
        jwk_before["kid"] == old_token, f"expected kid to equal the identityToken, got {jwk_before}"
    )

    new_identity = gov.rotate_key()
    must(
        new_identity.identity_token == old_token,
        "rotate-key must not change the long-lived identity_token",
    )

    jwks_after_resp = httpx.get(
        f"{GATEWAY_URL}/identities/{identity_id}/jwks",
        headers={"Authorization": f"Bearer {ctx.org_api_key}"},
        timeout=15.0,
    )
    must(
        jwks_after_resp.status_code == 200,
        f"jwks fetch failed post-rotation: {jwks_after_resp.status_code}",
    )
    jwk_after = jwks_after_resp.json()["keys"][0]
    must(jwk_after["kid"] == old_token, "kid (=identityToken) must stay the same across rotation")
    must(
        (jwk_after["x"], jwk_after["y"]) != (jwk_before["x"], jwk_before["y"]),
        "expected the JWK's public key coordinates to actually change after rotation",
    )

    # "a signed call still verifies" -- neither a session handshake nor a
    # tool check needs a real LLM provider key, so this is fully verified
    # here rather than SKIPped: gov.start() below performs a fresh signed
    # handshake using the NEW private key (the old key was invalidated
    # server-side the moment rotate_key() overwrote public_key).
    gov.start()
    try:
        gov._telemetry.flush_now()  # noqa: SLF001 -- forces the handshake to actually happen now
        must(
            gov.state.lifecycle_status == "active",
            f"expected the post-rotation handshake to succeed, got {gov.state}",
        )
        decision = gov.check_tool("live_check_post_rotation_tool", {"x": 1})
        must(
            decision.allowed,
            f"expected a signed tool check with the new key to succeed, got {decision}",
        )
    finally:
        gov.stop()


# ---------------------------------------------------------------------------
# Scenario 15: DENY visible in the admin call log (no LLM key needed)
# ---------------------------------------------------------------------------


@scenario(
    "call log: a policy-denied chat completion for a suspended identity shows policyOutcome=DENY via GET /api/v1/gateway/call-log"
)
def scenario_call_log_deny(ctx: Ctx) -> None:
    gov = fresh_governor(ctx, "call-log-deny")
    identity_id = gov.identity.identity_id  # type: ignore[union-attr]
    suspend_identity(ctx, identity_id, "AGDK live_check call-log DENY scenario")
    try:
        client = gov.httpx_client()
        try:
            resp = client.post("/chat/completions", json=chat_body())
        finally:
            client.close()
        must(
            resp.status_code == 403 and resp.json().get("error") == "policy_denied",
            f"expected a 403 policy_denied for a suspended identity, got {resp.status_code}: {resp.text}",
        )

        deadline = time.monotonic() + 10.0
        rows: list[dict[str, Any]] = []
        while time.monotonic() < deadline:
            rows = call_log_rows(ctx, identity_id)
            if any(r.get("policyOutcome") == "DENY" for r in rows):
                return
            time.sleep(0.5)
        raise AssertionError(
            f"no policyOutcome=DENY row appeared in the admin call log for identity {identity_id} "
            f"within 10s: last rows={rows}"
        )
    finally:
        restore_identity(ctx, identity_id)


@scenario(
    "messages route auth: an Anthropic SDK client authenticates to Gateway with Authorization Bearer"
)
def scenario_anthropic_auth(ctx: Ctx) -> None:
    """The Anthropic SDK sends `api_key=` as x-api-key, which Gateway never
    reads; anthropic_client_kwargs() therefore passes the org key as
    `auth_token=` (Authorization: Bearer). Proves the request gets past
    Gateway's auth and session gates; whether it completes depends on the
    tenant having an Anthropic provider, which is not required here."""
    try:
        import anthropic
    except ImportError as exc:
        raise Skip(f"anthropic SDK not installed in this venv: {exc}") from exc

    gov = fresh_governor(ctx, "anthropic-auth")
    gov.start()
    try:
        with gov.run("live-check-anthropic-auth"):
            client = anthropic.Anthropic(**gov.anthropic_client_kwargs(), http_client=gov.httpx_client())
            try:
                client.messages.create(
                    model=GATEWAY_MODEL or "claude-3-5-haiku-latest",
                    max_tokens=8,
                    messages=[{"role": "user", "content": "Reply with one word: pong"}],
                )
                print("    completed a real /v1/messages call")
            except anthropic.APIStatusError as exc:
                body = exc.body if isinstance(exc.body, dict) else {}
                code = body.get("error")
                must(
                    exc.status_code != 401
                    and code
                    not in (
                        "unauthorized",
                        "invalid_api_key",
                        "missing_api_key",
                        "session_required",
                    ),
                    f"Gateway rejected the Anthropic client's credentials: {exc.status_code} {body}",
                )
                print(
                    f"    past auth and session gates ({exc.status_code} {code}); no Anthropic provider needed for this check"
                )
    finally:
        gov.stop()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--filter", default=None, help="run only scenarios whose name contains this substring"
    )
    args = parser.parse_args()

    print(f"Matimo AGDK live_check -- gateway={GATEWAY_URL} backend={BACKEND_URL}")
    ctx = provision()

    print(
        "Preflight: confirming the tenant has an active Matimo Enterprise license via a real "
        "POST /v1/identities call (there is no lighter-weight check, and this script never "
        "activates a license itself)..."
    )
    try:
        probe = fresh_governor(ctx, "license-preflight")
    except GatewayError as exc:
        print(
            f"\nFATAL: the first real Gateway call failed ({exc.code}): {exc.message}\n"
            "This script never activates a Matimo Enterprise license, mints an API key, or "
            "creates any other credential -- fix the environment (activate a license for this "
            "tenant, or check MATIMO_API_KEY's scopes/validity) and re-run.",
            file=sys.stderr,
        )
        return 2
    print(
        f"  OK -- preflight identity {probe.identity.identity_id} registered "  # type: ignore[union-attr]
        "(identities are never deleted server-side, see docs/SERVER-CONTRACT.md section 10)"
    )

    results: list[ScenarioResult] = []
    for name, fn in _REGISTRY:
        if args.filter and args.filter.lower() not in name.lower():
            continue
        print(f"\n=== {name} ===")
        started = time.monotonic()
        try:
            fn(ctx)
        except Skip as exc:
            elapsed = time.monotonic() - started
            print(f"SKIP  ({elapsed:.1f}s) {exc}")
            results.append(ScenarioResult(name, "SKIP", str(exc), elapsed))
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: keep going after a failure
            elapsed = time.monotonic() - started
            tb = traceback.format_exc()
            print(f"FAIL  ({elapsed:.1f}s) {exc}\n{tb}")
            results.append(ScenarioResult(name, "FAIL", str(exc), elapsed))
        else:
            elapsed = time.monotonic() - started
            print(f"PASS  ({elapsed:.1f}s)")
            results.append(ScenarioResult(name, "PASS", "", elapsed))

    print("\nCleaning up provisioned resources...")
    for cleanup_fn in reversed(ctx.cleanup):
        with contextlib.suppress(Exception):
            cleanup_fn()

    passed = sum(1 for r in results if r.outcome == "PASS")
    failed = sum(1 for r in results if r.outcome == "FAIL")
    skipped = sum(1 for r in results if r.outcome == "SKIP")

    print("\n--- summary ---")
    for r in results:
        marker = {"PASS": "PASS", "FAIL": "FAIL", "SKIP": "SKIP"}[r.outcome]
        print(f"{marker:5s} {r.name}")
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
