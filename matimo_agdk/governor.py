"""Governor: the one public entry point.

Three lines to govern any agent::

    governor = Governor.from_env()
    governor.start()
    with governor.run("my-run"):
        result = governor.guard(my_tool_fn, name="search")(query="...")

Framework-agnostic at its core -- adapters in matimo_agdk.adapters wrap
this for LangChain/Google ADK/CrewAI/AutoGen, but none of that is required
to use Governor directly against a plain Python callable or any LLM SDK
that accepts a custom base_url and http client.
"""

from __future__ import annotations

import functools
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, TypeVar

import httpx

from ._compat import sdk_requires_httpx2
from ._retry_transport import AsyncSessionRetryTransport, SessionRetryTransport
from .config import GatewayConfig
from .exceptions import GatewayError, ToolDenied
from .identity import IdentityCredentials, JWSSigner, credentials_paths, save_credentials
from .session import SESSION_TOKEN_HEADER, AsyncSessionManager, SessionManager
from .telemetry import (
    AsyncTelemetryExporter,
    GovernanceState,
    TelemetryExporter,
    llm_span,
    run_span,
    tool_span,
)
from .tools import AsyncToolGovernor, ToolDecision, ToolGovernor
from .transport import AsyncGatewayHTTP, GatewayHTTP

R = TypeVar("R")

RUN_ID_HEADER = "X-Matimo-Run-Id"

# The active run id is context-local (asyncio task or thread), never an
# attribute on the governor: two concurrent `governor.run()` blocks must
# not see or reset each other's id (review finding, 2026-09-18).
_current_run: ContextVar[str | None] = ContextVar("matimo_agdk_current_run", default=None)


def current_run_id() -> str | None:
    """The run id of the innermost active `governor.run()` block in this
    task or thread, or None."""
    return _current_run.get()


REGISTER_PATH = "/identities"


_HTTPX2_HINT = (
    "This client needs the `httpx2` package (the HTTP library newer LLM SDKs such as "
    "anthropic >= 1.6 are built on). It is installed with those SDKs; otherwise "
    "`pip install httpx2`."
)


def _bind_hint(exc: ImportError) -> ImportError:
    return ImportError(f"{_HTTPX2_HINT} ({exc})")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _identity_from_response(
    data: dict[str, Any], *, base_url: str, fallback_created_at: str | None = None
) -> IdentityCredentials:
    return IdentityCredentials(
        identity_id=data["id"],
        identity_token=data["identityToken"],
        tenant_id=data["tenantId"],
        display_name=data["displayName"],
        external_framework=data.get("externalFramework"),
        private_key_pem=data["privateKeyPem"],
        base_url=base_url,
        public_key_fingerprint=data.get("publicKeyFingerprint"),
        created_at=data.get("createdAt") or fallback_created_at,
    )


def _positional_to_kwargs(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Best-effort reconstruction of "the tool's arguments" as a dict, for
    hashing and for the human-reviewer-facing `args` field on a PENDING
    tool check. Works cleanly for the common case (a tool called entirely
    with keyword arguments, which is how LangChain/ADK/CrewAI/AutoGen tool
    functions are normally invoked); positional arguments are given
    synthetic names. Documented limitation, not silently pretended away.
    """
    if not args:
        return dict(kwargs)
    merged = {f"arg{i}": v for i, v in enumerate(args)}
    merged.update(kwargs)
    return merged


class Governor:
    """Synchronous Governor -- one instance per agent process."""

    def __init__(self, config: GatewayConfig) -> None:
        self.config = config
        self._http = GatewayHTTP(config.base_url, config.api_key, timeout=config.http_timeout())
        self._identity: IdentityCredentials | None = None
        self._signer: JWSSigner | None = None
        self._session: SessionManager | None = None
        self._tools: ToolGovernor | None = None
        self._telemetry: TelemetryExporter | None = None
        self._started = False
        # True once this process wrote (or found) a credentials file for the
        # bound identity; rotate_key() only overwrites the file in that case,
        # so a persist=False identity never leaks into ~/.matimo/agents.
        self._persisted = False

        if config.has_identity():
            self._bind_identity(
                IdentityCredentials(
                    identity_id=config.identity_id or "",
                    identity_token=config.identity_token or "",
                    tenant_id=config.tenant_id or "",
                    display_name=config.agent_name,
                    external_framework=config.framework,
                    private_key_pem=config.private_key_pem or "",
                    base_url=config.base_url,
                )
            )

    # -- construction ----------------------------------------------------

    @classmethod
    def from_env(cls, **overrides: Any) -> Governor:
        return cls(GatewayConfig.load(**overrides))

    # -- identity / registration ------------------------------------------

    def _bind_identity(self, identity: IdentityCredentials) -> None:
        self._identity = identity
        self._signer = JWSSigner.from_credentials(identity)
        self._http.signer = self._signer
        self._session = SessionManager(self._http, self._signer, identity)
        self._tools = ToolGovernor(
            self._http,
            identity_token=identity.identity_token,
            identity_id=identity.identity_id,
            tenant_id=identity.tenant_id,
            external_framework=identity.external_framework,
        )

    @property
    def identity(self) -> IdentityCredentials | None:
        return self._identity

    def register(
        self,
        *,
        display_name: str | None = None,
        framework: str | None = None,
        allowed_tool_categories: list[str] | None = None,
        allowed_llm_models: list[str] | None = None,
        registration_metadata: dict[str, Any] | None = None,
        persist: bool = True,
    ) -> IdentityCredentials:
        """POST /v1/identities. Not idempotent server-side -- calling this
        twice creates two identities. Persists the one-time privateKeyPem
        locally unless persist=False."""
        body: dict[str, Any] = {
            "displayName": display_name or self.config.agent_name,
            "externalFramework": framework or self.config.framework,
        }
        if allowed_tool_categories is not None:
            body["allowedToolCategories"] = allowed_tool_categories
        if allowed_llm_models is not None:
            body["allowedLlmModels"] = allowed_llm_models
        if registration_metadata is not None:
            body["registrationMetadata"] = registration_metadata

        resp = self._http.request("POST", REGISTER_PATH, json_body=body, sign=False)
        identity = _identity_from_response(resp.data, base_url=self.config.base_url)
        if persist:
            save_credentials(identity, self.config.credentials_dir)
            self._persisted = True
        self._bind_identity(identity)
        return identity

    def rotate_key(self) -> IdentityCredentials:
        """POST /v1/identities/:agentId/rotate-key. The new private key is
        returned exactly once; persisted immediately. Every subsequent
        signed call (including the next session handshake) must use it --
        this method invalidates the cached session so the next call
        re-handshakes with the new key automatically."""
        if self._identity is None:
            raise GatewayError("cannot rotate a key before an identity is registered or loaded")
        resp = self._http.request(
            "POST", f"/identities/{self._identity.identity_id}/rotate-key", json_body={}, sign=False
        )
        identity = _identity_from_response(
            resp.data, base_url=self.config.base_url, fallback_created_at=self._identity.created_at
        )
        self._persist_rotated(identity)
        self._bind_identity(identity)
        return identity

    def _persist_rotated(self, identity: IdentityCredentials) -> None:
        """Writes the rotated key only where a credentials file already
        exists for this identity (registered with persist=True, or loaded
        from disk by from_env/the CLI). Otherwise the caller holds the only
        copy, exactly as with register(persist=False)."""
        meta_path, _ = credentials_paths(identity.display_name, self.config.credentials_dir)
        if self._persisted or meta_path.exists():
            save_credentials(identity, self.config.credentials_dir)
            self._persisted = True

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> Governor:
        if self._identity is None:
            raise GatewayError(
                "Governor has no identity: call governor.register(...) once, "
                "or run `matimo-agdk register` and then Governor.from_env()."
            )
        if self._started:
            return self
        heartbeat_interval = self.config.resolved_heartbeat_interval()
        self._telemetry = TelemetryExporter(
            self._http,
            self._session,
            flush_interval=self.config.telemetry_flush_interval,
            batch_size=self.config.telemetry_batch_size,
            queue_max=self.config.telemetry_queue_max,
            heartbeat_interval=heartbeat_interval,
            fail_open=self.config.fail_open_telemetry,
            heartbeat_resolver=self.config.resolved_heartbeat_interval,
        )
        self._telemetry.start()
        self._started = True
        return self

    def stop(self) -> None:
        """Flushes telemetry and stops the exporter. The HTTP client stays
        open so start() can be called again; call close() (or use the
        governor as a context manager) to release it."""
        if self._telemetry is not None:
            self._telemetry.stop()
        self._started = False

    def close(self) -> None:
        self.stop()
        self._http.close()

    def __enter__(self) -> Governor:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- governance state ----------------------------------------------------

    @property
    def state(self) -> GovernanceState:
        if self._telemetry is None:
            return GovernanceState()
        return self._telemetry.state

    def is_suspended(self) -> bool:
        return self.state.is_suspended

    def raise_if_suspended(self) -> None:
        if self._telemetry is not None:
            self._telemetry.raise_if_suspended()

    def on_suspend(self, callback: Callable[[GovernanceState], None]) -> None:
        if self._telemetry is None:
            raise GatewayError("call governor.start() before registering an on_suspend callback")
        self._telemetry._on_suspend = callback  # noqa: SLF001 -- single intended internal caller

    # -- runs and spans ----------------------------------------------------

    @contextmanager
    def run(self, name: str = "agent-run") -> Iterator[str]:
        run_id = uuid.uuid4().hex
        token = _current_run.set(run_id)
        started = time.monotonic()
        self._emit(
            run_span(run_id, name=name, status="running", session_id=run_id, started_at=_now_iso())
        )
        try:
            yield run_id
        except Exception:
            self._emit(
                run_span(
                    run_id,
                    name=name,
                    status="failed",
                    session_id=run_id,
                    duration_ms=int((time.monotonic() - started) * 1000),
                )
            )
            raise
        else:
            self._emit(
                run_span(
                    run_id,
                    name=name,
                    status="completed",
                    session_id=run_id,
                    duration_ms=int((time.monotonic() - started) * 1000),
                )
            )
        finally:
            _current_run.reset(token)

    def _emit(self, event: dict[str, Any]) -> None:
        if self._telemetry is not None:
            self._telemetry.submit(event)

    def run_span(
        self,
        run_id: str,
        *,
        status: str,
        name: str = "agent-run",
        started_at: str | None = None,
        duration_ms: int | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Emits a `kind:"run"` span for a run the framework owns (ADK's
        invocation id), for adapters that cannot wrap the run in
        `governor.run()`. `status="running"` opens it; a terminal status
        (`completed`/`failed`/`cancelled`) is the only thing that ends it
        server-side -- see docs/SERVER-CONTRACT.md section 7.3."""
        if started_at is None and status == "running":
            started_at = _now_iso()
        self._emit(
            run_span(
                run_id,
                name=name,
                status=status,
                session_id=run_id,
                started_at=started_at,
                duration_ms=duration_ms,
                attributes=attributes,
            )
        )

    def llm_span(self, **kwargs: Any) -> None:
        run_id = kwargs.pop("run_id", None) or _current_run.get()
        if run_id is None:
            raise GatewayError(
                "llm_span() needs an active governor.run() block or an explicit run_id"
            )
        kwargs.setdefault("session_id", run_id)
        self._emit(llm_span(run_id, **kwargs))

    def tool_span(self, tool_name: str, **kwargs: Any) -> None:
        run_id = kwargs.pop("run_id", None) or _current_run.get()
        if run_id is None:
            raise GatewayError(
                "tool_span() needs an active governor.run() block or an explicit run_id"
            )
        kwargs.setdefault("session_id", run_id)
        self._emit(tool_span(run_id, tool_name, **kwargs))

    # -- tool governance ----------------------------------------------------

    def check_tool(
        self, tool_name: str, args: dict[str, Any] | None = None, **kwargs: Any
    ) -> ToolDecision:
        if self._tools is None:
            raise GatewayError("Governor has no bound identity")
        return self._tools.check(tool_name, args, **kwargs)

    def await_decision(self, resume_token: str, **kwargs: Any) -> ToolDecision:
        if self._tools is None:
            raise GatewayError("Governor has no bound identity")
        return self._tools.await_decision(resume_token, **kwargs)

    def check_and_wait(
        self, tool_name: str, args: dict[str, Any] | None = None, **kwargs: Any
    ) -> ToolDecision:
        """check_tool(), then poll a PENDING to its final ALLOW/DENY. Never
        returns PENDING -- see ToolGovernor.check_and_wait()."""
        if self._tools is None:
            raise GatewayError("Governor has no bound identity")
        return self._tools.check_and_wait(tool_name, args, **kwargs)

    def set_tool_category(self, tool_name: str, category: str) -> None:
        if self._tools is None:
            raise GatewayError("Governor has no bound identity")
        self._tools.set_category(tool_name, category)

    def guard(
        self,
        fn: Callable[..., R] | None = None,
        *,
        name: str | None = None,
        category: str | None = None,
    ) -> Callable[..., R]:
        """Wraps a plain Python callable with: a policy check, a blocking
        wait on PENDING, a recorded tool span, and a best-effort result
        report. Usable directly or as a decorator::

            governor.guard(search, name="search")(query="...")

            @governor.guard(name="search", category="web")
            def search(query: str) -> str: ...

        Raises ToolDenied if the check (or the resolved PENDING decision)
        is DENY. The wrapped callable's own exceptions propagate unchanged
        after being recorded as a failed tool span.
        """

        def decorator(inner: Callable[..., R]) -> Callable[..., R]:
            tool_name: str = name if name is not None else str(getattr(inner, "__name__", "tool"))
            if self._tools is None:
                raise GatewayError("Governor has no bound identity")
            tools = self._tools

            @functools.wraps(inner)
            def wrapper(*args: Any, **kwargs: Any) -> R:
                if _current_run.get() is None:
                    # No governor.run() block: open an implicit one so the
                    # span has a run to land in, instead of running the
                    # tool and then failing in the finally.
                    with self.run(f"tool:{tool_name}"):
                        return wrapper(*args, **kwargs)
                call_args = _positional_to_kwargs(args, kwargs)
                decision = tools.check_and_wait(tool_name, call_args, category_hint=category)
                if decision.denied:
                    self.tool_span(tool_name, status="denied", duration_ms=0, arguments=call_args)
                    raise ToolDenied(decision.reason)

                started = time.monotonic()
                started_iso = _now_iso()
                status = "completed"
                error: str | None = None
                try:
                    result = inner(*args, **kwargs)
                except Exception as exc:
                    status = "error"
                    error = str(exc)
                    raise
                finally:
                    duration_ms = int((time.monotonic() - started) * 1000)
                    self.tool_span(
                        tool_name,
                        status=status,
                        started_at=started_iso,
                        duration_ms=duration_ms,
                        arguments=call_args,
                    )
                    if decision.resume_token:
                        tools.report_result(
                            decision.resume_token,
                            status=status,
                            duration_ms=duration_ms,
                            error=error,
                        )
                return result

            return wrapper

        if fn is not None:
            return decorator(fn)
        return decorator  # type: ignore[return-value]

    # -- LLM client helpers ----------------------------------------------

    def httpx_client(self) -> httpx.Client:
        """A pooled httpx.Client pre-wired to Gateway: base_url set, the
        org API key attached, and a request event hook that injects the
        current session token and a fresh Matimo-Agent-Signature on every
        outgoing request, computed over the exact bytes httpx is about to
        send.

        Point an OpenAI SDK's `http_client=` at this. A plain
        `default_headers=` cannot carry a per-request signature (the
        signature must cover each request's own body bytes) -- this
        client, via its request event hook, is the mechanism that makes
        that possible. For the Anthropic SDK use `anthropic_http_client()`:
        anthropic >= 1.6 is built on `httpx2` and rejects an `httpx.Client`.

        Also transparently re-handshakes exactly once on a live 401
        session_expired (e.g. another process called DELETE /v1/sessions,
        or the cached session was otherwise invalidated behind this
        process's back) -- found missing entirely during live verification
        against a real Gateway; see SessionRetryTransport's docstring.
        """
        session, signer = self._bound()
        transport = SessionRetryTransport(
            httpx.HTTPTransport(),
            session,
            signer,
            signing_enabled=self.config.signing_enabled,
        )
        return httpx.Client(
            base_url=self.config.base_url,
            headers={"Authorization": f"Bearer {self.config.api_key}"},
            event_hooks={"request": [self._request_hook()]},
            transport=transport,
            timeout=self.config.http_timeout(),
        )

    def httpx2_client(self) -> Any:
        """`httpx_client()` for SDKs built on `httpx2` (anthropic >= 1.6): the
        same live session token, per-request signature and transparent
        re-handshake, as an `httpx2.Client`. Needs the `httpx2` package."""
        try:
            from ._retry_transport_httpx2 import Httpx2SessionRetryTransport, httpx2
        except ImportError as exc:
            raise _bind_hint(exc) from exc
        session, signer = self._bound()
        transport = Httpx2SessionRetryTransport(
            httpx2.HTTPTransport(),
            session,
            signer,
            signing_enabled=self.config.signing_enabled,
        )
        return httpx2.Client(
            base_url=self.config.base_url,
            headers={"Authorization": f"Bearer {self.config.api_key}"},
            event_hooks={"request": [self._request_hook()]},
            transport=transport,
            timeout=self.config.http_timeout(httpx2),
        )

    def anthropic_http_client(self) -> Any:
        """The right `http_client=` for `anthropic.Anthropic(...)`, whichever
        HTTP library the installed anthropic release uses::

            client = anthropic.Anthropic(
                **governor.anthropic_client_kwargs(),
                http_client=governor.anthropic_http_client(),
            )
        """
        return self.httpx2_client() if sdk_requires_httpx2("anthropic") else self.httpx_client()

    def _bound(self) -> tuple[SessionManager, JWSSigner]:
        if self._session is None or self._identity is None or self._signer is None:
            raise GatewayError("Governor has no bound identity")
        return self._session, self._signer

    def _request_hook(self) -> Callable[[Any], None]:
        """The per-request hook shared by every client this governor builds
        (library-agnostic: it only touches `request.headers`/`.content`)."""
        session, signer = self._bound()
        config = self.config

        def _hook(request: Any) -> None:
            request.headers[SESSION_TOKEN_HEADER] = session.get_token()
            run_id = _current_run.get()
            if run_id:
                request.headers[RUN_ID_HEADER] = run_id
            if config.signing_enabled:
                jws = signer.sign_request(body_bytes=request.content or b"")
                request.headers["Matimo-Agent-Signature"] = jws

        return _hook

    def openai_client_kwargs(self) -> dict[str, Any]:
        """kwargs for `openai.OpenAI(**governor.openai_client_kwargs())`.

        `default_headers` alone cannot carry a per-request signature (it
        covers each request's own body bytes) -- also pass
        `http_client=governor.httpx_client()` if the tenant may ever
        enable requireSignedRequests, or simply always pass it: signing is
        free when unenforced (docs/SERVER-CONTRACT.md section 11 point 6).
        """
        return {
            "base_url": self.config.base_url,
            "api_key": self.config.api_key,
            "default_headers": self._default_llm_headers(),
        }

    def anthropic_client_kwargs(self) -> dict[str, Any]:
        """kwargs for `anthropic.Anthropic(**governor.anthropic_client_kwargs())`.
        Same signing caveat as openai_client_kwargs(): also pass
        `http_client=governor.anthropic_http_client()` for a live, signed client.

        Uses `auth_token`, not `api_key`: the Anthropic SDK sends `api_key`
        as `x-api-key`, which Gateway does not read; `auth_token` is sent
        as `Authorization: Bearer`, which is what Gateway authenticates."""
        return {
            "base_url": self.config.base_url,
            "auth_token": self.config.api_key,
            "default_headers": self._default_llm_headers(),
        }

    def bind_run_id(self, run_id: str | None) -> str | None:
        """Sets the run id attached to subsequent LLM-call headers and
        telemetry without opening a `governor.run()` span. Used by
        framework adapters whose runs are owned by the framework (ADK's
        invocation id). Returns the previous value so callers can restore
        it."""
        previous = _current_run.get()
        _current_run.set(run_id)
        return previous

    def request_headers(self, body: bytes | None = None) -> dict[str, str]:
        """Live per-request headers for an LLM call routed through Gateway:
        the current session token, the current run id (if inside
        `governor.run()`), and, when `body` is given and signing is
        enabled, a `Matimo-Agent-Signature` JWS over exactly those bytes.
        Framework adapters that can see each outbound request (CrewAI's
        interceptor, ADK's LiteLLMClient) call this per request instead of
        freezing headers at construction time."""
        headers = self._default_llm_headers()
        if body is not None and self.config.signing_enabled and self._signer is not None:
            headers["Matimo-Agent-Signature"] = self._signer.sign_request(body_bytes=body)
        return headers

    def _default_llm_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self._session is not None:
            headers[SESSION_TOKEN_HEADER] = self._session.get_token()
        run_id = _current_run.get()
        if run_id:
            headers[RUN_ID_HEADER] = run_id
        return headers


class AsyncGovernor:
    """Async twin of Governor -- one instance per agent process."""

    def __init__(self, config: GatewayConfig) -> None:
        self.config = config
        self._http = AsyncGatewayHTTP(
            config.base_url, config.api_key, timeout=config.http_timeout()
        )
        self._identity: IdentityCredentials | None = None
        self._signer: JWSSigner | None = None
        self._session: AsyncSessionManager | None = None
        self._tools: AsyncToolGovernor | None = None
        self._telemetry: AsyncTelemetryExporter | None = None
        self._started = False
        # True once this process wrote (or found) a credentials file for the
        # bound identity; rotate_key() only overwrites the file in that case,
        # so a persist=False identity never leaks into ~/.matimo/agents.
        self._persisted = False

        if config.has_identity():
            self._bind_identity(
                IdentityCredentials(
                    identity_id=config.identity_id or "",
                    identity_token=config.identity_token or "",
                    tenant_id=config.tenant_id or "",
                    display_name=config.agent_name,
                    external_framework=config.framework,
                    private_key_pem=config.private_key_pem or "",
                    base_url=config.base_url,
                )
            )

    @classmethod
    def from_env(cls, **overrides: Any) -> AsyncGovernor:
        return cls(GatewayConfig.load(**overrides))

    def _bind_identity(self, identity: IdentityCredentials) -> None:
        self._identity = identity
        self._signer = JWSSigner.from_credentials(identity)
        self._http.signer = self._signer
        self._session = AsyncSessionManager(self._http, self._signer, identity)
        self._tools = AsyncToolGovernor(
            self._http,
            identity_token=identity.identity_token,
            identity_id=identity.identity_id,
            tenant_id=identity.tenant_id,
            external_framework=identity.external_framework,
        )

    @property
    def identity(self) -> IdentityCredentials | None:
        return self._identity

    async def register(
        self,
        *,
        display_name: str | None = None,
        framework: str | None = None,
        allowed_tool_categories: list[str] | None = None,
        allowed_llm_models: list[str] | None = None,
        registration_metadata: dict[str, Any] | None = None,
        persist: bool = True,
    ) -> IdentityCredentials:
        body: dict[str, Any] = {
            "displayName": display_name or self.config.agent_name,
            "externalFramework": framework or self.config.framework,
        }
        if allowed_tool_categories is not None:
            body["allowedToolCategories"] = allowed_tool_categories
        if allowed_llm_models is not None:
            body["allowedLlmModels"] = allowed_llm_models
        if registration_metadata is not None:
            body["registrationMetadata"] = registration_metadata

        resp = await self._http.request("POST", REGISTER_PATH, json_body=body, sign=False)
        identity = _identity_from_response(resp.data, base_url=self.config.base_url)
        if persist:
            save_credentials(identity, self.config.credentials_dir)
            self._persisted = True
        self._bind_identity(identity)
        return identity

    async def rotate_key(self) -> IdentityCredentials:
        if self._identity is None:
            raise GatewayError("cannot rotate a key before an identity is registered or loaded")
        resp = await self._http.request(
            "POST", f"/identities/{self._identity.identity_id}/rotate-key", json_body={}, sign=False
        )
        identity = _identity_from_response(
            resp.data, base_url=self.config.base_url, fallback_created_at=self._identity.created_at
        )
        self._persist_rotated(identity)
        self._bind_identity(identity)
        return identity

    def _persist_rotated(self, identity: IdentityCredentials) -> None:
        """Writes the rotated key only where a credentials file already
        exists for this identity (registered with persist=True, or loaded
        from disk by from_env/the CLI). Otherwise the caller holds the only
        copy, exactly as with register(persist=False)."""
        meta_path, _ = credentials_paths(identity.display_name, self.config.credentials_dir)
        if self._persisted or meta_path.exists():
            save_credentials(identity, self.config.credentials_dir)
            self._persisted = True

    async def start(self) -> AsyncGovernor:
        if self._identity is None:
            raise GatewayError(
                "Governor has no identity: call governor.register(...) once, "
                "or run `matimo-agdk register` and then AsyncGovernor.from_env()."
            )
        if self._started:
            return self
        heartbeat_interval = self.config.resolved_heartbeat_interval()
        self._telemetry = AsyncTelemetryExporter(
            self._http,
            self._session,
            flush_interval=self.config.telemetry_flush_interval,
            batch_size=self.config.telemetry_batch_size,
            queue_max=self.config.telemetry_queue_max,
            heartbeat_interval=heartbeat_interval,
            fail_open=self.config.fail_open_telemetry,
            heartbeat_resolver=self.config.resolved_heartbeat_interval,
        )
        await self._telemetry.start()
        self._started = True
        return self

    async def stop(self) -> None:
        if self._telemetry is not None:
            await self._telemetry.stop()
        self._started = False

    async def aclose(self) -> None:
        await self.stop()
        await self._http.aclose()

    async def __aenter__(self) -> AsyncGovernor:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    @property
    def state(self) -> GovernanceState:
        if self._telemetry is None:
            return GovernanceState()
        return self._telemetry.state

    def is_suspended(self) -> bool:
        return self.state.is_suspended

    def raise_if_suspended(self) -> None:
        if self._telemetry is not None:
            self._telemetry.raise_if_suspended()

    def on_suspend(self, callback: Callable[[GovernanceState], None]) -> None:
        if self._telemetry is None:
            raise GatewayError(
                "call await governor.start() before registering an on_suspend callback"
            )
        self._telemetry._on_suspend = callback  # noqa: SLF001

    @asynccontextmanager
    async def run(self, name: str = "agent-run") -> AsyncIterator[str]:
        run_id = uuid.uuid4().hex
        token = _current_run.set(run_id)
        started = time.monotonic()
        self._emit(
            run_span(run_id, name=name, status="running", session_id=run_id, started_at=_now_iso())
        )
        try:
            yield run_id
        except Exception:
            self._emit(
                run_span(
                    run_id,
                    name=name,
                    status="failed",
                    session_id=run_id,
                    duration_ms=int((time.monotonic() - started) * 1000),
                )
            )
            raise
        else:
            self._emit(
                run_span(
                    run_id,
                    name=name,
                    status="completed",
                    session_id=run_id,
                    duration_ms=int((time.monotonic() - started) * 1000),
                )
            )
        finally:
            _current_run.reset(token)

    def _emit(self, event: dict[str, Any]) -> None:
        if self._telemetry is not None:
            self._telemetry.submit(event)

    def run_span(
        self,
        run_id: str,
        *,
        status: str,
        name: str = "agent-run",
        started_at: str | None = None,
        duration_ms: int | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Emits a `kind:"run"` span for a run the framework owns (ADK's
        invocation id), for adapters that cannot wrap the run in
        `governor.run()`. `status="running"` opens it; a terminal status
        (`completed`/`failed`/`cancelled`) is the only thing that ends it
        server-side -- see docs/SERVER-CONTRACT.md section 7.3."""
        if started_at is None and status == "running":
            started_at = _now_iso()
        self._emit(
            run_span(
                run_id,
                name=name,
                status=status,
                session_id=run_id,
                started_at=started_at,
                duration_ms=duration_ms,
                attributes=attributes,
            )
        )

    def llm_span(self, **kwargs: Any) -> None:
        run_id = kwargs.pop("run_id", None) or _current_run.get()
        if run_id is None:
            raise GatewayError(
                "llm_span() needs an active governor.run() block or an explicit run_id"
            )
        kwargs.setdefault("session_id", run_id)
        self._emit(llm_span(run_id, **kwargs))

    def tool_span(self, tool_name: str, **kwargs: Any) -> None:
        run_id = kwargs.pop("run_id", None) or _current_run.get()
        if run_id is None:
            raise GatewayError(
                "tool_span() needs an active governor.run() block or an explicit run_id"
            )
        kwargs.setdefault("session_id", run_id)
        self._emit(tool_span(run_id, tool_name, **kwargs))

    async def check_tool(
        self, tool_name: str, args: dict[str, Any] | None = None, **kwargs: Any
    ) -> ToolDecision:
        if self._tools is None:
            raise GatewayError("Governor has no bound identity")
        return await self._tools.check(tool_name, args, **kwargs)

    async def await_decision(self, resume_token: str, **kwargs: Any) -> ToolDecision:
        if self._tools is None:
            raise GatewayError("Governor has no bound identity")
        return await self._tools.await_decision(resume_token, **kwargs)

    async def check_and_wait(
        self, tool_name: str, args: dict[str, Any] | None = None, **kwargs: Any
    ) -> ToolDecision:
        """Async twin of Governor.check_and_wait()."""
        if self._tools is None:
            raise GatewayError("Governor has no bound identity")
        return await self._tools.check_and_wait(tool_name, args, **kwargs)

    async def set_tool_category(self, tool_name: str, category: str) -> None:
        if self._tools is None:
            raise GatewayError("Governor has no bound identity")
        await self._tools.set_category(tool_name, category)

    def guard(
        self,
        fn: Callable[..., Awaitable[R]] | None = None,
        *,
        name: str | None = None,
        category: str | None = None,
    ) -> Callable[..., Awaitable[R]]:
        """Async twin of Governor.guard() -- wraps an async callable."""

        def decorator(inner: Callable[..., Awaitable[R]]) -> Callable[..., Awaitable[R]]:
            tool_name: str = name if name is not None else str(getattr(inner, "__name__", "tool"))
            if self._tools is None:
                raise GatewayError("Governor has no bound identity")
            tools = self._tools

            @functools.wraps(inner)
            async def wrapper(*args: Any, **kwargs: Any) -> R:
                if _current_run.get() is None:
                    async with self.run(f"tool:{tool_name}"):
                        return await wrapper(*args, **kwargs)
                call_args = _positional_to_kwargs(args, kwargs)
                decision = await tools.check_and_wait(tool_name, call_args, category_hint=category)
                if decision.denied:
                    self.tool_span(tool_name, status="denied", duration_ms=0, arguments=call_args)
                    raise ToolDenied(decision.reason)

                started = time.monotonic()
                started_iso = _now_iso()
                status = "completed"
                error: str | None = None
                try:
                    result = await inner(*args, **kwargs)
                except Exception as exc:
                    status = "error"
                    error = str(exc)
                    raise
                finally:
                    duration_ms = int((time.monotonic() - started) * 1000)
                    self.tool_span(
                        tool_name,
                        status=status,
                        started_at=started_iso,
                        duration_ms=duration_ms,
                        arguments=call_args,
                    )
                    if decision.resume_token:
                        await tools.report_result(
                            decision.resume_token,
                            status=status,
                            duration_ms=duration_ms,
                            error=error,
                        )
                return result

            return wrapper

        if fn is not None:
            return decorator(fn)
        return decorator  # type: ignore[return-value]

    def httpx_async_client(self) -> httpx.AsyncClient:
        """Async twin of Governor.httpx_client(), including the same
        transparent re-handshake-on-401-session_expired behavior."""
        session, signer = self._bound()
        transport = AsyncSessionRetryTransport(
            httpx.AsyncHTTPTransport(),
            session,
            signer,
            signing_enabled=self.config.signing_enabled,
        )
        return httpx.AsyncClient(
            base_url=self.config.base_url,
            headers={"Authorization": f"Bearer {self.config.api_key}"},
            event_hooks={"request": [self._request_hook()]},
            transport=transport,
            timeout=self.config.http_timeout(),
        )

    def httpx2_async_client(self) -> Any:
        """Async twin of Governor.httpx2_client(): an `httpx2.AsyncClient` for
        SDKs built on `httpx2` (anthropic >= 1.6). Needs the `httpx2` package."""
        try:
            from ._retry_transport_httpx2 import Httpx2AsyncSessionRetryTransport, httpx2
        except ImportError as exc:
            raise _bind_hint(exc) from exc
        session, signer = self._bound()
        transport = Httpx2AsyncSessionRetryTransport(
            httpx2.AsyncHTTPTransport(),
            session,
            signer,
            signing_enabled=self.config.signing_enabled,
        )
        return httpx2.AsyncClient(
            base_url=self.config.base_url,
            headers={"Authorization": f"Bearer {self.config.api_key}"},
            event_hooks={"request": [self._request_hook()]},
            transport=transport,
            timeout=self.config.http_timeout(httpx2),
        )

    def anthropic_http_client(self) -> Any:
        """The right `http_client=` for `anthropic.AsyncAnthropic(...)`,
        whichever HTTP library the installed anthropic release uses."""
        return (
            self.httpx2_async_client()
            if sdk_requires_httpx2("anthropic")
            else self.httpx_async_client()
        )

    def _bound(self) -> tuple[AsyncSessionManager, JWSSigner]:
        if self._session is None or self._identity is None or self._signer is None:
            raise GatewayError("Governor has no bound identity")
        return self._session, self._signer

    def _request_hook(self) -> Callable[[Any], Awaitable[None]]:
        session, signer = self._bound()
        config = self.config

        async def _hook(request: Any) -> None:
            request.headers[SESSION_TOKEN_HEADER] = await session.get_token()
            run_id = _current_run.get()
            if run_id:
                request.headers[RUN_ID_HEADER] = run_id
            if config.signing_enabled:
                jws = signer.sign_request(body_bytes=request.content or b"")
                request.headers["Matimo-Agent-Signature"] = jws

        return _hook

    async def openai_client_kwargs(self) -> dict[str, Any]:
        return {
            "base_url": self.config.base_url,
            "api_key": self.config.api_key,
            "default_headers": await self._default_llm_headers(),
        }

    async def anthropic_client_kwargs(self) -> dict[str, Any]:
        return {
            "base_url": self.config.base_url,
            "auth_token": self.config.api_key,
            "default_headers": await self._default_llm_headers(),
        }

    def bind_run_id(self, run_id: str | None) -> str | None:
        """Sets the run id attached to subsequent LLM-call headers and
        telemetry without opening a `governor.run()` span. Used by
        framework adapters whose runs are owned by the framework (ADK's
        invocation id). Returns the previous value so callers can restore
        it."""
        previous = _current_run.get()
        _current_run.set(run_id)
        return previous

    async def request_headers(self, body: bytes | None = None) -> dict[str, str]:
        """Async twin of Governor.request_headers()."""
        headers = await self._default_llm_headers()
        if body is not None and self.config.signing_enabled and self._signer is not None:
            headers["Matimo-Agent-Signature"] = self._signer.sign_request(body_bytes=body)
        return headers

    async def _default_llm_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self._session is not None:
            headers[SESSION_TOKEN_HEADER] = await self._session.get_token()
        run_id = _current_run.get()
        if run_id:
            headers[RUN_ID_HEADER] = run_id
        return headers
