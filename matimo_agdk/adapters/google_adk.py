"""Google ADK adapter. Verified against google-adk==2.9.1 and openai==2.54.0.

Install with `pip install matimo-agdk[google-adk]`.

Framework string: registrations from this adapter use `"google-adk"`
(hyphen) -- the live Gateway `externalFramework` enum accepts both
`"google-adk"` and `"google_adk"` (a widening kept for compatibility, see
AGDK-SERVER-HEARTBEAT-REPORT.md section 3), but this adapter picks one
spelling and uses it consistently, per that report's own instruction to
whoever built this next. `GatewayConfig.framework`'s `Framework` literal
already uses the hyphenated form -- register with
`governor.register(framework="google-adk")` or
`matimo-agdk register --framework google-adk`.

Two pieces:

- `MatimoPlugin(governor, mode=...)` -- a `BasePlugin` subclass, registered
  once on the customer's `Runner(plugins=[MatimoPlugin(governor)])`.
  Method names/signatures verified directly against the installed
  `google.adk.plugins.base_plugin.BasePlugin`:
  `before_tool_callback(*, tool, tool_args, tool_context)`,
  `after_tool_callback(*, tool, tool_args, tool_context, result)`,
  `before_model_callback(*, callback_context, llm_request)`,
  `after_model_callback(*, callback_context, llm_response)`,
  `on_tool_error_callback(*, tool, tool_args, tool_context, error)`.
- `gateway_model(governor, model=...)` -- a `MatimoLlm` (`BaseLlm`) instance
  pointed at Gateway's OpenAI-compatible endpoint, talking to the real
  `openai` SDK directly -- no `litellm` anywhere in this path (see
  `_adk_openai.py`'s own module docstring for exactly why, and its "Scope
  relative to LiteLlm" section for what that trades away). Needs an
  `AsyncGovernor`, not a sync `Governor` -- see `gateway_model()`'s own
  docstring.

## What enforces what

`before_tool_callback` returning a non-`None` dict short-circuits ADK's
tool dispatch entirely (verified: this is ADK's own documented contract,
not an assumption) -- so, unlike LangChain, a DENY genuinely *can* be
returned gracefully here: `{"error": <reason>}` becomes the tool's result
as far as the agent's own reasoning loop is concerned, no exception, no
crashed run. `mode="govern"` (default) does exactly this: calls
`governor.check_tool()`/`await_decision()` before the tool dispatches, and
denies via the short-circuit dict rather than raising. `mode="observe"`
never calls `check_tool()` at all, only records spans.

A Gateway outage that leaves a tool check unanswered (fail-closed,
`ToolCheckUnavailable`) is returned the same way, as `{"error": <message>}`, so the
tool does not run and the run does not crash. With
`tool_check_failure_mode="fail_open_bounded"` the tool may run instead, and its span
is marked `matimo.degraded_mode` (see the core README).

Rapid suspend (`mode="govern"` only): `governor.raise_if_suspended()` is
called at the top of both `before_model_callback` and
`before_tool_callback` -- since neither ADK contract catches an arbitrary
exception raised from a plugin callback the way `before_tool_callback`'s
own dict-return contract does, this genuinely halts the run (matches the
"rapid suspend is a hard stop, not a graceful denial" positioning).

## LLM spans and gen_ai.* attributes

`before_model_callback`/`after_model_callback` bracket every model call
ADK makes, regardless of which underlying model backend is configured
(Gemini, `MatimoLlm`, or anything else) -- so telemetry works even for a
Gemini-native agent that never goes through `gateway_model()`/Gateway at
all (matching this SDK's honest "AGDK never sees your real LLM traffic
unless it goes through Gateway" framing: spans still get recorded, they
just won't correlate with a `matimo_gateway_call_log` row for cost/token
cross-checking the way a Gateway-routed call's telemetry does). Token
usage is read from `llm_response.usage_metadata`
(`prompt_token_count`/`candidates_token_count`), the real field names on
the installed `google.genai.types.GenerateContentResponseUsageMetadata`.

Every span's `run_id` is the ADK invocation's own `invocation_id`
(`callback_context.invocation_id`/`tool_context.invocation_id`), so LLM
and tool spans for one agent turn correlate automatically without the
caller wrapping anything in `governor.run()`.

## Run lifecycle

`before_run_callback` opens the run (`kind:"run"`, `status="running"`);
`after_run_callback` closes it `completed` and `on_run_error_callback`
closes it `failed`. Gateway only ends a run on an explicit terminal run
span (docs/SERVER-CONTRACT.md section 7.3), so without these the run stays
`running` until the staleness sweep. ADK skips `after_run_callback` when the
caller abandons the event stream early (e.g. `break` out of
`runner.run_async`); such a run also falls back to the staleness sweep.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
import uuid
from typing import Any

from pydantic import ConfigDict, PrivateAttr

from .._outage import degraded_attributes
from ..exceptions import GatewayError, ToolCheckUnavailable
from . import _adk_openai
from ._shared import (
    Mode,
    async_check_and_wait,
    async_raise_if_suspended,
    check_mode,
    emit_llm_span,
    emit_tool_span,
    is_async_governor,
)

try:
    from google.adk.models import LlmCapabilities
    from google.adk.models.base_llm import BaseLlm
    from google.adk.plugins.base_plugin import BasePlugin
except ImportError as exc:  # pragma: no cover - exercised only when the extra is missing
    raise ImportError(
        "matimo_agdk.adapters.google_adk requires the 'google-adk' extra: "
        "pip install matimo-agdk[google-adk]"
    ) from exc


class _QuietExpectedGatewayErrors(logging.Filter):
    """A `gateway_model()` DENY has no graceful-return contract the way a
    tool DENY does (`before_tool_callback`'s short-circuit dict -- see this
    module's own docstring) -- `MatimoLlm.generate_content_async()` must
    raise, and ADK's own node runner logs *every* exception that propagates
    out of a node/root run at ERROR level with a full traceback before
    re-raising it: `google.adk.workflow._node_runner.NodeRunner.run`'s
    `logger.exception("Node execution failed with exception")` and
    `google.adk.runners.Runner._cleanup_root_task`'s
    `logger.error('Root node %s failed.', ..., exc_info=True)` (verified
    against google-adk==2.9.1, the same version pin as the rest of this
    file). That is correct behavior for a genuine ADK-internal bug and pure
    noise for an expected `GatewayError` -- which every example's own
    `except GatewayError` already prints as one clean line. Drop only
    records whose exception is ours; anything else ADK logs at ERROR still
    goes through untouched.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        exc = record.exc_info[1] if record.exc_info else None
        return not isinstance(exc, GatewayError)


def _quiet_expected_gateway_errors() -> None:
    quiet_filter = _QuietExpectedGatewayErrors()
    for name in ("google.adk.runners", "google.adk.workflow._node_runner"):
        adk_logger = logging.getLogger(name)
        if not any(isinstance(f, _QuietExpectedGatewayErrors) for f in adk_logger.filters):
            adk_logger.addFilter(quiet_filter)


_quiet_expected_gateway_errors()


def _now() -> float:
    return time.monotonic()


def _usage_attributes(usage_metadata: Any) -> dict[str, Any]:
    if usage_metadata is None:
        return {}
    attrs: dict[str, Any] = {}
    prompt = getattr(usage_metadata, "prompt_token_count", None)
    completion = getattr(usage_metadata, "candidates_token_count", None)
    if prompt is not None:
        attrs["gen_ai.usage.input_tokens"] = prompt
    if completion is not None:
        attrs["gen_ai.usage.output_tokens"] = completion
    return attrs


class MatimoPlugin(BasePlugin):  # type: ignore[misc]
    """The one entry point: `Runner(plugins=[MatimoPlugin(governor)])`."""

    def __init__(
        self,
        governor: Any,
        mode: Mode = "govern",
        *,
        category: str | None = None,
        name: str = "matimo",
    ) -> None:
        super().__init__(name)
        self.governor = governor
        self.mode: Mode = check_mode(mode)
        self.category = category
        self._pending_llm: dict[str, tuple[str, float, str | None]] = {}
        self._previous_run: dict[str, str | None] = {}
        self._pending_tool: dict[str, tuple[float, str | None, dict[str, Any] | None]] = {}
        self._open_runs: dict[str, tuple[str, float]] = {}

    def _restore_run(self, invocation_id: str) -> None:
        if invocation_id in self._previous_run:
            self.governor.bind_run_id(self._previous_run.pop(invocation_id))

    # -- run lifecycle ------------------------------------------------------

    def _close_run(self, invocation_id: str, status: str) -> None:
        opened = self._open_runs.pop(invocation_id, None)
        if opened is None:
            return
        name, started = opened
        try:
            self.governor.run_span(
                invocation_id,
                status=status,
                name=name,
                duration_ms=int((_now() - started) * 1000),
            )
        except Exception:  # noqa: BLE001 -- telemetry must never break the caller's agent
            pass

    async def before_run_callback(self, *, invocation_context: Any) -> Any:
        # ADK owns the run, so there is no `governor.run()` block to open and
        # close it. Without an explicit terminal `kind:"run"` span Gateway
        # leaves the run `running` until the staleness sweep (contract 7.3).
        invocation_id = getattr(invocation_context, "invocation_id", None)
        if not invocation_id:
            return None
        agent_name = getattr(getattr(invocation_context, "agent", None), "name", None)
        name = str(agent_name) if agent_name else "adk-run"
        self._open_runs[invocation_id] = (name, _now())
        try:
            self.governor.run_span(invocation_id, status="running", name=name)
        except Exception:  # noqa: BLE001
            pass
        return None

    async def after_run_callback(self, *, invocation_context: Any) -> None:
        invocation_id = getattr(invocation_context, "invocation_id", None)
        if invocation_id:
            self._close_run(invocation_id, "completed")

    async def on_run_error_callback(self, *, invocation_context: Any, error: Exception) -> None:
        invocation_id = getattr(invocation_context, "invocation_id", None)
        if invocation_id:
            self._close_run(invocation_id, "failed")

    # -- model callbacks --------------------------------------------------

    async def before_model_callback(self, *, callback_context: Any, llm_request: Any) -> Any:
        if self.mode == "govern":
            await async_raise_if_suspended(self.governor)
        invocation_id = getattr(callback_context, "invocation_id", None) or "unknown-invocation"
        # ADK owns the run (its invocation id); bind it so the LLM call's
        # X-Matimo-Run-Id header (via gateway_model()'s live client) and the
        # call-log row correlate with this plugin's spans (2026-09-18).
        self._previous_run[invocation_id] = self.governor.bind_run_id(invocation_id)
        span_id = uuid.uuid4().hex
        model = getattr(llm_request, "model", None)
        self._pending_llm[invocation_id] = (span_id, _now(), model)
        return None

    async def after_model_callback(self, *, callback_context: Any, llm_response: Any) -> Any:
        invocation_id = getattr(callback_context, "invocation_id", None) or "unknown-invocation"
        self._restore_run(invocation_id)
        span_id, started, model = self._pending_llm.pop(
            invocation_id, (uuid.uuid4().hex, _now(), None)
        )
        finish_reason = getattr(llm_response, "finish_reason", None)
        emit_llm_span(
            self.governor,
            run_id=invocation_id,
            span_id=span_id,
            model=model,
            provider="google",
            status="completed",
            duration_ms=int((_now() - started) * 1000),
            finish_reasons=[str(finish_reason)] if finish_reason is not None else None,
            attributes=_usage_attributes(getattr(llm_response, "usage_metadata", None)) or None,
        )
        return None

    async def on_model_error_callback(
        self, *, callback_context: Any, llm_request: Any, error: Exception
    ) -> Any:
        invocation_id = getattr(callback_context, "invocation_id", None) or "unknown-invocation"
        self._restore_run(invocation_id)
        span_id, started, model = self._pending_llm.pop(
            invocation_id, (uuid.uuid4().hex, _now(), None)
        )
        emit_llm_span(
            self.governor,
            run_id=invocation_id,
            span_id=span_id,
            model=model,
            provider="google",
            status="error",
            duration_ms=int((_now() - started) * 1000),
        )
        return None

    # -- tool callbacks -----------------------------------------------------

    async def before_tool_callback(
        self, *, tool: Any, tool_args: dict[str, Any], tool_context: Any
    ) -> dict[str, Any] | None:
        call_id = getattr(tool_context, "function_call_id", None) or uuid.uuid4().hex
        if self.mode == "observe":
            self._pending_tool[call_id] = (_now(), None, None)
            return None

        await async_raise_if_suspended(self.governor)
        # The denied span must land in ADK's own run (the invocation), not a
        # run of its own: bind_run_id() only covers model calls.
        invocation_id = getattr(tool_context, "invocation_id", None)
        try:
            decision = await async_check_and_wait(
                self.governor,
                tool.name,
                dict(tool_args),
                category=self.category,
                run_id=invocation_id or None,
            )
        except ToolCheckUnavailable as exc:
            # Gateway could not answer and the failure mode is fail-closed: the tool
            # does not run. Same graceful short-circuit as a DENY, not a crashed run.
            return {"error": str(exc)}
        if decision.denied:
            # ADK's documented contract: a non-None dict from
            # before_tool_callback short-circuits dispatch and becomes the
            # tool's own result -- a graceful, recoverable DENY, not a
            # crashed run.
            return {"error": decision.reason or "tool call denied"}
        self._pending_tool[call_id] = (_now(), decision.resume_token, degraded_attributes(decision))
        return None

    async def after_tool_callback(
        self, *, tool: Any, tool_args: dict[str, Any], tool_context: Any, result: dict[str, Any]
    ) -> dict[str, Any] | None:
        call_id = getattr(tool_context, "function_call_id", None) or uuid.uuid4().hex
        started, resume_token, degraded = self._pending_tool.pop(call_id, (_now(), None, None))
        invocation_id = getattr(tool_context, "invocation_id", None) or "unknown-invocation"
        emit_tool_span(
            self.governor,
            tool.name,
            run_id=invocation_id,
            span_id=call_id,
            call_id=call_id,
            status="completed",
            duration_ms=int((_now() - started) * 1000),
            arguments=dict(tool_args),
            result=result,
            attributes=degraded,
        )
        if resume_token:
            await self._report_result_best_effort(resume_token, status="completed")
        return None

    async def on_tool_error_callback(
        self, *, tool: Any, tool_args: dict[str, Any], tool_context: Any, error: Exception
    ) -> dict[str, Any] | None:
        call_id = getattr(tool_context, "function_call_id", None) or uuid.uuid4().hex
        started, resume_token, degraded = self._pending_tool.pop(call_id, (_now(), None, None))
        invocation_id = getattr(tool_context, "invocation_id", None) or "unknown-invocation"
        emit_tool_span(
            self.governor,
            tool.name,
            run_id=invocation_id,
            span_id=call_id,
            call_id=call_id,
            status="error",
            duration_ms=int((_now() - started) * 1000),
            arguments=dict(tool_args),
            result=str(error),
            attributes=degraded,
        )
        if resume_token:
            await self._report_result_best_effort(resume_token, status="error", error=str(error))
        return None

    async def _report_result_best_effort(self, resume_token: str, **kwargs: Any) -> None:
        # Governor has no public report_result() -- it lives on the
        # private ToolGovernor/AsyncToolGovernor the Governor already
        # constructs (see tests/test_governor.py's own `governor._tools`
        # access for the same convention within this codebase). Best
        # effort, per docs/SERVER-CONTRACT.md section 8.3 -- swallow any
        # failure, including this attribute not existing on some future
        # Governor shape. Works for either a sync Governor (bridged via a
        # thread, since ToolGovernor.report_result is a blocking HTTP call)
        # or an AsyncGovernor (awaited directly).
        try:
            tools = self.governor._tools  # noqa: SLF001
            if tools is None:
                return
            report = tools.report_result
            if inspect.iscoroutinefunction(report):
                await report(resume_token, **kwargs)
            else:
                await asyncio.to_thread(report, resume_token, **kwargs)
        except Exception:  # noqa: BLE001
            pass


class MatimoLlm(BaseLlm):  # type: ignore[misc]
    """A `BaseLlm` that talks to Matimo Gateway's OpenAI-compatible endpoint
    directly via the real `openai` SDK -- see `gateway_model()`'s own
    docstring for why this replaced a `LiteLlm`-based implementation, and
    `_adk_openai.py`'s module docstring for the full request/response
    conversion and its disclosed scope limits. Built by `gateway_model()`;
    construct it directly only if you need to override `capabilities` or
    `supported_models` by subclassing it.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _client: Any = PrivateAttr()
    _extra_kwargs: dict[str, Any] = PrivateAttr(default_factory=dict)

    def __init__(self, model: str, *, governor: Any, **kwargs: Any) -> None:
        if not is_async_governor(governor):
            raise TypeError(
                "gateway_model() needs an AsyncGovernor -- generate_content_async() "
                "is fully async and needs an httpx.AsyncClient with live per-request "
                "signing, which only AsyncGovernor.httpx_async_client() provides. "
                "Use AsyncGovernor.from_env() instead of Governor.from_env()."
            )
        super().__init__(model=model)
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(
            base_url=governor.config.base_url,
            api_key=governor.config.api_key or "matimo-gateway",
            http_client=governor.httpx_async_client(),
        )
        self._extra_kwargs = kwargs

    @property
    def capabilities(self) -> LlmCapabilities:
        return LlmCapabilities(output_schema_and_tools=True)

    @classmethod
    def supported_models(cls) -> list[str]:
        # Constructed directly by gateway_model(), never resolved by name
        # through ADK's LlmRegistry -- no patterns to register.
        return []

    async def generate_content_async(self, llm_request: Any, stream: bool = False) -> Any:
        self._maybe_append_user_content(llm_request)
        _adk_openai._append_fallback_user_content_if_missing(llm_request)
        effective_model = llm_request.model or self.model
        kwargs = _adk_openai.build_completion_kwargs(llm_request, effective_model)
        kwargs.update(self._extra_kwargs)

        if stream:
            kwargs["stream"] = True
            kwargs["stream_options"] = {"include_usage": True}
            response = await self._client.chat.completions.create(**kwargs)
            async for llm_response in _adk_openai.stream_llm_responses(response):
                yield llm_response
        else:
            response = await self._client.chat.completions.create(**kwargs)
            yield _adk_openai.completion_to_llm_response(response)


def gateway_model(governor: Any, *, model: str | None = None, **kwargs: Any) -> Any:
    """Returns a `BaseLlm` pointed at Gateway's OpenAI-compatible endpoint,
    talking to the real `openai` SDK directly -- no `litellm` in this path
    (see `_adk_openai.py`'s own module docstring for why, and what that
    trades away relative to `google.adk.models.lite_llm.LiteLlm`, which this
    replaced).

    **Needs an `AsyncGovernor`, not a sync `Governor`.**
    `generate_content_async()` is fully async and needs an
    `httpx.AsyncClient` with a live per-request signing hook, which only
    `AsyncGovernor.httpx_async_client()` provides -- the same constraint
    AutoGen's `gateway_model_client()` already documents. Use
    `AsyncGovernor.from_env()` instead of `Governor.from_env()`.

    **Full per-request `Matimo-Agent-Signature` signing, unlike the old
    `LiteLlm`-based implementation.** `governor.httpx_async_client()`
    attaches the live session token, current `governor.run()` id, and (when
    signing is enabled) a signature over each request's exact body bytes,
    via the same transport-level hook every other Gateway call uses
    (`_retry_transport.py`) -- there is no serialization layer in between to
    lose it, unlike routing through `litellm`.

    **A denied call raises `PolicyDenied`, like every other adapter.** The
    real `openai` SDK's own `_base_client.request()` re-raises an
    already-`OpenAIError` exception (`GatewayError`'s base) untouched, so
    Governor's transport-raised typed exception reaches the caller exactly
    as raised -- no recovery/remapping step needed, unlike the `litellm`
    path this replaced.

    `model` defaults to `"matimo/auto"` (Gateway's own routing sentinel).
    Any other keyword arguments are merged into every
    `chat.completions.create()` call, letting an unrecognized/passthrough
    parameter go through `extra_body` explicitly if the real SDK's typed
    signature would otherwise reject it.
    """
    model_name = model or "matimo/auto"
    return MatimoLlm(model=model_name, governor=governor, **kwargs)
