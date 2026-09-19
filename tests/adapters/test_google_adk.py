from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import respx

from matimo_agdk.exceptions import AgentSuspendedLocally
from matimo_agdk.identity import IdentityCredentials
from matimo_agdk.tools import ToolDecision

from ..conftest import BASE_URL, future_iso
from .conftest import bound_async_governor


class _Tool:
    def __init__(self, name: str = "search") -> None:
        self.name = name


class _Ctx:
    def __init__(self, function_call_id: str = "call-1", invocation_id: str = "inv-1") -> None:
        self.function_call_id = function_call_id
        self.invocation_id = invocation_id


class _CallbackCtx:
    def __init__(self, invocation_id: str = "inv-1") -> None:
        self.invocation_id = invocation_id


class _LlmRequest:
    def __init__(self, model: str = "gemini-2.0-flash") -> None:
        self.model = model


class _UsageMetadata:
    def __init__(self, prompt: int = 12, candidates: int = 34) -> None:
        self.prompt_token_count = prompt
        self.candidates_token_count = candidates


class _LlmResponse:
    def __init__(self, finish_reason: str = "STOP", usage: _UsageMetadata | None = None) -> None:
        self.finish_reason = finish_reason
        self.usage_metadata = usage or _UsageMetadata()


def _mock_sessions_and_telemetry() -> None:
    respx.post(f"{BASE_URL}/sessions").mock(
        return_value=httpx.Response(
            201,
            json={
                "data": {"sessionToken": "tok", "expiresAt": future_iso(3600), "identityId": "x"}
            },
        )
    )
    respx.post(f"{BASE_URL}/telemetry/batch").mock(
        return_value=httpx.Response(200, json={"data": {"accepted": 1, "failed": []}})
    )


@pytest.mark.asyncio
async def test_observe_mode_never_calls_check_tool() -> None:
    from matimo_agdk.adapters.google_adk import MatimoPlugin

    gov = MagicMock()
    gov.check_tool = AsyncMock()
    gov.check_and_wait = AsyncMock()
    gov.tool_span = MagicMock()

    plugin = MatimoPlugin(gov, mode="observe")
    result = await plugin.before_tool_callback(
        tool=_Tool(), tool_args={"q": "hi"}, tool_context=_Ctx()
    )
    assert result is None
    gov.check_tool.assert_not_called()
    gov.check_and_wait.assert_not_called()

    await plugin.after_tool_callback(
        tool=_Tool(), tool_args={"q": "hi"}, tool_context=_Ctx(), result={"ok": True}
    )
    gov.tool_span.assert_called_once()
    assert gov.tool_span.call_args.kwargs["status"] == "completed"


@pytest.mark.asyncio
@respx.mock
async def test_govern_mode_allow_passes_through(
    identity: IdentityCredentials, credentials_dir
) -> None:
    from matimo_agdk.adapters.google_adk import MatimoPlugin

    _mock_sessions_and_telemetry()
    respx.post(f"{BASE_URL}/tools/check").mock(
        return_value=httpx.Response(200, json={"data": {"decision": "ALLOW"}})
    )
    governor = bound_async_governor(identity, credentials_dir)
    plugin = MatimoPlugin(governor, mode="govern")

    result = await plugin.before_tool_callback(
        tool=_Tool(), tool_args={"q": "hi"}, tool_context=_Ctx()
    )
    assert result is None


@pytest.mark.asyncio
@respx.mock
async def test_govern_mode_deny_returns_error_dict_not_a_crash(
    identity: IdentityCredentials, credentials_dir
) -> None:
    from matimo_agdk.adapters.google_adk import MatimoPlugin

    _mock_sessions_and_telemetry()
    respx.post(f"{BASE_URL}/tools/check").mock(
        return_value=httpx.Response(
            200, json={"data": {"decision": "DENY", "reason": "tool_category_not_allowed"}}
        )
    )
    governor = bound_async_governor(identity, credentials_dir)
    plugin = MatimoPlugin(governor, mode="govern")

    result = await plugin.before_tool_callback(
        tool=_Tool(), tool_args={"q": "hi"}, tool_context=_Ctx()
    )
    # ADK's own documented contract: a non-None dict short-circuits
    # dispatch and becomes the tool's result -- no exception, no crash.
    assert result == {"error": "tool_category_not_allowed"}


@pytest.mark.asyncio
@respx.mock
async def test_govern_mode_pending_then_approved_proceeds(
    identity: IdentityCredentials, credentials_dir
) -> None:
    from matimo_agdk.adapters.google_adk import MatimoPlugin

    _mock_sessions_and_telemetry()
    respx.post(f"{BASE_URL}/tools/check").mock(
        return_value=httpx.Response(
            200, json={"data": {"decision": "PENDING", "resumeToken": "rt-1"}}
        )
    )
    calls = {"n": 0}

    def status_responder(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, json={"data": {"decision": "PENDING"}})
        return httpx.Response(200, json={"data": {"decision": "ALLOW"}})

    respx.post(f"{BASE_URL}/tools/check/status").mock(side_effect=status_responder)
    respx.post(f"{BASE_URL}/tools/result").mock(
        return_value=httpx.Response(202, json={"data": {"accepted": True}})
    )
    governor = bound_async_governor(identity, credentials_dir)
    governor._tools.poll_interval = 0.01  # noqa: SLF001
    governor._tools.poll_max_interval = 0.02  # noqa: SLF001
    plugin = MatimoPlugin(governor, mode="govern")

    result = await plugin.before_tool_callback(
        tool=_Tool(), tool_args={"q": "hi"}, tool_context=_Ctx()
    )
    assert result is None
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_suspended_state_stops_before_tool_call() -> None:
    from matimo_agdk.adapters.google_adk import MatimoPlugin

    gov = MagicMock()
    gov.check_tool = AsyncMock(return_value=ToolDecision(decision="ALLOW"))
    gov.check_and_wait = AsyncMock(return_value=ToolDecision(decision="ALLOW"))
    gov.raise_if_suspended = MagicMock(side_effect=AgentSuspendedLocally("suspended", False))

    plugin = MatimoPlugin(gov, mode="govern")
    with pytest.raises(AgentSuspendedLocally):
        await plugin.before_tool_callback(tool=_Tool(), tool_args={}, tool_context=_Ctx())
    gov.check_tool.assert_not_called()
    gov.check_and_wait.assert_not_called()


@pytest.mark.asyncio
async def test_suspended_state_stops_before_llm_call() -> None:
    from matimo_agdk.adapters.google_adk import MatimoPlugin

    gov = MagicMock()
    gov.raise_if_suspended = MagicMock(side_effect=AgentSuspendedLocally("suspended", False))

    plugin = MatimoPlugin(gov, mode="govern")
    with pytest.raises(AgentSuspendedLocally):
        await plugin.before_model_callback(
            callback_context=_CallbackCtx(), llm_request=_LlmRequest()
        )


@pytest.mark.asyncio
async def test_llm_span_carries_model_and_usage() -> None:
    from matimo_agdk.adapters.google_adk import MatimoPlugin

    gov = MagicMock()
    gov.raise_if_suspended = MagicMock()
    gov.llm_span = MagicMock()

    plugin = MatimoPlugin(gov, mode="govern")
    ctx = _CallbackCtx(invocation_id="inv-42")
    await plugin.before_model_callback(
        callback_context=ctx, llm_request=_LlmRequest(model="gemini-2.0-flash")
    )
    await plugin.after_model_callback(
        callback_context=ctx,
        llm_response=_LlmResponse(finish_reason="STOP", usage=_UsageMetadata(7, 9)),
    )

    gov.llm_span.assert_called_once()
    kwargs = gov.llm_span.call_args.kwargs
    assert kwargs["model"] == "gemini-2.0-flash"
    assert kwargs["run_id"] == "inv-42"
    assert kwargs["attributes"]["gen_ai.usage.input_tokens"] == 7
    assert kwargs["attributes"]["gen_ai.usage.output_tokens"] == 9
    assert kwargs["finish_reasons"] == ["STOP"]


def test_gateway_model_returns_lite_llm_pointed_at_gateway() -> None:
    from matimo_agdk.adapters.google_adk import gateway_model

    gov = MagicMock()
    gov.config.base_url = BASE_URL
    gov.config.api_key = "org-key"
    gov.openai_client_kwargs.return_value = {"default_headers": {"X-Matimo-Session-Token": "tok"}}

    model = gateway_model(gov, model="gpt-4o-mini")
    assert model.model == "openai/gpt-4o-mini"


def test_gateway_model_client_injects_live_headers_per_call(monkeypatch) -> None:
    import asyncio

    import google.adk.models.lite_llm as lite_llm_module

    from matimo_agdk.adapters.google_adk import gateway_model

    gov = MagicMock()
    gov.config.base_url = BASE_URL
    gov.config.api_key = "org-key"
    gov.openai_client_kwargs.return_value = {"default_headers": {"X-Matimo-Session-Token": "tok"}}
    gov.request_headers = MagicMock(
        return_value={"X-Matimo-Session-Token": "live-tok", "X-Matimo-Run-Id": "run-7"}
    )
    seen: dict = {}

    async def fake_acompletion(**kw):
        seen.update(kw)
        return MagicMock()

    monkeypatch.setattr(lite_llm_module, "_ensure_litellm_imported", lambda: None)
    monkeypatch.setattr(lite_llm_module, "acompletion", fake_acompletion)

    model = gateway_model(gov, model="gpt-4o-mini")
    asyncio.run(
        model.llm_client.acompletion(
            "openai/gpt-4o-mini", [], None, extra_headers={"X-Static": "1"}
        )
    )
    assert seen["extra_headers"]["X-Matimo-Session-Token"] == "live-tok"
    assert seen["extra_headers"]["X-Matimo-Run-Id"] == "run-7"
    assert seen["extra_headers"]["X-Static"] == "1"


async def test_before_model_binds_invocation_id_as_current_run() -> None:
    from matimo_agdk.adapters.google_adk import MatimoPlugin

    gov = MagicMock()
    gov.raise_if_suspended = MagicMock()
    plugin = MatimoPlugin(gov, mode="govern")
    await plugin.before_model_callback(
        callback_context=_CallbackCtx(invocation_id="inv-42"),
        llm_request=_LlmRequest(model="gemini-2.0-flash"),
    )
    gov.bind_run_id.assert_called_once_with("inv-42")


class _RunCtx:
    def __init__(self, invocation_id: str = "e-inv-1", agent_name: str = "weather_agent") -> None:
        self.invocation_id = invocation_id
        self.agent = MagicMock()
        self.agent.name = agent_name


@pytest.mark.asyncio
async def test_run_opens_running_then_closes_completed() -> None:
    from matimo_agdk.adapters.google_adk import MatimoPlugin

    gov = MagicMock()
    plugin = MatimoPlugin(gov, mode="govern")
    ctx = _RunCtx()

    assert await plugin.before_run_callback(invocation_context=ctx) is None
    gov.run_span.assert_called_once_with("e-inv-1", status="running", name="weather_agent")

    gov.run_span.reset_mock()
    await plugin.after_run_callback(invocation_context=ctx)
    gov.run_span.assert_called_once()
    args, kwargs = gov.run_span.call_args
    assert args == ("e-inv-1",)
    assert kwargs["status"] == "completed"
    assert kwargs["name"] == "weather_agent"
    assert kwargs["duration_ms"] >= 0


@pytest.mark.asyncio
async def test_run_error_closes_failed_and_only_once() -> None:
    from matimo_agdk.adapters.google_adk import MatimoPlugin

    gov = MagicMock()
    plugin = MatimoPlugin(gov, mode="observe")
    ctx = _RunCtx()

    await plugin.before_run_callback(invocation_context=ctx)
    gov.run_span.reset_mock()
    await plugin.on_run_error_callback(invocation_context=ctx, error=RuntimeError("boom"))
    assert gov.run_span.call_args.kwargs["status"] == "failed"

    # A second terminal notification for the same invocation must not re-emit.
    gov.run_span.reset_mock()
    await plugin.after_run_callback(invocation_context=ctx)
    gov.run_span.assert_not_called()


@pytest.mark.asyncio
async def test_run_span_failure_never_breaks_the_agent() -> None:
    from matimo_agdk.adapters.google_adk import MatimoPlugin

    gov = MagicMock()
    gov.run_span.side_effect = RuntimeError("telemetry down")
    plugin = MatimoPlugin(gov, mode="govern")
    ctx = _RunCtx()

    assert await plugin.before_run_callback(invocation_context=ctx) is None
    await plugin.after_run_callback(invocation_context=ctx)


@pytest.mark.asyncio
async def test_governor_run_span_emits_kind_run_event() -> None:
    from matimo_agdk.governor import AsyncGovernor

    gov = AsyncGovernor.__new__(AsyncGovernor)
    gov._telemetry = MagicMock()
    gov.run_span("e-inv-9", status="completed", name="adk", duration_ms=12)

    event = gov._telemetry.submit.call_args.args[0]
    assert event["kind"] == "run"
    assert event["runId"] == "e-inv-9"
    assert event["sessionId"] == "e-inv-9"
    assert event["status"] == "completed"
    assert event["durationMs"] == 12


def _real_span_governor() -> tuple[object, MagicMock]:
    """A real AsyncGovernor whose exporter is a mock, so the *real* span
    builders run (a MagicMock governor accepts any kwargs and cannot catch a
    kwarg the builders don't know about -- which is how `span_id` was being
    silently dropped)."""
    from matimo_agdk.governor import AsyncGovernor

    gov = AsyncGovernor.__new__(AsyncGovernor)
    gov._telemetry = MagicMock()
    return gov, gov._telemetry


@pytest.mark.asyncio
async def test_llm_and_tool_spans_reach_the_exporter_through_real_builders() -> None:
    from matimo_agdk.adapters.google_adk import MatimoPlugin

    gov, telemetry = _real_span_governor()
    plugin = MatimoPlugin(gov, mode="observe")
    ctx = _CallbackCtx(invocation_id="e-inv-7")

    await plugin.before_model_callback(callback_context=ctx, llm_request=_LlmRequest("gpt-4o-mini"))
    await plugin.after_model_callback(callback_context=ctx, llm_response=_LlmResponse())
    await plugin.before_tool_callback(
        tool=_Tool("get_weather"), tool_args={"city": "x"}, tool_context=_Ctx("call-9", "e-inv-7")
    )
    await plugin.after_tool_callback(
        tool=_Tool("get_weather"),
        tool_args={"city": "x"},
        tool_context=_Ctx("call-9", "e-inv-7"),
        result={"ok": True},
    )

    events = [c.args[0] for c in telemetry.submit.call_args_list]
    by_kind = {e["kind"]: e for e in events}
    assert set(by_kind) == {"llm", "tool"}
    assert by_kind["llm"]["runId"] == "e-inv-7"
    assert by_kind["llm"]["spanId"]
    assert by_kind["tool"]["runId"] == "e-inv-7"
    assert by_kind["tool"]["spanId"] == "call-9"
