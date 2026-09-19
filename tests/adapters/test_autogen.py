from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import respx
from autogen_core import CancellationToken
from autogen_core.tools import FunctionTool

from matimo_agdk.exceptions import AgentSuspendedLocally, ToolDenied
from matimo_agdk.identity import IdentityCredentials
from matimo_agdk.tools import ToolDecision

from ..conftest import BASE_URL, future_iso
from .conftest import bound_async_governor, bound_governor


def _add(a: int, b: int) -> int:
    return a + b


def _fresh_tool() -> FunctionTool:
    return FunctionTool(_add, description="adds two numbers")


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
    from matimo_agdk.adapters.autogen import govern_tools

    gov = MagicMock()
    gov.check_tool = AsyncMock()
    gov.tool_span = MagicMock()

    tools = govern_tools([_fresh_tool()], gov, mode="observe")
    result = await tools[0].run_json({"a": 1, "b": 2}, CancellationToken())
    assert result == 3
    gov.check_tool.assert_not_called()
    gov.tool_span.assert_called_once()
    assert gov.tool_span.call_args.kwargs["status"] == "completed"


@pytest.mark.asyncio
@respx.mock
async def test_govern_mode_allow_passes_through_sync_governor(
    identity: IdentityCredentials, credentials_dir
) -> None:
    from matimo_agdk.adapters.autogen import govern_tools

    _mock_sessions_and_telemetry()
    respx.post(f"{BASE_URL}/tools/check").mock(
        return_value=httpx.Response(200, json={"data": {"decision": "ALLOW"}})
    )
    governor = bound_governor(identity, credentials_dir)
    tools = govern_tools([_fresh_tool()], governor, mode="govern")

    result = await tools[0].run_json({"a": 3, "b": 4}, CancellationToken())
    assert result == 7


@pytest.mark.asyncio
@respx.mock
async def test_govern_mode_allow_passes_through_async_governor(
    identity: IdentityCredentials, credentials_dir
) -> None:
    from matimo_agdk.adapters.autogen import govern_tools

    _mock_sessions_and_telemetry()
    respx.post(f"{BASE_URL}/tools/check").mock(
        return_value=httpx.Response(200, json={"data": {"decision": "ALLOW"}})
    )
    governor = bound_async_governor(identity, credentials_dir)
    tools = govern_tools([_fresh_tool()], governor, mode="govern")

    result = await tools[0].run_json({"a": 5, "b": 6}, CancellationToken())
    assert result == 11


@pytest.mark.asyncio
@respx.mock
async def test_govern_mode_deny_surfaces_as_tool_error_not_a_crash(
    identity: IdentityCredentials, credentials_dir
) -> None:
    from matimo_agdk.adapters.autogen import govern_tools

    _mock_sessions_and_telemetry()
    respx.post(f"{BASE_URL}/tools/check").mock(
        return_value=httpx.Response(
            200, json={"data": {"decision": "DENY", "reason": "tool_category_not_allowed"}}
        )
    )
    governor = bound_governor(identity, credentials_dir)
    tools = govern_tools([_fresh_tool()], governor, mode="govern")

    with pytest.raises(ToolDenied) as excinfo:
        await tools[0].run_json({"a": 1, "b": 1}, CancellationToken())
    assert excinfo.value.reason == "tool_category_not_allowed"


@pytest.mark.asyncio
@respx.mock
async def test_govern_mode_pending_then_approved_proceeds(
    identity: IdentityCredentials, credentials_dir
) -> None:
    from matimo_agdk.adapters.autogen import govern_tools

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
    governor = bound_governor(identity, credentials_dir)
    governor._tools.poll_interval = 0.01  # noqa: SLF001
    governor._tools.poll_max_interval = 0.02  # noqa: SLF001
    tools = govern_tools([_fresh_tool()], governor, mode="govern")

    result = await tools[0].run_json({"a": 10, "b": 20}, CancellationToken())
    assert result == 30
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_suspended_state_stops_before_tool_call() -> None:
    from matimo_agdk.adapters.autogen import govern_tools

    gov = MagicMock()
    gov.check_tool = AsyncMock(return_value=ToolDecision(decision="ALLOW"))
    gov.raise_if_suspended = MagicMock(side_effect=AgentSuspendedLocally("suspended", False))

    tools = govern_tools([_fresh_tool()], gov, mode="govern")
    with pytest.raises(AgentSuspendedLocally):
        await tools[0].run_json({"a": 1, "b": 1}, CancellationToken())
    gov.check_tool.assert_not_called()


def test_gateway_model_client_requires_async_governor() -> None:
    from matimo_agdk.adapters.autogen import gateway_model_client

    gov = MagicMock()
    gov.check_tool = MagicMock()  # sync -> not a coroutine function
    with pytest.raises(TypeError):
        gateway_model_client(gov)


def test_gateway_model_client_wires_signed_async_http_client() -> None:
    from matimo_agdk.adapters.autogen import gateway_model_client

    gov = MagicMock()

    async def acheck(*a, **k):
        return ToolDecision(decision="ALLOW")

    gov.check_tool = acheck
    gov.config.base_url = BASE_URL
    gov.config.api_key = "org-key"
    gov.httpx_async_client.return_value = httpx.AsyncClient(base_url=BASE_URL)

    client = gateway_model_client(gov, model="gpt-4o-mini")
    assert client is not None


def _fake_client_deps(gov: MagicMock) -> None:
    gov.config.base_url = BASE_URL
    gov.config.api_key = "org-key"
    gov.httpx_async_client.return_value = httpx.AsyncClient(base_url=BASE_URL)


@pytest.mark.asyncio
async def test_gateway_model_client_create_emits_llm_span_with_usage() -> None:
    from autogen_core.models import CreateResult, RequestUsage

    from matimo_agdk.adapters.autogen import gateway_model_client

    gov = MagicMock()

    async def acheck(*a, **k):
        return ToolDecision(decision="ALLOW")

    gov.check_tool = acheck
    gov.llm_span = MagicMock()
    _fake_client_deps(gov)

    client = gateway_model_client(gov, model="gpt-4o-mini")

    async def fake_create(*a, **k):
        return CreateResult(
            finish_reason="stop",
            content="hi",
            usage=RequestUsage(prompt_tokens=11, completion_tokens=4),
            cached=False,
        )

    from matimo_agdk.adapters.autogen import _wrap_model_create

    client.create = _wrap_model_create(fake_create, gov, "gpt-4o-mini")  # type: ignore[method-assign]

    result = await client.create([])
    assert result.content == "hi"

    gov.llm_span.assert_called_once()
    kwargs = gov.llm_span.call_args.kwargs
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["status"] == "completed"
    assert kwargs["finish_reasons"] == ["stop"]
    assert kwargs["attributes"]["gen_ai.usage.input_tokens"] == 11
    assert kwargs["attributes"]["gen_ai.usage.output_tokens"] == 4


@pytest.mark.asyncio
async def test_gateway_model_client_create_emits_error_span_on_exception() -> None:
    from matimo_agdk.adapters.autogen import _wrap_model_create

    gov = MagicMock()
    gov.llm_span = MagicMock()

    async def failing_create(*a, **k):
        raise RuntimeError("upstream_error")

    wrapped = _wrap_model_create(failing_create, gov, "gpt-4o-mini")
    with pytest.raises(RuntimeError):
        await wrapped([])

    kwargs = gov.llm_span.call_args.kwargs
    assert kwargs["status"] == "error"
    assert kwargs["model"] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_gateway_model_client_create_stream_emits_one_span_after_exhaustion() -> None:
    from autogen_core.models import CreateResult, RequestUsage

    from matimo_agdk.adapters.autogen import _wrap_model_create_stream

    gov = MagicMock()
    gov.llm_span = MagicMock()

    async def fake_stream(*a, **k):
        yield "chunk-1"
        yield "chunk-2"
        yield CreateResult(
            finish_reason="stop",
            content="done",
            usage=RequestUsage(prompt_tokens=2, completion_tokens=6),
            cached=False,
        )

    wrapped = _wrap_model_create_stream(fake_stream, gov, "gpt-4o-mini")
    items = [item async for item in wrapped([])]

    assert items[0] == "chunk-1"
    assert items[-1].content == "done"
    gov.llm_span.assert_called_once()
    kwargs = gov.llm_span.call_args.kwargs
    assert kwargs["status"] == "completed"
    assert kwargs["attributes"]["gen_ai.usage.output_tokens"] == 6


def test_gateway_model_client_wraps_create_and_create_stream() -> None:
    from matimo_agdk.adapters.autogen import gateway_model_client

    gov = MagicMock()

    async def acheck(*a, **k):
        return ToolDecision(decision="ALLOW")

    gov.check_tool = acheck
    _fake_client_deps(gov)

    client = gateway_model_client(gov, model="gpt-4o-mini")
    # functools.wraps() copies __name__ from the wrapped bound method, so
    # __wrapped__ (also set by functools.wraps) is what actually proves the
    # monkeypatch took effect.
    assert hasattr(client.create, "__wrapped__")
    assert hasattr(client.create_stream, "__wrapped__")


def test_model_call_correlates_with_governor_run(
    identity: IdentityCredentials, credentials_dir
) -> None:
    """No natural per-chat id exists for AutoGen's model client -- spans
    correlate only when the caller wraps the call in governor.run(), same
    as tool spans."""
    from matimo_agdk.adapters.autogen import _wrap_model_create

    governor = bound_governor(identity, credentials_dir)
    captured: list = []
    governor._emit = captured.append  # type: ignore[method-assign]  # noqa: SLF001

    async def fake_create(*a, **k):
        return None

    wrapped = _wrap_model_create(fake_create, governor, "gpt-4o-mini")

    async def run() -> None:
        with governor.run("my-chat-run") as run_id:
            await wrapped([])
        return run_id

    run_id = asyncio.run(run())

    llm_events = [e for e in captured if e.get("kind") == "llm"]
    assert llm_events, "expected an llm span to have been emitted"
    assert llm_events[0]["runId"] == run_id
