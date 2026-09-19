from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

import httpx
import pytest
import respx
from crewai.tools import BaseTool
from pydantic import BaseModel

from matimo_agdk.exceptions import AgentSuspendedLocally, ToolDenied
from matimo_agdk.identity import IdentityCredentials
from matimo_agdk.tools import ToolDecision

from ..conftest import BASE_URL, future_iso
from .conftest import bound_governor


class _Args(BaseModel):
    x: int


def _fresh_tool() -> BaseTool:
    class MyTool(BaseTool):
        name: str = "mytool"
        description: str = "test"
        args_schema: type[BaseModel] = _Args

        def _run(self, x: int) -> str:
            return f"ran {x}"

    return MyTool()


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


def test_observe_mode_never_calls_check_tool() -> None:
    from matimo_agdk.adapters.crewai import govern_tool

    gov = MagicMock()
    gov.check_tool = MagicMock()
    gov.tool_span = MagicMock()

    t = govern_tool(_fresh_tool(), gov, mode="observe")
    assert t.run(x=5) == "ran 5"
    gov.check_tool.assert_not_called()
    gov.tool_span.assert_called_once()
    assert gov.tool_span.call_args.kwargs["status"] == "completed"


@respx.mock
def test_govern_mode_allow_passes_through(identity: IdentityCredentials, credentials_dir) -> None:
    from matimo_agdk.adapters.crewai import govern_tool

    _mock_sessions_and_telemetry()
    respx.post(f"{BASE_URL}/tools/check").mock(
        return_value=httpx.Response(200, json={"data": {"decision": "ALLOW"}})
    )
    governor = bound_governor(identity, credentials_dir)
    t = govern_tool(_fresh_tool(), governor, mode="govern")

    assert t.run(x=9) == "ran 9"


@respx.mock
def test_govern_mode_deny_surfaces_as_tool_error_not_a_crash(
    identity: IdentityCredentials, credentials_dir
) -> None:
    from matimo_agdk.adapters.crewai import govern_tool

    _mock_sessions_and_telemetry()
    respx.post(f"{BASE_URL}/tools/check").mock(
        return_value=httpx.Response(
            200, json={"data": {"decision": "DENY", "reason": "tool_category_not_allowed"}}
        )
    )
    governor = bound_governor(identity, credentials_dir)
    t = govern_tool(_fresh_tool(), governor, mode="govern")

    with pytest.raises(ToolDenied) as excinfo:
        t.run(x=1)
    assert excinfo.value.reason == "tool_category_not_allowed"


@respx.mock
def test_govern_mode_pending_then_approved_proceeds(
    identity: IdentityCredentials, credentials_dir
) -> None:
    from matimo_agdk.adapters.crewai import govern_tool

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
    t = govern_tool(_fresh_tool(), governor, mode="govern")

    assert t.run(x=3) == "ran 3"
    assert calls["n"] == 2


def test_suspended_state_stops_before_tool_call() -> None:
    from matimo_agdk.adapters.crewai import govern_tool

    gov = MagicMock()
    gov.check_tool = MagicMock(return_value=ToolDecision(decision="ALLOW"))
    gov.raise_if_suspended.side_effect = AgentSuspendedLocally("suspended", False)

    t = govern_tool(_fresh_tool(), gov, mode="govern")
    with pytest.raises(AgentSuspendedLocally):
        t.run(x=1)
    gov.check_tool.assert_not_called()


def test_govern_crew_wraps_every_agent_tool() -> None:
    from matimo_agdk.adapters.crewai import govern_crew

    class FakeAgent:
        def __init__(self, tools: list) -> None:
            self.tools = tools

    gov = MagicMock()
    gov.check_tool.return_value = ToolDecision(decision="ALLOW")
    gov.tool_span = MagicMock()

    t1, t2 = _fresh_tool(), _fresh_tool()
    agents = [FakeAgent([t1]), FakeAgent([t2])]
    govern_crew(agents, gov, mode="govern")

    assert t1.run(x=1) == "ran 1"
    assert t2.run(x=2) == "ran 2"
    assert gov.check_tool.call_count == 2


def test_govern_crew_does_not_double_wrap_a_tool_shared_by_two_agents() -> None:
    from matimo_agdk.adapters.crewai import govern_crew

    class FakeAgent:
        def __init__(self, tools: list) -> None:
            self.tools = tools

    gov = MagicMock()
    gov.check_tool.return_value = ToolDecision(decision="ALLOW")
    gov.tool_span = MagicMock()

    shared = _fresh_tool()
    agents = [FakeAgent([shared]), FakeAgent([shared])]
    govern_crew(agents, gov, mode="govern")

    shared.run(x=1)
    assert gov.check_tool.call_count == 1


def test_gateway_llm_wires_base_url_and_static_session_header() -> None:
    from matimo_agdk.adapters.crewai import gateway_llm

    gov = MagicMock()
    gov.config.base_url = BASE_URL
    gov.config.api_key = "org-key"
    gov.openai_client_kwargs.return_value = {"default_headers": {"X-Matimo-Session-Token": "tok"}}

    llm = gateway_llm(gov, model="gpt-4o-mini")
    assert llm.base_url == BASE_URL
    assert llm.additional_params["extra_headers"]["X-Matimo-Session-Token"] == "tok"


def test_gateway_llm_installs_live_header_interceptor() -> None:
    import httpx

    from matimo_agdk.adapters.crewai import gateway_llm

    gov = MagicMock()
    gov.config.base_url = BASE_URL
    gov.config.api_key = "org-key"
    gov.openai_client_kwargs.return_value = {"default_headers": {"X-Matimo-Session-Token": "tok"}}
    gov.request_headers = MagicMock(
        return_value={
            "X-Matimo-Session-Token": "live-tok",
            "X-Matimo-Run-Id": "run-1",
            "Matimo-Agent-Signature": "jws",
        }
    )

    llm = gateway_llm(gov, model="gpt-4o-mini")
    assert llm.interceptor is not None
    req = httpx.Request("POST", f"{BASE_URL}/chat/completions", content=b'{"a":1}')
    out = llm.interceptor.on_outbound(req)
    gov.request_headers.assert_called_once_with(b'{"a":1}')
    assert out.headers["X-Matimo-Session-Token"] == "live-tok"
    assert out.headers["X-Matimo-Run-Id"] == "run-1"
    assert out.headers["Matimo-Agent-Signature"] == "jws"


def _fresh_interceptor(gov: MagicMock):
    from matimo_agdk.adapters.crewai import make_interceptor

    gov.request_headers = MagicMock(return_value={})
    return make_interceptor(gov)


def _chat_request(model: str = "gpt-4o-mini") -> httpx.Request:
    return httpx.Request(
        "POST",
        f"{BASE_URL}/chat/completions",
        content=json.dumps({"model": model}).encode(),
    )


def test_interceptor_emits_llm_span_with_model_and_usage() -> None:
    gov = MagicMock()
    gov.llm_span = MagicMock()
    interceptor = _fresh_interceptor(gov)

    interceptor.on_outbound(_chat_request())
    resp = httpx.Response(
        200,
        headers={"content-type": "application/json"},
        content=json.dumps(
            {
                "model": "gpt-4o-mini-2024-07-18",
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            }
        ).encode(),
    )
    interceptor.on_inbound(resp)

    gov.llm_span.assert_called_once()
    kwargs = gov.llm_span.call_args.kwargs
    assert kwargs["model"] == "gpt-4o-mini-2024-07-18"
    assert kwargs["status"] == "completed"
    assert kwargs["attributes"]["gen_ai.usage.input_tokens"] == 7
    assert kwargs["attributes"]["gen_ai.usage.output_tokens"] == 3


def test_interceptor_marks_non_2xx_response_as_error_span() -> None:
    gov = MagicMock()
    gov.llm_span = MagicMock()
    interceptor = _fresh_interceptor(gov)

    interceptor.on_outbound(_chat_request())
    resp = httpx.Response(
        500, headers={"content-type": "application/json"}, content=b'{"error":"boom"}'
    )
    interceptor.on_inbound(resp)

    assert gov.llm_span.call_args.kwargs["status"] == "error"
    # requested model still reported even though the error body has no "model" key
    assert gov.llm_span.call_args.kwargs["model"] == "gpt-4o-mini"


def test_interceptor_skips_body_read_for_streaming_response() -> None:
    gov = MagicMock()
    gov.llm_span = MagicMock()
    interceptor = _fresh_interceptor(gov)

    interceptor.on_outbound(_chat_request())
    resp = httpx.Response(200, headers={"content-type": "text/event-stream"})
    interceptor.on_inbound(resp)

    kwargs = gov.llm_span.call_args.kwargs
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["attributes"] is None


def test_interceptor_async_path_emits_llm_span() -> None:
    gov = MagicMock()
    gov.llm_span = MagicMock()
    interceptor = _fresh_interceptor(gov)
    req = _chat_request()

    async def run() -> None:
        await interceptor.aon_outbound(req)
        resp = httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=json.dumps(
                {"model": "gpt-4o-mini", "usage": {"prompt_tokens": 1, "completion_tokens": 2}}
            ).encode(),
        )
        await interceptor.aon_inbound(resp)

    asyncio.run(run())

    kwargs = gov.llm_span.call_args.kwargs
    assert kwargs["status"] == "completed"
    assert kwargs["attributes"]["gen_ai.usage.output_tokens"] == 2


def test_interceptor_correlates_with_governor_run(
    identity: IdentityCredentials, credentials_dir
) -> None:
    """No natural per-kickoff id exists for CrewAI -- spans correlate only
    when the caller wraps the call in governor.run(), same as tool spans."""
    from matimo_agdk.adapters.crewai import make_interceptor

    governor = bound_governor(identity, credentials_dir)
    governor.request_headers = MagicMock(return_value={})  # type: ignore[method-assign]
    captured: list = []
    governor._emit = captured.append  # type: ignore[method-assign]  # noqa: SLF001

    interceptor = make_interceptor(governor)
    req = _chat_request()
    resp = httpx.Response(200, headers={"content-type": "application/json"}, content=b"{}")

    with governor.run("my-crew-run") as run_id:
        interceptor.on_outbound(req)
        interceptor.on_inbound(resp)

    llm_events = [e for e in captured if e.get("kind") == "llm"]
    assert llm_events, "expected an llm span to have been emitted"
    assert llm_events[0]["runId"] == run_id


def test_interceptor_uncorrelated_without_governor_run(
    identity: IdentityCredentials, credentials_dir
) -> None:
    """Without a governor.run() wrapper, each LLM call gets its own fresh,
    uncorrelated run id -- the same documented fallback as tool spans."""
    from matimo_agdk.adapters.crewai import make_interceptor

    governor = bound_governor(identity, credentials_dir)
    governor.request_headers = MagicMock(return_value={})  # type: ignore[method-assign]
    captured: list = []
    governor._emit = captured.append  # type: ignore[method-assign]  # noqa: SLF001

    interceptor = make_interceptor(governor)
    req = _chat_request()
    resp = httpx.Response(200, headers={"content-type": "application/json"}, content=b"{}")

    interceptor.on_outbound(req)
    interceptor.on_inbound(resp)

    llm_events = [e for e in captured if e.get("kind") == "llm"]
    assert llm_events, "expected an llm span to have been emitted"
    assert llm_events[0]["runId"]  # a fresh id was generated, not blank
