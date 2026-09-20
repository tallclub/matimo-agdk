from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
import respx
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import ToolException, tool

from matimo_agdk.exceptions import AgentSuspendedLocally
from matimo_agdk.identity import IdentityCredentials
from matimo_agdk.tools import ToolDecision

from ..conftest import BASE_URL, future_iso
from .conftest import bound_governor


def _fresh_add_tool() -> Any:
    """A fresh StructuredTool per test -- govern_tools() mutates a tool's
    _run/_arun in place, so a module-level shared tool would get
    double-wrapped across tests."""

    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    return add


def _mock_sessions_and_telemetry(captured_batches: list[Any] | None = None) -> None:
    respx.post(f"{BASE_URL}/sessions").mock(
        return_value=httpx.Response(
            201,
            json={
                "data": {"sessionToken": "tok", "expiresAt": future_iso(3600), "identityId": "x"}
            },
        )
    )

    def responder(request: httpx.Request) -> httpx.Response:
        if captured_batches is not None:
            captured_batches.append(json.loads(request.content))
        return httpx.Response(200, json={"data": {"accepted": 1, "failed": []}})

    respx.post(f"{BASE_URL}/telemetry/batch").mock(side_effect=responder)


# ---------------------------------------------------------------------------
# govern_tools: observe / allow / deny / pending
# ---------------------------------------------------------------------------


def test_observe_mode_never_calls_check_tool() -> None:
    from matimo_agdk.adapters.langchain import govern_tools

    gov = MagicMock()
    gov.check_tool = MagicMock()
    gov.check_and_wait = MagicMock()
    gov.tool_span = MagicMock()

    tools = govern_tools([_fresh_add_tool()], gov, mode="observe")
    result = tools[0].invoke({"a": 1, "b": 2})

    assert result == 3
    gov.check_tool.assert_not_called()
    gov.check_and_wait.assert_not_called()
    gov.tool_span.assert_called_once()
    assert gov.tool_span.call_args.kwargs["status"] == "completed"


@respx.mock
def test_govern_mode_allow_passes_through(identity: IdentityCredentials, credentials_dir) -> None:
    from matimo_agdk.adapters.langchain import govern_tools

    _mock_sessions_and_telemetry()
    respx.post(f"{BASE_URL}/tools/check").mock(
        return_value=httpx.Response(200, json={"data": {"decision": "ALLOW"}})
    )
    governor = bound_governor(identity, credentials_dir)
    tools = govern_tools([_fresh_add_tool()], governor, mode="govern")

    result = tools[0].invoke({"a": 3, "b": 4})
    assert result == 7


@respx.mock
def test_govern_mode_deny_surfaces_as_tool_error_not_a_crash(
    identity: IdentityCredentials, credentials_dir
) -> None:
    """Regression test for a real bug found live-verifying against Gateway
    (2026-09-18 live verification, see CHANGELOG.md): govern_tools() used to raise this SDK's own
    ToolDenied on DENY, which langchain_core's BaseTool.run() cannot
    gracefully convert -- its handle_tool_error machinery special-cases
    exactly ToolException (and pydantic's ValidationError), so a bare
    ToolDenied always fell into the generic except-and-reraise branch and
    crashed the chain regardless of handle_tool_error. This test's own name
    was true of the *intent*, not the *code*, before that fix: it asserted
    `pytest.raises(ToolDenied)`, which is exactly the crash the name says
    shouldn't happen. Fixed to raise ToolException, and this test now
    actually asserts the graceful (non-crashing) behavior it describes."""
    from matimo_agdk.adapters.langchain import govern_tools

    _mock_sessions_and_telemetry()
    respx.post(f"{BASE_URL}/tools/check").mock(
        return_value=httpx.Response(
            200, json={"data": {"decision": "DENY", "reason": "tool_category_not_allowed"}}
        )
    )
    governor = bound_governor(identity, credentials_dir)
    add_tool = _fresh_add_tool()
    add_tool.handle_tool_error = True  # the standard LangChain opt-in this relies on
    tools = govern_tools([add_tool], governor, mode="govern")

    result = tools[0].invoke({"a": 1, "b": 1})
    assert result == "tool_category_not_allowed"


@respx.mock
def test_govern_mode_deny_raises_tool_exception_without_handle_tool_error(
    identity: IdentityCredentials, credentials_dir
) -> None:
    """Without handle_tool_error=True (LangChain's own default), a DENY
    still propagates -- as ToolException, not this SDK's ToolDenied. Locks
    in the exact exception type BaseTool.run() actually special-cases,
    verified by reading langchain_core/tools/base.py directly."""
    from matimo_agdk.adapters.langchain import govern_tools

    _mock_sessions_and_telemetry()
    respx.post(f"{BASE_URL}/tools/check").mock(
        return_value=httpx.Response(
            200, json={"data": {"decision": "DENY", "reason": "tool_category_not_allowed"}}
        )
    )
    governor = bound_governor(identity, credentials_dir)
    tools = govern_tools([_fresh_add_tool()], governor, mode="govern")

    with pytest.raises(ToolException) as excinfo:
        tools[0].invoke({"a": 1, "b": 1})
    assert str(excinfo.value) == "tool_category_not_allowed"


@respx.mock
def test_govern_mode_pending_then_approved_proceeds(
    identity: IdentityCredentials, credentials_dir
) -> None:
    from matimo_agdk.adapters.langchain import govern_tools

    _mock_sessions_and_telemetry()
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
    governor = bound_governor(identity, credentials_dir)
    governor._tools.poll_interval = 0.01  # noqa: SLF001
    governor._tools.poll_max_interval = 0.02  # noqa: SLF001
    tools = govern_tools([_fresh_add_tool()], governor, mode="govern")

    result = tools[0].invoke({"a": 10, "b": 20})
    assert result == 30


def test_suspended_state_stops_before_tool_call() -> None:
    from matimo_agdk.adapters.langchain import govern_tools

    gov = MagicMock()
    gov.check_tool = MagicMock(return_value=ToolDecision(decision="ALLOW"))
    gov.check_and_wait = MagicMock(return_value=ToolDecision(decision="ALLOW"))
    gov.raise_if_suspended.side_effect = AgentSuspendedLocally("suspended", False)

    tools = govern_tools([_fresh_add_tool()], gov, mode="govern")
    with pytest.raises(AgentSuspendedLocally):
        tools[0].invoke({"a": 1, "b": 1})
    gov.check_tool.assert_not_called()
    gov.check_and_wait.assert_not_called()


def test_suspended_state_stops_before_llm_boundary_via_callback() -> None:
    """RunnableLambda has no LLM call to trigger on_llm_start/
    on_chat_model_start, so exercise the callback's own suspend check
    directly at each of the three boundaries it guards."""
    from matimo_agdk.adapters.langchain import MatimoCallbackHandler

    run_id = __import__("uuid").uuid4()
    gov = MagicMock()
    gov.raise_if_suspended.side_effect = AgentSuspendedLocally("suspended", False)
    handler = MatimoCallbackHandler(gov, mode="govern")

    with pytest.raises(AgentSuspendedLocally):
        handler.on_chat_model_start({"kwargs": {}}, [[]], run_id=run_id, parent_run_id=None)
    with pytest.raises(AgentSuspendedLocally):
        handler.on_llm_start({}, ["prompt"], run_id=run_id, parent_run_id=None)
    with pytest.raises(AgentSuspendedLocally):
        handler.on_tool_start({"name": "search"}, "input", run_id=run_id, parent_run_id=None)


def test_tool_call_stops_before_dispatch_when_callback_and_tool_denied_agree() -> None:
    """End-to-end: a chain that actually calls a governed tool halts at
    on_tool_start (via the callback) before the tool's own governed _run
    would even get a chance to run."""
    from matimo_agdk.adapters.langchain import MatimoCallbackHandler, govern_tools

    gov = MagicMock()
    gov.raise_if_suspended.side_effect = AgentSuspendedLocally("suspended", False)
    handler = MatimoCallbackHandler(gov, mode="govern")
    tools = govern_tools([_fresh_add_tool()], gov, mode="govern")

    with pytest.raises(AgentSuspendedLocally):
        tools[0].invoke({"a": 1, "b": 1}, config={"callbacks": [handler]})


# ---------------------------------------------------------------------------
# LLM spans: gen_ai.request.model + usage
# ---------------------------------------------------------------------------


def test_llm_span_carries_model_and_usage() -> None:
    from matimo_agdk.adapters.langchain import MatimoCallbackHandler

    gov = MagicMock()
    gov.raise_if_suspended = MagicMock()
    gov.llm_span = MagicMock()
    handler = MatimoCallbackHandler(gov, mode="govern")

    run_id = __import__("uuid").uuid4()

    class FakeGeneration:
        generation_info = {"finish_reason": "stop"}

    class FakeResult:
        llm_output = {
            "model_name": "gpt-4o-mini",
            "token_usage": {"prompt_tokens": 12, "completion_tokens": 34},
        }
        generations = [[FakeGeneration()]]

    handler.on_chat_model_start({"kwargs": {}}, [[]], run_id=run_id, parent_run_id=None)
    handler.on_llm_end(FakeResult(), run_id=run_id, parent_run_id=None)

    gov.llm_span.assert_called_once()
    kwargs = gov.llm_span.call_args.kwargs
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["finish_reasons"] == ["stop"]
    assert kwargs["attributes"]["gen_ai.usage.input_tokens"] == 12
    assert kwargs["attributes"]["gen_ai.usage.output_tokens"] == 34
    assert kwargs["run_id"] == str(run_id)


def test_observe_mode_callback_never_checks_suspend() -> None:
    from matimo_agdk.adapters.langchain import MatimoCallbackHandler

    gov = MagicMock()
    gov.raise_if_suspended = MagicMock()
    handler = MatimoCallbackHandler(gov, mode="observe")

    chain = RunnableLambda(lambda x: x + 1)
    result = chain.invoke(1, config={"callbacks": [handler]})
    assert result == 2
    gov.raise_if_suspended.assert_not_called()


def test_gateway_chat_model_openai_wires_signed_http_client() -> None:
    from matimo_agdk.adapters.langchain import gateway_chat_model

    gov = MagicMock()
    gov.config.base_url = BASE_URL
    gov.config.api_key = "org-key"
    gov.openai_client_kwargs.return_value = {"default_headers": {"X-Matimo-Session-Token": "tok"}}
    gov.httpx_client.return_value = httpx.Client(base_url=BASE_URL)

    model = gateway_chat_model(gov, model="gpt-4o-mini", provider="openai")
    assert (
        model.openai_api_base == BASE_URL or str(model.root_client.base_url).rstrip("/") == BASE_URL
    )


def test_gateway_chat_model_anthropic_uses_static_headers_no_http_client_kwarg() -> None:
    from matimo_agdk.adapters.langchain import gateway_chat_model

    gov = MagicMock()
    gov.config.base_url = BASE_URL
    gov.config.api_key = "org-key"
    gov.anthropic_client_kwargs.return_value = {
        "default_headers": {"X-Matimo-Session-Token": "tok"}
    }

    model = gateway_chat_model(gov, model="claude-3-5-sonnet-20241022", provider="anthropic")
    assert model.default_headers["X-Matimo-Session-Token"] == "tok"


def test_gateway_chat_model_rejects_unknown_provider() -> None:
    from matimo_agdk.adapters.langchain import gateway_chat_model

    gov = MagicMock()
    with pytest.raises(ValueError):
        gateway_chat_model(gov, provider="bogus")


def test_llm_and_tool_spans_reach_the_exporter_through_real_builders() -> None:
    """A MagicMock governor accepts any kwargs, so it cannot catch a kwarg the
    span builders don't know about (`span_id`/`parent_span_id` were silently
    dropped that way). Run the real builders against a mock exporter."""
    import uuid

    from matimo_agdk.adapters.langchain import MatimoCallbackHandler
    from matimo_agdk.governor import Governor

    gov = Governor.__new__(Governor)
    gov._telemetry = MagicMock()
    handler = MatimoCallbackHandler(gov, mode="observe")

    chain_id, llm_id = uuid.uuid4(), uuid.uuid4()

    class FakeResult:
        llm_output = {"model_name": "gpt-4o-mini", "token_usage": {}}
        generations = [[]]

    handler.on_chain_start({}, {}, run_id=chain_id, parent_run_id=None)
    handler.on_chat_model_start({"kwargs": {}}, [[]], run_id=llm_id, parent_run_id=chain_id)
    handler.on_llm_end(FakeResult(), run_id=llm_id, parent_run_id=chain_id)

    events = [c.args[0] for c in gov._telemetry.submit.call_args_list]
    assert [e["kind"] for e in events] == ["run", "llm"]
    llm = events[1]
    assert llm["runId"] == str(chain_id)
    assert llm["spanId"] == str(llm_id)
    assert llm["parentSpanId"] == str(chain_id)


# ---------------------------------------------------------------------------
# Run lifecycle: Gateway only ends a run on a terminal kind:"run" span
# ---------------------------------------------------------------------------


def _real_governor_with_mock_exporter() -> Any:
    from matimo_agdk.governor import Governor

    gov = Governor.__new__(Governor)
    gov._telemetry = MagicMock()
    return gov


def _events(gov: Any) -> list[dict[str, Any]]:
    return [c.args[0] for c in gov._telemetry.submit.call_args_list]


class _FakeLLMResult:
    llm_output = {"model_name": "gpt-4o-mini", "token_usage": {}}
    generations = [[]]


def test_standalone_llm_call_opens_and_closes_its_own_run() -> None:
    """Regression: a parentless LLM call outside governor.run() used to emit a
    span under a fresh run_id and never a terminal run span, leaving the run
    `running` in Gateway."""
    import uuid

    from matimo_agdk.adapters.langchain import MatimoCallbackHandler

    gov = _real_governor_with_mock_exporter()
    handler = MatimoCallbackHandler(gov, mode="observe")
    llm_id = uuid.uuid4()

    handler.on_chat_model_start({"kwargs": {}}, [[]], run_id=llm_id, parent_run_id=None)
    handler.on_llm_end(_FakeLLMResult(), run_id=llm_id, parent_run_id=None)

    events = _events(gov)
    assert [(e["kind"], e["status"]) for e in events] == [
        ("run", "running"),
        ("llm", "completed"),
        ("run", "completed"),
    ]
    assert {e["runId"] for e in events} == {str(llm_id)}


def test_standalone_tool_call_opens_and_closes_its_own_run() -> None:
    import uuid

    from matimo_agdk.adapters.langchain import MatimoCallbackHandler

    gov = _real_governor_with_mock_exporter()
    handler = MatimoCallbackHandler(gov, mode="observe")
    tool_id = uuid.uuid4()

    handler.on_tool_start({"name": "add"}, "1+1", run_id=tool_id, parent_run_id=None)
    handler.on_tool_end("2", run_id=tool_id, parent_run_id=None, name="add")

    events = _events(gov)
    assert [(e["kind"], e["status"]) for e in events] == [
        ("run", "running"),
        ("tool", "completed"),
        ("run", "completed"),
    ]
    assert events[0]["name"] == "add"


def test_run_is_closed_failed_when_the_root_errors() -> None:
    import uuid

    from matimo_agdk.adapters.langchain import MatimoCallbackHandler

    gov = _real_governor_with_mock_exporter()
    handler = MatimoCallbackHandler(gov, mode="observe")
    chain_id = uuid.uuid4()

    handler.on_chain_start({}, {}, run_id=chain_id, parent_run_id=None)
    handler.on_chain_error(RuntimeError("boom"), run_id=chain_id, parent_run_id=None)

    assert [(e["kind"], e["status"]) for e in _events(gov)] == [
        ("run", "running"),
        ("run", "failed"),
    ]


def test_only_the_root_node_owns_the_run() -> None:
    """Chain -> LLM -> tool under one root: one run opened, one closed, and the
    terminal span comes after the child spans."""
    import uuid

    from matimo_agdk.adapters.langchain import MatimoCallbackHandler

    gov = _real_governor_with_mock_exporter()
    handler = MatimoCallbackHandler(gov, mode="observe")
    chain_id, llm_id, tool_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()

    handler.on_chain_start({}, {}, run_id=chain_id, parent_run_id=None)
    handler.on_chat_model_start({"kwargs": {}}, [[]], run_id=llm_id, parent_run_id=chain_id)
    handler.on_llm_end(_FakeLLMResult(), run_id=llm_id, parent_run_id=chain_id)
    handler.on_tool_start({"name": "add"}, "1", run_id=tool_id, parent_run_id=chain_id)
    handler.on_tool_end("2", run_id=tool_id, parent_run_id=chain_id, name="add")
    handler.on_chain_end({}, run_id=chain_id, parent_run_id=None)

    events = _events(gov)
    assert [(e["kind"], e["status"]) for e in events] == [
        ("run", "running"),
        ("llm", "completed"),
        ("tool", "completed"),
        ("run", "completed"),
    ]
    assert {e["runId"] for e in events} == {str(chain_id)}


def test_calls_inside_governor_run_join_it_and_open_no_run_of_their_own() -> None:
    """The bug from the live demo: two separate parentless calls inside one
    `governor.run()` each became their own never-closed run. They must join the
    ambient run, which governor.run() itself opens and closes."""
    import uuid

    from matimo_agdk.adapters.langchain import MatimoCallbackHandler

    gov = _real_governor_with_mock_exporter()
    handler = MatimoCallbackHandler(gov, mode="observe")
    llm_id, tool_id = uuid.uuid4(), uuid.uuid4()

    with gov.run("demo") as ambient:
        handler.on_chat_model_start({"kwargs": {}}, [[]], run_id=llm_id, parent_run_id=None)
        handler.on_llm_end(_FakeLLMResult(), run_id=llm_id, parent_run_id=None)
        handler.on_tool_start({"name": "add"}, "1", run_id=tool_id, parent_run_id=None)
        handler.on_tool_end("2", run_id=tool_id, parent_run_id=None, name="add")

    events = _events(gov)
    assert [(e["kind"], e["status"]) for e in events] == [
        ("run", "running"),
        ("llm", "completed"),
        ("tool", "completed"),
        ("run", "completed"),
    ]
    assert {e["runId"] for e in events} == {ambient}
    # The per-node ids are still distinct span ids under that one run.
    assert events[1]["spanId"] == str(llm_id)
    assert events[2]["spanId"] == str(tool_id)
