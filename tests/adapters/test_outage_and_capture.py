"""Cross-adapter behaviour added with the tool-check outage work:

- a fail-closed outage (ToolCheckUnavailable) surfaces as a recoverable tool
  error wherever a DENY does, and never runs the tool;
- a fail-open call is marked `matimo.degraded_mode` on its tool span;
- `capture_tool_results` is the one switch for the result on a tool span, for
  every adapter (off by default);
- LangChain reports one span per tool call, and maps the LLM provider.

Each scenario drives the adapter against a real Governor (real span builders)
with a mock exporter and a stubbed policy check: no network.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from autogen_core import CancellationToken
from autogen_core.tools import FunctionTool
from langchain_core.outputs import LLMResult
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import ToolException, tool

from matimo_agdk.config import GatewayConfig
from matimo_agdk.exceptions import ToolCheckUnavailable
from matimo_agdk.governor import AsyncGovernor, Governor
from matimo_agdk.tools import ToolDecision

SECRET = "sk-" + "a" * 30
DEGRADED = ToolDecision(
    decision="ALLOW",
    reason="gateway_unavailable_fail_open",
    degraded=True,
    degraded_age_seconds=12.5,
)
ALLOW = ToolDecision(decision="ALLOW")


def make_governor(
    *,
    capture: bool = False,
    decision: ToolDecision = ALLOW,
    outage: bool = False,
    asynchronous: bool = False,
) -> Any:
    """A real Governor/AsyncGovernor with a mock exporter and a stubbed check."""
    cls = AsyncGovernor if asynchronous else Governor
    gov: Any = cls.__new__(cls)
    gov.config = GatewayConfig(capture_tool_results=capture)
    gov._telemetry = MagicMock()
    gov.raise_if_suspended = MagicMock()
    gov.request_headers = MagicMock(return_value={})
    stub = AsyncMock if asynchronous else MagicMock
    if outage:
        gov.check_and_wait = stub(side_effect=ToolCheckUnavailable("gateway unreachable"))
    else:
        gov.check_and_wait = stub(return_value=decision)
    return gov


def events(gov: Any) -> list[dict[str, Any]]:
    return [c.args[0] for c in gov._telemetry.submit.call_args_list]


def tool_events(gov: Any) -> list[dict[str, Any]]:
    return [e for e in events(gov) if e["kind"] == "tool"]


def attrs_of(event: dict[str, Any]) -> dict[str, Any]:
    return dict(event.get("attributes") or {})


def add_tool(*, asynchronous: bool = False) -> Any:
    """A LangChain tool. An AsyncGovernor needs an async tool: for a sync-only
    tool `ainvoke()` runs `invoke()` in an executor, which is a sync call site."""
    if asynchronous:

        @tool
        async def add_async(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        add_async.name = "add"
        add_async.handle_tool_error = True
        return add_async

    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    add.handle_tool_error = True
    return add


# ---------------------------------------------------------------------------
# Outage: a recoverable tool error, tool never runs
# ---------------------------------------------------------------------------


def test_langchain_outage_is_a_recoverable_tool_observation() -> None:
    from matimo_agdk.adapters.langchain import govern_tools

    ran: list[int] = []

    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        ran.append(1)
        return a + b

    add.handle_tool_error = True
    gov = make_governor(outage=True)
    out = govern_tools([add], gov)[0].invoke({"a": 1, "b": 2})
    assert "gateway unreachable" in out  # the model sees this observation
    assert ran == []


def test_langchain_outage_raises_toolexception_when_the_tool_does_not_handle_errors() -> None:
    from matimo_agdk.adapters.langchain import govern_tools

    add = add_tool()
    add.handle_tool_error = False
    with pytest.raises(ToolException, match="gateway unreachable"):
        govern_tools([add], make_governor(outage=True))[0].invoke({"a": 1, "b": 2})


@pytest.mark.asyncio
async def test_langchain_async_outage_is_a_recoverable_tool_observation() -> None:
    from matimo_agdk.adapters.langchain import govern_tools

    gov = make_governor(outage=True, asynchronous=True)
    out = await govern_tools([add_tool(asynchronous=True)], gov)[0].ainvoke({"a": 1, "b": 2})
    assert "gateway unreachable" in out


@pytest.mark.asyncio
async def test_adk_outage_short_circuits_with_an_error_dict() -> None:
    from matimo_agdk.adapters.google_adk import MatimoPlugin

    gov = make_governor(outage=True, asynchronous=True)
    plugin = MatimoPlugin(gov)
    result = await plugin.before_tool_callback(
        tool=_AdkTool(), tool_args={"city": "x"}, tool_context=_AdkCtx()
    )
    assert isinstance(result, dict)
    assert "gateway unreachable" in result["error"]
    assert tool_events(gov) == []


@pytest.mark.asyncio
async def test_adk_outage_with_a_sync_governor() -> None:
    from matimo_agdk.adapters.google_adk import MatimoPlugin

    plugin = MatimoPlugin(make_governor(outage=True))
    result = await plugin.before_tool_callback(
        tool=_AdkTool(), tool_args={}, tool_context=_AdkCtx()
    )
    assert isinstance(result, dict) and "error" in result


@pytest.mark.asyncio
async def test_autogen_outage_raises_out_of_run_and_never_runs_the_tool() -> None:
    from matimo_agdk.adapters.autogen import govern_tools

    ran: list[int] = []

    def add(a: int, b: int) -> int:
        ran.append(1)
        return a + b

    gov = make_governor(outage=True, asynchronous=True)
    governed = govern_tools([FunctionTool(add, description="adds")], gov)[0]
    with pytest.raises(ToolCheckUnavailable):
        await governed.run_json({"a": 1, "b": 2}, CancellationToken())
    assert ran == []


def test_crewai_outage_raises_out_of_run_and_never_runs_the_tool() -> None:
    tool_ = _crew_tool()
    from matimo_agdk.adapters.crewai import govern_tool

    gov = make_governor(outage=True)
    with pytest.raises(ToolCheckUnavailable):
        govern_tool(tool_, gov).run(x=1)
    assert tool_calls == []


@pytest.mark.asyncio
async def test_generic_outage_async_never_runs_the_tool() -> None:
    from matimo_agdk.adapters.generic import govern

    ran: list[int] = []

    async def fetch(q: str) -> str:
        ran.append(1)
        return q

    with pytest.raises(ToolCheckUnavailable):
        await govern(fetch, make_governor(outage=True))(q="x")
    assert ran == []


# ---------------------------------------------------------------------------
# Degraded (fail-open) calls are marked on the span
# ---------------------------------------------------------------------------


def _assert_degraded(gov: Any) -> None:
    spans = tool_events(gov)
    assert len(spans) == 1
    attrs = attrs_of(spans[0])
    assert attrs["matimo.degraded_mode"] is True
    assert attrs["matimo.degraded_cache_age_seconds"] == 12.5


def _assert_not_degraded(gov: Any) -> None:
    for span in tool_events(gov):
        assert "matimo.degraded_mode" not in attrs_of(span)


def test_langchain_degraded_span_sync() -> None:
    from matimo_agdk.adapters.langchain import govern_tools

    gov = make_governor(decision=DEGRADED)
    assert govern_tools([add_tool()], gov)[0].invoke({"a": 1, "b": 2}) == 3
    _assert_degraded(gov)


@pytest.mark.asyncio
async def test_langchain_degraded_span_async() -> None:
    from matimo_agdk.adapters.langchain import govern_tools

    gov = make_governor(decision=DEGRADED, asynchronous=True)
    assert await govern_tools([add_tool(asynchronous=True)], gov)[0].ainvoke({"a": 1, "b": 2}) == 3
    _assert_degraded(gov)


@pytest.mark.asyncio
async def test_adk_degraded_span() -> None:
    from matimo_agdk.adapters.google_adk import MatimoPlugin

    gov = make_governor(decision=DEGRADED, asynchronous=True)
    plugin = MatimoPlugin(gov)
    ctx, t = _AdkCtx(), _AdkTool()
    assert await plugin.before_tool_callback(tool=t, tool_args={}, tool_context=ctx) is None
    await plugin.after_tool_callback(tool=t, tool_args={}, tool_context=ctx, result={"ok": 1})
    _assert_degraded(gov)


@pytest.mark.asyncio
async def test_adk_degraded_span_survives_a_tool_error() -> None:
    from matimo_agdk.adapters.google_adk import MatimoPlugin

    gov = make_governor(decision=DEGRADED, asynchronous=True)
    plugin = MatimoPlugin(gov)
    ctx, t = _AdkCtx(), _AdkTool()
    await plugin.before_tool_callback(tool=t, tool_args={}, tool_context=ctx)
    await plugin.on_tool_error_callback(
        tool=t, tool_args={}, tool_context=ctx, error=RuntimeError("boom")
    )
    _assert_degraded(gov)


@pytest.mark.asyncio
async def test_autogen_degraded_span() -> None:
    from matimo_agdk.adapters.autogen import govern_tools

    gov = make_governor(decision=DEGRADED, asynchronous=True)
    governed = govern_tools([FunctionTool(_add, description="adds")], gov)[0]
    assert await governed.run_json({"a": 1, "b": 2}, CancellationToken()) == 3
    _assert_degraded(gov)


def test_crewai_degraded_span() -> None:
    tool_ = _crew_tool()
    from matimo_agdk.adapters.crewai import govern_tool

    gov = make_governor(decision=DEGRADED)
    govern_tool(tool_, gov).run(x=1)
    _assert_degraded(gov)


@pytest.mark.asyncio
async def test_generic_degraded_span_async() -> None:
    from matimo_agdk.adapters.generic import govern

    async def fetch(q: str) -> str:
        return q

    gov = make_governor(decision=DEGRADED)
    await govern(fetch, gov)(q="x")
    _assert_degraded(gov)


def test_a_normal_allow_is_not_marked_degraded() -> None:
    from matimo_agdk.adapters.langchain import govern_tools

    gov = make_governor(decision=ALLOW)
    govern_tools([add_tool()], gov)[0].invoke({"a": 1, "b": 2})
    assert len(tool_events(gov)) == 1
    _assert_not_degraded(gov)


# ---------------------------------------------------------------------------
# capture_tool_results: one switch, every adapter, off by default
# ---------------------------------------------------------------------------

RESULT_KEY = "gen_ai.tool.call.result"


def _result_of(gov: Any) -> Any:
    spans = tool_events(gov)
    assert len(spans) == 1, spans
    return attrs_of(spans[0]).get(RESULT_KEY)


def _run_langchain_wrapper(gov: Any) -> None:
    from matimo_agdk.adapters.langchain import govern_tools

    @tool
    def echo(text: str) -> str:
        """Echo."""
        return f"ok {text}"

    govern_tools([echo], gov)[0].invoke({"text": "hi"})


def _run_langchain_handler_only(gov: Any) -> None:
    from matimo_agdk.adapters.langchain import MatimoCallbackHandler

    @tool
    def echo(text: str) -> str:
        """Echo."""
        return f"ok {text}"

    echo.invoke({"text": "hi"}, config={"callbacks": [MatimoCallbackHandler(gov, mode="observe")]})


def _run_generic_observe(gov: Any) -> None:
    from matimo_agdk.adapters.generic import govern

    govern(lambda text: f"ok {text}", gov, mode="observe")(text="hi")


def _run_generic_govern_guard(gov: Any) -> None:
    from matimo_agdk.adapters.generic import govern

    # A sync callable under a sync Governor is delegated to Governor.guard(), which
    # needs a real ToolGovernor; give it one whose check always allows.
    gov._tools = MagicMock()
    gov._tools.check_and_wait = MagicMock(return_value=ALLOW)
    with gov.run("r"):
        govern(lambda text: f"ok {text}", gov)(text="hi")


def _run_crewai(gov: Any) -> None:
    tool_ = _crew_tool()
    from matimo_agdk.adapters.crewai import govern_tool

    govern_tool(tool_, gov, mode="observe").run(x=1)


CAPTURE_SCENARIOS = {
    "langchain_wrapper": _run_langchain_wrapper,
    "langchain_handler_only": _run_langchain_handler_only,
    "generic_observe": _run_generic_observe,
    "generic_guard": _run_generic_govern_guard,
    "crewai": _run_crewai,
}


@pytest.mark.parametrize("name", CAPTURE_SCENARIOS)
def test_no_adapter_sends_the_result_by_default(name: str) -> None:
    gov = make_governor(capture=False)
    CAPTURE_SCENARIOS[name](gov)
    assert _result_of(gov) is None


@pytest.mark.parametrize("name", CAPTURE_SCENARIOS)
def test_every_adapter_sends_the_result_when_capture_is_on(name: str) -> None:
    gov = make_governor(capture=True)
    CAPTURE_SCENARIOS[name](gov)
    assert _result_of(gov) in {"ok hi", "ran 1"}


@pytest.mark.asyncio
async def test_adk_result_is_gated_by_capture() -> None:
    from matimo_agdk.adapters.google_adk import MatimoPlugin

    for capture in (False, True):
        gov = make_governor(capture=capture, asynchronous=True)
        plugin = MatimoPlugin(gov)
        ctx, t = _AdkCtx(), _AdkTool()
        await plugin.before_tool_callback(tool=t, tool_args={}, tool_context=ctx)
        await plugin.after_tool_callback(
            tool=t, tool_args={}, tool_context=ctx, result={"answer": 42}
        )
        assert (_result_of(gov) is not None) is capture
        if capture:
            assert "42" in _result_of(gov)


@pytest.mark.asyncio
async def test_autogen_result_is_gated_by_capture() -> None:
    from matimo_agdk.adapters.autogen import govern_tools

    for capture in (False, True):
        gov = make_governor(capture=capture, asynchronous=True)
        governed = govern_tools([FunctionTool(_add, description="adds")], gov)[0]
        await governed.run_json({"a": 2, "b": 5}, CancellationToken())
        assert (_result_of(gov) is not None) is capture
        if capture:
            assert _result_of(gov) == "7"


@pytest.mark.asyncio
async def test_generic_bridged_async_result_is_gated_by_capture() -> None:
    from matimo_agdk.adapters.generic import govern

    async def fetch(q: str) -> str:
        return f"got {q}"

    for capture in (False, True):
        gov = make_governor(capture=capture)
        await govern(fetch, gov)(q="x")
        assert (_result_of(gov) is not None) is capture


@pytest.mark.asyncio
async def test_guard_async_result_is_gated_by_capture() -> None:
    for capture in (False, True):
        gov = make_governor(capture=capture, asynchronous=True)
        gov._tools = MagicMock()
        gov._tools.check_and_wait = AsyncMock(return_value=ALLOW)

        async def fetch(q: str) -> str:
            return f"got {q}"

        async with gov.run("r"):
            await gov.guard(fetch, name="fetch")(q="x")
        assert (_result_of(gov) is not None) is capture


def test_captured_result_is_redacted_and_cut_to_500_characters() -> None:
    gov = make_governor(capture=True)
    gov._tools = MagicMock()
    gov._tools.check_and_wait = MagicMock(return_value=ALLOW)

    # A secret straddling the 500-character mark must be masked whole, not cut in half.
    long_result = "x" * 494 + " " + SECRET + " " + "y" * 100
    with gov.run("r"):
        gov.guard(lambda: long_result, name="t")()
    result = _result_of(gov)
    assert len(result) <= 500
    assert "sk-" not in result
    assert "aaaa" not in result

    gov = make_governor(capture=True)
    gov._tools = MagicMock()
    gov._tools.check_and_wait = MagicMock(return_value=ALLOW)
    with gov.run("r"):
        gov.guard(lambda: {"password": "hunter2", "note": f"key {SECRET}"}, name="t")()
    result = _result_of(gov)
    assert "hunter2" not in result
    assert SECRET not in result
    assert "REDACTED" in result


def test_captured_error_text_is_scrubbed_when_a_tool_raises() -> None:
    gov = make_governor(capture=True)
    gov._tools = MagicMock()
    gov._tools.check_and_wait = MagicMock(return_value=ALLOW)

    def boom() -> None:
        raise RuntimeError(f"connect failed with {SECRET}")

    with gov.run("r"), pytest.raises(RuntimeError):
        gov.guard(boom, name="t")()
    span = tool_events(gov)[0]
    assert span["status"] == "error"
    assert SECRET not in attrs_of(span)[RESULT_KEY]


# ---------------------------------------------------------------------------
# LangChain: one span per tool call
# ---------------------------------------------------------------------------


def _lc_pair(gov: Any) -> tuple[Any, Any]:
    from matimo_agdk.adapters.langchain import MatimoCallbackHandler, govern_tools

    return MatimoCallbackHandler(gov), govern_tools([add_tool()], gov)[0]


def test_wrapper_and_handler_together_emit_exactly_one_tool_span() -> None:
    gov = make_governor(capture=True)
    handler, wrapped = _lc_pair(gov)
    assert wrapped.invoke({"a": 1, "b": 2}, config={"callbacks": [handler]}) == 3

    spans = tool_events(gov)
    assert len(spans) == 1
    span = spans[0]
    assert span["status"] == "completed"
    attrs = attrs_of(span)
    assert attrs["gen_ai.tool.call.arguments"] == {"a": 1, "b": 2}
    assert attrs[RESULT_KEY] == "3"
    # Correlated with the run the handler opened, under LangChain's own span id.
    runs = [e for e in events(gov) if e["kind"] == "run"]
    assert [r["status"] for r in runs] == ["running", "completed"]
    assert span["runId"] == runs[0]["runId"] == span["spanId"]


def test_the_single_span_joins_the_enclosing_chain_run() -> None:
    gov = make_governor()
    handler, wrapped = _lc_pair(gov)
    chain = RunnableLambda(lambda x: wrapped.invoke(x))
    chain.invoke({"a": 1, "b": 2}, config={"callbacks": [handler]})

    span = tool_events(gov)[0]
    assert len(tool_events(gov)) == 1
    assert span["runId"] != span["spanId"]  # rooted at the chain, not at the tool
    assert span["parentSpanId"] == span["runId"]


def test_a_denied_call_is_reported_once_as_denied() -> None:
    gov = make_governor(decision=ToolDecision(decision="DENY", reason="nope"))
    handler, wrapped = _lc_pair(gov)
    out = wrapped.invoke({"a": 1, "b": 2}, config={"callbacks": [handler]})
    assert out == "nope"  # the handled ToolException, as the model sees it

    spans = tool_events(gov)
    assert len(spans) == 1
    assert spans[0]["status"] == "denied"
    assert spans[0]["spanId"]  # tied to LangChain's own run id


def test_an_error_is_reported_once_as_error() -> None:
    from matimo_agdk.adapters.langchain import MatimoCallbackHandler, govern_tools

    @tool
    def broken(a: int) -> int:
        """Always fails."""
        raise RuntimeError("bad")

    gov = make_governor()
    with pytest.raises(RuntimeError):
        govern_tools([broken], gov)[0].invoke(
            {"a": 1}, config={"callbacks": [MatimoCallbackHandler(gov)]}
        )
    spans = tool_events(gov)
    assert len(spans) == 1
    assert spans[0]["status"] == "error"


def test_an_unwrapped_tool_still_gets_the_handlers_span() -> None:
    from matimo_agdk.adapters.langchain import MatimoCallbackHandler

    gov = make_governor()
    add_tool().invoke({"a": 1, "b": 2}, config={"callbacks": [MatimoCallbackHandler(gov)]})
    spans = tool_events(gov)
    assert len(spans) == 1
    assert spans[0]["name"] == "add"
    assert spans[0]["status"] == "completed"


def test_wrapped_and_unwrapped_tools_in_one_run_each_get_one_span() -> None:
    from matimo_agdk.adapters.langchain import MatimoCallbackHandler, govern_tools

    gov = make_governor()
    handler = MatimoCallbackHandler(gov)

    @tool
    def wrapped_one(a: int) -> int:
        """Wrapped."""
        return a

    @tool
    def plain_one(a: int) -> int:
        """Plain."""
        return a

    governed = govern_tools([wrapped_one], gov)[0]
    with gov_run(gov):
        governed.invoke({"a": 1}, config={"callbacks": [handler]})
        plain_one.invoke({"a": 2}, config={"callbacks": [handler]})
    spans = tool_events(gov)
    assert sorted(s["name"] for s in spans) == ["plain_one", "wrapped_one"]


@pytest.mark.asyncio
async def test_async_wrapper_and_handler_emit_one_span() -> None:
    from matimo_agdk.adapters.langchain import AsyncMatimoCallbackHandler, govern_tools

    gov = make_governor(asynchronous=True, capture=True)
    handler = AsyncMatimoCallbackHandler(gov)
    wrapped = govern_tools([add_tool(asynchronous=True)], gov)[0]
    assert await wrapped.ainvoke({"a": 4, "b": 5}, config={"callbacks": [handler]}) == 9

    spans = tool_events(gov)
    assert len(spans) == 1
    assert attrs_of(spans[0])[RESULT_KEY] == "9"
    assert spans[0]["runId"] == spans[0]["spanId"]


def test_no_tool_call_state_leaks_after_calls() -> None:
    from matimo_agdk.adapters import langchain as lc

    gov = make_governor()
    handler, wrapped = _lc_pair(gov)
    for _ in range(3):
        wrapped.invoke({"a": 1, "b": 2}, config={"callbacks": [handler]})
    assert lc._TOOL_CALLS == {}


def test_the_wrapper_without_a_handler_still_emits_its_span() -> None:
    from matimo_agdk.adapters.langchain import govern_tools

    gov = make_governor()
    govern_tools([add_tool()], gov)[0].invoke({"a": 1, "b": 2})
    assert len(tool_events(gov)) == 1


# ---------------------------------------------------------------------------
# LangChain: LLM provider
# ---------------------------------------------------------------------------


def _llm_span(handler: Any, gov: Any, serialized: dict[str, Any] | None) -> dict[str, Any]:
    run_id = uuid.uuid4()
    handler.on_llm_start(serialized, ["hi"], run_id=run_id)
    handler.on_llm_end(LLMResult(generations=[[]]), run_id=run_id)
    return [e for e in events(gov) if e["kind"] == "llm"][-1]


@pytest.mark.parametrize(
    ("class_name", "provider"),
    [
        ("ChatOpenAI", "openai"),
        ("OpenAI", "openai"),
        ("ChatAnthropic", "anthropic"),
        ("ChatGoogleGenerativeAI", "google"),
        ("ChatVertexAI", "google"),
    ],
)
def test_llm_span_provider_from_the_serialized_class(class_name: str, provider: str) -> None:
    from matimo_agdk.adapters.langchain import MatimoCallbackHandler

    gov = make_governor()
    span = _llm_span(
        MatimoCallbackHandler(gov), gov, {"id": ["langchain", "chat_models", "x", class_name]}
    )
    assert attrs_of(span)["gen_ai.provider.name"] == provider


@pytest.mark.parametrize(
    "serialized",
    [
        {"id": ["langchain", "chat_models", "x", "ChatSomethingElse"]},
        {"id": []},
        {},
        None,
    ],
)
def test_an_unknown_llm_class_gives_no_provider(serialized: dict[str, Any] | None) -> None:
    from matimo_agdk.adapters.langchain import MatimoCallbackHandler

    gov = make_governor()
    span = _llm_span(MatimoCallbackHandler(gov), gov, serialized)
    assert "gen_ai.provider.name" not in attrs_of(span)


def test_provider_is_kept_per_call_and_survives_an_llm_error() -> None:
    from matimo_agdk.adapters.langchain import MatimoCallbackHandler

    gov = make_governor()
    handler = MatimoCallbackHandler(gov)
    first, second = uuid.uuid4(), uuid.uuid4()
    handler.on_chat_model_start({"id": ["a", "ChatAnthropic"]}, [[]], run_id=first)
    handler.on_chat_model_start({"id": ["a", "ChatOpenAI"]}, [[]], run_id=second)
    handler.on_llm_error(RuntimeError("x"), run_id=first)
    handler.on_llm_end(LLMResult(generations=[[]]), run_id=second)

    llm = [e for e in events(gov) if e["kind"] == "llm"]
    assert [attrs_of(e)["gen_ai.provider.name"] for e in llm] == ["anthropic", "openai"]
    assert llm[0]["status"] == "error"


@pytest.mark.asyncio
async def test_async_handler_maps_the_provider_too() -> None:
    from matimo_agdk.adapters.langchain import AsyncMatimoCallbackHandler

    gov = make_governor(asynchronous=True)
    handler = AsyncMatimoCallbackHandler(gov)
    run_id = uuid.uuid4()
    await handler.on_llm_start({"id": ["a", "ChatAnthropic"]}, ["hi"], run_id=run_id)
    await handler.on_llm_end(LLMResult(generations=[[]]), run_id=run_id)
    llm = [e for e in events(gov) if e["kind"] == "llm"][0]
    assert attrs_of(llm)["gen_ai.provider.name"] == "anthropic"


# ---------------------------------------------------------------------------
# helpers for the framework-specific scenarios above
# ---------------------------------------------------------------------------


def _add(a: int, b: int) -> int:
    return a + b


class _AdkTool:
    name = "search"


class _AdkCtx:
    function_call_id = "call-1"
    invocation_id = "inv-1"


tool_calls: list[int] = []


def _crew_tool() -> Any:
    pytest.importorskip("crewai")
    from crewai.tools import BaseTool
    from pydantic import BaseModel

    class Args(BaseModel):
        x: int

    class MyTool(BaseTool):
        name: str = "mytool"
        description: str = "test"
        args_schema: type[BaseModel] = Args

        def _run(self, x: int) -> str:
            tool_calls.append(x)
            return f"ran {x}"

    tool_calls.clear()
    return MyTool()


def gov_run(gov: Any) -> Any:
    return gov.run("test-run")
