"""Cross-adapter invariant: no adapter may leave a run `running` in Gateway.

Gateway only ends a run on an explicit terminal `kind:"run"` span
(docs/SERVER-CONTRACT.md 7.3), so every run id that appears in emitted
telemetry must be opened once and closed once, with the close last. Each
scenario below drives one adapter's path *outside* any `governor.run()` block --
the case where an adapter has to own the run itself -- against a real
`Governor` and a mock exporter, then all of them are held to the same check.

Adding an adapter or a new span-emitting path? Add a scenario here.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from autogen_core import CancellationToken
from autogen_core.tools import FunctionTool
from crewai.tools import BaseTool as CrewBaseTool
from langchain_core.tools import ToolException, tool
from pydantic import BaseModel

from matimo_agdk.config import GatewayConfig
from matimo_agdk.exceptions import ToolDenied
from matimo_agdk.governor import Governor
from matimo_agdk.tools import ToolDecision


def _governor(decision: str = "ALLOW") -> Governor:
    """A real Governor (real span builders, real `run_span()`) with a mock
    exporter and a stubbed policy check -- no network."""
    gov = Governor.__new__(Governor)
    gov.config = GatewayConfig()
    gov._telemetry = MagicMock()
    gov.check_and_wait = MagicMock(  # type: ignore[method-assign]
        return_value=ToolDecision(decision=decision, reason="nope" if decision == "DENY" else None)
    )
    gov.raise_if_suspended = MagicMock()  # type: ignore[method-assign]
    gov.request_headers = MagicMock(return_value={})  # type: ignore[method-assign]
    return gov


def _events(gov: Governor) -> list[dict[str, Any]]:
    return [c.args[0] for c in gov._telemetry.submit.call_args_list]  # type: ignore[union-attr]


def assert_every_run_is_closed(events: list[dict[str, Any]]) -> None:
    assert events, "the scenario emitted nothing"
    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in events:
        by_run[e["runId"]].append(e)
    for run_id, spans in by_run.items():
        runs = [s for s in spans if s["kind"] == "run"]
        opens = [s for s in runs if s["status"] == "running"]
        closes = [s for s in runs if s["status"] != "running"]
        assert len(opens) == 1, f"run {run_id} opened {len(opens)} times: {spans}"
        assert len(closes) == 1, f"run {run_id} closed {len(closes)} times: {spans}"
        assert spans[0] is opens[0], f"run {run_id}: a span arrived before the run opened"
        assert spans[-1] is closes[0], f"run {run_id}: a span arrived after the run closed"


# ---------------------------------------------------------------------------
# Scenarios: one per adapter path that emits spans
# ---------------------------------------------------------------------------

Scenario = Callable[[], Awaitable[Governor]]
SCENARIOS: dict[str, Scenario] = {}


def scenario(fn: Scenario) -> Scenario:
    SCENARIOS[fn.__name__.removeprefix("_")] = fn
    return fn


@scenario
async def _generic_observe_sync() -> Governor:
    from matimo_agdk.adapters.generic import govern

    gov = _governor()
    assert govern(lambda x: x + 1, gov, mode="observe")(1) == 2
    return gov


@scenario
async def _generic_observe_async() -> Governor:
    from matimo_agdk.adapters.generic import govern

    async def add(x: int) -> int:
        return x + 1

    gov = _governor()
    assert await govern(add, gov, mode="observe")(1) == 2
    return gov


@scenario
async def _generic_govern_async_on_sync_governor() -> Governor:
    from matimo_agdk.adapters.generic import govern

    async def add(x: int) -> int:
        return x + 1

    gov = _governor()
    assert await govern(add, gov)(1) == 2
    return gov


@scenario
async def _generic_govern_async_denied() -> Governor:
    from matimo_agdk.adapters.generic import govern

    async def add(x: int) -> int:
        return x + 1

    gov = _governor("DENY")
    with pytest.raises(ToolDenied):
        await govern(add, gov)(1)
    return gov


@scenario
async def _langchain_govern_tool() -> Governor:
    from matimo_agdk.adapters.langchain import govern_tools

    @tool
    def add(a: int, b: int) -> int:
        """Add."""
        return a + b

    gov = _governor()
    assert govern_tools([add], gov)[0].invoke({"a": 1, "b": 2}) == 3
    return gov


@scenario
async def _langchain_govern_tool_denied() -> Governor:
    from matimo_agdk.adapters.langchain import govern_tools

    @tool
    def add(a: int, b: int) -> int:
        """Add."""
        return a + b

    gov = _governor("DENY")
    with pytest.raises(ToolException):
        govern_tools([add], gov)[0].invoke({"a": 1, "b": 2})
    return gov


@scenario
async def _langchain_callback_llm_and_tool_under_one_handler() -> Governor:
    """Two parentless calls on one handler: the case the live demo hit."""
    import uuid

    from matimo_agdk.adapters.langchain import MatimoCallbackHandler

    class Result:
        llm_output = {"model_name": "m", "token_usage": {}}
        generations = [[]]

    gov = _governor()
    handler = MatimoCallbackHandler(gov, mode="observe")
    llm_id, tool_id = uuid.uuid4(), uuid.uuid4()
    handler.on_chat_model_start({}, [[]], run_id=llm_id, parent_run_id=None)
    handler.on_llm_end(Result(), run_id=llm_id, parent_run_id=None)
    handler.on_tool_start({"name": "t"}, "x", run_id=tool_id, parent_run_id=None)
    handler.on_tool_end("y", run_id=tool_id, parent_run_id=None, name="t")
    return gov


class _Args(BaseModel):
    x: int


def _crew_tool() -> CrewBaseTool:
    class MyTool(CrewBaseTool):
        name: str = "mytool"
        description: str = "test"
        args_schema: type[BaseModel] = _Args

        def _run(self, x: int) -> str:
            return f"ran {x}"

    return MyTool()


@scenario
async def _crewai_govern_tool() -> Governor:
    from matimo_agdk.adapters.crewai import govern_tool

    gov = _governor()
    assert govern_tool(_crew_tool(), gov).run(x=5) == "ran 5"
    return gov


@scenario
async def _crewai_govern_tool_denied() -> Governor:
    from matimo_agdk.adapters.crewai import govern_tool

    gov = _governor("DENY")
    with pytest.raises(ToolDenied):
        govern_tool(_crew_tool(), gov).run(x=5)
    return gov


@scenario
async def _crewai_llm_interceptor() -> Governor:
    from matimo_agdk.adapters.crewai import make_interceptor

    gov = _governor()
    interceptor = make_interceptor(gov)
    request = interceptor.on_outbound(
        httpx.Request("POST", "http://gw/v1/chat/completions", json={"model": "gpt-4o-mini"})
    )
    assert request is not None
    interceptor.on_inbound(httpx.Response(200, json={"model": "gpt-4o-mini", "usage": {}}))
    return gov


def _autogen_tool() -> FunctionTool:
    def add(a: int, b: int) -> int:
        return a + b

    return FunctionTool(add, description="adds")


@scenario
async def _autogen_govern_tool() -> Governor:
    from matimo_agdk.adapters.autogen import govern_tools

    gov = _governor()
    tools = govern_tools([_autogen_tool()], gov)
    assert await tools[0].run_json({"a": 1, "b": 2}, CancellationToken()) == 3
    return gov


@scenario
async def _autogen_govern_tool_denied() -> Governor:
    from matimo_agdk.adapters.autogen import govern_tools

    gov = _governor("DENY")
    tools = govern_tools([_autogen_tool()], gov)
    with pytest.raises(ToolDenied):
        await tools[0].run_json({"a": 1, "b": 2}, CancellationToken())
    return gov


@scenario
async def _autogen_model_create() -> Governor:
    from matimo_agdk.adapters.autogen import _wrap_model_create

    async def create(*args: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(finish_reason="stop", usage=None)

    gov = _governor()
    await _wrap_model_create(create, gov, "gpt-4o-mini")()
    return gov


@scenario
async def _google_adk_denied_tool_stays_in_the_invocation_run() -> Governor:
    from matimo_agdk.adapters.google_adk import MatimoPlugin

    gov = _governor("DENY")
    plugin = MatimoPlugin(gov)
    invocation = SimpleNamespace(invocation_id="inv-1", agent=SimpleNamespace(name="a"))
    tool_ctx = SimpleNamespace(function_call_id="call-1", invocation_id="inv-1")

    await plugin.before_run_callback(invocation_context=invocation)
    denied = await plugin.before_tool_callback(
        tool=SimpleNamespace(name="t"), tool_args={"q": "x"}, tool_context=tool_ctx
    )
    assert denied == {"error": "nope"}
    await plugin.after_run_callback(invocation_context=invocation)
    return gov


@pytest.mark.asyncio
@pytest.mark.parametrize("name", sorted(SCENARIOS))
async def test_no_adapter_leaves_a_run_running(name: str) -> None:
    gov = await SCENARIOS[name]()
    assert_every_run_is_closed(_events(gov))


@pytest.mark.asyncio
async def test_google_adk_denied_tool_span_uses_the_invocation_run_id() -> None:
    gov = await SCENARIOS["google_adk_denied_tool_stays_in_the_invocation_run"]()
    assert {e["runId"] for e in _events(gov)} == {"inv-1"}


@pytest.mark.asyncio
async def test_a_failed_tool_closes_its_run_failed() -> None:
    """The terminal status must reflect the outcome, not just exist."""
    gov = await SCENARIOS["langchain_govern_tool_denied"]()
    closes = [e for e in _events(gov) if e["kind"] == "run" and e["status"] != "running"]
    assert [c["status"] for c in closes] == ["failed"]


@pytest.mark.asyncio
async def test_govern_twice_is_idempotent_for_every_adapter() -> None:
    """LangChain, CrewAI and AutoGen already skip an already-governed tool;
    the generic adapter must too, or one call is policy-checked twice."""
    from matimo_agdk.adapters.generic import govern

    gov = _governor()

    def add(x: int) -> int:
        return x + 1

    once = govern(add, gov, mode="observe")
    assert govern(once, gov, mode="observe") is once
    once(1)
    assert len([e for e in _events(gov) if e["kind"] == "tool"]) == 1
