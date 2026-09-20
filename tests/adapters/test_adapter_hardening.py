"""Regression tests for adapter defects found in the 2026-09-20 review.

LangChain and CrewAI are imported inside the tests that need them, so this
module also collects on an interpreter where an extra is not installed (CI
runs Python 3.14 without CrewAI).
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import pytest

from matimo_agdk.tools import ToolDecision

from .conftest import bound_async_governor


class _CountingGovernor:
    """Duck-typed sync Governor that only counts policy checks."""

    def __init__(self) -> None:
        self.checks = 0

    def check_tool(self, *_a: Any, **_k: Any) -> None: ...

    def check_and_wait(
        self, name: str, args: dict[str, Any], category_hint: str | None = None
    ) -> ToolDecision:
        self.checks += 1
        return ToolDecision("ALLOW")

    def raise_if_suspended(self) -> None: ...

    def tool_span(self, *_a: Any, **_k: Any) -> None: ...

    def llm_span(self, *_a: Any, **_k: Any) -> None: ...


# ---------------------------------------------------------------------------
# LangChain
# ---------------------------------------------------------------------------


def _run_only_tool() -> Any:
    from langchain_core.tools import BaseTool

    class RunOnly(BaseTool):
        name: str = "run_only"
        description: str = "echoes"

        def _run(self, q: str) -> str:
            return q

    return RunOnly()


async def test_langchain_async_call_of_a_run_only_tool_is_checked_once() -> None:
    """Reproduced before the fix: LangChain routes the default `_arun` back
    through the wrapped `_run`, so one call raised two policy checks (and would
    have asked a human to approve it twice)."""
    from matimo_agdk.adapters.langchain import govern_tools

    governor = _CountingGovernor()
    tool = govern_tools([_run_only_tool()], governor)[0]
    assert await tool.ainvoke({"q": "hi"}) == "hi"
    assert governor.checks == 1


async def test_langchain_async_call_of_a_structured_tool_without_a_coroutine_is_checked_once() -> (
    None
):
    from langchain_core.tools import tool as tool_decorator

    from matimo_agdk.adapters.langchain import govern_tools

    @tool_decorator
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    governor = _CountingGovernor()
    governed = govern_tools([add], governor)[0]
    assert await governed.ainvoke({"a": 1, "b": 2}) == 3
    assert governor.checks == 1


def test_langchain_govern_tools_twice_does_not_stack_checks() -> None:
    from matimo_agdk.adapters.langchain import govern_tools

    governor = _CountingGovernor()
    tool = _run_only_tool()
    govern_tools([tool], governor)
    govern_tools([tool], governor)
    assert tool.invoke({"q": "hi"}) == "hi"
    assert governor.checks == 1


def test_langchain_sync_call_is_still_checked_once() -> None:
    from matimo_agdk.adapters.langchain import govern_tools

    governor = _CountingGovernor()
    tool = govern_tools([_run_only_tool()], governor)[0]
    assert tool.invoke({"q": "hi"}) == "hi"
    assert governor.checks == 1


def test_langchain_failed_chains_do_not_leak_run_tree_entries() -> None:
    from matimo_agdk.adapters.langchain import MatimoCallbackHandler

    handler = MatimoCallbackHandler(_CountingGovernor(), mode="observe")
    for _ in range(50):
        run_id = uuid.uuid4()
        handler.on_chain_start({}, {}, run_id=run_id)
        handler.on_chain_error(RuntimeError("boom"), run_id=run_id)
    assert handler._impl.tree._root_of == {}
    assert handler._impl.tree._started_at == {}


async def test_langchain_async_failed_chains_do_not_leak_run_tree_entries() -> None:
    from matimo_agdk.adapters.langchain import AsyncMatimoCallbackHandler

    handler = AsyncMatimoCallbackHandler(_CountingGovernor(), mode="observe")
    run_id = uuid.uuid4()
    await handler.on_chain_start({}, {}, run_id=run_id)
    await handler.on_chain_error(RuntimeError("boom"), run_id=run_id)
    assert handler._impl.tree._root_of == {}


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_langchain_gateway_chat_model_rejects_an_async_governor_clearly(
    provider: str, identity: Any, credentials_dir: Any
) -> None:
    from matimo_agdk.adapters.langchain import gateway_chat_model

    governor = bound_async_governor(identity, credentials_dir)
    with pytest.raises(TypeError, match="sync Governor"):
        gateway_chat_model(governor, provider=provider)


# ---------------------------------------------------------------------------
# CrewAI
# ---------------------------------------------------------------------------


def test_crewai_default_arun_is_not_wrapped_so_no_check_runs_for_a_call_that_cannot_execute() -> (
    None
):
    """CrewAI's default `_arun` raises NotImplementedError. The old MRO test
    wrapped it anyway, so an async call ran a policy check (and could open an
    approval request) before failing."""
    pytest.importorskip("crewai")
    from crewai.tools import BaseTool as CrewBaseTool

    from matimo_agdk.adapters.crewai import govern_tool

    class SyncOnly(CrewBaseTool):
        name: str = "sync_only"
        description: str = "sync only"

        def _run(self, q: str) -> str:
            return q

    governor = _CountingGovernor()
    tool = govern_tool(SyncOnly(), governor)
    assert tool._run(q="hi") == "hi"
    assert governor.checks == 1
    with pytest.raises(NotImplementedError):
        asyncio.run(tool._arun(q="hi"))
    assert governor.checks == 1  # the doomed async call was not policy-checked


def test_crewai_gateway_llm_rejects_an_async_governor_clearly(
    identity: Any, credentials_dir: Any
) -> None:
    pytest.importorskip("crewai")
    from matimo_agdk.adapters.crewai import gateway_llm

    with pytest.raises(TypeError, match="sync Governor"):
        gateway_llm(bound_async_governor(identity, credentials_dir))


# ---------------------------------------------------------------------------
# AutoGen
# ---------------------------------------------------------------------------


async def test_autogen_govern_tools_twice_does_not_stack_checks() -> None:
    pytest.importorskip("autogen_core")
    from autogen_core import CancellationToken
    from autogen_core.tools import FunctionTool

    from matimo_agdk.adapters.autogen import govern_tools

    def add(a: int, b: int) -> int:
        """Add."""
        return a + b

    tool = FunctionTool(add, description="add")
    governor = _CountingGovernor()
    govern_tools([tool], governor)
    govern_tools([tool], governor)
    result = await tool.run_json({"a": 1, "b": 2}, CancellationToken())
    assert result == 3
    assert governor.checks == 1
