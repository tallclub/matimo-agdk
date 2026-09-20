"""AutoGen adapter. Verified against autogen-core==0.7.5,
autogen-agentchat==0.7.5, autogen-ext==0.7.5 (the modern, 0.4+-generation
"AG2/AutoGen" rewrite -- `autogen_core`/`autogen_agentchat`/`autogen_ext`,
Microsoft's actively maintained line).

Install with `pip install matimo-agdk[autogen]`.

**Legacy `pyautogen` 0.2-style `ConversableAgent`/`register_function` is
NOT supported by this module, and this is a real finding, not a shortcut:**
as of this writing, the `pyautogen` PyPI package (version 0.10.0, the
latest available) ships a completely empty `__init__.py` -- verified
directly, not assumed -- `import pyautogen` succeeds but exposes nothing.
The legacy `ConversableAgent`/`register_function` API that name used to
provide now lives in the community `ag2` package, which this build does
not install or test. If you need to govern a `pyautogen`/`ag2`-style
agent's registered functions, wrap each one directly with
`matimo_agdk.adapters.generic.govern()` before passing it to
`register_function`/`register_for_execution` -- that works with any plain
callable regardless of framework and needs no AutoGen-specific code here.

Two pieces, for the modern generation:

- `govern_tools(tools, governor, mode=...)` -- wraps each
  `autogen_core.tools.BaseTool`'s `run` method in place (covers
  `FunctionTool` and any other `BaseTool` subclass; `run_json`, the method
  AutoGen's own agents actually call during tool dispatch, always calls
  `self.run(...)` internally -- verified by reading
  `autogen_core/tools/_base.py` -- so patching `run` governs both call
  paths with one wrapper).
- `gateway_model_client(governor, model=..., **kwargs)` -- an
  `OpenAIChatCompletionClient` pointed at Gateway. Needs an `AsyncGovernor`
  (autogen_core's model clients are async-only, and only `AsyncGovernor`
  exposes the `httpx.AsyncClient` a real per-request signature needs).

## What enforces what

`autogen_core.tools.BaseTool.run()` is a plain method call inside the
agent's own tool-dispatch code -- there is no callback-veto mechanism to
work around here (unlike LangChain). `mode="govern"` (default) raises
`ToolDenied` from inside the wrapped `run()` on a policy DENY;
AutoGen's own tool-call handling (in `autogen_agentchat`'s
`AssistantAgent`/`Handoff` tool-execution path) catches an exception from
a tool call and turns it into a `FunctionExecutionResult` with `is_error`
set, which is fed back to the model as a normal tool result rather than
crashing the run -- the same graceful-recovery shape as the other
adapters. `mode="observe"` only records tool spans, never calls
`check_tool()`.

Rapid suspend (`mode="govern"` only): `governor.raise_if_suspended()` is
called before each wrapped tool's real body runs.

## LLM spans and gen_ai.* attributes

`gateway_model_client()` wraps the returned `OpenAIChatCompletionClient`'s
own `create()`/`create_stream()` (per-instance monkeypatch, the same
technique `govern_tools()` uses on a `BaseTool.run`) rather than hooking
`httpx_async_client()` -- `CreateResult`/the streamed chunks already carry
`usage`/`finish_reason` as typed fields, so there is no need to re-parse a
raw response body the way the CrewAI adapter must. One LLM span is emitted
per `create()` call and per `create_stream()` generator (after it is fully
consumed or raises): `duration_ms`, `status` (`"completed"`/`"error"`), the
configured `model`, and `gen_ai.usage.*`/`finish_reasons` from the result
AutoGen itself already computed.

**Run-id correlation.** A `ChatCompletionClient` is shared across however
many agents/teams use it -- AutoGen gives this wrapper no natural
per-chat/per-run id the way ADK's invocation_id or LangChain's
run_id/parent_run_id tree does. Every span here (LLM and tool alike) uses
the ambient `governor.run()` id when one is active, and otherwise gives
each call a one-span run of its own, opened and closed around it -- the
same fallback `emit_llm_span()`/`emit_tool_span()` use everywhere in this
SDK. **Wrap your `agent.on_messages(...)`/`team.run(...)` call in `async
with governor.run("my-run"):`** to get one correlated run per chat in the
Gateway Observability Hub; without it, every LLM call and every tool call
shows up as its own separate, uncorrelated (but completed, not stuck
`running`) run.
"""

from __future__ import annotations

import functools
import time
from typing import Any

from ..exceptions import ToolDenied
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
    import autogen_core  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised only when the extra is missing
    raise ImportError(
        "matimo_agdk.adapters.autogen requires the 'autogen' extra: "
        "pip install matimo-agdk[autogen]"
    ) from exc


def _now() -> float:
    return time.monotonic()


def _args_to_dict(args: Any) -> dict[str, Any]:
    if hasattr(args, "model_dump"):
        return args.model_dump()
    if isinstance(args, dict):
        return dict(args)
    return {"args": args}


def _wrap_run(inner: Any, tool_name: str, governor: Any, mode: Mode, category: str | None) -> Any:
    @functools.wraps(inner)
    async def wrapper(args: Any, cancellation_token: Any = None, *a: Any, **kw: Any) -> Any:
        call_args = _args_to_dict(args)
        if mode == "govern":
            await async_raise_if_suspended(governor)
            decision = await async_check_and_wait(governor, tool_name, call_args, category=category)
            if decision.denied:
                raise ToolDenied(decision.reason)
        started = _now()
        status = "completed"
        try:
            return await inner(args, cancellation_token, *a, **kw)
        except Exception:
            status = "error"
            raise
        finally:
            emit_tool_span(
                governor,
                tool_name,
                status=status,
                duration_ms=int((_now() - started) * 1000),
                arguments=call_args,
            )

    return wrapper


def govern_tools(
    tools: list[Any],
    governor: Any,
    *,
    mode: Mode = "govern",
    category: str | None = None,
) -> list[Any]:
    """Wraps each `autogen_core.tools.BaseTool`'s `run` method in place.
    Mutates and returns the same list -- call this once, right after
    building your tool list, before handing it to an agent.

    Works with either a sync `Governor` (bridged via a thread so the
    async `run()` call never blocks the event loop) or an `AsyncGovernor`
    (awaited directly).
    """
    check_mode(mode)
    for t in tools:
        if getattr(t, "_matimo_governed", False):
            continue  # govern_tools() twice must not stack two checks on one call
        name = getattr(t, "name", None) or type(t).__name__
        base_run = t.run
        t.run = _wrap_run(base_run, name, governor, mode, category)  # type: ignore[method-assign]
        try:
            t._matimo_governed = True  # noqa: SLF001
        except Exception:  # noqa: BLE001 -- a stricter model config may reject this; harmless
            pass
    return tools


def _usage_attributes(result: Any) -> dict[str, Any]:
    usage = getattr(result, "usage", None)
    if usage is None:
        return {}
    attrs: dict[str, Any] = {}
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    if prompt is not None:
        attrs["gen_ai.usage.input_tokens"] = prompt
    if completion is not None:
        attrs["gen_ai.usage.output_tokens"] = completion
    return attrs


def _emit_model_call_span(
    governor: Any, model_name: str, status: str, started: float, result: Any
) -> None:
    finish_reason = getattr(result, "finish_reason", None) if result is not None else None
    emit_llm_span(
        governor,
        model=model_name,
        provider="openai",
        status=status,
        duration_ms=int((_now() - started) * 1000),
        finish_reasons=[str(finish_reason)] if finish_reason is not None else None,
        attributes=_usage_attributes(result) or None,
    )


def _wrap_model_create(inner: Any, governor: Any, model_name: str) -> Any:
    @functools.wraps(inner)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        started = _now()
        status = "completed"
        result: Any = None
        try:
            result = await inner(*args, **kwargs)
            return result
        except Exception:
            status = "error"
            raise
        finally:
            _emit_model_call_span(governor, model_name, status, started, result)

    return wrapper


def _wrap_model_create_stream(inner: Any, governor: Any, model_name: str) -> Any:
    from autogen_core.models import CreateResult

    @functools.wraps(inner)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        started = _now()
        status = "completed"
        final_result: Any = None
        try:
            async for item in inner(*args, **kwargs):
                if isinstance(item, CreateResult):
                    final_result = item
                yield item
        except Exception:
            status = "error"
            raise
        finally:
            _emit_model_call_span(governor, model_name, status, started, final_result)

    return wrapper


def gateway_model_client(
    governor: Any,
    *,
    model: str = "matimo/auto",
    model_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Returns an `autogen_ext.models.openai.OpenAIChatCompletionClient`
    pointed at Gateway, with full per-request `Matimo-Agent-Signature`
    signing.

    Needs an `AsyncGovernor`, not a sync `Governor`: autogen_core's model
    clients are async-only and need an `httpx.AsyncClient`, which only
    `AsyncGovernor.httpx_async_client()` provides -- verified that
    `OpenAIChatCompletionClient.__init__` accepts an arbitrary `**kwargs`
    dict filtered against `AsyncOpenAI.__init__`'s own keyword-only
    argument names (`openai_init_kwargs =
    set(inspect.getfullargspec(AsyncOpenAI.__init__).kwonlyargs)` in the
    installed `autogen_ext/models/openai/_openai_client.py`), which
    includes `http_client` -- so, unlike CrewAI's `LLM` and ADK's
    `LiteLlm`, this path gets the *same* full signing support as
    LangChain's `gateway_chat_model(provider="openai")`, not a
    session-header-only fallback.

    `model_info` is required by `OpenAIChatCompletionClient` for any model
    name it doesn't already recognize as a stock OpenAI model -- which
    includes Gateway's own `"matimo/auto"` routing sentinel and any
    non-OpenAI model pin. Defaults to a maximally-generic profile
    (`function_calling=True`, `vision=False`, `json_output=True`,
    `structured_output=False`, `family=ModelFamily.UNKNOWN`) since Gateway
    can route to any BYOK backend; pass your own `model_info` if you know
    the actual resolved model's real capabilities (e.g. vision support) and
    want AutoGen to rely on that.
    """
    if not is_async_governor(governor):
        raise TypeError(
            "gateway_model_client() needs an AsyncGovernor -- autogen_ext's "
            "OpenAIChatCompletionClient is async-only and needs an httpx.AsyncClient, "
            "which only AsyncGovernor.httpx_async_client() provides. "
            "Use AsyncGovernor.from_env() instead of Governor.from_env()."
        )
    from autogen_core.models import ModelFamily
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    resolved_model_info: Any = model_info or {
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "structured_output": False,
        "family": ModelFamily.UNKNOWN,
    }
    client = OpenAIChatCompletionClient(
        model=model,
        base_url=governor.config.base_url,
        api_key=governor.config.api_key or "matimo-gateway",
        # `http_client` is a real, working kwarg at runtime -- it is
        # filtered through `AsyncOpenAI.__init__`'s own kwonlyargs (see
        # this function's docstring) rather than declared on
        # OpenAIChatCompletionClient's own typed signature, which is why
        # mypy doesn't see it.
        http_client=governor.httpx_async_client(),  # type: ignore[call-arg]
        model_info=resolved_model_info,
        **kwargs,
    )
    # Per-instance monkeypatch (see this module's docstring, "LLM spans and
    # gen_ai.* attributes") -- OpenAIChatCompletionClient is a plain class,
    # not pydantic, so a direct instance attribute assignment shadows the
    # class method for this client without needing CrewAI's private-attr
    # workaround.
    client.create = _wrap_model_create(client.create, governor, model)  # type: ignore[method-assign]
    client.create_stream = _wrap_model_create_stream(  # type: ignore[method-assign]
        client.create_stream, governor, model
    )
    return client
