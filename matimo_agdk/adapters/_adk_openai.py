"""Converts between ADK's `LlmRequest`/`LlmResponse` and the real `openai`
SDK's chat-completions wire types, so `google_adk.gateway_model()` can talk to
Matimo Gateway directly -- no `litellm` anywhere in this path.

## Why this exists instead of `google.adk.models.lite_llm.LiteLlm`

`LiteLlm` (~3,500 lines) is Google's own bridge from ADK to `litellm`, which
in turn bridges to dozens of heterogeneous backends: Anthropic-native
thinking-block signatures, Gemma's `tool_responses` role, Ollama message
flattening, DeepSeek's text-parsed tool calls, Vertex/Gemini-via-litellm
quirks, and more. None of that applies here: Gateway always presents one
wire format to this adapter -- an OpenAI-compatible `/v1/chat/completions`
endpoint (every other adapter in this package makes the same assumption,
e.g. `crewai.py`'s `custom_openai=True`) -- regardless of which backend it
proxies to server-side. Routing through `litellm` to reach that one format
bought nothing but `litellm`'s own unconditional exception remapping
(`litellm_core_utils/exception_mapping_utils.py`'s `exception_type()` has no
"is this already one of my own recognized types?" escape hatch the way the
openai/anthropic SDKs' own `request()` methods do), which is what silently
turned a Gateway policy DENY into a bare `litellm.exceptions.APIError`
instead of this SDK's own `PolicyDenied`. Talking to the real `openai` SDK
directly instead means:

- A denied/blocked call raises this SDK's own `PolicyDenied`/`RateLimited`/
  etc. exactly like every other adapter -- no recovery shim needed, because
  `openai`'s own `_base_client.request()` re-raises an already-`OpenAIError`
  exception (`GatewayError`'s base, see `matimo_agdk.exceptions`) untouched.
- Full per-request `Matimo-Agent-Signature` signing, the same
  `governor.httpx_async_client()` gives AutoGen's `gateway_model_client()` --
  `LiteLlm` never had a hook for this at all (see that adapter's docstring).

## Scope relative to `LiteLlm`

Covered: multi-turn text, images (inline bytes or http(s) URLs), audio
(inline bytes), function/tool calling in both directions, streaming and
non-streaming, structured output (`response_schema` -> strict
`json_schema`), `reasoning_content`/`reasoning`-style thought passthrough
(the OpenAI-compatible convention several backends use, e.g. Azure/Foundry,
LM Studio, vLLM), and the same missing-tool-result / empty-turn healing
`LiteLlm` does.

Deliberately not covered, because there is no coherent OpenAI-wire-format
equivalent to reach through a single OpenAI-compatible endpoint (this is a
disclosed limitation, not an oversight -- see this package's own convention
of disclosing exactly this shape of gap, e.g. `langchain.py`'s
`ChatAnthropic` "session-header-only" note):

- Gemini-native output blobs (inline-generated images in the response) --
  OpenAI chat completions has no equivalent response shape.
- Anthropic's `thinking_blocks` with cross-turn signature preservation --
  that requires embedding signed blocks back into the *outbound* message on
  the next turn, an Anthropic-native requirement.
- `thinking_config` (Gemini's `include_thoughts`/`thinking_budget`) --
  OpenAI's nearest analog, `reasoning_effort`, is a different shape (a
  three-level enum, not a token budget) and the mapping would be a guess.
- Remote (non-http) file references and the OpenAI Files API upload flow --
  large documents go through inline `file_data` base64 only.
- ADK's `connect()` (Gemini Live / bidi streaming) -- Gateway's contract is
  request/response chat completions, not a live session protocol.
"""

from __future__ import annotations

import ast
import base64
import json
import re
from collections.abc import AsyncGenerator
from typing import Any

from google.adk.models.interactions_utils import extract_system_instruction
from google.genai import types

_FINISH_REASON_MAPPING: dict[str, types.FinishReason] = {
    "length": types.FinishReason.MAX_TOKENS,
    "stop": types.FinishReason.STOP,
    "tool_calls": types.FinishReason.STOP,
    "function_call": types.FinishReason.STOP,
    "content_filter": types.FinishReason.SAFETY,
}

_MISSING_TOOL_RESULT_MESSAGE = (
    "Error: Missing tool result (tool execution may have been interrupted "
    "before a response was recorded)."
)

# MIME major-type -> the openai content-part type it becomes. Deliberately no
# "video": OpenAI chat completions has no video_url content part -- that is a
# litellm cross-provider normalization for backends this adapter never talks
# to directly.
_MEDIA_CONTENT_KIND_BY_MAJOR_MIME_TYPE = {
    "image": "image_url",
    "audio": "input_audio",
}

# Document types OpenAI's chat completions API accepts inline as base64
# `file_data` (no upload/file_id round trip -- see this module's own
# docstring for why that flow is out of scope).
_SUPPORTED_INLINE_FILE_MIME_TYPES = frozenset({
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/json",
})

_UNQUOTED_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _map_finish_reason(finish_reason: str | None) -> types.FinishReason | None:
    if not finish_reason:
        return None
    return _FINISH_REASON_MAPPING.get(finish_reason.lower(), types.FinishReason.OTHER)


def _finish_reason_to_error_message(finish_reason: types.FinishReason) -> str:
    if finish_reason == types.FinishReason.MAX_TOKENS:
        return "Maximum tokens reached"
    return f"Finished with {finish_reason.name}"


def _quote_unquoted_json_object_keys(value: str) -> str:
    """Quotes simple unquoted object keys without touching string contents.

    Ported verbatim from `google.adk.models.lite_llm` (self-contained, no
    litellm dependency): some OpenAI-compatible backends stream a "complete"
    tool call whose finalized argument payload has unquoted keys (a Python
    dict repr, not JSON). This repairs only that shape.
    """
    result = []
    i = 0
    in_string = False
    string_quote = ""
    escaped = False

    while i < len(value):
        char = value[i]
        if in_string:
            result.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == string_quote:
                in_string = False
                string_quote = ""
            i += 1
            continue

        if char in {'"', "'"}:
            in_string = True
            string_quote = char
            result.append(char)
            i += 1
            continue

        if char in "{,":
            result.append(char)
            i += 1
            whitespace_start = i
            while i < len(value) and value[i].isspace():
                i += 1
            result.append(value[whitespace_start:i])

            key_match = _UNQUOTED_KEY_RE.match(value, i)
            if key_match:
                key_end = key_match.end()
                colon_index = key_end
                while colon_index < len(value) and value[colon_index].isspace():
                    colon_index += 1
                if colon_index < len(value) and value[colon_index] == ":":
                    result.append(f'"{key_match.group(0)}"')
                    result.append(value[key_end:colon_index])
                    i = colon_index
                    continue
            continue

        result.append(char)
        i += 1

    return "".join(result)


def _parse_tool_call_arguments(arguments: Any) -> Any:
    """Parses tool-call arguments leniently, same repair ladder as `LiteLlm`:
    strict JSON first, then a Python-literal fallback, then the unquoted-key
    repair above. Raises the original `json.JSONDecodeError` if nothing works."""
    if not arguments:
        return {}
    if not isinstance(arguments, str):
        return arguments

    try:
        return json.loads(arguments)
    except json.JSONDecodeError as exc:
        json_error = exc

    try:
        return ast.literal_eval(arguments)
    except (SyntaxError, ValueError):
        pass

    repaired = _quote_unquoted_json_object_keys(arguments)
    if repaired != arguments:
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(repaired)
            except (SyntaxError, ValueError):
                pass

    raise json_error


def _lowercase_schema_types(value: object) -> None:
    """Lowercases JSON Schema `type` strings in place (`types.Schema` dumps
    them as the uppercase enum name, e.g. `STRING`; JSON Schema consumers
    expect `string`). Local port of `google.adk.utils._schema_utils
    .lowercase_schema_types` to avoid depending on an underscore-prefixed
    (private) ADK module path."""
    if isinstance(value, list):
        for item in value:
            _lowercase_schema_types(item)
        return
    if not isinstance(value, dict):
        return
    schema_type = value.get("type")
    if isinstance(schema_type, str):
        value["type"] = schema_type.lower()
    elif isinstance(schema_type, list):
        value["type"] = [t.lower() if isinstance(t, str) else t for t in schema_type]
    for key in ("properties", "$defs"):
        for sub in (value.get(key) or {}).values():
            _lowercase_schema_types(sub)
    for key in ("anyOf", "oneOf", "allOf"):
        for item in value.get(key) or []:
            _lowercase_schema_types(item)
    if isinstance(value.get("items"), dict):
        _lowercase_schema_types(value["items"])


def _schema_to_dict(schema: types.Schema | dict[str, Any]) -> dict[str, Any]:
    """Recursively converts a `types.Schema` (or a plain dict) into a pure
    JSON Schema dict. Ported from `lite_llm.py`'s identical, already
    provider-agnostic helper."""
    if isinstance(schema, types.Schema):
        schema_dict = schema.model_dump(by_alias=True, exclude_none=True)
    else:
        schema_dict = dict(schema)

    enum_values = schema_dict.get("enum")
    if isinstance(enum_values, (list, tuple)):
        schema_dict["enum"] = [v for v in enum_values if v is not None]

    if schema_dict.get("type") is not None:
        t = schema_dict["type"]
        if isinstance(t, types.Type):
            schema_dict["type"] = str(t.value).lower()
        elif isinstance(t, str):
            schema_dict["type"] = t.lower()
        elif isinstance(t, (list, tuple)):
            schema_dict["type"] = [
                item.value.lower()
                if isinstance(item, types.Type)
                else (item.lower() if isinstance(item, str) else item)
                for item in t
            ]
        else:
            schema_dict["type"] = str(t).lower()

    if "items" in schema_dict and isinstance(schema_dict["items"], (types.Schema, dict)):
        schema_dict["items"] = _schema_to_dict(schema_dict["items"])

    any_of = schema_dict.pop("any_of", None) or schema_dict.get("anyOf")
    if any_of is not None:
        schema_dict["anyOf"] = [
            _schema_to_dict(item) if isinstance(item, (types.Schema, dict)) else item
            for item in any_of
        ]

    if "properties" in schema_dict:
        schema_dict["properties"] = {
            key: (_schema_to_dict(value) if isinstance(value, (types.Schema, dict)) else value)
            for key, value in schema_dict["properties"].items()
        }

    additional_properties = schema_dict.pop("additional_properties", None) or schema_dict.get(
        "additionalProperties"
    )
    if additional_properties is not None:
        schema_dict["additionalProperties"] = (
            _schema_to_dict(additional_properties)
            if isinstance(additional_properties, (types.Schema, dict))
            else additional_properties
        )

    return schema_dict


def _enforce_strict_openai_schema(schema: dict[str, Any]) -> None:
    """Mutates a JSON Schema dict in place to satisfy OpenAI strict structured
    outputs: `additionalProperties: false` on every object, every property
    required, no sibling keywords next to `$ref`. Ported verbatim from
    `lite_llm.py` (already provider-agnostic)."""
    if not isinstance(schema, dict):
        return

    if "$ref" in schema:
        for key in list(schema.keys()):
            if key != "$ref":
                del schema[key]
        return

    if schema.get("type") == "object" and "properties" in schema:
        schema["additionalProperties"] = False
        schema["required"] = sorted(schema["properties"].keys())

    for defn in schema.get("$defs", {}).values():
        _enforce_strict_openai_schema(defn)
    for prop in schema.get("properties", {}).values():
        _enforce_strict_openai_schema(prop)
    for key in ("anyOf", "oneOf", "allOf"):
        for item in schema.get(key, []):
            _enforce_strict_openai_schema(item)
    if isinstance(schema.get("items"), dict):
        _enforce_strict_openai_schema(schema["items"])


def _function_declaration_to_tool_param(
    function_declaration: types.FunctionDeclaration,
) -> dict[str, Any]:
    """Converts a `types.FunctionDeclaration` to an OpenAI `tools[]` entry."""
    assert function_declaration.name

    if function_declaration.parameters_json_schema:
        import copy

        parameters = copy.deepcopy(function_declaration.parameters_json_schema)
        _lowercase_schema_types(parameters)
    elif function_declaration.parameters:
        parameters = _schema_to_dict(function_declaration.parameters)
        parameters.setdefault("type", "object")
        parameters.setdefault("properties", {})
    else:
        parameters = {"type": "object", "properties": {}}

    tool_param: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": function_declaration.name,
            "description": function_declaration.description or "",
            "parameters": parameters,
        },
    }

    required_fields = (
        function_declaration.parameters.required
        if not function_declaration.parameters_json_schema and function_declaration.parameters
        else None
    )
    if required_fields and "required" not in tool_param["function"]["parameters"]:
        tool_param["function"]["parameters"]["required"] = required_fields

    return tool_param


def _response_format_from_schema(response_schema: types.SchemaUnion) -> dict[str, Any] | None:
    """Converts an ADK `response_schema` into OpenAI's strict `json_schema`
    `response_format`. Adapted from `lite_llm.py`'s `_to_litellm_response_format`
    with its Gemini-specific branch removed -- Gateway's wire format is always
    OpenAI's, never Gemini's native `response_schema` key."""
    import copy

    schema_name = "response"
    if isinstance(response_schema, dict):
        schema_dict = copy.deepcopy(response_schema)
        if "title" in schema_dict:
            schema_name = str(schema_dict["title"])
    elif isinstance(response_schema, type) and hasattr(response_schema, "model_json_schema"):
        schema_dict = response_schema.model_json_schema()
        schema_name = response_schema.__name__
    elif isinstance(response_schema, types.Schema):
        schema_dict = copy.deepcopy(
            response_schema.model_dump(by_alias=True, exclude_none=True, mode="json")
        )
        if "title" in schema_dict:
            schema_name = str(schema_dict["title"])
    elif hasattr(response_schema, "model_dump"):
        schema_dict = copy.deepcopy(response_schema.model_dump(exclude_none=True, mode="json"))
        schema_name = response_schema.__class__.__name__
    else:
        return None

    _lowercase_schema_types(schema_dict)
    _enforce_strict_openai_schema(schema_dict)
    return {
        "type": "json_schema",
        "json_schema": {"name": schema_name, "strict": True, "schema": schema_dict},
    }


def _part_has_payload(part: types.Part) -> bool:
    if part.text:
        return True
    if part.inline_data and part.inline_data.data:
        return True
    if part.file_data and part.file_data.file_uri:
        return True
    if part.function_response:
        return True
    return False


def _append_fallback_user_content_if_missing(llm_request: Any) -> None:
    """Ensures the last user turn carries real content, so the endpoint never
    sees an empty user message. Ported from `lite_llm.py`."""
    for content in reversed(llm_request.contents):
        if content.role == "user":
            parts = content.parts or []
            if any(_part_has_payload(p) for p in parts):
                return
            parts.append(
                types.Part.from_text(
                    text="Handle the requests as specified in the System Instruction."
                )
            )
            content.parts = parts
            return
    llm_request.contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text="Handle the requests as specified in the System Instruction."
                )
            ],
        )
    )


def _mime_content_kind(mime_type: str) -> str | None:
    major = mime_type.split(";", 1)[0].strip().lower().split("/", 1)[0]
    return _MEDIA_CONTENT_KIND_BY_MAJOR_MIME_TYPE.get(major)


def _audio_format_from_mime_type(mime_type: str) -> str:
    subtype = mime_type.split(";", 1)[0].strip().lower().split("/", 1)[1]
    if subtype.startswith("x-"):
        subtype = subtype[2:]
    if subtype == "mpeg":
        return "mp3"
    if subtype in ("wave", "vnd.wave"):
        return "wav"
    return subtype


def _decode_inline_text(raw_bytes: bytes) -> str:
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return raw_bytes.decode("latin-1", errors="replace")


def _parts_to_openai_content(parts: list[types.Part]) -> str | list[dict[str, Any]]:
    """Converts non-function-call/-response parts into OpenAI message
    `content` -- a bare string when there is exactly one text part (some
    OpenAI-compatible backends reject a single-element content array), a
    list of typed content parts otherwise."""
    if len(parts) == 1 and parts[0].text:
        return parts[0].text

    content: list[dict[str, Any]] = []
    for part in parts:
        if part.text:
            content.append({"type": "text", "text": part.text})
            continue

        if part.inline_data and part.inline_data.data and part.inline_data.mime_type:
            mime_type = part.inline_data.mime_type.split(";", 1)[0].strip().lower()
            if mime_type.startswith("text/"):
                content.append(
                    {"type": "text", "text": _decode_inline_text(part.inline_data.data)}
                )
                continue
            b64 = base64.b64encode(part.inline_data.data).decode("utf-8")
            kind = _mime_content_kind(mime_type)
            if kind == "image_url":
                content.append(
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}}
                )
            elif kind == "input_audio":
                content.append(
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": b64,
                            "format": _audio_format_from_mime_type(mime_type),
                        },
                    }
                )
            elif mime_type in _SUPPORTED_INLINE_FILE_MIME_TYPES:
                content.append(
                    {
                        "type": "file",
                        "file": {"file_data": f"data:{mime_type};base64,{b64}"},
                    }
                )
            else:
                raise ValueError(
                    f"gateway_model() does not support an inline content part with MIME "
                    f"type {part.inline_data.mime_type!r} -- see _adk_openai.py's own "
                    "module docstring for what is and isn't supported."
                )
            continue

        if part.file_data and part.file_data.file_uri:
            uri = part.file_data.file_uri
            if uri.startswith("http://") or uri.startswith("https://"):
                mime_type = (part.file_data.mime_type or "").split(";", 1)[0].strip().lower()
                if _mime_content_kind(mime_type) == "image_url" or not mime_type:
                    content.append({"type": "image_url", "image_url": {"url": uri}})
                    continue
            raise ValueError(
                f"gateway_model() cannot resolve file_uri {uri!r}: only inline data and "
                "http(s) image URLs are supported through Gateway's OpenAI-compatible "
                "endpoint (no OpenAI Files-API upload flow, no remote non-image fetch) "
                "-- see _adk_openai.py's own module docstring."
            )

    return content


def _content_to_openai_messages(content: types.Content) -> list[dict[str, Any]]:
    """Converts one ADK `types.Content` turn into one or more OpenAI chat
    messages (a tool-result turn can expand into several `tool` messages)."""
    parts = content.parts or []
    if not parts:
        return []

    tool_messages: list[dict[str, Any]] = []
    non_tool_parts: list[types.Part] = []
    for part in parts:
        if part.function_response:
            fr = part.function_response
            response_text = (
                fr.response if isinstance(fr.response, str) else json.dumps(fr.response)
            )
            tool_messages.append(
                {"role": "tool", "tool_call_id": fr.id or "", "content": response_text}
            )
        else:
            non_tool_parts.append(part)

    if tool_messages and not non_tool_parts:
        return tool_messages
    if tool_messages and non_tool_parts:
        return tool_messages + _content_to_openai_messages(
            types.Content(role=content.role, parts=non_tool_parts)
        )

    if content.role == "user":
        user_parts = [p for p in parts if not p.thought]
        return [{"role": "user", "content": _parts_to_openai_content(user_parts)}]

    # assistant/model turn
    tool_calls: list[dict[str, Any]] = []
    content_parts: list[types.Part] = []
    reasoning_texts: list[str] = []
    for part in parts:
        if part.function_call:
            fc = part.function_call
            if not fc.name:
                raise ValueError("gateway_model() function calls require a name")
            tool_calls.append(
                {
                    "type": "function",
                    "id": fc.id or "",
                    "function": {"name": fc.name, "arguments": json.dumps(fc.args or {})},
                }
            )
        elif part.thought:
            if part.text:
                reasoning_texts.append(part.text)
        else:
            content_parts.append(part)

    message: dict[str, Any] = {"role": "assistant"}
    if content_parts:
        final_content = _parts_to_openai_content(content_parts)
        if (
            isinstance(final_content, list)
            and len(final_content) == 1
            and "text" in final_content[0]
        ):
            message["content"] = final_content[0]["text"]
        else:
            message["content"] = final_content
    else:
        message["content"] = None
    if tool_calls:
        message["tool_calls"] = tool_calls
    if reasoning_texts:
        # The OpenAI-compatible convention several backends (Azure/Foundry, LM
        # Studio, vLLM) use for round-tripping prior reasoning; see this
        # module's own docstring for what is deliberately not covered
        # (Anthropic's signed thinking_blocks).
        message["reasoning_content"] = "\n".join(reasoning_texts)
    return [message]


def _ensure_tool_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Inserts placeholder `tool` messages for any assistant tool call that
    was never followed by a result -- OpenAI-compatible backends reject a
    history where a tool call has no matching response before the next
    non-tool message. Ported from `lite_llm.py`, minus its Gemma-specific
    `tool_responses` role (Gateway's wire format is always plain `tool`)."""
    if not messages:
        return messages

    def _placeholder(tool_call_id: str) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": _MISSING_TOOL_RESULT_MESSAGE,
        }

    healed: list[dict[str, Any]] = []
    pending_ids: list[str] = []

    for message in messages:
        role = message.get("role")
        if pending_ids and role != "tool":
            healed.extend(_placeholder(tool_call_id) for tool_call_id in pending_ids)
            pending_ids = []

        if role == "assistant":
            pending_ids = [tc["id"] for tc in message.get("tool_calls") or [] if tc.get("id")]
        elif role == "tool":
            tool_call_id = message.get("tool_call_id")
            if tool_call_id in pending_ids:
                pending_ids.remove(tool_call_id)

        healed.append(message)

    if pending_ids:
        healed.extend(_placeholder(tool_call_id) for tool_call_id in pending_ids)
    return healed


def build_completion_kwargs(llm_request: Any, model: str) -> dict[str, Any]:
    """Converts an `LlmRequest` into `openai.AsyncOpenAI().chat.completions
    .create(**kwargs)` keyword arguments. Only sets keys with a real value --
    the real SDK's typed signature uses an `Omit` sentinel default, not
    `None`, so an explicit `None` is not equivalent to leaving a parameter
    out."""
    messages: list[dict[str, Any]] = []
    for content in llm_request.contents or []:
        messages.extend(_content_to_openai_messages(content))

    system_instruction = extract_system_instruction(llm_request.config)
    if system_instruction:
        messages.insert(0, {"role": "system", "content": system_instruction})
    messages = _ensure_tool_results(messages)

    kwargs: dict[str, Any] = {"model": model, "messages": messages}

    tools: list[dict[str, Any]] = []
    if llm_request.config and llm_request.config.tools:
        for tool in llm_request.config.tools:
            if not isinstance(tool, types.Tool):
                continue
            if tool.function_declarations:
                tools.extend(
                    _function_declaration_to_tool_param(fd) for fd in tool.function_declarations
                )
    if tools:
        kwargs["tools"] = tools

    if llm_request.config and llm_request.config.response_schema:
        response_format = _response_format_from_schema(llm_request.config.response_schema)
        if response_format:
            kwargs["response_format"] = response_format

    if llm_request.config:
        config_dict = llm_request.config.model_dump(exclude_none=True)
        param_mapping = {"max_output_tokens": "max_completion_tokens", "stop_sequences": "stop"}
        extra_body: dict[str, Any] = {}
        for key in (
            "temperature",
            "max_output_tokens",
            "top_p",
            "seed",
            "stop_sequences",
            "presence_penalty",
            "frequency_penalty",
        ):
            if key in config_dict:
                kwargs[param_mapping.get(key, key)] = config_dict[key]
        # top_k has no OpenAI chat-completions equivalent; forward it via
        # extra_body so a Gateway-side backend that does understand it still
        # gets it, without the real SDK rejecting an unknown typed kwarg.
        if "top_k" in config_dict:
            extra_body["top_k"] = config_dict["top_k"]
        if extra_body:
            kwargs["extra_body"] = extra_body

    if tools and llm_request.config and llm_request.config.tool_config:
        function_calling_config = llm_request.config.tool_config.function_calling_config
        if function_calling_config:
            mode = function_calling_config.mode
            if mode == types.FunctionCallingConfigMode.ANY:
                kwargs["tool_choice"] = "required"
            elif mode == types.FunctionCallingConfigMode.NONE:
                kwargs["tool_choice"] = "none"
            # AUTO -> leave unset, provider default.

    return kwargs


def _usage_metadata(usage: Any) -> types.GenerateContentResponseUsageMetadata | None:
    """Builds ADK's usage-metadata shape from a real `openai.types.completion
    _usage.CompletionUsage` -- a single canonical shape, unlike `lite_llm.py`'s
    multi-provider `_extract_*` helpers, since the real SDK always returns one."""
    if usage is None:
        return None
    cached = 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        cached = getattr(details, "cached_tokens", None) or 0
    reasoning_tokens = 0
    completion_details = getattr(usage, "completion_tokens_details", None)
    if completion_details is not None:
        reasoning_tokens = getattr(completion_details, "reasoning_tokens", None) or 0
    return types.GenerateContentResponseUsageMetadata(
        prompt_token_count=usage.prompt_tokens,
        candidates_token_count=usage.completion_tokens,
        total_token_count=usage.total_tokens,
        cached_content_token_count=cached or None,
        thoughts_token_count=reasoning_tokens or None,
    )


def _reasoning_text(message: Any) -> str | None:
    """The OpenAI-compatible-convention reasoning text some backends attach
    to a message/delta beyond the strict schema (`reasoning_content`, then
    `reasoning`) -- see this module's own docstring for the Anthropic
    `thinking_blocks` shape this deliberately does not handle."""
    text = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
    return text if isinstance(text, str) and text else None


def _message_to_llm_response(
    *,
    content: str | None,
    tool_calls: list[tuple[str, str, str]],
    reasoning_text: str | None,
    model_version: str | None,
    is_partial: bool = False,
) -> Any:
    """`tool_calls` is `(id, name, raw_json_arguments)` tuples -- a plain
    shape both the non-streaming path (real SDK `ChatCompletionMessageToolCall`
    objects) and the streaming path (accumulated-by-index dicts) convert into,
    so this one function handles both without needing a fake SDK-shaped object."""
    from google.adk.models.llm_response import LlmResponse

    parts: list[types.Part] = []
    if reasoning_text:
        parts.append(types.Part(text=reasoning_text, thought=True))
    if content:
        parts.append(types.Part.from_text(text=content))
    for tool_call_id, name, raw_arguments in tool_calls:
        try:
            args = _parse_tool_call_arguments(raw_arguments)
        except json.JSONDecodeError:
            args = {}
        part = types.Part.from_function_call(name=name, args=args)
        assert part.function_call is not None
        part.function_call.id = tool_call_id
        parts.append(part)

    return LlmResponse(
        content=types.Content(role="model", parts=parts),
        partial=is_partial,
        model_version=model_version,
    )


def completion_to_llm_response(response: Any) -> Any:
    """Non-streaming path: converts an `openai.types.chat.ChatCompletion`
    into an `LlmResponse`."""
    from google.adk.models.llm_response import LlmResponse

    choices = response.choices or []
    if not choices:
        return LlmResponse(
            content=types.Content(role="model", parts=[]), model_version=response.model
        )

    choice = choices[0]
    message = choice.message
    tool_calls = [
        (tc.id, tc.function.name, tc.function.arguments) for tc in message.tool_calls or []
    ]
    llm_response = _message_to_llm_response(
        content=message.content,
        tool_calls=tool_calls,
        reasoning_text=_reasoning_text(message),
        model_version=response.model,
    )

    mapped_finish_reason = _map_finish_reason(choice.finish_reason)
    if mapped_finish_reason:
        llm_response.finish_reason = mapped_finish_reason
        if mapped_finish_reason != types.FinishReason.STOP:
            llm_response.error_code = mapped_finish_reason
            llm_response.error_message = _finish_reason_to_error_message(mapped_finish_reason)

    usage_metadata = _usage_metadata(response.usage)
    if usage_metadata:
        llm_response.usage_metadata = usage_metadata

    return llm_response


async def stream_llm_responses(stream: Any) -> AsyncGenerator[Any]:
    """Streaming path: consumes an `AsyncStream[ChatCompletionChunk]`, yields
    a partial `LlmResponse` per text/reasoning delta (matching
    `BaseLlm.generate_content_async()`'s documented streaming contract), then
    one final non-partial `LlmResponse` aggregating the whole turn.

    Function-call argument fragments are accumulated by the delta's own
    `index` -- the real OpenAI API always sends this correctly for a single
    well-behaved backend, unlike `lite_llm.py`'s defensive
    `_BraceDepthTracker` workaround for third-party backends with known-buggy
    indexing across the many providers litellm fans out to.
    """
    from google.adk.models.llm_response import LlmResponse

    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    function_calls: dict[int, dict[str, Any]] = {}
    model_version: str | None = None
    finish_reason: str | None = None
    usage_metadata: types.GenerateContentResponseUsageMetadata | None = None

    async for chunk in stream:
        model_version = chunk.model or model_version
        chunk_usage = _usage_metadata(getattr(chunk, "usage", None))
        if chunk_usage:
            usage_metadata = chunk_usage

        choices = chunk.choices or []
        if not choices:
            continue
        choice = choices[0]
        if choice.finish_reason:
            finish_reason = choice.finish_reason
        delta = choice.delta

        reasoning_text = _reasoning_text(delta)
        if reasoning_text:
            reasoning_parts.append(reasoning_text)
            thought_part = types.Part(text=reasoning_text, thought=True)
            yield LlmResponse(
                content=types.Content(role="model", parts=[thought_part]),
                partial=True,
                model_version=model_version,
            )

        if delta.content:
            text_parts.append(delta.content)
            text_part = types.Part.from_text(text=delta.content)
            yield LlmResponse(
                content=types.Content(role="model", parts=[text_part]),
                partial=True,
                model_version=model_version,
            )

        for tool_call_delta in delta.tool_calls or []:
            index = tool_call_delta.index
            entry = function_calls.setdefault(index, {"id": None, "name": "", "args_parts": []})
            if tool_call_delta.id:
                entry["id"] = tool_call_delta.id
            function = tool_call_delta.function
            if function is not None:
                if function.name:
                    entry["name"] += function.name
                if function.arguments:
                    entry["args_parts"].append(function.arguments)

    final_content = "".join(text_parts) or None
    final_reasoning = "\n".join(reasoning_parts) or None
    tool_calls: list[tuple[str, str, str]] = []
    for index in sorted(function_calls):
        entry = function_calls[index]
        if not entry["id"] and not entry["name"]:
            continue
        tool_calls.append(
            (entry["id"] or str(index), entry["name"], "".join(entry["args_parts"]))
        )

    llm_response = _message_to_llm_response(
        content=final_content,
        tool_calls=tool_calls,
        reasoning_text=final_reasoning,
        model_version=model_version,
    )
    mapped_finish_reason = _map_finish_reason(finish_reason)
    if mapped_finish_reason:
        llm_response.finish_reason = mapped_finish_reason
        if mapped_finish_reason != types.FinishReason.STOP:
            llm_response.error_code = mapped_finish_reason
            llm_response.error_message = _finish_reason_to_error_message(mapped_finish_reason)
    if usage_metadata:
        llm_response.usage_metadata = usage_metadata
    yield llm_response
