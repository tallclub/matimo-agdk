"""Recursive, word-based redaction (review finding 2026-09-18: top-level
keys only, and substring markers that matched `keyword`, `monkey`,
`tokenizer`)."""

from __future__ import annotations

from matimo_agdk._redact import REDACTED, is_sensitive_key, redact
from matimo_agdk.telemetry import redact_attributes, tool_span
from matimo_agdk.tools import redact_args


def test_nested_secret_inside_arguments_is_masked() -> None:
    out = redact_args(
        {"query": "x", "credentials": {"password": "hunter2"}, "nested": {"a": {"apiKey": "k"}}}
    )
    assert out["credentials"] == REDACTED
    assert out["nested"]["a"]["apiKey"] == REDACTED
    assert out["query"] == "x"


def test_lists_of_dicts_are_walked() -> None:
    out = redact({"items": [{"token": "t", "name": "n"}, "plain"]})
    assert out["items"][0]["token"] == REDACTED
    assert out["items"][0]["name"] == "n"
    assert out["items"][1] == "plain"


def test_word_boundaries_avoid_false_positives() -> None:
    assert not is_sensitive_key("keyword")
    assert not is_sensitive_key("monkey")
    assert not is_sensitive_key("tokenizer")
    assert is_sensitive_key("api_key")
    assert is_sensitive_key("apiKey")
    assert is_sensitive_key("privateKeyPem")
    assert is_sensitive_key("Authorization")
    assert is_sensitive_key("x-access-token")


def test_tool_span_arguments_are_redacted_before_send() -> None:
    event = tool_span("run-1", "wire", arguments={"amount": 5, "auth": {"password": "p"}})
    args = event["attributes"]["gen_ai.tool.call.arguments"]
    assert args["amount"] == 5
    assert args["auth"] == REDACTED


def test_redact_attributes_still_truncates_and_masks_top_level() -> None:
    out = redact_attributes({"note": "x" * 3000, "secret": "s"})
    assert out["note"].endswith("...[TRUNCATED]")
    assert out["secret"] == REDACTED


# ---------------------------------------------------------------------------
# Everything that leaves the process must be JSON (found by the contract tests)
# ---------------------------------------------------------------------------


def test_non_json_values_are_sent_as_text_not_left_to_break_serialization() -> None:
    import datetime as dt
    import json
    import uuid
    from pathlib import Path

    value = {
        "when": dt.datetime(2026, 9, 20, 10, 0, tzinfo=dt.UTC),
        "day": dt.date(2026, 9, 20),
        "path": Path("a/b"),
        "id": uuid.UUID(int=1),
        "tags": {"only"},
        "raw": b"x",
        "nan": float("nan"),
        "inf": float("inf"),
        "ok": [1, 2.5, True, None, "s"],
    }
    out = redact(value)
    json.dumps(out, allow_nan=False)  # would raise on a datetime, a Path, or NaN
    assert out["when"] == "2026-09-20T10:00:00+00:00"
    assert out["tags"] == ["only"]
    assert out["nan"] == "nan"
    assert out["ok"] == [1, 2.5, True, None, "s"]


def test_a_hostile_str_does_not_break_redaction() -> None:
    class Bad:
        def __str__(self) -> str:
            raise RuntimeError("no")

    assert redact({"x": Bad()}) == {"x": "<Bad>"}


def test_text_of_a_non_json_value_is_scrubbed_and_cut() -> None:
    class Holder:
        def __str__(self) -> str:
            return "key sk-" + "a" * 30 + " " + "z" * 5000

    out = redact(Holder(), max_string=100)
    assert "sk-" not in out
    assert out.endswith("...[TRUNCATED]")


def test_build_event_clamps_fields_to_the_servers_limits() -> None:
    from matimo_agdk.telemetry import build_event

    event = build_event(
        run_id="r" * 300,
        kind="tool",
        session_id="s" * 300,
        span_id="p" * 300,
        parent_span_id="q" * 300,
        name="n" * 300,
        status="completed-and-then-some-more",
        duration_ms=-5,
    )
    assert len(event["runId"]) == 120
    assert len(event["sessionId"]) == len(event["spanId"]) == len(event["parentSpanId"]) == 120
    assert len(event["name"]) == 255
    assert len(event["status"]) == 20
    assert event["durationMs"] == 0
    assert build_event(run_id="r", kind="tool", duration_ms=12.9)["durationMs"] == 12  # type: ignore[arg-type]


def test_telemetry_batch_size_is_capped_at_the_servers_limit() -> None:
    import pytest
    from pydantic import ValidationError

    from matimo_agdk.config import GatewayConfig

    assert GatewayConfig(telemetry_batch_size=500).telemetry_batch_size == 500
    with pytest.raises(ValidationError):
        GatewayConfig(telemetry_batch_size=501)
