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
