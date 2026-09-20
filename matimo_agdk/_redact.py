"""Local, best-effort redaction applied before anything leaves the process.

Runs on top of, not instead of, Gateway's own server-side masking
(docs/SERVER-CONTRACT.md section 7.1). Key-based and recursive: a secret
nested three levels down inside a tool's argument dict is still masked.
Keys are matched on whole word components (split on separators and
camelCase), so `api_key`, `apiKey`, `privateKeyPem` and `Authorization`
are masked while `keyword`, `monkey` and `tokenizer` are not.

String values are also scrubbed for a few unmistakable secret shapes (PEM
private keys, `Bearer` credentials, Matimo/OpenAI/GitHub/AWS key prefixes),
so a secret that reaches an exception message or a free-text field under an
innocuous key is still masked. This is a backstop, not a classifier: it will
not catch an arbitrary password in prose.
"""

from __future__ import annotations

import re
from typing import Any

REDACTED = "[REDACTED]"
DEFAULT_MAX_STRING = 2000
_MAX_DEPTH = 8

_SENSITIVE_TOKENS = frozenset(
    {
        "password",
        "passwd",
        "pwd",
        "secret",
        "secrets",
        "token",
        "tokens",
        "authorization",
        "auth",
        "apikey",
        "credential",
        "credentials",
        "privatekey",
        "accesskey",
        "secretkey",
        "bearer",
        "cookie",
        "key",
        "keys",
        "jwt",
        "signature",
    }
)
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_SPLIT = re.compile(r"[^a-z0-9]+")

# Ordered: the PEM patterns run first so a whole key body is masked in one piece.
_SECRET_VALUE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
        REDACTED,
    ),
    # A PEM cut off by truncation has no END line; mask everything after BEGIN.
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*", re.S), REDACTED),
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"), "Bearer " + REDACTED),
    (re.compile(r"\bme-(?:live|test|sess|id)-[A-Za-z0-9_-]{8,}"), REDACTED),
    (re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}"), REDACTED),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"), REDACTED),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), REDACTED),
)


def scrub_string(text: str) -> str:
    """Masks unmistakable secret shapes inside a free-form string."""
    for pattern, replacement in _SECRET_VALUE_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def is_sensitive_key(key: str) -> bool:
    tokens = [t for t in _SPLIT.split(_CAMEL.sub("_", key).lower()) if t]
    return any(t in _SENSITIVE_TOKENS for t in tokens)


def redact(value: Any, *, max_string: int = DEFAULT_MAX_STRING, _depth: int = 0) -> Any:
    """Returns a redacted copy of `value` (dicts, lists and tuples are
    walked; other values are returned as-is, long strings truncated)."""
    if _depth > _MAX_DEPTH:
        return "...[TRUNCATED]"
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            key = str(k)
            if is_sensitive_key(key):
                out[key] = REDACTED
            else:
                out[key] = redact(v, max_string=max_string, _depth=_depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        return [redact(v, max_string=max_string, _depth=_depth + 1) for v in value]
    if isinstance(value, str):
        value = scrub_string(value)
        if len(value) > max_string:
            return value[:max_string] + "...[TRUNCATED]"
    return value
