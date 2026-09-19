"""Local, best-effort redaction applied before anything leaves the process.

Runs on top of, not instead of, Gateway's own server-side masking
(docs/SERVER-CONTRACT.md section 7.1). Key-based and recursive: a secret
nested three levels down inside a tool's argument dict is still masked.
Keys are matched on whole word components (split on separators and
camelCase), so `api_key`, `apiKey`, `privateKeyPem` and `Authorization`
are masked while `keyword`, `monkey` and `tokenizer` are not.
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
    if isinstance(value, str) and len(value) > max_string:
        return value[:max_string] + "...[TRUNCATED]"
    return value
