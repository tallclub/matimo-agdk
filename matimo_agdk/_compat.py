"""Detecting which HTTP library an installed LLM SDK is built on."""

from __future__ import annotations

import re
from importlib import metadata

# "httpx2" as a whole distribution name: not "httpx2-extras", not "httpx".
_HTTPX2_REQUIREMENT = re.compile(r"httpx2(?![\w.-])", re.IGNORECASE)


def sdk_requires_httpx2(distribution: str) -> bool:
    """True if `distribution` (e.g. "anthropic") hard-requires `httpx2`.

    Reads the installed package's declared requirements, which is public,
    stable metadata, rather than poking at the SDK's private modules. A
    requirement that only applies to an optional extra (`extra == "httpx2"`)
    does not count: such an SDK still works with a plain `httpx.Client`.
    """
    try:
        requirements = metadata.requires(distribution) or []
    except metadata.PackageNotFoundError:
        return False
    for requirement in requirements:
        spec, _, marker = requirement.partition(";")
        if "extra" in marker:
            continue
        if _HTTPX2_REQUIREMENT.match(spec.strip()):
            return True
    return False
