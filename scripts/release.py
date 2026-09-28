"""Release helpers used by .github/workflows/release.yml (stdlib only).

    python scripts/release.py check [vX.Y.Z]   tag, pyproject.toml and __version__ must agree
    python scripts/release.py notes X.Y.Z      print that version's CHANGELOG.md section

`check` prints `key=value` lines (version, prerelease) so the workflow can append them to
$GITHUB_OUTPUT. Run it locally before tagging: `python scripts/release.py check`.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Final releases only: 1.2.3. Anything else that PEP 440 allows (0.2.0rc1, 1.0.0a2, ...) is a
# prerelease; tags must spell it exactly as pyproject.toml does (v0.2.0rc1, not v0.2.0-rc.1).
FINAL_VERSION = re.compile(r"^\d+(\.\d+)*$")


def _pyproject_version() -> str:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def _module_version() -> str:
    text = (ROOT / "matimo_agdk" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        raise SystemExit("could not find __version__ in matimo_agdk/__init__.py")
    return match.group(1)


def cmd_check(tag: str | None) -> int:
    version = _pyproject_version()
    module_version = _module_version()
    errors: list[str] = []

    if module_version != version:
        errors.append(
            f"pyproject.toml version ({version}) != matimo_agdk.__version__ ({module_version})"
        )
    if tag is not None:
        if not tag.startswith("v"):
            errors.append(f"tag {tag!r} must start with 'v' (e.g. v{version})")
        elif tag[1:] != version:
            errors.append(
                f"tag {tag!r} does not match pyproject.toml version {version} (want v{version})"
            )

    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"version={version}")
    print(f"prerelease={'false' if FINAL_VERSION.match(version) else 'true'}")
    return 0


def cmd_notes(version: str) -> int:
    lines = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8").splitlines()
    heading = re.compile(rf"^## \[{re.escape(version)}\]")
    start = next((i for i, line in enumerate(lines) if heading.match(line)), None)
    if start is None:
        print(f"error: no '## [{version}]' section in CHANGELOG.md", file=sys.stderr)
        return 1
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## [")), len(lines))
    print("\n".join(lines[start + 1 : end]).strip())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_check = sub.add_parser("check", help="verify tag, pyproject.toml and __version__ agree")
    p_check.add_argument(
        "tag", nargs="?", help="release tag, e.g. v0.2.0 (omit to check files only)"
    )
    p_notes = sub.add_parser("notes", help="print one version's CHANGELOG.md section")
    p_notes.add_argument("version", help="version without the leading v, e.g. 0.2.0")
    args = parser.parse_args(argv)
    if args.command == "check":
        return cmd_check(args.tag)
    return cmd_notes(args.version)


if __name__ == "__main__":
    raise SystemExit(main())
