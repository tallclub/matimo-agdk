"""`matimo-agdk` command-line entry point.

register    -- one-time identity registration, writes the credentials file.
status      -- prints GovernanceState via one empty heartbeat poll.
rotate-key  -- rotates the signing key for an already-registered identity.
doctor      -- checks connectivity, handshake, and a signing round trip.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from .config import DEFAULT_BASE_URL, GatewayConfig
from .exceptions import GatewayError
from .governor import Governor
from .identity import credentials_paths


def _build_config(args: argparse.Namespace) -> GatewayConfig:
    overrides: dict[str, Any] = {}
    if getattr(args, "gateway_url", None):
        overrides["base_url"] = args.gateway_url
    if getattr(args, "api_key", None):
        overrides["api_key"] = args.api_key
    if getattr(args, "framework", None):
        overrides["framework"] = args.framework
    return GatewayConfig.load(agent_name=getattr(args, "name", None), **overrides)


def cmd_register(args: argparse.Namespace) -> int:
    config = _build_config(args)
    if not config.api_key:
        print("error: no org API key. Pass --api-key or set MATIMO_API_KEY.", file=sys.stderr)
        return 1
    config.agent_name = args.name
    governor = Governor(config)
    try:
        identity = governor.register(
            display_name=args.name,
            framework=args.framework,
            allowed_tool_categories=args.tool_category or None,
        )
    except GatewayError as exc:
        print(f"registration failed: {exc.message}", file=sys.stderr)
        return 1
    meta_path, key_path = credentials_paths(identity.display_name, config.credentials_dir)
    print(f"Registered identity {identity.identity_id} ({identity.display_name})")
    print(f"  identity_token: {identity.identity_token}")
    print(f"  framework:      {identity.external_framework}")
    print(f"  metadata:       {meta_path}")
    print(f"  private key:    {key_path} (never printed, never retrievable again)")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    config = _build_config(args)
    governor = Governor(config)
    if governor.identity is None:
        print("error: no identity found. Run `matimo register` first.", file=sys.stderr)
        return 1
    try:
        governor.start()
        assert governor._telemetry is not None  # noqa: SLF001 -- guaranteed by start()
        governor._telemetry.flush_now()  # noqa: SLF001 -- CLI-internal, forces one heartbeat now
        state = governor.state
    except GatewayError as exc:
        print(f"status check failed: {exc.message}", file=sys.stderr)
        return 1
    finally:
        governor.stop()
    print(
        f"identity:            {governor.identity.identity_id} ({governor.identity.display_name})"
    )
    print(f"lifecycle_status:    {state.lifecycle_status}")
    print(f"emergency_stop:      {state.emergency_stop}")
    print(f"telemetry_mode:      {state.telemetry_mode}")
    print(f"staleness_minutes:   {state.telemetry_staleness_minutes}")
    print(f"server_time:         {state.server_time}")
    print(f"suspended (locally): {state.is_suspended}")
    return 0


def cmd_rotate_key(args: argparse.Namespace) -> int:
    config = _build_config(args)
    governor = Governor(config)
    if governor.identity is None:
        print("error: no identity found. Run `matimo register` first.", file=sys.stderr)
        return 1
    try:
        identity = governor.rotate_key()
    except GatewayError as exc:
        print(f"key rotation failed: {exc.message}", file=sys.stderr)
        return 1
    meta_path, key_path = credentials_paths(identity.display_name, config.credentials_dir)
    print(f"Rotated key for identity {identity.identity_id}")
    print(f"  private key: {key_path} (overwritten, never printed)")
    print("  Re-handshake (any subsequent Governor.start()) will use the new key automatically.")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    config = _build_config(args)
    print(f"Gateway URL: {config.base_url}")
    if not config.api_key:
        print("  [FAIL] no org API key configured")
        return 1
    print("  [ok] org API key present")

    governor = Governor(config)
    if governor.identity is None:
        print("  [FAIL] no identity found -- run `matimo register` first")
        return 1
    print(f"  [ok] identity loaded: {governor.identity.identity_id}")

    try:
        assert governor._session is not None  # noqa: SLF001 -- guaranteed: governor.identity is set
        token = governor._session.get_token()  # noqa: SLF001 -- exercises the signed handshake
        print(f"  [ok] session handshake succeeded (token prefix: {token[:12]}...)")
    except GatewayError as exc:
        print(f"  [FAIL] session handshake failed: {exc.message}")
        return 1

    try:
        governor.start()
        assert governor._telemetry is not None  # noqa: SLF001 -- guaranteed by start()
        governor._telemetry.flush_now()  # noqa: SLF001
        state = governor.state
        print(f"  [ok] telemetry heartbeat succeeded: lifecycle_status={state.lifecycle_status}")
    except GatewayError as exc:
        print(f"  [FAIL] telemetry heartbeat failed: {exc.message}")
        return 1
    finally:
        governor.stop()

    print("doctor: all checks passed")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="matimo-agdk", description="Matimo Agent Governance Development Kit CLI"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--gateway-url", default=None, help=f"Gateway base URL (default {DEFAULT_BASE_URL})"
    )
    common.add_argument("--api-key", default=None, help="Org API key (or set MATIMO_API_KEY)")

    p_register = sub.add_parser("register", parents=[common], help="Register a new agent identity")
    p_register.add_argument(
        "--name", required=True, help="Agent display name (also the local credentials file key)"
    )
    p_register.add_argument(
        "--framework",
        default="custom",
        choices=["langchain", "google-adk", "crewai", "autogen", "custom"],
        help="Framework this agent runs on",
    )
    p_register.add_argument(
        "--tool-category", action="append", default=[], help="Allowed tool category (repeatable)"
    )
    p_register.set_defaults(func=cmd_register)

    p_status = sub.add_parser("status", parents=[common], help="Print current GovernanceState")
    p_status.add_argument(
        "--name", default=None, help="Agent name (defaults to MATIMO_AGENT_NAME or 'matimo-agent')"
    )
    p_status.set_defaults(func=cmd_status)

    p_rotate = sub.add_parser(
        "rotate-key", parents=[common], help="Rotate this identity's signing key"
    )
    p_rotate.add_argument("--name", default=None)
    p_rotate.set_defaults(func=cmd_rotate_key)

    p_doctor = sub.add_parser(
        "doctor", parents=[common], help="Check connectivity, handshake, and signing"
    )
    p_doctor.add_argument("--name", default=None)
    p_doctor.set_defaults(func=cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
