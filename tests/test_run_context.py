"""Run ids must be context-local (review finding 2026-09-18: a plain
instance attribute let concurrent runs see and reset each other's id)."""

from __future__ import annotations

import asyncio
import threading

from matimo_agdk.config import GatewayConfig
from matimo_agdk.governor import AsyncGovernor, Governor, current_run_id
from matimo_agdk.identity import IdentityCredentials

from .conftest import BASE_URL


def _config(identity: IdentityCredentials) -> GatewayConfig:
    return GatewayConfig(
        base_url=BASE_URL,
        api_key="org-key",
        identity_token=identity.identity_token,
        identity_id=identity.identity_id,
        tenant_id=identity.tenant_id,
        private_key_pem=identity.private_key_pem,
        agent_name=identity.display_name,
    )


async def test_concurrent_async_runs_keep_their_own_ids(identity: IdentityCredentials) -> None:
    governor = AsyncGovernor(_config(identity))
    seen: dict[str, list[str | None]] = {"a": [], "b": []}

    async def worker(label: str, hold: float) -> None:
        async with governor.run(label) as run_id:
            seen[label].append(run_id)
            await asyncio.sleep(hold)
            seen[label].append(current_run_id())
            await asyncio.sleep(hold)
            seen[label].append(current_run_id())

    await asyncio.gather(worker("a", 0.02), worker("b", 0.01))
    for label in ("a", "b"):
        own = seen[label][0]
        assert seen[label] == [own, own, own]
    assert seen["a"][0] != seen["b"][0]
    assert current_run_id() is None


def test_threads_do_not_share_the_current_run(identity: IdentityCredentials) -> None:
    governor = Governor(_config(identity))
    results: dict[str, str | None] = {}
    barrier = threading.Barrier(2)

    def worker(label: str) -> None:
        with governor.run(label) as run_id:
            barrier.wait()
            results[label] = current_run_id()
            assert results[label] == run_id

    threads = [threading.Thread(target=worker, args=(n,)) for n in ("t1", "t2")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results["t1"] != results["t2"]
    assert current_run_id() is None


def test_bind_run_id_returns_previous_and_is_restorable(identity: IdentityCredentials) -> None:
    governor = Governor(_config(identity))
    assert governor.bind_run_id("inv-1") is None
    assert current_run_id() == "inv-1"
    assert governor.bind_run_id(None) == "inv-1"
    assert current_run_id() is None
