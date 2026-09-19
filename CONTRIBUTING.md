# Contributing

## Setup

```bash
uv sync --dev
```

## Common commands

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format .
uv run mypy matimo_agdk
```

## Conventions

- Python >= 3.13. No em-dash character anywhere in code, comments, or
  docs -- use a comma, period, or colon instead.
- No framework dependency in `matimo_agdk` core -- adapters live in
  `matimo_agdk/adapters/<framework>.py` behind an optional extra.
- Tests live under `tests/`, mirroring the module they cover
  (`tests/test_identity.py` for `matimo_agdk/identity.py`, etc.). Use
  `respx` to mock httpx calls -- never hit a real network in a unit test.
- Latency/overhead claims must be measured, never invented. Label anything
  unmeasured as unmeasured.

## Releasing

Maintainers: see [docs/RELEASING.md](docs/RELEASING.md). Pushing a `vX.Y.Z` tag publishes to PyPI.
