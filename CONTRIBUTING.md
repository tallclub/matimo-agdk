# Contributing

## Setup

```bash
uv sync --all-extras --group dev
```

`--all-extras` matters: the adapter tests import LangChain, Google ADK, CrewAI and
AutoGen. CrewAI does not import on Python 3.14 yet, so on 3.14 use
`uv sync --extra langchain --extra google-adk --extra autogen --group dev` and run
`pytest --ignore=tests/adapters/test_crewai.py`.

## Before you push

CI runs exactly these, so run them first:

```bash
uv run ruff check .
uv run ruff format --check .      # `uv run ruff format .` to fix
uv run mypy matimo_agdk
uv run pytest -q
```

## Conventions

- Python >= 3.13. No em-dash character anywhere in code, comments, or
  docs -- use a comma, period, or colon instead.
- Commit messages follow Conventional Commits, checked by commitlint on every
  pull request (`commitlint.config.cjs`): `type(scope): subject`, with `type`
  one of `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`,
  `ci`, `revert`, `example`.
- No framework dependency in `matimo_agdk` core -- adapters live in
  `matimo_agdk/adapters/<framework>.py` behind an optional extra.
- Tests live under `tests/`, mirroring the module they cover
  (`tests/test_identity.py` for `matimo_agdk/identity.py`, etc.). Never hit a
  real network in a unit test: mock HTTP with `respx` or an in-process fake.
- A bug fix comes with a regression test that fails without the fix.
- Governance code fails closed: when a check cannot be understood (an unknown
  decision, a malformed response), the guarded tool must not run.
- Latency/overhead claims must be measured, never invented. Label anything
  unmeasured as unmeasured.

## Releasing

Maintainers: see [docs/RELEASING.md](docs/RELEASING.md). Pushing a `vX.Y.Z` tag publishes to PyPI.
