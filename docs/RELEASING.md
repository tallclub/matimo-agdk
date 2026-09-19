# Releasing matimo-agdk

Pushing a version tag publishes the package to PyPI and creates the GitHub Release. The
workflow is [`.github/workflows/release.yml`](../.github/workflows/release.yml).

| Tag | Result |
|---|---|
| `v0.2.0` | Final release: PyPI + GitHub Release |
| `v0.2.0rc1`, `v0.2.0a1`, `v0.2.0b1` | Prerelease: PyPI (pip skips it unless `--pre`) + GitHub Release marked "pre-release" |

The tag must be exactly `v` plus the version in `pyproject.toml`, spelled the way PEP 440
normalises it (`v0.2.0rc1`, not `v0.2.0-rc.1`). The workflow refuses to publish otherwise.

## What the workflow does

1. **validate**: the tag commit is on `main`; the tag, `pyproject.toml` `version` and
   `matimo_agdk.__version__` all agree; `CHANGELOG.md` has a non-empty `## [X.Y.Z]` section.
2. **ci**: the same jobs that run on every PR (`ruff`, `ruff format --check`, `mypy`, `pytest`
   on Python 3.13 and 3.14), plus a build of the sdist and wheel, `twine check --strict`, and a
   smoke test that installs the wheel into a clean venv and runs `import matimo_agdk` and
   `matimo-agdk --help`.
3. **publish-pypi**: uploads exactly the files the `ci` job built and tested, using PyPI trusted
   publishing (OIDC). No PyPI token is stored in GitHub.
4. **github-release**: after PyPI accepts the upload, creates the GitHub Release with the
   `CHANGELOG.md` section as its notes and the wheel and sdist attached.

Any failure stops the pipeline before the next step, so nothing is published from a red build.

## One-time setup

Do these once, before the first tag.

1. **PyPI pending publisher.** On pypi.org: Account settings, Publishing, "Add a new pending
   publisher", with:
   - PyPI project name: `matimo-agdk`
   - Owner: `tallclub`
   - Repository: `matimo-agdk`
   - Workflow name: `release.yml`
   - Environment name: `pypi`

   The first successful publish creates the project and converts the pending publisher into a
   normal one. Confirm the name `matimo-agdk` is still free on PyPI before relying on it.
2. **GitHub environment.** Repository Settings, Environments, New environment named `pypi`.
   Recommended: add yourself as a required reviewer, so every publish waits for a manual approval
   click after CI is green.
3. **Tag protection (recommended).** Repository Settings, Rules, Rulesets: restrict creation of
   tags matching `v*` to maintainers. Anyone who can push a tag can otherwise start a release.

## Cutting a release

1. Bump the version in **both** `pyproject.toml` and `matimo_agdk/__init__.py`.
2. Add a `## [X.Y.Z] - YYYY-MM-DD` section to `CHANGELOG.md`. It becomes the GitHub Release notes.
3. Check locally:

   ```bash
   python scripts/release.py check          # version files agree
   python scripts/release.py notes X.Y.Z    # preview the release notes
   uv run ruff check . && uv run ruff format --check . && uv run mypy matimo_agdk && uv run pytest -q
   ```

4. Merge to `main` through a PR.
5. Tag the merged commit and push the tag:

   ```bash
   git checkout main && git pull
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

6. Watch the run under Actions, approve the `pypi` environment if you added reviewers, then check
   `pip install matimo-agdk==X.Y.Z` in a fresh environment.

To rehearse without publishing: Actions, release, "Run workflow". A manual run validates the
version files, runs the full CI gate and builds the package, and never publishes.

## When something goes wrong

- **Failed before `publish-pypi`** (validation, tests, build): nothing was published. Fix on `main`,
  then move the tag: `git tag -d vX.Y.Z && git push origin :refs/tags/vX.Y.Z`, re-tag the new
  commit and push.
- **`github-release` failed after PyPI succeeded**: re-run just that job from the Actions UI.
  Do not re-tag; the version is already on PyPI.
- **Published a bad build**: PyPI never lets a version be re-uploaded. Yank it on pypi.org
  (Manage, Releases) and ship the fix as the next version.

## After the first release

The 0.1.0 docs say the package is not on PyPI yet. Once `0.1.0` (or whichever version goes out
first) is live, update: the "Not yet published to PyPI" line in `CHANGELOG.md`, the install
instructions in `README.md`, and the install section of `docs/USER-MANUAL.md`.
