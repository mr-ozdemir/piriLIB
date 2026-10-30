# Contributing

Thank you for helping. Bugs and ideas go to the
[issue tracker](https://github.com/mr-ozdemir/piriLIB/issues); security issues
are reported privately, as described in [SECURITY.md](SECURITY.md). Everyone
taking part follows the [code of conduct](CODE_OF_CONDUCT.md).

## Development setup

Create a virtual environment, activate it (`.venv\Scripts\activate` on Windows,
`source .venv/bin/activate` elsewhere) and install the package with the
development tools. pip 25.1 or newer is needed for `--group`.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e . --group dev
```

## Checks

Every pull request must pass the same checks as CI:

```bash
ruff check . && ruff format --check . && mypy
pytest
mkdocs build --strict
```

`mkdocs` needs the documentation tools: `python -m pip install --group docs`.
`mkdocs serve` shows a live preview at http://127.0.0.1:8000.

After changing the specification models, regenerate the JSON Schema:

```bash
python -m pirilib.spec.schema schemas/linehaul-0.2.schema.json
```

## Code and tests

- Code fits the package structure: engines and data are classes, entry points
  are thin functions.
- No `#` comment lines. Explanations go into docstrings.
- Every change in behaviour comes with tests. [docs/testing.md](docs/testing.md)
  describes the test levels and how a test is written.
- User-visible changes are added to `CHANGELOG.md` under `[Unreleased]`.

## Branches and pull requests

| Branch | Purpose |
|---|---|
| `master` | Released code, changed only by merging `develop` |
| `develop` | Finished and reviewed work |
| personal branches | Each contributor's own work |

1. Work on your own branch and push it.
2. Open a pull request into `develop`. Another contributor reviews and merges it.
3. When `develop` is ready for a release, it is merged into `master`.

Commit messages are short and in the imperative, for example
"Add wave splitting to forecasts". Every change has one author.

## Releasing

Versions are written in PEP 440 form: `0.1.0a1`, `0.1.0b1` and `0.1.0rc1` are
pre-releases, `0.1.0` is a final release. A pre-release is published to
TestPyPI only, a final release to TestPyPI and then to PyPI.

1. Set `__version__` in `src/pirilib/_version.py`.
2. In `CHANGELOG.md`, add `## [X.Y.Z] - YYYY-MM-DD` under `## [Unreleased]`
   and update the link references.
3. Merge into `develop`, then `develop` into `master`.
4. Tag the merge commit on `master` and push the tag:

   ```bash
   git switch master
   git pull
   git tag -a v0.1.0a1 -m "pirilib 0.1.0a1"
   git push origin v0.1.0a1
   ```

The tag starts `.github/workflows/release.yml`: it checks that the tag matches
`__version__`, runs the lint and the tests, builds and installs the wheel, and
publishes to TestPyPI with trusted publishing. A final version then waits for
an approval in the `pypi` environment and is published to PyPI. A version
cannot be uploaded twice; after a wrong upload, release the next number.

One-time setup before the first tag:

1. On test.pypi.org and pypi.org, add a trusted publisher for `pirilib`:
   repository `mr-ozdemir/piriLIB`, workflow `release.yml`, environment
   `testpypi` on TestPyPI and `pypi` on PyPI.
2. In the repository settings, create the environments `testpypi` and `pypi`,
   and add a required reviewer to `pypi`.
