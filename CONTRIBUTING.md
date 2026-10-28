# Contributing

## Branches

| Branch | Purpose |
|---|---|
| `master` | Released code. Changed only by merging `develop`. |
| `develop` | Work that is done and reviewed collects here. |
| personal branches | Each contributor works on their own branch. |

1. Commit on your own branch and push it.
2. Open a pull request into `develop`. Another contributor reviews and merges it.
3. When `develop` is ready for a release, it is merged into `master`.

Keep commit messages short and in the imperative, for example
"Add wave splitting to forecasts". Every task has one author, who makes its
commits.

## Before a pull request

```bash
ruff check . && ruff format --check . && mypy
pytest
```

New tests follow [docs/testing.md](docs/testing.md): test levels, how to
write a test, fuzz, model-based, performance and mutation testing.

The code has no `#` comment lines; explanations go into docstrings. Decisions
that change behaviour are written to `docs/kararlar.md`, measurements to
`docs/olcumler.md`.

## Releasing

Versions are written in PEP 440 form: `0.1.0a1`, `0.1.0b1` and `0.1.0rc1` are
pre-releases, `0.1.0` is a final release. A pre-release is published to
TestPyPI only, a final release to TestPyPI and then to PyPI.

1. On your branch, set `__version__` in `src/pirilib/_version.py`.
2. In `CHANGELOG.md`, add the heading `## [X.Y.Z] - YYYY-MM-DD` with the
   planned tag date right under `## [Unreleased]`, so that the entries
   collected so far belong to the release, and update the link references
   below the introduction.
3. Open a pull request into `develop`, then merge `develop` into `master`.
4. Tag the merge commit on `master` with `v` and the version, and push the tag:

   ```bash
   git switch master
   git pull
   git tag -a v0.1.0a1 -m "pirilib 0.1.0a1"
   git push origin v0.1.0a1
   ```

The tag starts `.github/workflows/release.yml`. It stops unless the tag
matches `__version__`, runs the lint and the tests, builds the sdist and the
wheel, installs the wheel and runs `pirilib --help`, and publishes to
TestPyPI. For a final version it then waits for an approval in the `pypi`
environment and publishes the same files to PyPI. Publishing uses trusted
publishing, so the repository stores no tokens. A version cannot be uploaded
twice; after a wrong upload, release the next number.

One-time setup before the first tag:

1. On test.pypi.org and on pypi.org, add a trusted publisher for the project
   `pirilib` (a pending publisher while the project does not exist there yet):
   repository `mr-ozdemir/piriLIB`, workflow `release.yml`, environment
   `testpypi` on TestPyPI and `pypi` on PyPI.
2. In the GitHub repository settings, create the environments `testpypi` and
   `pypi`, and add yourself as a required reviewer of `pypi`.
