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
"Add wave splitting to forecasts". For work done together, add a
`Co-authored-by: Name <email>` line at the end of the message.

## Before a pull request

```bash
ruff check . && ruff format --check . && mypy
pytest
```

The code has no `#` comment lines; explanations go into docstrings. Decisions
that change behaviour are written to `docs/kararlar.md`, measurements to
`docs/olcumler.md`.
