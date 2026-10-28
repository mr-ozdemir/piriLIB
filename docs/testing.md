# Testing

The test suite follows a test-writing protocol taken from the course
"Software Testing and Quality Assurance" (YZ304/BS450): test levels, unit
testing practice, test doubles, larger tests, coverage criteria, fuzzing,
model-based, performance and mutation testing. This page says how each of them
is used in pirilib and how to run them.

## Running the tests

```bash
pytest
pytest -m unit
pytest -m "integration or system"
pytest -m "fuzz or model"
pytest --performance -m performance
coverage run -m pytest && coverage report
```

| Variable or option | Effect |
|---|---|
| `HYPOTHESIS_PROFILE=dev` | default: 100 examples per property, random |
| `HYPOTHESIS_PROFILE=ci` | the same examples on every run (CI uses it) |
| `HYPOTHESIS_PROFILE=fuzz` | 2000 examples per property, for fuzz and model tests |
| `--performance` | also run the performance tests, which time the solvers |

## Test levels

Every test is marked with its level; `tests/conftest.py` sets the marker from
the module.

| Marker | What it tests | Examples |
|---|---|---|
| `unit` | one unit through its public API, in isolation, fast | `test_plan.py`, `test_objective.py` |
| `integration` | several components, files or optional packages together | `test_planio.py`, `test_exact.py`, `test_schedule.py` |
| `system` | the installed package as a user drives it: command line, documentation examples | `test_cli.py`, `test_docs_examples.py` |

Most tests are unit tests; integration and system tests cover what unit tests
cannot see, such as files, optional packages and whole command-line runs.

## Writing a test

- Test a behaviour, not a method, through the public API; the result is
  checked, not how it was obtained, so the test survives internal changes.
  This matters because the line-haul problem is only the first problem type and
  the internals will change.
- One behaviour per test, named after it: `test_unservable_demand_is_left_when_cheaper`.
- Arrange, act, assert (given, when, then), and write the expected value as a
  constant.
- No logic in a test body: no `for`, `while`, `if`, `try`, conditional
  expression, loop or filter comprehension, `and`/`or` condition, or `any()` and
  `all()` over results. A test should read as a plain statement of what is true,
  not as code that itself needs a test. Repeated cases become
  `@pytest.mark.parametrize`; a check over many values becomes one equality or set
  assertion (`np.testing.assert_array_equal`, `set(...) == {...}`); a fuzz oracle
  "returns or raises the library's error" becomes `contextlib.suppress(PlanError)`.
- Helpers and fixtures that build inputs may contain logic but never compute the
  expected value; a helper with logic gets a test of its own. Pytest hooks in
  `tests/conftest.py` are infrastructure, not tests.
- Compare expected and actual with an assertion that fits the relation
  (`pytest.approx` for floats, `np.testing.assert_allclose` for arrays,
  `pytest.raises(SpecError, match=...)` for errors, by type first).
- Choose inputs by equivalence classes and boundaries: one value from every
  class, both sides of every threshold, and the empty, minimal and largest
  cases.
- Tests are independent and repeatable: they may run in any order, use
  `tmp_path` for files and fixed seeds for every random choice.
- No `#` comments; a docstring explains a test when its name cannot.

## Test doubles

Use the real collaborator when it is fast and deterministic, as most of pirilib
is. A double replaces only what is slow, random or outside the process: a stub
returns prepared values, a fake is a small working replacement (and gets tests
of its own), a mock checks the calls made to it. Optional packages that are not
installed are not mocked; their tests are skipped with `pytest.importorskip`.

## Coverage

Coverage is measured with branch coverage (`[tool.coverage.run] branch = true`).
CI fails below 95 %; on 2026-10-28 the suite covered 96.5 % of 5929 statements
and 1532 branches. Coverage shows what no test runs; it is not the goal and
not a proof of quality. A line that is run but whose result is never checked is
covered and still untested, which mutation testing finds.

## Fuzz tests

`tests/fuzz` feeds the readers of outside input (plan tables, problem
documents, YAML files) many generated inputs: random rows from an alphabet of
separators and special characters, and valid inputs with one value mutated.
The oracle is the documented contract: the reader returns its result or raises
`PlanError` or `SpecError`; any other exception is a defect. Each fuzz test has
a companion test that checks its seed input is valid.

## Model-based tests

`tests/model` gives the same inputs to the library and to a small model of what
it must do, and compares them after every step. The ledger of handling capacity
and vehicle slots is tested against a plain table; every sequence starts from a
fresh ledger.

## Performance tests

`tests/performance` times the solvers on fixed synthetic problems. They run
only with `--performance`: after a warm-up run each case is solved five times,
the median must stay within a budget several times the measured time, and the
mean, minimum and maximum are written to the JUnit XML report
(`--junitxml=performance.xml`) so they can be followed across releases.

## Mutation testing

Mutation testing measures how well the tests find faults: small changes
(mutants) are made to the code one at a time and the tests must fail on each.
`mutation/evaluation.toml` configures cosmic-ray for `evaluation.py`, the
evaluator every other test relies on:

```bash
pip install -e . --group test --group mutation
cosmic-ray baseline mutation/evaluation.toml
cosmic-ray init mutation/evaluation.toml session.sqlite
cosmic-ray exec mutation/evaluation.toml session.sqlite
cr-report session.sqlite
cr-rate session.sqlite --estimate
```

A surviving mutant either needs a new test that tells it apart from the
original, or is equivalent to the original; an equivalent mutant is written
down with the reason.

## Continuous integration

| Workflow | When | What |
|---|---|---|
| CI | every push and pull request | lint, docs, the whole suite on Linux, Windows and macOS; branch coverage on Linux, Python 3.12 |
| Quality | Mondays 03:00 UTC and on demand | fuzz and model tests with the `fuzz` profile, performance tests, mutation testing |
