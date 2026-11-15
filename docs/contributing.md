# Contributing

PyDESeq2 is a living project and contributions are welcome.
The project is hosted [on GitHub](https://github.com/scverse/PyDESeq2), and contributors are expected to follow the [Python code of conduct](https://www.python.org/psf/codeofconduct/).

## Reporting a bug or requesting a feature

Search the [issue tracker](https://github.com/scverse/PyDESeq2/issues) first.
If the issue already exists, a thumbs up helps us prioritise it, and a comment helps if you have something to add.

If it does not exist, describe how you reached it, with a code snippet where possible, and include the version you are running:

```python
import pydeseq2

pydeseq2.__version__
```

## Setting up a development environment

Fork the repository on GitHub, then clone your fork:

```bash
git clone git@github.com:<your-github-username>/PyDESeq2.git
cd PyDESeq2
```

Install the project with its development, test and documentation dependencies:

```bash
uv sync --group dev --group test --group doc
```

Add the upstream remote so you can keep your fork current:

```bash
git remote add upstream git@github.com:scverse/PyDESeq2.git
git fetch upstream
```

Install the pre-commit hooks, which run ruff and mypy on every commit:

```bash
uvx prek install
```

## Opening a pull request

Branch off an up-to-date `main`, commit your changes, push the branch to your fork, and [open a pull request from it](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork).
If the pull request addresses an open issue, link it by writing `#{issue_number}` in the description.
Say what the change does and why, so a reviewer does not have to reconstruct it from the diff.

## Running the tests

The test matrix is defined by the hatch environments in `pyproject.toml`, so the same command runs locally and in CI:

```bash
uvx hatch test
```

To run a single environment:

```bash
uvx hatch run hatch-test.py3.12-stable:pytest
```

The `hatch-test.py3.14-deseq2` environment also compares PyDESeq2 with DESeq2, which it runs through rpy2.
It needs R and the DESeq2 version pinned in `tests/deseq2_environment.yaml`, and `R_HOME` has to point to that R installation:

```bash
conda env create --name deseq2 --file tests/deseq2_environment.yaml
R_HOME=$(conda run --name deseq2 R RHOME) uvx hatch run hatch-test.py3.14-deseq2:pytest tests/test_r_deseq2.py
```

In the other environments, these tests are skipped.

## Building the documentation

```bash
uvx hatch run docs:build
```

The rendered HTML lands in `docs/_build/html`.
`uvx hatch run docs:open` opens it in a browser, and `uvx hatch run docs:clean` removes the generated files.
The build treats warnings as errors, so a broken cross-reference fails it.
