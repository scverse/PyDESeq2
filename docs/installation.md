# Installation

PyDESeq2 requires Python 3.12 or newer.

::::{tab-set}
:::{tab-item} pip
```bash
pip install pydeseq2
```
:::

:::{tab-item} uv
```bash
uv add pydeseq2
```
:::

:::{tab-item} conda
```bash
conda install bioconda::pydeseq2
```
:::
::::

This installs everything needed to run a differential expression analysis: dispersion and log fold-change estimation, Wald tests, and the plots.

Minimum supported versions of Python and of the core scientific Python packages follow [SPEC 0](https://scientific-python.org/specs/spec-0000/).
Python versions are dropped three years after their initial release, and core package versions two years after theirs.
The exact lower bounds live in `pyproject.toml` and are installed for you.

## Development install

```bash
git clone https://github.com/scverse/PyDESeq2.git
cd PyDESeq2
uv sync --group dev --group test --group doc
```

The {doc}`contributing guide <contributing>` describes the environments, the test matrix and the docs build.
