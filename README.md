<img src="docs/_static/pydeseq2_logo_green.png" width="600">

#
[![pypi version](https://img.shields.io/pypi/v/pydeseq2)](https://pypi.org/project/pydeseq2)
[![pypiDownloads](https://static.pepy.tech/badge/pydeseq2)](https://pepy.tech/project/pydeseq2)
[![condaDownloads](https://img.shields.io/conda/dn/bioconda/pydeseq2?logo=Anaconda)](https://anaconda.org/bioconda/pydeseq2)
[![license](https://img.shields.io/pypi/l/pydeseq2)](LICENSE)

PyDESeq2 is a python implementation of the [DESeq2](https://bioconductor.org/packages/release/bioc/html/DESeq2.html)
method [1] for differential expression analysis (DEA) with bulk RNA-seq data, originally in R.
It aims to facilitate DEA experiments for python users.

As PyDESeq2 is a re-implementation of [DESeq2](https://bioconductor.org/packages/release/bioc/html/DESeq2.html) from
scratch, you may experience some differences in terms of retrieved values or available features.

Currently, available features broadly correspond to the default settings of DESeq2 (v1.34.0) for single-factor and
multi-factor analysis (with categorical or continuous factors) using Wald tests.
We plan to implement more in the future.
In case there is a feature you would particularly like to be implemented, feel free to open an issue.

## Table of Contents
- [PyDESeq2](#pydeseq2)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Requirements](#requirements)
  - [Getting started](#getting-started)
    - [Documentation](#documentation)
    - [Data](#data)
  - [Contributing](#contributing)
    - [1 - Download the repository](#1---download-the-repository)
    - [2 - Create a conda environment](#2---create-a-conda-environment)
  - [Development roadmap](#development-roadmap)
  - [Citing this work](#citing-this-work)
  - [References](#references)
  - [License](#license)

## Installation

### PyPI

`PyDESeq2` can be installed from PyPI using `pip`:

```bash
pip install pydeseq2
```

We recommend installing within a conda environment:

```bash
conda create -n pydeseq2
conda activate pydeseq2
conda install pip
pip install pydeseq2
```

You can also add it to your projects through `uv`:

```bash
uv add pydeseq2
```

### Bioconda

`PyDESeq2` can also be installed from Bioconda with `conda`:

`conda install -c bioconda pydeseq2`

If you're interested in contributing or want access to the development version, please see the [contributing](#contributing) section.

### Requirements

The list of package version requirements is available in `pyproject.toml`.

For reference, the code is being tested in a github workflow (CI) with python
3.12 to 3.14 and the latest versions of the following packages:

```
- anndata
- formulaic
- numpy
- pandas
- scikit-learn
- scipy
- formulaic-contrasts
- matplotlib
```

Please don't hesitate to open an issue in case you encounter any issue due to possible deprecations.


## Getting started

The [Getting Started](https://pydeseq2.readthedocs.io/en/latest/auto_examples/index.html) section of the documentation
contains downloadable examples on how to use PyDESeq2.


### Documentation

The documentation is hosted [here on ReadTheDocs](https://pydeseq2.readthedocs.io/en/latest/).
If you want to have the latest version of the documentation, you can build it from source.
Please go to the dedicated [README.md](https://github.com/owkin/PyDESeq2/blob/main/docs/README.md) for information on how to do so.

### Data

The quick start examples use synthetic data, provided in this repo (see [datasets](https://github.com/owkin/PyDESeq2/blob/main/datasets/README.md).)

The experiments described in the [PyDESeq2 article](https://academic.oup.com/bioinformatics/article/39/9/btad547/7260507) rely on data
from [The Cancer Genome Atlas](https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga),
which may be obtained from this [portal](https://portal.gdc.cancer.gov/).

## Contributing

Please the [Contributing](https://pydeseq2.readthedocs.io/en/latest/usage/contributing.html) section of the
documentation to see how you can contribute to PyDESeq2.

### 1 - Download the repository

`git clone https://github.com/owkin/PyDESeq2.git`

### 2 - Create a uv environment

Run `uv venv --python 3.13` (or higher python version) to create the `pydeseq2` environment and then activate it:
`source .venv/bin/activate`.

`cd` to the root of the repo and run `uv sync --extra dev --extra doc` to install in developer mode.

Then, run `pre-commit install`.

The `pre-commit` tool will automatically run [ruff](https://docs.astral.sh/ruff/), [black](https://black.readthedocs.io/en/stable/), and [mypy](https://mypy.readthedocs.io/en/stable/).

PyDESeq2 is a living project and any contributions are welcome! Feel free to open new PRs or issues.

## Credits

PyDESeq2 has been originally developed by Boris Muzellec, Maria Teleńczuk, Vincent Cabeli, and Mathieu Andreux and funded by [Owkin](https://www.owkin.com/).
In Dec 2025, the maintenance of PyDESeq2 was taken over by the scverse community.

## Citing this work

```
@article{muzellec2023pydeseq2,
  title={PyDESeq2: a python package for bulk RNA-seq differential expression analysis},
  author={Muzellec, Boris and Telenczuk, Maria and Cabeli, Vincent and Andreux, Mathieu},
  year={2023},
  doi = {10.1093/bioinformatics/btad547},
  journal={Bioinformatics},
}
```

## References

[1] Love, M. I., Huber, W., & Anders, S. (2014). "Moderated estimation of fold
        change and dispersion for RNA-seq data with DESeq2." Genome biology, 15(12), 1-21.
        <https://genomebiology.biomedcentral.com/articles/10.1186/s13059-014-0550-8>

[2] Zhu, A., Ibrahim, J. G., & Love, M. I. (2019).
        "Heavy-tailed prior distributions for sequence count data:
        removing the noise and preserving large differences."
        Bioinformatics, 35(12), 2084-2092.
        <https://academic.oup.com/bioinformatics/article/35/12/2084/5159452>

## License

PyDESeq2 is released under an [MIT license](https://github.com/owkin/PyDESeq2/blob/main/LICENSE).
