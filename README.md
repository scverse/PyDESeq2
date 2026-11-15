<img src="https://raw.githubusercontent.com/scverse/PyDESeq2/main/docs/_static/pydeseq2_logo_green.png" width="600">

# PyDESeq2

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]
[![PyPI][badge-pypi]][pypi]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/scverse/PyDESeq2/test.yaml?branch=main
[badge-docs]: https://app.readthedocs.org/projects/pydeseq2/badge/
[badge-pypi]: https://img.shields.io/pypi/v/pydeseq2

PyDESeq2 is a Python implementation of the [DESeq2][] method for differential expression analysis with bulk RNA-seq data, originally written in R.
It works on [AnnData][] objects and is part of the [scverse ecosystem][scverse].

Current features broadly correspond to the default settings of DESeq2 for single-factor and multi-factor analysis, with categorical or continuous factors, using Wald tests, plus an optional [apeGLM][] log fold-change shrinkage step.
Because PyDESeq2 is a reimplementation from scratch, you may see differences in retrieved values or available features.

## Getting started

Please refer to the [documentation][], in particular the [tutorials][] and the [API documentation][].

## Installation

You need to have Python 3.12 or newer installed on your system.
There are several options to install PyDESeq2:

1. Install the latest release of `pydeseq2` from [PyPI][]:

   ```bash
   pip install pydeseq2
   ```

2. Install the latest release from [Bioconda][]:

   ```bash
   conda install bioconda::pydeseq2
   ```

3. Install the latest development version:

   ```bash
   pip install git+https://github.com/scverse/PyDESeq2.git@main
   ```

## Release notes

See the [release notes on GitHub][changelog].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

## Citation

If you use PyDESeq2, please cite the PyDESeq2 paper:

```bibtex
@article{muzellec2023pydeseq2,
  title={PyDESeq2: a python package for bulk RNA-seq differential expression analysis},
  author={Muzellec, Boris and Telenczuk, Maria and Cabeli, Vincent and Andreux, Mathieu},
  year={2023},
  doi = {10.1093/bioinformatics/btad547},
  journal={Bioinformatics},
}
```

The [citation page][] of the documentation lists the methods that PyDESeq2 implements.

## Credits

PyDESeq2 has been originally developed by Boris Muzellec, Maria Teleńczuk, Vincent Cabeli, and Mathieu Andreux and funded by [Owkin][].
In Dec 2025, the maintenance of PyDESeq2 was taken over by the scverse community.

[deseq2]: https://bioconductor.org/packages/release/bioc/html/DESeq2.html
[anndata]: https://anndata.readthedocs.io
[scverse]: https://scverse.org
[apeglm]: https://academic.oup.com/bioinformatics/article/35/12/2084/5159452
[tests]: https://github.com/scverse/PyDESeq2/actions/workflows/test.yaml
[documentation]: https://pydeseq2.readthedocs.io
[tutorials]: https://pydeseq2.readthedocs.io/en/latest/auto_examples/index.html
[api documentation]: https://pydeseq2.readthedocs.io/en/latest/api.html
[citation page]: https://pydeseq2.readthedocs.io/en/latest/about/cite.html
[pypi]: https://pypi.org/project/pydeseq2
[bioconda]: https://anaconda.org/bioconda/pydeseq2
[changelog]: https://github.com/scverse/PyDESeq2/releases
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/scverse/PyDESeq2/issues
[owkin]: https://www.owkin.com/
