# PyDESeq2

PyDESeq2 is a Python implementation of the [DESeq2](https://bioconductor.org/packages/release/bioc/html/DESeq2.html) method {cite:p}`love2014moderated` for differential expression analysis with bulk RNA-seq data, originally written in R.
It works on {class}`~anndata.AnnData` objects and is part of the [scverse ecosystem](https://scverse.org).

Because PyDESeq2 is a reimplementation from scratch, you may see differences in retrieved values or available features.
Current features broadly correspond to the default settings of DESeq2 (v1.34.0) for single-factor and multi-factor analysis, with categorical or continuous factors, using Wald tests, plus an optional [apeGLM](https://academic.oup.com/bioinformatics/article/35/12/2084/5159452) log fold-change shrinkage step {cite:p}`zhu2019heavy`.
If there is a feature you would like to see, open an issue on GitHub.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`desktop-download;1.5em;sd-mr-1` Installation
:link: installation
:link-type: doc

Install PyDESeq2 with pip or conda.
:::

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` Tutorials
:link: auto_examples/index
:link-type: doc

Runnable examples, from a minimal pipeline to a step-by-step walkthrough.
:::

:::{grid-item-card} {octicon}`code-square;1.5em;sd-mr-1` API reference
:link: api
:link-type: doc

Every class and function, with its parameters and return values.
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` About PyDESeq2
:link: about/background
:link-type: doc

What DESeq2 does, what PyDESeq2 covers, and where the two differ.
:::

:::{grid-item-card} {octicon}`git-pull-request;1.5em;sd-mr-1` Contributing
:link: contributing
:link-type: doc

Set up a development environment and open a pull request.
:::

:::{grid-item-card} {octicon}`mark-github;1.5em;sd-mr-1` GitHub
:link: https://github.com/scverse/PyDESeq2

Read the source, report a bug, or open a pull request.
:::

::::

## NumFOCUS

[//]: # "numfocus-fiscal-sponsor-attribution"

PyDESeq2 is part of the scverse® project ([website](https://scverse.org), [governance](https://scverse.org/about/roles)) and is fiscally sponsored by [NumFOCUS](https://numfocus.org/).
If you like scverse® and want to support our mission, please consider making a tax-deductible [donation](https://numfocus.org/donate-to-scverse) to help the project pay for developer time, professional services, travel, workshops, and a variety of other needs.

```{toctree}
:caption: Getting started
:hidden:
:maxdepth: 1

installation
Quickstart <auto_examples/plot_minimal_pydeseq2_pipeline>
```

```{toctree}
:caption: Guides
:hidden:
:maxdepth: 2

auto_examples/index
```

```{toctree}
:caption: Reference
:hidden:
:maxdepth: 1

api
about/background
```

```{toctree}
:caption: Project
:hidden:
:maxdepth: 1

contributing
about/cite
references
GitHub <https://github.com/scverse/PyDESeq2>
Discourse <https://discourse.scverse.org/>
```
