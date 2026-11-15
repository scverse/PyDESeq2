"""
Step-by-step PyDESeq2 workflow
===============================

This notebook details all the steps of the PyDESeq2 pipeline.

It allows you to run the PyDESeq2 pipeline on the synthetic data provided in this repository.

If this is your first contact with PyDESeq2, we recommend you first have a look at the :doc:`standard workflow example <plot_minimal_pydeseq2_pipeline>`.

.. contents:: Contents
    :local:
    :depth: 3

We start by importing required packages and setting up an optional path to save results.
"""

import os
import pickle as pkl

import numpy as np

from pydeseq2.datasets import load_example_data
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats

SAVE = False  # whether to save the outputs of this notebook

if SAVE:
    # Replace this with the path to directory where you would like results to be
    # saved
    OUTPUT_PATH = "../output_files/synthetic_example"
    os.makedirs(OUTPUT_PATH, exist_ok=True)  # Create path if it doesn't exist

# %%
# Data loading
# ------------
#
# Note that we are also explaining this step in the 'getting started' example.
# To perform differential expression analysis (DEA), PyDESeq2 requires two types of
# inputs:
#
#   * A count matrix of shape 'number of samples' x 'number of genes', containing
#     read counts (non-negative integers),
#   * Metadata (or "column" data) of shape 'number of samples' x
#     'number of variables', containing sample annotations that will be used
#     to split the data in cohorts.
#
# Both should be provided as `pandas dataframes
# <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_.
#
# .. currentmodule:: pydeseq2
#
# To illustrate the required data format, we load a synthetic example dataset
# that may be
# obtained through PyDESeq2's API using :func:`datasets.load_example_data`.
# You may replace it with your own dataset.

counts_df = load_example_data(
    modality="raw_counts",
    dataset="synthetic",
    debug=False,
)

metadata = load_example_data(
    modality="metadata",
    dataset="synthetic",
    debug=False,
)

# %%
# 1. Read counts modeling
# -----------------------
# Read counts modeling with the :class:`DeseqDataSet
# <dds.DeseqDataSet>` class
#
# The :class:`DeseqDataSet <dds.DeseqDataSet>` class has two mandatory
# arguments, ``counts`` and
# ``metadata``, as well as a set of optional keyword arguments, among which:
#
# - ``design``: a string representing a
#   `formulaic <https://github.com/matthewwardrop/formulaic>`_ formula, or a design
#   matrix (ndarray), that will be used to model the data.
# - ``refit_cooks``: whether to refit cooks outliers – this is advised, in general.
#
# .. note::
#   in the case of the provided synthetic data, there won't be any Cooks
#   outliers.

inference = DefaultInference(n_cpus=8)
dds = DeseqDataSet(
    counts=counts_df,
    metadata=metadata,
    design="~condition",  # compare samples based on the "condition"
    # column ("B" vs "A")
    refit_cooks=True,
    inference=inference,
)

# %%
# Compute normalization factors
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

dds.fit_size_factors()

dds.obs["size_factors"]

# %%
# Fit genewise dispersions
# ^^^^^^^^^^^^^^^^^^^^^^^^

dds.fit_genewise_dispersions()

dds.var["genewise_dispersions"]

# %%
# Fit dispersion trend coefficients
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The ``fit_type`` argument of ``DeseqDataSet`` selects the trend.
# It is ``"parametric"`` by default, and can also be ``"mean"`` or ``"local"``, a local regression of log dispersions over log means.

dds.fit_dispersion_trend()
dds.uns["trend_coeffs"]
dds.var["fitted_dispersions"]

# %%
# Dispersion priors
# ^^^^^^^^^^^^^^^^^

dds.fit_dispersion_prior()
print(f"logres_prior={dds.uns['_squared_logres']}, sigma_prior={dds.uns['prior_disp_var']}")

# %%
# MAP Dispersions
# ^^^^^^^^^^^^^^^
# The `fit_MAP_dispersions` method filters the genes for which dispersion
# shrinkage is applied.
# Indeed, for genes whose MLE dispersions are too high above the trend curve,
# the original MLE value is kept.
# The final values of the dispersions that are used for downstream analysis is
# stored in `dds.dispersions`.

dds.fit_MAP_dispersions()
dds.var["MAP_dispersions"]
dds.var["dispersions"]

# %%
# Fit log fold changes
# ^^^^^^^^^^^^^^^^^^^^
# Note that in the `DeseqDataSet` object, the log-fold changes are stored in
# natural
# log scale, but that the results dataframe output by the `summary` method of
# `DeseqStats` displays LFCs in log2 scale (see later on).

dds.fit_LFC()
dds.varm["LFC"]

# %%
# Calculate Cooks distances and refit
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Note that this step is optional.

dds.calculate_cooks()
if dds.refit_cooks:
    # Replace outlier counts
    dds.refit()

# %%
# Save everything

if SAVE:
    with open(os.path.join(OUTPUT_PATH, "dds_detailed_pipe.pkl"), "wb") as f:
        pkl.dump(dds, f)

# %%
# 2. Statistical analysis
# -----------------------
# Statistical analysis with the :class:`DeseqStats <ds.DeseqStats>` class.
# The `DeseqDataSet` class has two unique mandatory arguments, `dds`, which should
# be a *fitted* `DeseqDataSet` object, and ``contrast``, which should be a list
# of 3 strings of the form ``["factor", "tested_level", "reference_level"]`` or a
# numerical contrast vector, as well as a set of optional keyword
# arguments, among which:
#
# - `alpha`: the p-value and adjusted p-value significance threshold
# - `cooks_filter`: whether to filter p-values based on cooks outliers
# - `independent_filter`: whether to perform independent filtering to correct
#   p-value trends.

ds = DeseqStats(
    dds,
    contrast=np.array([0, 1]),
    alpha=0.05,
    cooks_filter=True,
    independent_filter=True,
)

# %%
# Wald tests
# ^^^^^^^^^^

ds.run_wald_test()
ds.p_values

# %%
# Cooks filtering
# ^^^^^^^^^^^^^^^
# This is optional
#
# .. note::
#   Note: in the case of the provided synthetic data, there won't be any
#   outliers.

if ds.cooks_filter:
    ds._cooks_filtering()
ds.p_values

# %%
# P-value adjustment
# ^^^^^^^^^^^^^^^^^^

if ds.independent_filter:
    ds._independent_filtering()
else:
    ds._p_value_adjustment()

ds.padj

# %%
# Building a results dataframe
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# This dataframe is stored in the `results_df` attribute of the `DeseqStats`
# class.

ds.summary()

# %%
# Save everything if SAVE is set to True

if SAVE:
    with open(os.path.join(OUTPUT_PATH, "stat_results_detailed_pipe.pkl"), "wb") as f:
        pkl.dump(ds, f)

# %%
# LFC Shrinkage
# ^^^^^^^^^^^^^
# For visualization or post-processing purposes, it might be suitable to perform
# LFC shrinkage. This is implemented by the `lfc_shrink` method.

ds.lfc_shrink(coeff="condition[T.B]")


# %%
# Save everything

if SAVE:
    with open(os.path.join(OUTPUT_PATH, "shrunk_results_detailed_pipe.pkl"), "wb") as f:
        pkl.dump(ds, f)
