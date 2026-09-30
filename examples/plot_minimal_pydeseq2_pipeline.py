"""
A simple PyDESeq2 workflow
===========================

In this example, we show how to perform a simple differential expression analysis on bulk RNAseq data, using PyDESeq2.

.. contents:: Contents
    :local:
    :depth: 3

We start by importing required packages and setting up an optional path to save results.

"""

# %%

import os
import pickle as pkl

from pydeseq2.datasets import load_example_data
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats

SAVE = False  # whether to save the outputs of this notebook

if SAVE:
    # Replace this with the path to directory where you would like results to be saved
    OUTPUT_PATH = "../output_files/synthetic_example"
    os.makedirs(OUTPUT_PATH, exist_ok=True)  # Create path if it doesn't exist

# %%
# Data loading
# ------------
#
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
# To illustrate the required data format, we load a synthetic example dataset that may be
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

print(counts_df)

# %%
print(metadata)


# %%
# In this example, the metadata data contains two columns, ``condition`` and ``group``,
# representing two types of bi-level annotations. In the first part, we will only use the
# ``condition`` factor. Later on, we'll see how to use both the `condition` and the
# ``group`` factors in our analysis (see :ref:`multifactor_ref`).

# %%
# Data filtering
# ^^^^^^^^^^^^^^
#
# Before proceeding with DEA, it is good practice to preprocess your data, e.g. to remove
# samples for which annotations are missing and exclude genes with very low levels of
# expression. This is not necessary in the case of our synthetic data, but don't forget
# this step if you are using real data. To this end you can use the code below.
#
# We start by removing samples for which ``condition`` is ``NaN``. If you are using
# another dataset, do not forget to change "condition" for the column of ``metadata``
# you wish to use as a design factor in your analysis.

samples_to_keep = ~metadata.condition.isna()
counts_df = counts_df.loc[samples_to_keep]
metadata = metadata.loc[samples_to_keep]

# %%
# .. note::
#   In the case where the design factor contains ``NaN`` entries, PyDESeq2 will throw an
#   error when intializing a :class:`DeseqDataSet <dds.DeseqDataSet>`.

# %%
# Next, we filter out genes that have less than 10 read counts in total. Note again that
# there are no such genes in this synthetic dataset.

genes_to_keep = counts_df.columns[counts_df.sum(axis=0) >= 10]
counts_df = counts_df[genes_to_keep]

# %%
# Now that we have loaded and filtered our data, we may proceed with the differential
# analysis.


# %%
# Single factor analysis
# --------------------------
#
# In this first analysis, we ignore the ``group`` variable and use the ``condition``
# column as our design factor. That is, we compare gene expressions of samples that have
# ``condition B`` to those that have ``condition A``.
#

# %%
# .. currentmodule:: pydeseq2.dds
#
# Read counts modeling with the :class:`DeseqDataSet` class
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We start by creating a :class:`DeseqDataSet`
# object from the count and metadata data.
# A :class:`DeseqDataSet` fits dispersion and
# log-fold change (LFC) parameters from the data, and stores them.
#

inference = DefaultInference(n_cpus=8)
dds = DeseqDataSet(
    counts=counts_df,
    metadata=metadata,
    design="~condition",
    refit_cooks=True,
    inference=inference,
    # n_cpus=8, # n_cpus can be specified here or in the inference object
)

# %%
# A :class:`DeseqDataSet` has two mandatory
# arguments: a ``counts`` and a ``metadata`` dataframe, like the ones we've loaded in the
# first part of this tutorial.
#
# Next, we should specify a ``design``, i.e. a Wilkinson formula that describes the
# design, or directly a design matrix. Here we provide a formula, which is a string
# that `formulaic <https://github.com/matthewwardrop/formulaic>`_ should be able to
# parse.
#
# .. note::
#   The ``"condition"`` factor in ``design`` corresponds to a column
#   from the ``metadata`` dataframe we loaded earlier.
#   You might need to change it according to your own dataset.
#
# Several other arguments may be optionally specified (see the :doc:`API documentation
# </generated/pydeseq2.dds.DeseqDataSet>`).
# Among those, the ``refit_cooks`` argument (set to ``True`` by default), controls
# whether Cooks outlier should be refitted (which is advised, in general) and ``n_cpus``
# sets the number of CPUs to use for computation. Here, we use 8 threads. Feel free to
# adapt this to your setup or to set to ``None`` to use all available CPUs.
#
# .. note::
#     In the case of the provided synthetic data, there won't be any Cooks outliers.
#
# Once a :class:`DeseqDataSet` was initialized,
# we may run the :meth:`deseq2() <DeseqDataSet.deseq2>` method
# to fit dispersions and LFCs.
#
#


dds.deseq2()


if SAVE:
    with open(os.path.join(OUTPUT_PATH, "dds.pkl"), "wb") as f:
        pkl.dump(dds, f)

# %%
# The :class:`DeseqDataSet` class extends the
# :class:`AnnData <anndata.AnnData>`
# class.

print(dds)

# %%
# Hence, parameters are stored according to the :class:`AnnData <anndata.AnnData>` data
# structure, with key-based data fields. In particular,
#
# - ``X`` stores the count data,
# - ``obs`` stores 1D sample-level data, such as design factors and ``"size_factors"``,
# - ``obsm`` stores multi-dimensional sample-level data, such as ``"design_matrix"``,
# - ``var`` stores 1D gene-level data, such as gene names and ``"dispersions"``,
# - ``varm`` stores multi-dimensional gene-level data, such as ``"LFC"``.
#
#
# As an example, here is how we would access dispersions and LFCs
# (in natural log scale):

# %%

print(dds.var["dispersions"])

# %%

print(dds.varm["LFC"])

# %%
# .. currentmodule:: pydeseq2.ds
#
# Statistical analysis with the :class:`DeseqStats` class
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Now that dispersions and LFCs were fitted, we may proceed with statistical tests to
# compute p-values and adjusted p-values for differential expresion. This is the role of
# the :class:`DeseqStats` class. It has two mandatory arguments:
#
# - ``dds``, which should be a *fitted* :class:`DeseqDataSet <pydeseq2.dds.DeseqDataSet>`
#   object,
# - ``contrast``, which is a list of three strings of the form
#   ``["variable", "tested_level", "control_level"]``, or directly a contrast vector.
#

ds = DeseqStats(dds, contrast=["condition", "B", "A"], inference=inference)

# %%
# It also has a set of optional keyword arguments (see the :doc:`API documentation
# </generated/pydeseq2.ds.DeseqStats>`), among which:
#
# - ``alpha``: the p-value and adjusted p-value significance threshold (``0.05``
#   by default),
# - ``cooks_filter``: whether to filter p-values based on cooks outliers
#   (``True`` by default),
# - ``independent_filter``: whether to perform independent filtering to correct
#   p-value trends (``True`` by default).
#
# In the :ref:`section on multifactor analysis<multifactor_ref>`, we will also see how
# to use the ``contrast`` argument to specify according to which variable samples should
# be compared.

# %%
# .. _wald_ref:
#
# Wald test
# """"""""""
#
# PyDESeq2 computes p-values using Wald tests. This can be done using the
# :meth:`summary() <DeseqStats.summary>` method, which runs the whole statistical
# analysis, cooks filtering and multiple testing adjustement included.

ds.summary()

if SAVE:
    with open(os.path.join(OUTPUT_PATH, "ds.pkl"), "wb") as f:
        pkl.dump(ds, f)

# %%
# The results are then stored in the ``results_df`` attribute (``ds.results_df``).

# %%
# Optional: threshold-based tests
# """"""""""""""""""""""""""""""""
#
# The user can specify a (log2) log fold change under the null hypothesis and an
# alternative hypothesis to re-compute Wald statistics and p-values.
# The alternative hypothesis corresponds to what the user wants to find rather than the
# null hypothesis. It can take one of the values
# ``["greaterAbs", "lessAbs", "greater", "less"]``.

ds.summary(lfc_null=0.1, alt_hypothesis="greaterAbs")
ds.plot_MA(s=20)

# %%
# LFC shrinkage
# """""""""""""
#
# For visualization or post-processing purposes, it might be suitable to perform
# LFC shrinkage. This is implemented by the :meth:`lfc_shrink() <DeseqStats.lfc_shrink>`
# method, which takes as argument the name of the coefficient to shrink (i.e., the name
# of one of the columns of the design matrix ``dds.obsm["design_matrix"]``).
# For instance, to shrink the LFCs of the ``condition B`` samples, we would run:

ds.lfc_shrink(coeff="condition[T.B]")

if SAVE:
    with open(os.path.join(OUTPUT_PATH, "shrunk_results.pkl"), "wb") as f:
        pkl.dump(ds, f)

# %%
# .. note::
#   Running :meth:`lfc_shrink() <DeseqStats.lfc_shrink>` will overwrite a
#   :class:`DeseqStats`' log fold changes (and standard errors) with shrunk values.
#   This can be checked using the ``shrunk_LFCs`` flag.
#

print(ds.shrunk_LFCs)  # Will be True only if lfc_shrink() was run.

# %%
# .. _reference_level_ref:
#
# Choosing a reference level for shrinkage
# """"""""""""""""""""""""""""""""""""""""""""""""""
#
# A statistical ``contrast`` can compare any two levels, but
# :meth:`lfc_shrink() <DeseqStats.lfc_shrink>` requires a single fitted coefficient.
# For example, with three conditions A, B and C and A as the reference, C versus B
# is a difference of coefficients, not a column to pass to ``lfc_shrink``.
# Choose the required reference in the design formula before fitting instead.
# The deprecated ``ref_level`` argument has no effect.
#
# Using our two-condition synthetic data, we choose B as the reference and fit a
# new dataset to obtain the A-versus-B coefficient. ``C()`` marks a variable as
# categorical; this also matters when category labels are numbers (e.g. batch IDs).
# ``contr.treatment(base='B')`` selects the reference category. See Formulaic's
# `treatment coding documentation
# <https://matthewwardrop.github.io/formulaic/latest/guides/contrasts/#treatment-aka-dummy>`_.

dds_ref_b = DeseqDataSet(
    counts=counts_df,
    metadata=metadata,
    design="~ C(condition, contr.treatment(base='B'))",
    inference=inference,
)
dds_ref_b.deseq2()

# %%
# Formula-based coding changes the coefficient names. Inspect the fitted columns
# rather than reusing ``"condition[T.B]"`` from the previous design. We also choose
# the matching A-versus-B contrast, so the Wald results and shrunk LFCs describe
# the same comparison. Do not replace the design matrix of an already fitted
# object without refitting its coefficients.

print(dds_ref_b.varm["LFC"].columns)
ds_A_vs_B = DeseqStats(dds_ref_b, contrast=["condition", "A", "B"], inference=inference)
ds_A_vs_B.summary()
ds_A_vs_B.lfc_shrink(coeff="C(condition, contr.treatment(base='B'))[T.A]")

# %%
# .. _multifactor_ref:
#
# Multifactor analysis
# ---------------------
#
# .. currentmodule:: pydeseq2.dds
#
# So far, we have only used the ``condition`` column of ``metadata``, which divides
# samples between conditions ``A`` and ``B``. Yet, ``metadata`` contains second
# column, which separates samples according to ``group`` ``X`` and ``Y``.

print(metadata)

# %%
# The goal of multifactor analysis is to use *both* variables to fit LFCs.
#
# Read counts modeling
# ^^^^^^^^^^^^^^^^^^^^^
#
# To perform multifactor analysis with PyDESeq2, we start by inializing a
# :class:`DeseqDataSet` as previously, but we provide several variables we would like
# to use in the ``design`` argument.
#

dds = DeseqDataSet(
    counts=counts_df,
    metadata=metadata,
    design="~group + condition",
    refit_cooks=True,
    inference=inference,
)
# %%
# As for the single-factor analysis, we fit dispersions and LFCs using the
# :meth:`deseq2() <DeseqDataSet.deseq2>` method.

dds.deseq2()

# %%
# Now, if we print log fold changes, we will have two columns in addition to the
# intercept: one corresponding to the ``group`` variable, and the other to ``condition``.

print(dds.varm["LFC"])

# %%
# .. currentmodule:: pydeseq2.ds
#
# Statistical analysis
# ^^^^^^^^^^^^^^^^^^^^
#
# P-values are computed as earlier from a :class:`DeseqStats` object with the
# :meth:`summary() <DeseqStats.summary>` method. The ``contrast`` argument will allow us
# to determines which variable we want to obtain LFCs and pvalues for.
# It is a list of three strings of the form
# ``["variable", "tested level", "reference level"]``
# As an example, to compare the condition B to the condition A, we set
# ``contrast=["condition", "B", "A"]``.
#

ds_B_vs_A = DeseqStats(dds, contrast=["condition", "B", "A"], inference=inference)

# %%
#
# Let us fit p-values:
#

ds_B_vs_A.summary()


# %%
# As we can see, although we are comparing the same cohorts (condition B vs A), the
# results differ from the :ref:`single-factor analysis <wald_ref>`. This is because the
# model uses information from both the ``condition`` and ``group`` variables.
#
# Let us now evaluate differential expression according to group Y vs X. To do so,
# we create a new :class:`DeseqStats` from the same
# :class:`DeseqDataSet <pydeseq2.dds.DeseqDataSet>`
# with ``contrast=["group", "Y", "X"]``, and run the analysis again.

ds_Y_vs_X = DeseqStats(dds, contrast=["group", "Y", "X"], inference=inference)
ds_Y_vs_X.summary()

# %%
# LFC shrinkage (multifactor)
# """""""""""""""""""""""""""
#
# In a multifactor setting, LFC shrinkage works as in the single-factor case, but will
# only shrink the LFCs of a :class:`DeseqStats` object based on its
# ``contrast`` argument.

ds_B_vs_A.lfc_shrink(coeff="condition[T.B]")
