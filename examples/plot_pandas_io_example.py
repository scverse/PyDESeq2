"""
Loading data and saving results with pandas and pickle
======================================================

In this example, we show how load data in order to perform a DEA analysis with PyDESeq2, and how to save its results, using `pandas <https://pandas.pydata.org/>`_ and `pickle <https://docs.python.org/3/library/pickle.html>`_.

We refer to the :doc:`getting started example <plot_minimal_pydeseq2_pipeline>` for more details on the analysis itself.

.. contents:: Contents
    :local:
    :depth: 3

We start by importing required packages and setting up a path to save results.

"""

import os
import pickle as pkl

import pandas as pd

from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats

# Replace this with the path to directory where you would like results to be saved
OUTPUT_PATH = "../output_files/synthetic_example/"
os.makedirs(OUTPUT_PATH, exist_ok=True)  # Create path if it doesn't exist


# %%
# Data loading with pandas
# ------------------------
#
# To perform differential expression analysis (DEA), PyDESeq2 requires two types of
# inputs:
#
#   * A count matrix of shape 'number of samples' x 'number of genes', containing
#     read counts (non-negative integers),
#   * Metadata (or annotations, or "column" data) of shape 'number of samples' x
#     'number of variables', containing sample annotations that will be used
#     to split the data in cohorts.
#
# Both should be provided as `pandas dataframes
# <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_.
#
# .. currentmodule:: pydeseq2
#
# Here, we show how to load CSV corresponding to counts and annotations as pandas
# dataframes, using `pandas.read_csv()
# <https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html#pandas.read_csv>`_.
#
# We assume that ``DATA_PATH`` is a directory containing a ``test_counts.csv`` and a
# ``test_metadata.csv`` file.

# Replace this with the path to your dataset
DATA_PATH = "https://raw.githubusercontent.com/owkin/PyDESeq2/main/datasets/synthetic/"

# %% Loading counts
counts_df = pd.read_csv(os.path.join(DATA_PATH, "test_counts.csv"), index_col=0)
print(counts_df)

# %%
# Note that the counts data is in a 'number of genes' x 'number of samples' format,
# whereas 'number of samples' x 'number of genes' is required. To fix this issue, we
# transpose the counts dataframe.

counts_df = counts_df.T
print(counts_df)

# %% Loading annotations
metadata = pd.read_csv(os.path.join(DATA_PATH, "test_metadata.csv"), index_col=0)
print(metadata)


# %%
# In this example, the metadata data contains two columns, ``condition`` and ``group``,
# representing two types of bi-level annotations. Here, we will only use the
# ``condition`` factor.

# %%
# Data filtering
# ^^^^^^^^^^^^^^
#
# Before proceeding with DEA, we start by preprocessing the data, as in the
# :doc:`getting started example <plot_minimal_pydeseq2_pipeline>`.

samples_to_keep = ~metadata.condition.isna()
counts_df = counts_df.loc[samples_to_keep]
metadata = metadata.loc[samples_to_keep]

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
# As in the :doc:`getting started example <plot_minimal_pydeseq2_pipeline>`, we ignore
# the ``group`` variable and use the ``condition`` column as our design factor.
#

# %%
# .. currentmodule:: pydeseq2.dds
#
# Read counts modeling with the :class:`DeseqDataSet` class
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We start by creating a :class:`DeseqDataSet`
# object from the count and metadata data that were just loaded.
#

inference = DefaultInference(n_cpus=8)
dds = DeseqDataSet(
    counts=counts_df,
    metadata=metadata,
    design="~condition",
    refit_cooks=True,
    inference=inference,
)

# %%
# Once a :class:`DeseqDataSet` was initialized,
# we may run the :meth:`deseq2() <DeseqDataSet.deseq2>` method
# to fit dispersions and LFCs.


dds.deseq2()


# %%
# The :class:`DeseqDataSet` class extends the
# :class:`AnnData <anndata.AnnData>`
# class.

print(dds)


# %%
# After removing unpicklable DeseqDataSet attributes, we can save the corresponding
# AnnData object. This can be done using the
# :meth:`to_picklable_anndata() <DeseqDataSet.to_picklable_anndata>` method.
with open(os.path.join(OUTPUT_PATH, "result_adata.pkl"), "wb") as f:
    pkl.dump(dds.to_picklable_anndata(), f)


# %%
# Parameters are stored according to the :class:`AnnData <anndata.AnnData>` data
# structure, with key-based data fields. In particular,
#
# - ``X`` stores the count data,
# - ``obs`` stores 1D sample-level data, such as design factors and ``"size_factors"``,
# - ``obsm`` stores multi-dimensional sample-level data, such as ``"design_matrix"``,
# - ``var`` stores 1D gene-level data, such as gene names and ``"dispersions"``,
# - ``varm`` stores multi-dimensional gene-level data, such as ``"LFC"``.
#
# As an example, here is how we would access dispersions and LFCs
# (in natural log scale):

# %%

print(dds.var["dispersions"])

# %%

print(dds.varm["LFC"])

# %%
# .. currentmodule:: pydeseq2
#
# Statistical analysis with the :class:`DeseqStats <ds.DeseqStats>` class
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Now that dispersions and LFCs were fitted, we may proceed with statistical tests to
# compute p-values and adjusted p-values for differential expresion. This is the role of
# the :class:`DeseqStats <ds.DeseqStats>` class.

ds = DeseqStats(dds, contrast=["condition", "B", "A"], inference=inference)

# %%
# PyDESeq2 computes p-values using Wald tests. This can be done using the
# :meth:`summary() <ds.DeseqStats.summary>` method, which runs the whole statistical
# analysis, cooks filtering and multiple testing adjustement included.

ds.summary()

# %%
# The results are then stored in the ``results_df`` attribute (``stat_res.results_df``).
# As with as :class:`DeseqDataSet <dds.DeseqDataSet>`, the whole
# :class:`DeseqStats <ds.DeseqStats>` object may be saved using pickle.
# However, it is often more convenient to have the results as a CSV.
# Hence, we may export ``stat_res.results_df`` as CSV, using `pandas.DataFrame.to_csv()
# <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html>`_.

ds.results_df.to_csv(os.path.join(OUTPUT_PATH, "results.csv"))
