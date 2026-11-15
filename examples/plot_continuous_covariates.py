"""
Continuous covariates
=====================

In addition to categorical factors, the design of a PyDESeq2 analysis may include continuous covariates, such as an age, a dose or a quality metric.
This tutorial shows how to add a continuous covariate to the design, and how to test its effect on gene expression.
"""

import numpy as np
import pandas as pd

from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

# %%
# Data loading
# ------------
#
# We use a synthetic dataset of 100 samples and 10 genes, in which the expression of some genes depends on a continuous covariate called ``measurement``.
# As in the :doc:`pandas IO example <plot_pandas_io_example>`, the counts and the metadata are read from CSV files.

DATA_PATH = "https://raw.githubusercontent.com/scverse/PyDESeq2/main/tests/data/continuous/"

counts_df = pd.read_csv(DATA_PATH + "test_counts.csv", index_col=0).T
metadata = pd.read_csv(DATA_PATH + "test_metadata.csv", index_col=0)
metadata

# %%
# Design
# ------
#
# A continuous covariate is added to the design formula like any other factor.
# Numeric columns of the metadata are treated as continuous, and the other columns as categorical.
# Here, we model the counts with the categorical ``condition`` and the continuous ``measurement``.

dds = DeseqDataSet(
    counts=counts_df,
    metadata=metadata,
    design="~condition + measurement",
    quiet=True,
)

# %%
# A continuous covariate takes up a single column of the design matrix, which holds its values.

dds.obsm["design_matrix"]

# %%
# .. note::
#   To treat a numeric column as categorical, e.g. a batch number, wrap it in ``C()`` in the formula, as in ``"~C(batch) + measurement"``.
#
# Dispersions and LFCs are then fitted as usual.

dds.deseq2()

# %%
# Statistical analysis
# --------------------
#
# The effect of a continuous covariate is tested with a contrast vector, which has one entry per column of the design matrix.
# To test ``measurement``, we set its entry to 1 and the other entries to 0.

contrast = np.array([0, 0, 1])
ds = DeseqStats(dds, contrast=contrast, quiet=True)
ds.summary()
ds.results_df

# %%
# The log2 fold change of a continuous covariate is the change in log2 expression when the covariate increases by one unit.
# For instance, the expression of ``gene1`` is multiplied by about :math:`2^{0.9} \approx 1.9` each time ``measurement`` increases by 1.
#
# The categorical factors of the design are tested as in the :doc:`getting started example <plot_minimal_pydeseq2_pipeline>`, and their LFCs are adjusted for the covariate.

ds_condition = DeseqStats(dds, contrast=["condition", "B", "A"], quiet=True)
ds_condition.summary()
ds_condition.results_df

# %%
# LFC shrinkage
# -------------
#
# The LFCs of a continuous covariate are shrunk by passing the name of its column of the design matrix to :meth:`lfc_shrink() <pydeseq2.ds.DeseqStats.lfc_shrink>`.

ds.lfc_shrink(coeff="measurement")
ds.results_df
