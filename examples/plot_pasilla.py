"""
The pasilla dataset
===================

This tutorial runs a differential expression analysis on the pasilla RNA-seq dataset of `Brooks et al. (2011) <https://doi.org/10.1101/gr.108662.110>`_, which the DESeq2 vignette also uses.
The experiment knocks down the *pasilla* gene in *Drosophila melanogaster* cells, and compares 3 treated samples to 4 untreated ones.
"""

import pandas as pd

from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

# %%
# Data loading
# ------------
#
# The counts and the sample annotations are distributed with the `pasilla Bioconductor package <https://bioconductor.org/packages/pasilla/>`_.
# We read them from a fixed commit of its GitHub mirror.

DATA_PATH = "https://raw.githubusercontent.com/bioc/pasilla/3c91e8df8add3549129e46b7a7a078fda4e2755a/inst/extdata/"

counts_df = pd.read_csv(DATA_PATH + "pasilla_gene_counts.tsv", sep="\t", index_col="gene_id").T
metadata = pd.read_csv(DATA_PATH + "pasilla_sample_annotation.csv", index_col="file")

# %%
# The annotation names the samples with an ``fb`` suffix that the counts do not have.
# Besides the treatment, it records whether each library was sequenced with single-end or paired-end reads, which we keep as a second factor.
# Making ``condition`` an ordered categorical sets ``untreated`` as its reference level.

metadata.index = metadata.index.str.removesuffix("fb")
metadata = metadata[["condition", "type"]]
metadata["type"] = metadata["type"].map({"single-read": "single", "paired-end": "paired"})
metadata["condition"] = pd.Categorical(metadata["condition"], categories=["untreated", "treated"])
counts_df = counts_df.loc[metadata.index]
metadata

# %%
# Genes with fewer than 10 reads in total carry little information and are removed beforehand, as in the DESeq2 vignette.

counts_df = counts_df.loc[:, counts_df.sum() >= 10]
counts_df.shape

# %%
# Model fitting
# -------------
#
# The design includes the library type, so that the effect of the treatment is estimated within samples of the same type.

dds = DeseqDataSet(counts=counts_df, metadata=metadata, design="~type + condition", quiet=True)
dds.deseq2()

# %%
# Statistical analysis
# --------------------
#
# We compare treated to untreated samples.

ds = DeseqStats(dds, contrast=["condition", "treated", "untreated"], quiet=True)
ds.summary()

# %%
# Around 1100 genes are differentially expressed at an adjusted p-value of 0.05, close to the 1086 genes that DESeq2 finds on the same data.

res = ds.results_df
res[res.padj < 0.05].sort_values("padj").head(10)

# %%
# LFC shrinkage
# -------------
#
# The LFCs of lowly expressed genes are noisy.
# Shrinking them with :meth:`lfc_shrink() <pydeseq2.ds.DeseqStats.lfc_shrink>` makes them easier to rank and to plot.

ds.lfc_shrink(coeff="condition[T.treated]")
ds.plot_MA(s=2)
