# Transcript-length normalization

For unscaled estimated gene counts imported from transcript-level quantifiers (`countsFromAbundance="no"` in tximport), pass the matching average transcript lengths alongside the counts.
Both matrices must use samples as rows and genes as columns.
When using data frames, their labels and ordering must be identical; arrays must already use the same ordering:

```python
dds = DeseqDataSet(
    counts=estimated_counts,
    metadata=metadata,
    transcript_lengths=average_transcript_lengths,
    design="~condition",
)
dds.deseq2()
```

[`pytximport`](https://github.com/complextissue/pytximport), described by {cite:t}`kuehl2024pytximport`, can produce a compatible AnnData object directly:

```python
from pydeseq2.dds import DeseqDataSet
from pytximport import tximport

sample_names = ["sample_1", "sample_2"]
txi = tximport(
    [f"{sample}/quant.sf" for sample in sample_names],
    data_type="salmon",
    transcript_gene_map=transcript_gene_map,
    counts_from_abundance=None,
    output_type="anndata",
)
txi.obs = metadata.loc[sample_names].copy()
dds = DeseqDataSet(adata=txi, transcript_lengths=txi.obsm["length"], design="~condition")
dds.deseq2()
```

Length correction is opt-in: stored lengths alone do not activate it.
Use only unscaled estimated counts (`counts_from_abundance=None`) with `transcript_lengths`; scaled counts must not receive a second length correction.
Without explicit lengths, the usual integer-count requirements apply.
The file list and metadata rows must use the same sample order.
For backed AnnData, call `adata.to_memory()` first.
AnnData does not align the gene dimension of `obsm["length"]` when subsetting genes; apply the same selection and order to the lengths before passing them explicitly.
Constructing a new dataset from length-normalized AnnData starts a fresh fit; pass lengths again to retain length correction.

Following the [`tximport`](https://bioconductor.org/packages/release/bioc/html/tximport.html) and [`DESeq2`](https://bioconductor.org/packages/release/bioc/html/DESeq2.html) workflow {cite:p}`love2014moderated,soneson2015tximport`, PyDESeq2 rounds estimated counts and combines transcript-length offsets with median-of-ratios library-size normalization.
Average lengths and the resulting sample-by-gene factors are available in `dds.layers["avg_tx_length"]` and `dds.layers["normalization_factors"]`.
The `ratio` and `poscounts` size-factor methods are supported; `iterative` is not currently compatible with transcript-length offsets.
