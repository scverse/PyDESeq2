## Fixtures for `tests/test_unit_data*.py`

Inputs, written by `make_counts.py` (run from the repository root) from
`tests.helpers.simulate_counts`:

- `vst_counts.csv`, `vst_metadata.csv`: 10 samples x 154 genes, `~condition`, with two
  all-zero and two single-sample genes among the others.
- `sparse_counts.csv`, `sparse_metadata.csv`: 8 samples x 120 genes, every gene with
  at least one zero count, one gene all zero.

DESeq2 outputs, written by `make_fixtures.R` (run from this folder) with R 4.5.3 and
DESeq2 1.50.2 (bioconda):

- `vst_blind_parametric.csv`, `vst_blind_mean.csv`: `varianceStabilizingTransformation`
  with `blind = TRUE` and `fitType` parametric or mean.
- `vst_design_parametric.csv`: the same with `blind = FALSE` after `estimateDispersions`.
- `sparse_size_factors.csv`: `estimateSizeFactors` with `type = "poscounts"` and
  `type = "iterate"`.

Count matrices are samples x genes, as `DeseqDataSet` takes them.
