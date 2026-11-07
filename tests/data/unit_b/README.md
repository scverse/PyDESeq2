## Dispersion fixtures from DESeq2 1.50.2

Used by `tests/test_unit_dispersions_dds.py::test_trend_prior_and_outliers_match_deseq2`.
Each case directory holds the input counts (`counts.csv`, genes x samples), the sample
metadata (`metadata.csv`) and DESeq2's output for them, written by `generate.R`:

- `r_dispersions.csv`: per gene, `dispGeneEst`, `dispGeneIter`, `dispFit`, `dispMAP`,
  `dispIter`, `dispOutlier` and `dispersion` from `estimateDispersions(dds,
  fitType = "parametric")` after `estimateSizeFactors(dds)`;
- `r_dispersion_fit.csv`: the size factors, the trend coefficients (`a0` is
  `asymptDisp`, `a1` is `extraPois`), `varLogDispEsts` and `dispPriorVar`.

The first line of each output file names the DESeq2 and R versions (DESeq2 1.50.2, R
4.5.3, from bioconda; see `tests/README_NEW_TESTS.md` for the environment).

| Case | Design | Samples | Genes |
|---|---|---|---|
| `two_group` | `~condition` | 12 | 150 simulated, 2 all zero, 3 with high dispersion |
| `continuous` | `~condition + age` | 16 | 120 simulated, 3 with high dispersion |

The counts come from `tests.helpers.simulate_counts`; the three `high` genes are drawn
with the same seed and samples (so the same size factors) and dispersion
`1 + 4 / mean`, so that the cases contain dispersion outliers:

```python
import pandas as pd
from tests import helpers

cases = {
    "two_group": dict(n_genes=150, n_samples=12, seed=31, disp_scatter=0.5, n_all_zero=2),
    "continuous": dict(n_genes=120, n_samples=16, seed=32, disp_scatter=0.5, continuous=("age",)),
}
for name, kw in cases.items():
    sim = helpers.simulate_counts(**kw)
    shared = {k: v for k, v in kw.items() if k not in ("n_genes", "n_all_zero", "disp_scatter")}
    high = helpers.simulate_counts(n_genes=3, trend=(1.0, 4.0), mean_range=(200, 2000), **shared)
    high_counts = high.counts.rename(columns=lambda c: c.replace("gene", "high"))
    pd.concat([sim.counts, high_counts], axis=1).T.to_csv(f"{name}/counts.csv")
    metadata = sim.metadata.copy()
    if "age" in metadata:
        metadata["age"] = metadata["age"].round(6)
    metadata.to_csv(f"{name}/metadata.csv")
```

Then, in each case directory:

```bash
Rscript ../generate.R "~condition"            # two_group
Rscript ../generate.R "~condition + age"      # continuous
```
