"""Write the input counts of the area A fixtures (tests/data/unit_a/).

Run from the repository root: python tests/data/unit_a/make_counts.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tests import helpers

out = Path(__file__).resolve().parent

# vst: two conditions, DE genes, two all-zero and two single-sample genes, columns
# shuffled so that the special genes sit between regular ones.
sim = helpers.simulate_counts(
    n_genes=150, n_samples=10, seed=21, disp_scatter=0.3, n_all_zero=2, n_single_sample=2
)
order = np.random.RandomState(21).permutation(sim.counts.shape[1])
sim.counts.iloc[:, order].to_csv(out / "vst_counts.csv")
sim.metadata.to_csv(out / "vst_metadata.csv")

# sparse: low means so that most genes have zeros; every gene gets at least one zero
# (its smallest count is set to zero), one gene is all zero. Genes whose positive
# counts are all 1 are dropped: PyDESeq2's poscounts excludes them (it keeps genes
# with logmeans > 0) and DESeq2 does not, so the two would not be comparable.
sim = helpers.simulate_counts(
    n_genes=120,
    n_samples=8,
    seed=22,
    mean_range=(2.0, 200.0),
    frac_de=0.0,
    disp_scatter=0.3,
    size_factor_sd=0.4,
    n_all_zero=1,
)
counts = sim.counts.copy()
for gene in counts.columns:
    y = counts[gene]
    if (y > 0).all():
        counts.loc[y.idxmin(), gene] = 0
positive = counts.where(counts > 0)
counts = counts.loc[:, ~(positive.max() == 1)]
order = np.random.RandomState(22).permutation(counts.shape[1])
counts = counts.iloc[:, order]
counts.to_csv(out / "sparse_counts.csv")
sim.metadata.to_csv(out / "sparse_metadata.csv")
