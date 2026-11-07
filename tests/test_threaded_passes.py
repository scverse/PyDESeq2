"""Outputs with ``n_cpus`` above 1 against the same calls with ``n_cpus=1``.

From 2**18 count entries (samples x genes) and more than 1,024 genes up,
``DeseqDataSet`` with ``n_cpus`` above 1 runs its passes over the count matrix (size
factors, method-of-moments dispersions, Cook's distances) on threads. With
``n_cpus=1`` none of this runs. Each test builds an input just above that size and
checks that every output of the public calls equals the ``n_cpus=1`` call's, NaN
positions included.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
from tests import helpers

pytestmark = helpers.PANDAS_DEPRECATION_FILTERS

# 24 samples x 11,200 genes: the 11,050 genes that are not all zero hold 265,200
# count entries, just above 2**18 = 262,144.
N_SAMPLES = 24
N_GENES = 11_200
N_CPUS = 2


def simulate(seed, *, continuous=(), n_outliers=300):
    """Return simulated counts and metadata of ``N_SAMPLES`` x ``N_GENES``.

    Two groups of 12 samples, so outliers can be replaced (``min_replicates=7``),
    with all-zero genes, genes positive in one sample, genes with one outlier count
    and genes with counts near 1e7.
    """
    n_all_zero, n_single_sample, n_large = 150, 30, 20
    sim = helpers.simulate_counts(
        n_genes=N_GENES - n_all_zero - n_single_sample - n_large,
        n_samples=N_SAMPLES,
        seed=seed,
        continuous=continuous,
        disp_scatter=0.3,
        n_all_zero=n_all_zero,
        n_single_sample=n_single_sample,
        n_large=n_large,
        n_outliers=n_outliers,
    )
    assert (sim.counts.to_numpy() != 0).any(axis=0).sum() * N_SAMPLES >= 2**18
    return sim.counts, sim.metadata, sim.design


def assert_identical(actual, expected, name):
    """Check that two outputs hold the same values, dtypes and labels."""
    if isinstance(expected, pd.DataFrame):
        assert isinstance(actual, pd.DataFrame), name
        pd.testing.assert_index_equal(actual.index, expected.index)
        pd.testing.assert_index_equal(actual.columns, expected.columns)
        for i in range(expected.shape[1]):
            assert_identical(
                actual.iloc[:, i], expected.iloc[:, i], f"{name}[{expected.columns[i]}]"
            )
    elif isinstance(expected, pd.Series):
        assert isinstance(actual, pd.Series), name
        pd.testing.assert_index_equal(actual.index, expected.index)
        assert actual.dtype == expected.dtype, name
        if isinstance(expected.dtype, np.dtype):
            np.testing.assert_array_equal(
                actual.to_numpy(), expected.to_numpy(), err_msg=name
            )
        else:
            missing = expected.isna().to_numpy()
            np.testing.assert_array_equal(
                actual.isna().to_numpy(), missing, err_msg=name
            )
            np.testing.assert_array_equal(
                actual.to_numpy()[~missing], expected.to_numpy()[~missing], err_msg=name
            )
    else:
        actual, expected = np.asarray(actual), np.asarray(expected)
        assert actual.dtype == expected.dtype, name
        np.testing.assert_array_equal(actual, expected, err_msg=name)


def assert_same_dds(dds, ref):
    """Check every ``obs``, ``var``, ``obsm``, ``varm``, ``layers`` and ``uns`` entry."""
    assert_identical(dds.obs, ref.obs, "obs")
    assert_identical(dds.var, ref.var, "var")
    for name in ("obsm", "varm", "layers", "uns"):
        got, want = getattr(dds, name), getattr(ref, name)
        assert sorted(map(str, got.keys())) == sorted(map(str, want.keys())), name
        for key in want.keys():
            assert_identical(got[key], want[key], f"{name}[{key}]")


def fit(counts, metadata, design, n_cpus, steps=False, **kwargs):
    """Fit a ``DeseqDataSet``, with ``deseq2()`` or one public step at a time."""
    dds = DeseqDataSet(
        counts=counts,
        metadata=metadata,
        design=design,
        n_cpus=n_cpus,
        quiet=True,
        **kwargs,
    )
    if not steps:
        dds.deseq2()
        return dds
    dds.fit_size_factors()
    dds.fit_genewise_dispersions()
    dds.fit_dispersion_trend()
    dds.fit_dispersion_prior()
    dds.fit_MAP_dispersions()
    dds.fit_LFC()
    dds.calculate_cooks()
    if dds.refit_cooks:
        dds.refit()
    dds.cooks_outlier()
    return dds


def stats(dds, contrast, coeff, n_cpus):
    """Return the results before and after ``lfc_shrink(coeff)``."""
    ds = DeseqStats(dds, contrast=contrast, n_cpus=n_cpus, quiet=True)
    ds.summary()
    results = ds.results_df.copy()
    with warnings.catch_warnings():
        # The shrinkage objective overflows exp for a few simulated genes, in this
        # process with n_cpus=1 and in the joblib workers otherwise.
        warnings.filterwarnings("ignore", "overflow encountered in exp", RuntimeWarning)
        ds.lfc_shrink(coeff=coeff)
    return results, ds.results_df


@pytest.mark.parametrize("steps", [False, True], ids=["deseq2", "steps"])
def test_two_groups(steps):
    """Two groups: refit of replaced outliers, and the two-column Wald test."""
    counts, metadata, design = simulate(seed=1)
    contrast = ["condition", "B", "A"]
    coeff = "condition[T.B]"
    ref = fit(counts, metadata, design, n_cpus=1, steps=steps)
    assert ref.var["refitted"].sum() > 0
    dds = fit(counts, metadata, design, n_cpus=N_CPUS, steps=steps)
    assert_same_dds(dds, ref)
    for got, want in zip(
        stats(dds, contrast, coeff, n_cpus=N_CPUS),
        stats(ref, contrast, coeff, n_cpus=1),
        strict=True,
    ):
        assert_identical(got, want, "results_df")


def test_continuous_covariate():
    """A continuous covariate: IRLS means, trimmed variances, the Wald test's means."""
    counts, metadata, design = simulate(seed=2, continuous=("x",))
    contrast = ["condition", "B", "A"]
    coeff = "x"
    ref = fit(counts, metadata, design, n_cpus=1)
    dds = fit(counts, metadata, design, n_cpus=N_CPUS)
    assert_same_dds(dds, ref)
    for got, want in zip(
        stats(dds, contrast, coeff, n_cpus=N_CPUS),
        stats(ref, contrast, coeff, n_cpus=1),
        strict=True,
    ):
        assert_identical(got, want, "results_df")


def test_many_samples():
    """260 samples x 1,100 genes: two chunks of genes, many samples per group."""
    sim = helpers.simulate_counts(
        n_genes=1_090, n_samples=260, seed=7, n_all_zero=10, n_outliers=50
    )
    assert sim.counts.size >= 2**18
    assert (sim.counts.to_numpy() != 0).any(axis=0).sum() > 1_024
    contrast = ["condition", "B", "A"]
    coeff = "condition[T.B]"
    ref = fit(sim.counts, sim.metadata, sim.design, n_cpus=1)
    dds = fit(sim.counts, sim.metadata, sim.design, n_cpus=N_CPUS)
    assert_same_dds(dds, ref)
    for got, want in zip(
        stats(dds, contrast, coeff, n_cpus=N_CPUS),
        stats(ref, contrast, coeff, n_cpus=1),
        strict=True,
    ):
        assert_identical(got, want, "results_df")


@pytest.mark.parametrize("use_design", [False, True])
def test_vst(use_design):
    """The variance-stabilizing transform and its fit."""
    counts, metadata, design = simulate(seed=3, n_outliers=0)
    result = {}
    for n_cpus in (1, N_CPUS):
        dds = DeseqDataSet(
            counts=counts, metadata=metadata, design=design, n_cpus=n_cpus, quiet=True
        )
        dds.vst(use_design=use_design)
        result[n_cpus] = dds
    assert_same_dds(result[N_CPUS], result[1])


def test_duplicated_gene_names():
    """Duplicated gene names, through the steps that accept them."""
    counts, metadata, design = simulate(seed=5)
    counts.columns = [f"gene{i % (N_GENES // 2)}" for i in range(N_GENES)]
    result = {}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Variable names are not unique")
        for n_cpus in (1, N_CPUS):
            dds = DeseqDataSet(
                counts=counts,
                metadata=metadata,
                design=design,
                n_cpus=n_cpus,
                quiet=True,
            )
            dds.fit_size_factors()
            dds.fit_genewise_dispersions()
            result[n_cpus] = dds
    assert_same_dds(result[N_CPUS], result[1])


class Float32Dispersions(DefaultInference):
    """Return the dispersions of :meth:`DefaultInference.alpha_mle` as float32."""

    def alpha_mle(self, *args, **kwargs):
        """Fit the dispersions and return them as float32."""
        dispersions, converged = super().alpha_mle(*args, **kwargs)
        return dispersions.astype(np.float32), converged


def test_custom_inference_float32_dispersions():
    """A custom inference that returns float32 dispersions.

    The fits run as separate calls and their dispersions are written into ``var``
    by label.
    """
    counts, metadata, design = simulate(seed=6)
    contrast = ["condition", "B", "A"]
    coeff = "condition[T.B]"
    result = {}
    for n_cpus in (1, N_CPUS):
        dds = fit(
            counts, metadata, design, n_cpus, inference=Float32Dispersions(n_cpus=n_cpus)
        )
        result[n_cpus] = dds, stats(dds, contrast, coeff, n_cpus)
    assert result[1][0].var["refitted"].sum() > 0
    assert_same_dds(result[N_CPUS][0], result[1][0])
    for got, want in zip(result[N_CPUS][1], result[1][1], strict=True):
        assert_identical(got, want, "results_df")
