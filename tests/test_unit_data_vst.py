"""Area A: the variance stabilizing transformation (``DeseqDataSet.vst``,
``vst_fit``, ``vst_transform``).

Expected values are built step by step outside PyDESeq2: median-of-ratios size
factors, gene-wise dispersions from the reference minimizer in ``helpers``, DESeq2's
parametric trend fit, and the VST from the integral of the inverse standard deviation.
DESeq2 1.50.2's output on the same counts is checked at a looser tolerance.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pydeseq2.default_inference import DefaultInference
from tests import helpers

pytestmark = helpers.PANDAS_DEPRECATION_FILTERS

DATA = Path(__file__).parent / "data" / "unit_a"
MIN_DISP = 1e-8

# fit_alpha_mle stops at a projected gradient of 1e-5 in log dispersion: up to 9.5e-5
# relative error against the reference minimizer (README_NEW_TESTS.md); 10x margin.
DISPERSION_RTOL = 1e-3
# PyDESeq2 fits the trend by L-BFGS-B on the gamma loss at default tolerances. Against
# the Newton fit below, on the same gene-wise dispersions, the coefficients agreed to
# 3.7e-4 relative over ten fits (five datasets, with and without the design); 2e-3
# leaves a 5x margin.
TREND_RTOL = 2e-3
# The VST is a closed form of the normalized counts and two coefficients; the
# reference below is a different closed form of the same function, so the two agree
# to rounding.
FORMULA_RTOL = 1e-10
# When L-BFGS-B reports failure on a gene-wise fit, fit_alpha_mle returns
# grid_fit_alpha's value instead, which is within half a step of its fine grid of the
# minimizer (up to 2.1e-3 relative at max_disp 10). L-BFGS-B reports failure (an
# abnormal line search) on a few genes whose fit ends at its gradient tolerance, and
# which genes do moves with rounding: with their means changed by 1e-11 relative, one
# to four genes per dataset below failed in some of 200 fits (OpenBLAS Haswell and
# SkylakeX kernels), and gene30 of the continuous-covariate data fails on some Windows
# CI runners. Such genes are compared at one full fine step, which covers the half
# step and the reference's own error, and at most this many per comparison.
MAX_GENEWISE_FALLBACKS = 2


def grid_fallback_rtol(max_disp, min_disp=MIN_DISP, grid_length=100):
    """One step of ``grid_fit_alpha``'s fine grid, as a relative tolerance.

    The coarse grid has ``grid_length`` points from ``log(min_disp)`` to
    ``log(max_disp)``; the fine grid spans the two coarse cells around the coarse
    minimum with ``grid_length`` points, so for a unimodal objective its minimum is
    within half a fine step of the minimizer.
    """
    delta = np.log(max_disp / min_disp) / (grid_length - 1)
    return np.expm1(2 * delta / (grid_length - 1))


def assert_genewise_close(actual, expected, *, rtol, max_disp):
    """Gene-wise dispersions within ``rtol``, except at most
    ``MAX_GENEWISE_FALLBACKS`` genes, which are within :func:`grid_fallback_rtol`
    (genes that took the grid fallback; see ``MAX_GENEWISE_FALLBACKS``)."""
    rel = np.abs(actual.to_numpy() / expected.to_numpy() - 1)
    fallback = (rel > rtol) & (rel <= grid_fallback_rtol(max_disp))
    assert fallback.sum() <= MAX_GENEWISE_FALLBACKS, list(actual.index[fallback])
    helpers.assert_fields_close(
        actual[~fallback],
        expected[~fallback],
        rtol=rtol,
        name="vst_genewise_dispersions",
    )


def vst_from_variance(normed_counts, a0, a1):
    """VST for the variance function ``v(q) = q + (a0 + a1 / q) q**2``.

    DESeq2 defines the transformation by ``d vst / dq = sqrt(a0) / (log(2) sqrt(v))``
    with ``vst ~ log2(q)`` for large ``q``. With ``q = s**2`` the integrand becomes
    ``2 sqrt(a0) / (log(2) sqrt(1 + a1 + a0 s**2))``, whose integral is
    ``2 asinh(s sqrt(a0 / (1 + a1))) / log(2)``; the constant
    ``log2((1 + a1) / (4 a0))`` gives the large-``q`` asymptote. The mean trend
    (constant dispersion ``d``) is the case ``a0 = d``, ``a1 = 0``.
    """
    q = np.asarray(normed_counts, dtype=float)
    c = 1.0 + a1
    return (np.log(c / (4 * a0)) + 2 * np.arcsinh(np.sqrt(a0 * q / c))) / np.log(2)


def reference_genewise(counts, size_factors, groups):
    """Gene-wise dispersions for a one-factor design (``groups``), or intercept-only.

    The means are the group means of the normalized counts times the size factor,
    floored at ``min_mu=0.5`` (a linear model on the group indicators), and the
    dispersion is the reference minimizer of the Cox-Reid adjusted likelihood, over
    ``[min_disp, max(10, n_samples)]``. Returns the dispersions and a mask of genes
    that are all zero or whose starting value (DESeq2's minimum of the rough and
    moments estimates) is at ``min_disp``; PyDESeq2, like DESeq2, leaves the latter at
    ``min_disp`` whatever the minimizer (a known issue), so they are not compared.
    """
    y = np.asarray(counts, dtype=float)
    n = y.shape[0]
    groups = np.zeros(n, dtype=int) if groups is None else np.asarray(groups)
    levels = np.unique(groups)
    x = (groups[:, None] == levels[None, :]).astype(float)
    normed = y / size_factors[:, None]
    mu = helpers.lin_reg_mu(y, size_factors, x)
    rough = helpers.rough_dispersions(normed, x)
    with np.errstate(divide="ignore", invalid="ignore"):
        moments = helpers.moments_dispersions(normed, size_factors)
    all_zero = (y == 0).all(axis=0)
    skip = all_zero | ~(np.minimum(rough, np.nan_to_num(moments)) > MIN_DISP)

    dispersions = np.full(y.shape[1], np.nan)
    for j in np.flatnonzero(~skip):
        dispersions[j] = helpers.reference_dispersion_mle(
            y[:, j], mu[:, j], x, min_disp=MIN_DISP, max_disp=max(10.0, n)
        )
    return dispersions, skip


def read_fixture():
    counts = pd.read_csv(DATA / "vst_counts.csv", index_col=0)
    metadata = pd.read_csv(DATA / "vst_metadata.csv", index_col=0)
    return counts, metadata


def check_genewise(dds, counts, groups):
    """Compare ``var["vst_genewise_dispersions"]`` with :func:`reference_genewise`."""
    sf = helpers.median_of_ratios(counts)
    expected, skip = reference_genewise(counts.to_numpy(), sf, groups)
    actual = dds.var["vst_genewise_dispersions"]
    all_zero = (counts == 0).all(axis=0).to_numpy()
    np.testing.assert_array_equal(actual.isna().to_numpy(), all_zero)
    assert (~skip).sum() >= 0.8 * len(skip)
    assert_genewise_close(
        actual[~skip],
        pd.Series(expected[~skip], index=actual.index[~skip]),
        rtol=DISPERSION_RTOL,
        max_disp=max(10.0, len(counts)),
    )


def check_trend(dds, counts):
    """``uns["vst_trend_coeffs"]`` is DESeq2's trend fit of the gene-wise dispersions."""
    sf = helpers.median_of_ratios(counts)
    disps = dds.var["vst_genewise_dispersions"].to_numpy()
    means = (counts.to_numpy() / sf[:, None]).mean(axis=0)
    fitted = ~np.isnan(disps)
    expected, _ = helpers.reference_trend(
        means[fitted], disps[fitted], min_disp=MIN_DISP
    )
    np.testing.assert_allclose(
        helpers.trend_coeffs(dds.uns["vst_trend_coeffs"]).to_numpy(),
        expected,
        rtol=TREND_RTOL,
    )


@pytest.mark.parametrize("use_design", [False, True], ids=["blind", "design"])
def test_vst_parametric_matches_reference(use_design):
    """Size factors, gene-wise dispersions, trend and VST, each against its reference.

    Blind, the dispersions are fitted with an intercept-only design; with the design,
    with ``~condition``. The design matrix is the original one afterwards.
    """
    counts, metadata = read_fixture()
    dds = helpers.make_dds(counts, metadata)
    design = pd.DataFrame(dds.obsm["design_matrix"]).copy()
    dds.vst(use_design=use_design)

    sf = helpers.median_of_ratios(counts)
    np.testing.assert_allclose(dds.obs["size_factors"], sf, rtol=1e-12)
    check_genewise(dds, counts, metadata["condition"] if use_design else None)
    check_trend(dds, counts)

    a0, a1 = helpers.trend_coeffs(dds.uns["vst_trend_coeffs"])
    np.testing.assert_allclose(
        dds.layers["vst_counts"],
        vst_from_variance(counts.to_numpy() / sf[:, None], a0, a1),
        rtol=FORMULA_RTOL,
    )
    pd.testing.assert_frame_equal(pd.DataFrame(dds.obsm["design_matrix"]), design)


@pytest.mark.parametrize("use_design", [False, True], ids=["blind", "design"])
def test_vst_mean_fit_matches_reference(use_design):
    """Mean trend: the VST of a constant dispersion, the trimmed mean (0.1% from each
    end) of the gene-wise dispersions above ``10 * min_disp``. With the design, the
    dispersions use ``~condition`` and a warning says the design only matters for the
    parametric fit."""
    counts, metadata = read_fixture()
    dds = helpers.make_dds(counts, metadata)
    if use_design:
        with pytest.warns(UserWarning, match="use_design=True is only useful"):
            dds.vst(use_design=True, fit_type="mean")
    else:
        dds.vst(use_design=False, fit_type="mean")

    check_genewise(dds, counts, metadata["condition"] if use_design else None)
    disps = dds.var["vst_genewise_dispersions"].to_numpy()
    disps = np.sort(disps[disps > 10 * MIN_DISP])
    k = int(0.001 * len(disps))
    mean_disp = disps[k : len(disps) - k].mean()
    sf = helpers.median_of_ratios(counts)
    np.testing.assert_allclose(
        dds.layers["vst_counts"],
        vst_from_variance(counts.to_numpy() / sf[:, None], mean_disp, 0.0),
        rtol=FORMULA_RTOL,
    )


def test_vst_mean_fit_leaves_out_dispersions_at_min_disp():
    """Genes whose counts follow the size factors closely (less variance than a
    Poisson) have their dispersion minimizer at ``min_disp``; the mean trend leaves
    them out, as DESeq2's ``dispGeneEst > 10 * minDisp`` does."""
    sim = helpers.simulate_counts(n_genes=60, n_samples=8, seed=303)
    flat = np.rint(np.outer(sim.size_factors, [20.0, 50.0, 100.0])).astype(int)
    counts = pd.concat(
        [
            sim.counts,
            pd.DataFrame(
                flat, index=sim.counts.index, columns=["flat1", "flat2", "flat3"]
            ),
        ],
        axis=1,
    )
    dds = helpers.make_dds(counts, sim.metadata)
    dds.vst(use_design=False, fit_type="mean")

    disps = dds.var["vst_genewise_dispersions"]
    low = disps <= 10 * MIN_DISP
    assert low[["flat1", "flat2", "flat3"]].all()
    mean_disp = disps[~low].mean()
    sf = helpers.median_of_ratios(counts)
    np.testing.assert_allclose(
        dds.layers["vst_counts"],
        vst_from_variance(counts.to_numpy() / sf[:, None], mean_disp, 0.0),
        rtol=FORMULA_RTOL,
    )


# PyDESeq2's gene-wise dispersions reach the minimizer of the Cox-Reid objective
# (DISPERSION_RTOL); DESeq2's line search stops short of it on high-count genes (up to
# 12% in dispersion, 0.03 in objective on this data, checked against the reference
# minimizer), which moves the trend and the VST. Measured differences: 1.8e-3 (blind
# parametric), 4.1e-3 (with the design), 1.9e-3 (mean); 1e-2 leaves a 2.4x margin.
DESEQ2_VST_RTOL = 1e-2


@pytest.mark.parametrize(
    "fixture, kwargs",
    [
        ("vst_blind_parametric.csv", {"use_design": False}),
        ("vst_design_parametric.csv", {"use_design": True}),
        ("vst_blind_mean.csv", {"use_design": False, "fit_type": "mean"}),
    ],
)
def test_vst_matches_deseq2(fixture, kwargs):
    """Against DESeq2 1.50.2's ``varianceStabilizingTransformation`` (blind) and, with
    the design, after ``estimateDispersions`` (``blind = FALSE``)."""
    counts, metadata = read_fixture()
    expected = pd.read_csv(DATA / fixture, index_col=0)
    dds = helpers.make_dds(counts, metadata)
    dds.vst(**kwargs)
    helpers.assert_fields_close(
        pd.DataFrame(
            dds.layers["vst_counts"], index=counts.index, columns=counts.columns
        ),
        expected,
        rtol=DESEQ2_VST_RTOL,
        name="vst_counts",
    )


def test_vst_fit_then_transform_equals_vst():
    """``vst`` is ``vst_fit`` followed by ``vst_transform`` (same computation)."""
    counts, metadata = read_fixture()
    one = helpers.make_dds(counts, metadata)
    one.vst(use_design=False)
    two = helpers.make_dds(counts, metadata)
    two.vst_fit(use_design=False)
    np.testing.assert_allclose(
        two.vst_transform(), one.layers["vst_counts"], rtol=FORMULA_RTOL
    )


def test_vst_after_deseq2_leaves_its_results():
    """A blind VST after ``deseq2`` gives the VST of a fresh dataset and leaves the
    size factors, dispersions, LFCs and design of the fit as they were (the VST
    writes to its own fields; the values are not recomputed, so they agree to
    rounding at most)."""
    sim = helpers.simulate_counts(
        n_genes=60, n_samples=8, seed=304, n_all_zero=1, disp_scatter=0.3
    )
    fresh = helpers.make_dds(sim.counts, sim.metadata)
    fresh.vst(use_design=False)

    dds = helpers.make_dds(sim.counts, sim.metadata)
    dds.deseq2()
    columns = ["genewise_dispersions", "fitted_dispersions", "MAP_dispersions"]
    before = (
        dds.obs["size_factors"].copy(),
        dds.var[[*columns, "dispersions"]].copy(),
        dds.varm["LFC"].copy(),
        pd.DataFrame(dds.obsm["design_matrix"]).copy(),
        helpers.trend_coeffs(dds.uns["trend_coeffs"]),
    )
    dds.vst(use_design=False)

    np.testing.assert_allclose(
        dds.layers["vst_counts"], fresh.layers["vst_counts"], rtol=FORMULA_RTOL
    )
    sf, var, lfc, design, trend = before
    helpers.assert_fields_close(dds.obs["size_factors"], sf, rtol=1e-12, name="sf")
    helpers.assert_fields_close(
        dds.var[[*columns, "dispersions"]], var, rtol=1e-12, name="var"
    )
    helpers.assert_fields_close(dds.varm["LFC"], lfc, rtol=1e-12, name="LFC")
    pd.testing.assert_frame_equal(pd.DataFrame(dds.obsm["design_matrix"]), design)
    pd.testing.assert_series_equal(helpers.trend_coeffs(dds.uns["trend_coeffs"]), trend)


def _train_test():
    """Twelve samples, means of at least 50 so that no test count is zero."""
    sim = helpers.simulate_counts(
        n_genes=80, n_samples=12, seed=301, mean_range=(50.0, 5000.0), disp_scatter=0.3
    )
    train = sim.counts.iloc[:8]
    test = sim.counts.iloc[8:]
    assert (test.to_numpy() > 0).all()
    return train, sim.metadata.iloc[:8], test


@pytest.mark.parametrize("fit_type", ["parametric", "mean"])
def test_vst_transform_new_counts(fit_type):
    """New samples are normalized with the training data's log geometric means and
    transformed with the trained coefficients."""
    train, metadata, test = _train_test()
    dds = helpers.make_dds(train, metadata, fit_type=fit_type)
    with pytest.raises(RuntimeError, match="vst_fit method should be called"):
        dds.vst_transform(test.to_numpy())
    dds.vst_fit(use_design=False)
    result = dds.vst_transform(test.to_numpy())

    log_train = np.log(train.to_numpy(dtype=float))
    usable = np.isfinite(log_train).all(axis=0)
    logmeans = log_train[:, usable].mean(axis=0)
    log_test = np.log(test.to_numpy(dtype=float))[:, usable]
    test_sf = np.exp(np.median(log_test - logmeans, axis=1))
    normed = test.to_numpy() / test_sf[:, None]

    if fit_type == "parametric":
        a0, a1 = helpers.trend_coeffs(dds.uns["vst_trend_coeffs"])
    else:
        disps = dds.var["vst_genewise_dispersions"].to_numpy()
        a0, a1 = disps[disps > 10 * MIN_DISP].mean(), 0.0
    np.testing.assert_allclose(
        result, vst_from_variance(normed, a0, a1), rtol=FORMULA_RTOL
    )


def test_vst_transform_after_iterative_size_factors():
    """Without log geometric means (iterative size factors), the new counts are
    normalized by their own median of ratios, with a warning."""
    train, metadata, test = _train_test()
    dds = helpers.make_dds(train, metadata, size_factors_fit_type="iterative")
    dds.vst_fit(use_design=False)
    with pytest.warns(UserWarning, match="size factors were fitted iteratively"):
        result = dds.vst_transform(test.to_numpy())
    a0, a1 = helpers.trend_coeffs(dds.uns["vst_trend_coeffs"])
    sf = helpers.median_of_ratios(test)
    np.testing.assert_allclose(
        result, vst_from_variance(test.to_numpy() / sf[:, None], a0, a1), rtol=1e-10
    )


# A correct rewrite may fit genes in batches whose composition changes with gene
# order, n_cpus or batch_size; each gene's fit then stops at a slightly different
# point inside the solver tolerance (DISPERSION_RTOL), which moves the trend within
# TREND_RTOL and the VST by less than 1e-3 relative (dVST/da0 is 1/(a0 log 2)).
INVARIANCE_RTOL = {"dispersions": DISPERSION_RTOL, "trend": TREND_RTOL, "vst": 1e-3}


def _vst_outputs(dds):
    return (
        dds.var["vst_genewise_dispersions"],
        helpers.trend_coeffs(dds.uns["vst_trend_coeffs"]).to_numpy(),
        pd.DataFrame(
            dds.layers["vst_counts"], index=dds.obs_names, columns=dds.var_names
        ),
    )


def _assert_vst_outputs_close(actual, expected):
    disp, trend, vst = actual
    disp_e, trend_e, vst_e = expected
    helpers.assert_fields_close(
        disp, disp_e, rtol=INVARIANCE_RTOL["dispersions"], name="dispersions"
    )
    np.testing.assert_allclose(trend, trend_e, rtol=INVARIANCE_RTOL["trend"])
    helpers.assert_fields_close(vst, vst_e, rtol=INVARIANCE_RTOL["vst"], name="vst")


def test_vst_follows_gene_order():
    """Permuting genes permutes every VST output; the all-zero and single-sample genes
    move to other positions."""
    counts, metadata = read_fixture()
    dds = helpers.make_dds(counts, metadata)
    dds.vst(use_design=True)
    expected = _vst_outputs(dds)

    order = np.random.RandomState(3).permutation(counts.shape[1])
    permuted = counts.iloc[:, order]
    dds_p = helpers.make_dds(permuted, metadata)
    dds_p.vst(use_design=True)
    disp, trend, vst = _vst_outputs(dds_p)
    _assert_vst_outputs_close(
        (disp.loc[counts.columns], trend, vst.loc[:, counts.columns]), expected
    )


def test_vst_n_cpus_and_batch_size_invariance():
    """Two workers with batches of 7 genes give the results of one worker with the
    default batches (154 genes: 22 batches, all-zero genes inside batches)."""
    counts, metadata = read_fixture()
    results = []
    for inference in [
        DefaultInference(n_cpus=1),
        DefaultInference(n_cpus=2, batch_size=7),
    ]:
        dds = helpers.make_dds(counts, metadata, inference=inference)
        dds.vst(use_design=True)
        results.append(_vst_outputs(dds))
    _assert_vst_outputs_close(results[1], results[0])


def test_vst_with_continuous_covariate():
    """With a continuous covariate the means come from IRLS (the design has more
    distinct rows than columns). Checked against the reference: the NB GLM fitted
    with the method-of-moments dispersion, then the dispersion MLE given those means.

    IRLS stops at a relative deviance change of 1e-8, which leaves up to 3.4e-4 in
    natural-log coefficients (README_NEW_TESTS.md), and the dispersion MLE moves with
    the means it is given: the dispersions agreed to 1.9e-4 relative here, so they are
    compared at 2e-3, except genes that took the grid fallback
    (``MAX_GENEWISE_FALLBACKS``).
    """
    sim = helpers.simulate_counts(
        n_genes=60, n_samples=10, seed=302, continuous=("age",), disp_scatter=0.3
    )
    dds = helpers.make_dds(sim.counts, sim.metadata, design=sim.design)
    dds.vst(use_design=True)

    y = sim.counts.to_numpy(dtype=float)
    x = sim.design_matrix.to_numpy()
    n, p = x.shape
    sf = helpers.median_of_ratios(sim.counts)
    normed = y / sf[:, None]
    fitted = np.maximum(x @ np.linalg.lstsq(x, normed, rcond=None)[0], 1.0)
    rough = np.maximum(
        (((normed - fitted) ** 2 - fitted) / fitted**2).sum(0) / (n - p), 0
    )
    means = normed.mean(axis=0)
    moments = (normed.var(axis=0, ddof=1) - (1 / sf).mean() * means) / means**2
    start = np.clip(np.minimum(rough, moments), MIN_DISP, max(10.0, n))
    keep = np.minimum(rough, moments) > MIN_DISP
    assert keep.sum() >= 50

    expected = np.full(y.shape[1], np.nan)
    for j in np.flatnonzero(keep):
        beta = helpers.reference_glm_mle(y[:, j], sf, x, start[j])
        mu = np.maximum(sf * np.exp(x @ beta), 0.5)
        expected[j] = helpers.reference_dispersion_mle(
            y[:, j], mu, x, min_disp=MIN_DISP, max_disp=max(10.0, n)
        )
    actual = dds.var["vst_genewise_dispersions"]
    assert_genewise_close(
        actual[keep],
        pd.Series(expected[keep], index=actual.index[keep]),
        rtol=2e-3,
        max_disp=max(10.0, n),
    )
    check_trend(dds, sim.counts)
