"""Area C: GLM fitting, Cook's distances and refit (see tests/TARGETS.md).

Expected values come from the references in ``tests/helpers.py`` (the NB likelihood,
the Newton minimizer of the ridge-penalized GLM objective, the dispersion objective
minimized by grid plus Brent) and from short reimplementations of DESeq2's formulas
below (trimmed means with ``scipy.stats.trim_mean``, the hat diagonal, the robust
method-of-moments dispersion, Cook's distance and the outlier replacement).
"""

import functools
from collections import Counter

import numpy as np
import pandas as pd
import pytest
from scipy import optimize
from scipy import stats

from pydeseq2 import dispersions
from pydeseq2 import distributions
from pydeseq2 import glm
from pydeseq2 import grid_search
from pydeseq2 import stats as deseq_stats
from pydeseq2.default_inference import DefaultInference
from tests import helpers

pytestmark = helpers.PANDAS_DEPRECATION_FILTERS

# IRLS stops when the relative change in deviance falls below beta_tol=1e-8. Against
# the Newton reference that left natural-log coefficient errors up to 2.2e-5 with two
# coefficients and 3.4e-4 with five (README_NEW_TESTS.md); 1e-3 keeps a 3x margin.
LFC_ATOL = 1e-3
# fit_alpha_mle stops L-BFGS-B at a projected gradient of 1e-5 in log dispersion:
# relative errors up to 9.5e-5 against the reference (README); 10x margin.
DISPERSION_RTOL = 1e-3
# Cook's distances recomputed from PyDESeq2's own LFCs and dispersions differ only by
# rounding in the 2-5 dimensional inverses (measured 1.6e-14); 1e-8 leaves room for a
# different but equivalent order of operations.
COOKS_RTOL = 1e-8
MIN_MU = 0.5
RIDGE = 1e-6

# Two groups of 7 (A, odd samples) and 6 (B, even samples): only A reaches the
# default min_replicates=7, so outliers in B are flagged but not replaced. Eight
# genes get one outlier count each; ``single_sample1`` (positive only in sample1, in
# A) becomes all zero after replacement; ``single_sample2`` (positive only in
# sample2, in B) is refitted on unchanged counts.
TWO_GROUP = {
    "n_genes": 60,
    "n_samples": 13,
    "seed": 21,
    "n_outliers": 8,
    "n_all_zero": 1,
    "n_single_sample": 2,
    "n_large": 1,
    "disp_scatter": 0.3,
}
# Two factors and a continuous covariate (five coefficients). Every design row is
# unique, so no cell has three replicates: Cook's distances use trimmed_variance.
CONTINUOUS = {
    "n_genes": 40,
    "n_samples": 18,
    "seed": 22,
    "factors": {"condition": ["A", "B"], "group": ["X", "Y", "Z"]},
    "continuous": ("age",),
    "n_outliers": 3,
    "n_all_zero": 1,
    "disp_scatter": 0.3,
}
# Six cells of sizes 3, 3, 3, 3, 2, 2: the two-sample cells are left out of the
# robust variance but kept in the mean.
MIXED_CELLS = {
    "n_genes": 40,
    "n_samples": 16,
    "seed": 23,
    "factors": {"condition": ["A", "B"], "group": ["X", "Y", "Z"]},
    "n_outliers": 3,
    "disp_scatter": 0.3,
}


# References


def _hat_diagonal(x, mu, disp, min_mu=MIN_MU):
    """Diagonal of ``W^1/2 X (X^T W X + ridge I)^-1 X^T W^1/2``.

    ``W = mu / (1 + disp mu)`` with ``mu`` floored at ``min_mu``, as in DESeq2's
    fitBeta; the ridge is the 1e-6 IRLS adds to every coefficient.
    """
    mu = np.maximum(mu, min_mu)
    w = mu / (1.0 + disp * mu)
    xtwx = (x.T * w) @ x + RIDGE * np.eye(x.shape[1])
    return w * np.einsum("ij,jk,ik->i", x, np.linalg.inv(xtwx), x)


def _trim_and_scale(n_samples):
    """DESeq2's trim fraction and variance scale for a cell of ``n_samples``."""
    if n_samples < 4:
        return 1 / 3, 2.04
    if n_samples < 24:
        return 1 / 4, 1.86
    return 1 / 8, 1.51


def _robust_dispersion_reference(normed_counts, design_matrix):
    """DESeq2's robustMethodOfMomentsDisp, written with ``scipy.stats.trim_mean``.

    Samples are grouped by identical design rows. When some cell has three or more
    samples, the variance is the maximum over those cells of a scaled trimmed
    variance (trim 1/3, 1/4, 1/8 and scale 2.04, 1.86, 1.51 for cells of 3, 4-23 and
    24 or more samples); otherwise it is 1.51 times the 1/8-trimmed variance of all
    samples. The mean is over all samples, and the result is floored at 0.04.
    """
    rows = [tuple(r) for r in np.asarray(design_matrix, dtype=float)]
    size = Counter(rows)
    cells = sorted({r for r in rows if size[r] >= 3})
    if cells:
        variance = np.full(normed_counts.shape[1], -np.inf)
        for cell in cells:
            sub = normed_counts[[r == cell for r in rows]]
            trim, scale = _trim_and_scale(len(sub))
            centre = stats.trim_mean(sub, trim, axis=0)
            cell_var = scale * stats.trim_mean((sub - centre) ** 2, trim, axis=0)
            variance = np.maximum(variance, cell_var)
    else:
        centre = stats.trim_mean(normed_counts, 1 / 8, axis=0)
        variance = 1.51 * stats.trim_mean((normed_counts - centre) ** 2, 1 / 8, axis=0)
    mean = normed_counts.mean(axis=0)
    return np.maximum((variance - mean) / mean**2, 0.04)


def _cooks_reference(dds):
    """Cook's distances from a fitted dataset's public outputs.

    DESeq2's calculateCooksDistance: ``(y - mu)^2 / (mu + a mu^2) / p * h / (1 - h)^2``
    with ``mu = s exp(X beta)`` from ``varm["LFC"]``, ``a`` the robust dispersion and
    ``h`` the hat diagonal at ``var["dispersions"]``. All-zero genes are NaN.
    """
    x = dds.obsm["design_matrix"].to_numpy(dtype=float)
    sf = dds.obs["size_factors"].to_numpy()
    non_zero = dds.var["non_zero"].to_numpy()
    y = np.asarray(dds.X, dtype=float)[:, non_zero]
    beta = dds.varm["LFC"].to_numpy()[non_zero]
    disp = dds.var["dispersions"].to_numpy()[non_zero]
    mu = sf[:, None] * np.exp(x @ beta.T)
    robust = _robust_dispersion_reference(y / sf[:, None], x)
    cooks = np.full(dds.X.shape, np.nan)
    for k, j in enumerate(np.flatnonzero(non_zero)):
        h = _hat_diagonal(x, mu[:, k], disp[k])
        pearson_sq = (y[:, k] - mu[:, k]) ** 2 / (mu[:, k] + robust[k] * mu[:, k] ** 2)
        cooks[:, j] = pearson_sq / x.shape[1] * h / (1 - h) ** 2
    return cooks


def _group_means(values, groups):
    out = np.empty_like(values, dtype=float)
    for level in np.unique(groups):
        out[groups == level] = values[groups == level].mean(axis=0)
    return out


def _starts_at_min_disp(y, sf, groups, n_coefs, min_disp=1e-8):
    """Whether DESeq2's starting dispersion (min of rough and moments) is floored.

    Such genes stay at ``min_disp`` (known issue, README_NEW_TESTS.md), so they are
    left out of comparisons with the reference minimizer.
    """
    normed = y / sf
    fitted = np.maximum(_group_means(normed, groups), 1.0)
    rough = (((normed - fitted) ** 2 - fitted) / fitted**2).sum() / (len(y) - n_coefs)
    moments = (normed.var(ddof=1) - (1 / sf).mean() * normed.mean()) / normed.mean() ** 2
    return min(max(rough, 0.0), moments) <= min_disp


def _lfc_reference_or_none(y, sf, x, disp):
    """``helpers.reference_glm_mle``, or None where IRLS's ``min_mu`` floor changes the
    fit.

    A fitted mean below ``min_mu`` moves IRLS's fixed point away from the minimizer
    (README_NEW_TESTS.md); a factor of 2 margin keeps genes clear of the floor.
    """
    if (y == 0).all() or _has_all_zero_group(y, x):
        return None
    beta = helpers.reference_glm_mle(y, sf, x, disp)
    if (sf * np.exp(x @ beta)).min() < 2 * MIN_MU:
        return None
    return beta


def _has_all_zero_group(y, x):
    rows = [tuple(r) for r in x]
    return any(
        all(y[i] == 0 for i, r in enumerate(rows) if r == cell) for cell in set(rows)
    )


@functools.cache
def _fitted(case, refit_cooks=True, fit_type="parametric", permutation_seed=None, **kw):
    """Run ``deseq2()`` on a simulated case; cached, so tests must not modify it."""
    params = {
        "TWO_GROUP": TWO_GROUP,
        "CONTINUOUS": CONTINUOUS,
        "MIXED_CELLS": MIXED_CELLS,
    }
    sim = helpers.simulate_counts(**params[case])
    counts = sim.counts
    if permutation_seed is not None:
        order = np.random.RandomState(permutation_seed).permutation(counts.shape[1])
        counts = counts.iloc[:, order]
    dds = helpers.make_dds(
        counts,
        sim.metadata,
        design=sim.design,
        refit_cooks=refit_cooks,
        fit_type=fit_type,
        **dict(kw),
    )
    dds.deseq2()
    return sim, dds


def _assert_same_fit(actual, expected, *, name, flags=("replaced", "refitted")):
    """Public outputs of two ``deseq2()`` runs agree, gene by gene (labels aligned).

    A gene whose MAP fit takes the grid fallback in one run only is left out of the
    MAP-dependent comparisons (``helpers.map_fallback_flips``); at most one may.
    """
    flips = helpers.map_fallback_flips(actual, expected, rtol=DISPERSION_RTOL)
    assert len(flips) <= 1, (name, list(flips))
    genes = expected.var_names.difference(flips, sort=False)
    for field in ["genewise_dispersions", "fitted_dispersions"]:
        helpers.assert_fields_close(
            actual.var[field].loc[expected.var_names],
            expected.var[field],
            rtol=DISPERSION_RTOL,
            name=f"{name} {field}",
        )
    for field in ["MAP_dispersions", "dispersions"]:
        helpers.assert_fields_close(
            actual.var[field].loc[genes],
            expected.var[field].loc[genes],
            rtol=DISPERSION_RTOL,
            name=f"{name} {field}",
        )
    helpers.assert_fields_close(
        actual.varm["LFC"].loc[genes],
        expected.varm["LFC"].loc[genes],
        rtol=0,
        atol=LFC_ATOL,
        name=f"{name} LFC",
    )
    for flag in ["non_zero", *flags]:
        assert actual.var[flag].loc[genes].equals(expected.var[flag].loc[genes]), (
            name,
            flag,
        )
    # Cook's distances scale with (y - mu)^2, so an LFC difference d changes them by
    # about 2 d mu / |y - mu| relative: 1e-3 relative for typical residuals, and an
    # absolute 1e-6 for the near-zero distances of near-zero residuals (the
    # replacement cutoff is about 7).
    cooks_actual = pd.DataFrame(
        actual.layers["cooks"], index=actual.obs_names, columns=actual.var_names
    )[genes]
    cooks_expected = pd.DataFrame(
        expected.layers["cooks"], index=expected.obs_names, columns=expected.var_names
    )[genes]
    helpers.assert_fields_close(
        cooks_actual.T,
        cooks_expected.T,
        rtol=1e-3,
        atol=1e-6,
        name=f"{name} cooks",
    )


# distributions.nb_nll


@pytest.mark.parametrize(
    "alpha, rtol",
    [(0.01, 1e-12), (0.2, 1e-12), (1.0, 1e-12), (5.0, 1e-12), (1e-4, 1e-10)],
)
def test_nb_nll_matches_reference(alpha, rtol):
    """Against ``helpers.nb_loglik`` (betaln form, independent of PyDESeq2).

    At alpha=1e-4 nb_nll subtracts gammaln(1e4 + y) - gammaln(1e4) and
    n/alpha log(alpha) from terms of size 1e5: measured 3.2e-11 absolute (6e-13
    relative) on a total of 54, so 1e-10 relative. Elsewhere measured at most
    1.4e-13 relative, so 1e-12.
    """
    y = np.array([0, 1, 3, 12, 40, 250, 3000])
    mu = np.array([0.4, 2.0, 2.5, 10.0, 55.0, 180.0, 2600.0])
    expected = -helpers.nb_loglik(y, mu, alpha)
    assert distributions.nb_nll(y, mu, alpha) == pytest.approx(expected, rel=rtol)


def test_nb_nll_hand_worked():
    """Geometric case worked in ``helpers.nb_example``: 7 log 3 - 4 log 2."""
    example = helpers.nb_example()
    nll = distributions.nb_nll(example["counts"], example["mu"], example["alpha"])
    assert nll == pytest.approx(example["nll"], rel=1e-13)


def test_nb_nll_vectorized_over_genes():
    """With one dispersion per gene, counts and means are samples x genes."""
    sim = helpers.simulate_counts(n_genes=5, n_samples=7, seed=31, disp_scatter=0.5)
    y, mu = sim.counts.to_numpy(), sim.mu.to_numpy()
    alpha = sim.dispersions.to_numpy()
    nll = distributions.nb_nll(y, mu, alpha)
    expected = -helpers.nb_loglik(y, mu, alpha[None, :], axis=0)
    assert nll.shape == (5,)
    # Dispersions 0.08-0.25: as in test_nb_nll_matches_reference, 1e-12.
    np.testing.assert_allclose(nll, expected, rtol=1e-12)


# stats.trimmed_mean and the trimmed variances


def test_trimmed_mean_hand_worked():
    """floor(n * trim) values are cut from each end of the sorted data.

    Sorted: -40 1 2 3 4 5 6 7 9 100. trim 0.1 cuts one from each end: 37/8. trim 0.25
    cuts floor(2.5) = 2: 27/6. trim 0 keeps all: 97/10.
    """
    x = np.array([9.0, -40.0, 3.0, 1.0, 7.0, 5.0, 100.0, 2.0, 4.0, 6.0])
    assert deseq_stats.trimmed_mean(x, trim=0.1) == pytest.approx(37 / 8, rel=1e-15)
    assert deseq_stats.trimmed_mean(x, trim=0.25) == pytest.approx(27 / 6, rel=1e-15)
    assert deseq_stats.trimmed_mean(x, trim=0.0) == pytest.approx(97 / 10, rel=1e-15)
    assert deseq_stats.trimmed_mean(x) == pytest.approx(37 / 8, rel=1e-15)

    # Along an axis: the same column twice, the second scaled by 2.
    m = np.column_stack([x, 2 * x])
    np.testing.assert_allclose(
        deseq_stats.trimmed_mean(m, trim=0.25, axis=0), [27 / 6, 27 / 3], rtol=1e-15
    )
    np.testing.assert_allclose(
        deseq_stats.trimmed_mean(m.T, trim=0.25, axis=1), [27 / 6, 27 / 3], rtol=1e-15
    )


def test_trimmed_variance_hand_worked():
    """1/8 trim on 8 samples cuts one value per end, then scales by 1.51.

    x = 0 2 3 4 5 6 7 30: trimmed mean 27/6 = 4.5; squared errors sorted 0.25 0.25
    2.25 2.25 6.25 6.25 20.25 650.25, trimmed mean 37.5/6 = 6.25; times 1.51 =
    9.4375. A second gene, twice the first, has four times the variance. With
    trim 1/4 two values go from each end: centre 18/4 = 4.5 again, squared errors
    2.25 2.25 6.25 6.25 kept, mean 4.25, times 1.51 = 6.4175.
    """
    x = np.array([0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 30.0])
    counts = np.column_stack([x, 2 * x])
    np.testing.assert_allclose(
        dispersions.trimmed_variance(counts), [9.4375, 4 * 9.4375], rtol=1e-14
    )
    assert dispersions.trimmed_variance(x) == pytest.approx(9.4375, rel=1e-14)
    # axis=1 is not tested: the trimmed mean is subtracted without restoring the
    # reduced axis, so rows are centred on the wrong means (PyDESeq2 uses axis=0).
    np.testing.assert_allclose(
        dispersions.trimmed_variance(counts, trim=0.25, axis=0),
        [6.4175, 4 * 6.4175],
        rtol=1e-14,
    )


def test_trimmed_cell_variance_hand_worked():
    """Per cell: trimmed mean, scaled trimmed mean of squared errors; max over cells.

    Cell 0 has 3 samples (trim 1/3, scale 2.04), cell 1 has 4 (trim 1/4, scale 1.86);
    the samples are interleaved. Gene 1: cell 0 (1, 2, 10) gives centre 2, squared
    errors (1, 0, 64), trimmed mean 1, so 2.04; cell 1 (1, 2, 3, 10) gives centre
    2.5, squared errors (2.25, 0.25, 0.25, 56.25), trimmed mean 1.25, so 2.325.
    Gene 2: cell 0 (0, 6, 3) gives centre 3, squared errors (9, 9, 0), so 18.36;
    cell 1 (5, 5, 5, 5) gives 0.
    """
    cells = pd.Series([0, 1, 0, 1, 1, 0, 1])
    counts = np.array(
        [[1, 0], [1, 5], [2, 6], [2, 5], [3, 5], [10, 3], [10, 5]], dtype=float
    )
    np.testing.assert_allclose(
        dispersions.trimmed_cell_variance(counts, cells), [2.325, 18.36], rtol=1e-14
    )


def test_trimmed_cell_variance_all_trim_bins():
    """Cells of 3, 5 and 24 samples (all three trim bins) against scipy's trim_mean."""
    sim = helpers.simulate_counts(n_genes=6, n_samples=32, seed=32, disp_scatter=0.5)
    counts = sim.counts.to_numpy(dtype=float)
    labels = np.array([0] * 3 + [1] * 5 + [2] * 24)[
        np.random.RandomState(3).permutation(32)
    ]
    expected = np.full(counts.shape[1], -np.inf)
    for level in [0, 1, 2]:
        sub = counts[labels == level]
        trim, scale = _trim_and_scale(len(sub))
        centre = stats.trim_mean(sub, trim, axis=0)
        expected = np.maximum(
            expected, scale * stats.trim_mean((sub - centre) ** 2, trim, axis=0)
        )
    # Sums over at most 24 values in another order: rounding only.
    np.testing.assert_allclose(
        dispersions.trimmed_cell_variance(counts, pd.Series(labels)),
        expected,
        rtol=1e-12,
    )


def test_robust_method_of_moments_disp_hand_worked():
    """Two cells of three: variance from the cells, mean from all samples.

    Cell A (1, 2, 10) gives 2.04 and cell B (4, 6, 8) gives 2.04 * 4 = 8.16, so
    v = 8.16; m = 31/6; alpha = (8.16 - 31/6) / (31/6)^2. The second gene is
    Poisson-like (v < m) and is floored at 0.04.
    """
    design = pd.DataFrame(
        {"Intercept": 1.0, "condition[T.B]": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]},
        index=[f"s{i}" for i in range(6)],
    )
    counts = np.array([[1, 20], [2, 21], [10, 19], [4, 20], [6, 20], [8, 21]], float)
    m = 31 / 6
    np.testing.assert_allclose(
        dispersions.robust_method_of_moments_disp(counts, design),
        [(8.16 - m) / m**2, 0.04],
        rtol=1e-13,
    )


@pytest.mark.parametrize("case", ["TWO_GROUP", "CONTINUOUS", "MIXED_CELLS"])
def test_robust_method_of_moments_disp_matches_reference(case):
    """Both branches (cell variance, trimmed variance) and cells left out.

    TWO_GROUP has cells of 7 and 6; CONTINUOUS has no repeated design row, so all
    samples go to trimmed_variance; MIXED_CELLS has cells of 3, 3, 3, 3, 2, 2 and
    the two-sample cells only count towards the mean.
    """
    params = {
        "TWO_GROUP": TWO_GROUP,
        "CONTINUOUS": CONTINUOUS,
        "MIXED_CELLS": MIXED_CELLS,
    }
    sim = helpers.simulate_counts(**{**params[case], "n_all_zero": 0})
    normed = sim.counts.to_numpy() / sim.size_factors.to_numpy()[:, None]
    design = sim.design_matrix
    np.testing.assert_allclose(
        dispersions.robust_method_of_moments_disp(normed, design),
        _robust_dispersion_reference(normed, design),
        rtol=1e-12,
    )


# glm.fit_lin_mu and DefaultInference.lin_reg_mu


def test_fit_lin_mu_two_groups():
    """With one indicator per group, the fit is the group mean of y / s, times s.

    Group B is all zero, so its means are floored at ``min_mu``.
    """
    sf = np.array([0.8, 1.2, 1.0, 0.9, 1.1, 1.05])
    x = np.column_stack([np.ones(6), [0, 0, 0, 1, 1, 1]]).astype(float)
    y = np.array([30, 45, 38, 0, 0, 0])
    mean_a = np.mean(y[:3] / sf[:3])
    expected = np.concatenate([sf[:3] * mean_a, [MIN_MU] * 3])
    # Least squares on a 0/1 design: rounding only.
    np.testing.assert_allclose(
        glm.fit_lin_mu(y, sf, x, min_mu=MIN_MU), expected, rtol=1e-12
    )
    np.testing.assert_allclose(
        glm.fit_lin_mu(y, sf, x, min_mu=0.1)[3:], [0.1] * 3, rtol=1e-12
    )


def test_fit_lin_mu_continuous():
    """A continuous covariate: ``s * X lstsq(X, y / s)``, floored at ``min_mu``."""
    sim = helpers.simulate_counts(n_genes=3, n_samples=9, seed=33, continuous=("age",))
    x, sf = sim.design_matrix.to_numpy(), sim.size_factors.to_numpy()
    for gene in sim.counts.columns:
        y = sim.counts[gene].to_numpy()
        coef = np.linalg.lstsq(x, y / sf, rcond=None)[0]
        expected = np.maximum(sf * (x @ coef), MIN_MU)
        # Two least-squares solvers on a 9 x 3 design with condition number < 10.
        np.testing.assert_allclose(glm.fit_lin_mu(y, sf, x), expected, rtol=1e-10)


@pytest.mark.parametrize("n_genes", [1, 7])
def test_lin_reg_mu_matches_per_gene_fit(inference, n_genes):
    """Samples x genes, column j the least-squares fit of gene j (one gene included)."""
    sim = helpers.simulate_counts(
        n_genes=n_genes, n_samples=8, seed=34, continuous=("age",)
    )
    x, sf = sim.design_matrix.to_numpy(), sim.size_factors.to_numpy()
    y = sim.counts.to_numpy()
    mu = inference.lin_reg_mu(counts=y, size_factors=sf, design_matrix=x, min_mu=MIN_MU)
    coef = np.linalg.lstsq(x, y / sf[:, None], rcond=None)[0]
    expected = np.maximum(sf[:, None] * (x @ coef), MIN_MU)
    assert mu.shape == (8, n_genes)
    np.testing.assert_allclose(mu, expected, rtol=1e-10)


# glm.irls_solver


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_samples": 6, "seed": 1},
        {
            "n_samples": 10,
            "seed": 2,
            "factors": {"condition": ["A", "B"], "group": ["X", "Y"]},
            "continuous": ("age",),
        },
        {
            "n_samples": 18,
            "seed": 3,
            "factors": {"condition": ["A", "B"], "group": ["X", "Y", "Z"]},
            "continuous": ("age",),
        },
    ],
    ids=["two_groups", "two_factors_age", "five_coefficients"],
)
def test_irls_solver_matches_reference(kwargs):
    """beta against the Newton minimizer; mu and the hat diagonal from that beta."""
    sim = helpers.simulate_counts(n_genes=12, disp_scatter=0.3, **kwargs)
    x, sf = sim.design_matrix.to_numpy(), sim.size_factors.to_numpy()
    n_compared = 0
    for gene in sim.counts.columns:
        y, disp = sim.counts[gene].to_numpy(), sim.dispersions[gene]
        expected = _lfc_reference_or_none(y, sf, x, disp)
        if expected is None:
            continue
        n_compared += 1
        beta, mu, hat, converged = glm.irls_solver(y, sf, x, disp)
        assert converged, gene
        np.testing.assert_allclose(beta, expected, rtol=0, atol=LFC_ATOL, err_msg=gene)
        # mu is returned without the min_mu floor; same beta, so rounding only.
        np.testing.assert_allclose(mu, sf * np.exp(x @ beta), rtol=1e-12, err_msg=gene)
        # Same beta, so rounding only (measured 3e-16).
        np.testing.assert_allclose(
            hat, _hat_diagonal(x, sf * np.exp(x @ beta), disp), rtol=1e-8, err_msg=gene
        )
        # The hat matrix is a projection of rank p, up to the 1e-6 ridge.
        assert hat.sum() == pytest.approx(x.shape[1], rel=1e-4)
    assert n_compared >= 10


def test_irls_solver_all_zero_group_closed_form():
    """Group B all zero: IRLS's ``min_mu`` floor gives a closed-form fixed point.

    With every B mean floored at 0.5, the working response there is
    ``log(0.5 / s) - 1`` with equal weights, so the fitted B log mean is
    ``mean(log(0.5 / s_B)) - 1`` (which keeps the means below 0.5); group A's mean
    is its own NB MLE. IRLS (with its 1e-6 ridge) differs from this by 4e-6
    (measured).
    """
    sf = np.array([0.8, 1.2, 1.0, 0.9, 1.1, 1.05])
    x = np.column_stack([np.ones(6), [0, 0, 0, 1, 1, 1]]).astype(float)
    y = np.array([30, 45, 38, 0, 0, 0])
    disp = 0.1
    beta, mu, hat, converged = glm.irls_solver(y, sf, x, disp)

    b0 = helpers.reference_glm_mle(y[:3], sf[:3], np.ones((3, 1)), disp)[0]
    b_group_b = np.mean(np.log(MIN_MU / sf[3:])) - 1
    assert converged
    np.testing.assert_allclose(beta, [b0, b_group_b - b0], rtol=0, atol=LFC_ATOL)
    # The returned means are not floored.
    np.testing.assert_allclose(mu, sf * np.exp(x @ beta), rtol=1e-12)
    assert (mu[3:] < MIN_MU).all()
    np.testing.assert_allclose(hat, _hat_diagonal(x, mu, disp), rtol=1e-8)


@pytest.mark.parametrize("sign", [1, -1])
def test_irls_solver_lfc_at_bound(sign):
    """An LFC beyond ``max_beta`` switches to L-BFGS-B, which stops at the bound.

    The covariate is +-0.1, so the fourfold change is an unbounded slope of 6.7
    (6.9 with the sign flipped); with bounds +-4 the slope sits at the bound and the
    intercept minimizes the objective with the slope fixed (a bounded scalar
    search, xatol 1e-12). Measured intercept differences: 9e-10.
    """
    sf = np.array([0.8, 1.2, 1.0, 0.9, 1.1, 1.05])
    x = np.column_stack([np.ones(6), sign * np.array([-0.1] * 3 + [0.1] * 3)])
    y = np.array([9, 13, 10, 40, 47, 36])
    disp = 0.05
    beta, mu, hat, converged = glm.irls_solver(y, sf, x, disp, min_beta=-4, max_beta=4)

    objective = helpers.glm_objective(y, sf, x, disp)
    intercept = optimize.minimize_scalar(
        lambda b0: objective(np.array([b0, 4.0 * sign])),
        bounds=(0, 4),
        method="bounded",
        options={"xatol": 1e-12},
    ).x
    assert converged
    # L-BFGS-B projects onto the bound; a solver that stops on a projected gradient
    # may stop just inside it, so LFC_ATOL (the unbounded slope is 2.7 away).
    assert beta[1] == pytest.approx(4.0 * sign, abs=LFC_ATOL)
    assert beta[0] == pytest.approx(intercept, abs=LFC_ATOL)
    np.testing.assert_allclose(mu, sf * np.exp(x @ beta), rtol=1e-12)


@pytest.mark.parametrize("optimizer", ["L-BFGS-B", "BFGS"])
def test_irls_solver_optimizer_path_matches_reference(optimizer):
    """``maxiter=1`` hands over to the optimizer at once; it must reach the MLE.

    Both stop at a gradient norm of 1e-5; measured coefficient errors 2.7e-6
    (L-BFGS-B) and 3.7e-7 (BFGS), so LFC_ATOL holds with a wide margin.
    """
    sim = helpers.simulate_counts(n_genes=6, n_samples=8, seed=35, disp_scatter=0.3)
    x, sf = sim.design_matrix.to_numpy(), sim.size_factors.to_numpy()
    for gene in sim.counts.columns:
        y, disp = sim.counts[gene].to_numpy(), sim.dispersions[gene]
        expected = _lfc_reference_or_none(y, sf, x, disp)
        if expected is None:
            continue
        beta, mu, hat, converged = glm.irls_solver(
            y, sf, x, disp, maxiter=1, optimizer=optimizer
        )
        assert converged, gene
        np.testing.assert_allclose(beta, expected, rtol=0, atol=LFC_ATOL, err_msg=gene)
        np.testing.assert_allclose(
            hat, _hat_diagonal(x, sf * np.exp(x @ beta), disp), rtol=1e-8
        )


def _floored_zero_group_case():
    sf = np.array([0.8, 1.2, 1.0, 0.9, 1.1, 1.05])
    x = np.column_stack([np.ones(6), [0, 0, 0, 1, 1, 1]]).astype(float)
    y = np.array([30, 45, 38, 0, 0, 0])
    return y, sf, x, 0.1


def test_irls_solver_grid_fallback_when_optimizer_fails():
    """Optimizer failure with two coefficients falls back to the grid search.

    With group B all zero and ``maxiter=1``, the optimizer works on the objective
    with means floored at 0.5, which is flat in the B direction while the gradient
    it is given is not, so the line search fails. The fallback grid's minimizer of
    that objective puts group A at its MLE and B's largest mean just at the floor:
    ``b0 + b1 = log(0.5 / max s_B)``, the ridge pulling b1 up to it. The fine grid
    spacing is 2 * (60/59) / 59 = 0.034; 0.1 allows three steps. The failed
    optimizer stops near b1 = -5.9, 1.5 away.
    """
    y, sf, x, disp = _floored_zero_group_case()
    beta, mu, hat, converged = glm.irls_solver(y, sf, x, disp, maxiter=1)

    b0 = helpers.reference_glm_mle(y[:3], sf[:3], np.ones((3, 1)), disp)[0]
    assert not converged
    assert beta[0] == pytest.approx(b0, abs=0.1)
    assert beta[0] + beta[1] == pytest.approx(np.log(MIN_MU / sf[3:].max()), abs=0.1)
    np.testing.assert_allclose(mu, sf * np.exp(x @ beta), rtol=1e-12)
    np.testing.assert_allclose(hat, _hat_diagonal(x, mu, disp), rtol=1e-8)


def test_irls_solver_no_grid_fallback_beyond_two_coefficients():
    """With three coefficients a failed optimizer is reported, not replaced."""
    sf = np.array([0.8, 1.2, 1.0, 0.9, 1.1, 1.05, 0.95, 1.0])
    x = np.column_stack(
        [
            np.ones(8),
            [0, 0, 0, 0, 1, 1, 1, 1],
            [0.3, -1.2, 0.5, 0.9, -0.4, 1.1, 0.2, -0.6],
        ]
    )
    y = np.array([30, 45, 38, 50, 0, 0, 0, 0])
    beta, mu, hat, converged = glm.irls_solver(y, sf, x, 0.1, maxiter=1)
    assert not converged
    assert np.isfinite(beta).all() and (np.abs(beta) <= 30).all()
    np.testing.assert_allclose(mu, sf * np.exp(x @ beta), rtol=1e-12)


def test_irls_solver_rank_deficient_design():
    """A duplicated column: the ridge picks the minimum-norm split of the coefficient.

    The fitted means equal the full-rank fit's, and the duplicated coefficient is
    split in half; the penalized objective has a unique minimizer, which the Newton
    reference also finds on the rank-deficient design.
    """
    sim = helpers.simulate_counts(n_genes=5, n_samples=8, seed=36)
    x, sf = sim.design_matrix.to_numpy(), sim.size_factors.to_numpy()
    x_dup = np.column_stack([x, x[:, 1]])
    for gene in sim.counts.columns:
        y, disp = sim.counts[gene].to_numpy(), sim.dispersions[gene]
        full = helpers.reference_glm_mle(y, sf, x, disp)
        beta, mu, hat, converged = glm.irls_solver(y, sf, x_dup, disp)
        assert converged
        np.testing.assert_allclose(
            beta, [full[0], full[1] / 2, full[1] / 2], rtol=0, atol=LFC_ATOL
        )
        np.testing.assert_allclose(
            beta, helpers.reference_glm_mle(y, sf, x_dup, disp), rtol=0, atol=LFC_ATOL
        )
        # exp of an LFC_ATOL error in log scale.
        np.testing.assert_allclose(mu, sf * np.exp(x @ full), rtol=LFC_ATOL)
        # Rank 2, so the hat diagonal sums to 2.
        assert hat.sum() == pytest.approx(2.0, rel=1e-4)


def test_irls_solver_near_singular_design():
    """A covariate equal to the condition plus 0.1 x age (condition number 32).

    Individual coefficients are ill-determined (up to 17 in size, errors up to
    5.4e-4 measured), so the objective and the fitted means are compared: relative
    objective gap 1e-7 (measured 2.9e-10; README: up to 1.3e-8 for IRLS), means
    within 1e-3 (measured 7.3e-5).
    """
    sim = helpers.simulate_counts(
        n_genes=8, n_samples=10, seed=6, continuous=("age",), disp_scatter=0.3
    )
    x = sim.design_matrix.to_numpy().copy()
    sf = sim.size_factors.to_numpy()
    x[:, 2] = x[:, 1] + 0.1 * x[:, 2]
    for gene in sim.counts.columns:
        y, disp = sim.counts[gene].to_numpy(), sim.dispersions[gene]
        expected = helpers.reference_glm_mle(y, sf, x, disp)
        objective = helpers.glm_objective(y, sf, x, disp)
        beta, mu, hat, converged = glm.irls_solver(y, sf, x, disp)
        assert converged
        assert objective(beta) == pytest.approx(objective(expected), rel=1e-7)
        np.testing.assert_allclose(mu, sf * np.exp(x @ expected), rtol=1e-3)


# grid_search.grid_fit_beta


def test_grid_fit_beta_near_minimizer():
    """The two-stage grid lands within its fine spacing of the MLE.

    Coarse grid: 60 points on [-30, 30]; fine grid: 60 points over +-1 coarse step,
    spacing 0.034. Measured coordinate errors up to 0.025; 0.05 allows 1.5 steps.
    The grid point's objective cannot be below the minimum.
    """
    sim = helpers.simulate_counts(n_genes=10, n_samples=8, seed=4, disp_scatter=0.3)
    x, sf = sim.design_matrix.to_numpy(), sim.size_factors.to_numpy()
    n_compared = 0
    for gene in sim.counts.columns:
        y, disp = sim.counts[gene].to_numpy(), sim.dispersions[gene]
        expected = _lfc_reference_or_none(y, sf, x, disp)
        if expected is None:
            continue
        n_compared += 1
        beta = grid_search.grid_fit_beta(y, sf, x, disp)
        objective = helpers.glm_objective(y, sf, x, disp)
        assert beta.shape == (2,)
        np.testing.assert_allclose(beta, expected, rtol=0, atol=0.05, err_msg=gene)
        assert objective(beta) >= objective(expected) - 1e-9 * abs(objective(expected))
    assert n_compared >= 8


def test_grid_fit_beta_floored_objective():
    """On the floored objective of an all-zero group (see the fallback test)."""
    y, sf, x, disp = _floored_zero_group_case()
    beta = grid_search.grid_fit_beta(y, sf, x, disp)
    b0 = helpers.reference_glm_mle(y[:3], sf[:3], np.ones((3, 1)), disp)[0]
    assert beta[0] == pytest.approx(b0, abs=0.1)
    assert beta[0] + beta[1] == pytest.approx(np.log(MIN_MU / sf[3:].max()), abs=0.1)


# DefaultInference.irls


def test_inference_irls_matches_reference_per_gene(inference):
    """Shapes, per-gene results and per-gene dispersions in one call."""
    sim = helpers.simulate_counts(
        n_genes=9, n_samples=10, seed=37, continuous=("age",), disp_scatter=0.5
    )
    x, sf = sim.design_matrix.to_numpy(), sim.size_factors.to_numpy()
    y, disp = sim.counts.to_numpy(), sim.dispersions.to_numpy()
    beta, mu, hat, converged = inference.irls(
        counts=y,
        size_factors=sf,
        design_matrix=x,
        disp=disp,
        min_mu=MIN_MU,
        beta_tol=1e-8,
    )
    assert beta.shape == (9, 3) and mu.shape == (10, 9) and hat.shape == (10, 9)
    assert converged.shape == (9,) and converged.all()
    np.testing.assert_allclose(mu, sf[:, None] * np.exp(x @ beta.T), rtol=1e-12)
    n_compared = 0
    for j in range(9):
        np.testing.assert_allclose(
            hat[:, j], _hat_diagonal(x, mu[:, j], disp[j]), rtol=1e-8
        )
        expected = _lfc_reference_or_none(y[:, j], sf, x, disp[j])
        if expected is not None:
            n_compared += 1
            np.testing.assert_allclose(beta[j], expected, rtol=0, atol=LFC_ATOL)
    assert n_compared >= 7


def test_inference_irls_mixed_convergence_and_bounds(inference):
    """Per-gene fallbacks inside one call: flags and results stay with their gene.

    With ``maxiter=1`` every gene goes to L-BFGS-B. The all-zero-group gene fails
    and takes the grid fallback (``converged`` False); the others converge to their
    MLE. With ``max_beta=4`` the steep gene stops at the bound.
    """
    y_zero, sf, x, _ = _floored_zero_group_case()
    y = np.column_stack(
        [
            [30, 45, 38, 80, 95, 70],
            y_zero,
            [500, 610, 480, 300, 260, 330],
            [9, 13, 10, 40, 47, 36],
        ]
    )
    disp = np.array([0.05, 0.1, 0.08, 0.05])
    beta, mu, hat, converged = inference.irls(
        counts=y,
        size_factors=sf,
        design_matrix=x,
        disp=disp,
        min_mu=MIN_MU,
        beta_tol=1e-8,
        maxiter=1,
    )
    np.testing.assert_array_equal(converged, [True, False, True, True])
    for j in [0, 2, 3]:
        expected = helpers.reference_glm_mle(y[:, j], sf, x, disp[j])
        np.testing.assert_allclose(beta[j], expected, rtol=0, atol=LFC_ATOL)
    b0 = helpers.reference_glm_mle(y_zero[:3], sf[:3], np.ones((3, 1)), 0.1)[0]
    assert beta[1, 0] == pytest.approx(b0, abs=0.1)
    assert beta[1].sum() == pytest.approx(np.log(MIN_MU / sf[3:].max()), abs=0.1)

    # The last gene's LFC is log(4) / 1 ~ 1.4 with a 0/1 design; scale its covariate
    # down by 4 so that the slope (about 5.6) passes max_beta=4.
    x_steep = x.copy()
    x_steep[:, 1] *= 0.25
    beta_b, _, _, conv_b = inference.irls(
        counts=y[:, [0, 3]],
        size_factors=sf,
        design_matrix=x_steep,
        disp=disp[[0, 3]],
        min_mu=MIN_MU,
        beta_tol=1e-8,
        min_beta=-4,
        max_beta=4,
    )
    assert conv_b.all()
    assert beta_b[1, 1] == pytest.approx(4.0, abs=LFC_ATOL)  # at the bound
    np.testing.assert_allclose(
        beta_b[0],
        helpers.reference_glm_mle(y[:, 0], sf, x_steep, disp[0]),
        rtol=0,
        atol=LFC_ATOL,
    )


def test_inference_irls_gene_order_and_single_gene(inference):
    """Permuting the genes permutes the results; a one-gene call matches its column."""
    sim = helpers.simulate_counts(n_genes=8, n_samples=9, seed=38, disp_scatter=0.4)
    x, sf = sim.design_matrix.to_numpy(), sim.size_factors.to_numpy()
    y, disp = sim.counts.to_numpy(), sim.dispersions.to_numpy()
    kw = {"size_factors": sf, "design_matrix": x, "min_mu": MIN_MU, "beta_tol": 1e-8}
    beta, mu, hat, conv = inference.irls(counts=y, disp=disp, **kw)
    order = np.array([5, 2, 7, 0, 3, 6, 1, 4])
    beta_p, mu_p, hat_p, conv_p = inference.irls(
        counts=y[:, order], disp=disp[order], **kw
    )
    # Same per-gene problem: the solver's own precision at most.
    np.testing.assert_allclose(beta_p, beta[order], rtol=0, atol=LFC_ATOL)
    np.testing.assert_allclose(mu_p, mu[:, order], rtol=LFC_ATOL)
    np.testing.assert_allclose(hat_p, hat[:, order], rtol=1e-3)
    np.testing.assert_array_equal(conv_p, conv[order])
    for j in [0, 7]:
        beta_1, mu_1, hat_1, conv_1 = inference.irls(
            counts=y[:, [j]], disp=disp[[j]], **kw
        )
        assert beta_1.shape == (1, 2) and mu_1.shape == (9, 1) and hat_1.shape == (9, 1)
        np.testing.assert_allclose(beta_1[0], beta[j], rtol=0, atol=LFC_ATOL)
        np.testing.assert_allclose(mu_1[:, 0], mu[:, j], rtol=LFC_ATOL)
        assert conv_1[0] == conv[j]


# DeseqDataSet.fit_LFC


@pytest.mark.parametrize("case", ["TWO_GROUP", "CONTINUOUS"])
def test_fit_lfc_matches_reference(case):
    """``varm["LFC"]`` against the Newton reference at the fitted dispersions.

    Natural log, one column per design column; all-zero genes are NaN. Genes whose
    fit reaches the ``min_mu`` floor are left out (README_NEW_TESTS.md).
    """
    sim, dds = _fitted(case, refit_cooks=False)
    x = sim.design_matrix.to_numpy()
    sf = dds.obs["size_factors"].to_numpy()
    lfc = dds.varm["LFC"]
    assert list(lfc.columns) == list(sim.design_matrix.columns)
    assert list(lfc.index) == list(sim.counts.columns)
    all_zero = (sim.counts == 0).all(axis=0).to_numpy()
    assert lfc.to_numpy()[all_zero].size and np.isnan(lfc.to_numpy()[all_zero]).all()
    assert np.isfinite(lfc.to_numpy()[~all_zero]).all()

    expected, genes = [], []
    for gene in sim.counts.columns[~all_zero]:
        y = sim.counts[gene].to_numpy()
        beta = _lfc_reference_or_none(y, sf, x, dds.var.loc[gene, "dispersions"])
        if beta is not None:
            expected.append(beta)
            genes.append(gene)
    assert len(genes) >= 0.85 * (~all_zero).sum()
    helpers.assert_fields_close(
        lfc.loc[genes],
        pd.DataFrame(expected, index=genes, columns=lfc.columns),
        rtol=0,
        atol=LFC_ATOL,
        name="LFC",
    )


def test_fit_lfc_two_genes():
    """Two genes (the dispersion trend is the mean of two estimates)."""
    sim = helpers.simulate_counts(n_genes=2, n_samples=8, seed=3)
    dds = helpers.make_dds(sim.counts, sim.metadata, fit_type="mean")
    dds.deseq2()
    x, sf = sim.design_matrix.to_numpy(), dds.obs["size_factors"].to_numpy()
    for gene in sim.counts.columns:
        y = sim.counts[gene].to_numpy()
        expected = helpers.reference_glm_mle(y, sf, x, dds.var.loc[gene, "dispersions"])
        np.testing.assert_allclose(
            dds.varm["LFC"].loc[gene].to_numpy(), expected, rtol=0, atol=LFC_ATOL
        )
    np.testing.assert_allclose(
        dds.layers["cooks"], _cooks_reference(dds), rtol=COOKS_RTOL
    )


# DeseqDataSet.calculate_cooks


@pytest.mark.parametrize("case", ["TWO_GROUP", "CONTINUOUS", "MIXED_CELLS"])
def test_calculate_cooks_matches_formula(case):
    """``layers["cooks"]`` from the formula applied to the run's public outputs.

    The three cases take the three robust-dispersion paths (see
    test_robust_method_of_moments_disp_matches_reference). All-zero genes are NaN.
    """
    _, dds = _fitted(case, refit_cooks=False)
    expected = _cooks_reference(dds)
    actual = dds.layers["cooks"]
    assert actual.shape == dds.X.shape
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_allclose(actual, expected, rtol=COOKS_RTOL)


def test_calculate_cooks_flags_simulated_outliers():
    """Every simulated outlier (100x its mean) has the largest distance of its gene.

    A statistical check against the simulator's truth (``sim.outliers``); it holds
    for all eight outliers of this seed.
    """
    sim, dds = _fitted("TWO_GROUP", refit_cooks=False)
    cooks = pd.DataFrame(dds.layers["cooks"], index=dds.obs_names, columns=dds.var_names)
    n, p = sim.design_matrix.shape
    cutoff = stats.f.ppf(0.99, p, n - p)
    for gene in sim.outliers.columns[sim.outliers.any(axis=0)]:
        sample = sim.outliers.index[sim.outliers[gene]][0]
        assert cooks[gene].idxmax() == sample, gene
        assert cooks.loc[sample, gene] > cutoff, gene


# DeseqDataSet.refit


def _replacement_reference(sim, dds, min_replicates=7):
    """Counts after DESeq2's replaceOutliers, by hand.

    Entries with Cook's distance above the 0.99 quantile of F(p, n - p), in samples
    whose design cell has ``min_replicates`` or more samples, become the 0.2-trimmed
    mean of the gene's normalized counts times the sample's size factor, truncated
    to an integer.
    """
    x = sim.design_matrix.to_numpy()
    n, p = x.shape
    y = sim.counts.to_numpy()
    sf = dds.obs["size_factors"].to_numpy()
    rows = [tuple(r) for r in x]
    size = Counter(rows)
    replaceable = np.array([size[r] >= min_replicates for r in rows])
    cutoff = stats.f.ppf(0.99, p, n - p)
    flagged = np.nan_to_num(dds.layers["cooks"], nan=-np.inf) > cutoff
    product = stats.trim_mean(y / sf[:, None], 0.2, axis=0)[None, :] * sf[:, None]
    replaced_counts = np.where(
        flagged & replaceable[:, None], np.trunc(product).astype(np.int64), y
    )
    return {
        "cutoff": cutoff,
        "flagged": flagged,
        "replaceable": replaceable,
        "product": product,
        "counts": replaced_counts,
    }


@pytest.mark.parametrize("fit_type", ["parametric", "mean"])
def test_refit_replaces_outliers_and_refits(fit_type):
    """Which genes are replaced and refit, and the refit genes' new estimates.

    Expected values for a refit gene come from its replaced counts: gene-wise
    dispersion from the reference minimizer at the group-mean fit, the trend (not
    refit) at the new mean of normalized counts, the MAP dispersion with the
    original prior variance, the gene-wise value instead when it lies more than two
    log-residual SDs above the trend, and the LFC from the Newton reference. Refit
    genes whose counts did not change (their outlier is in a non-replaceable sample)
    get their first-pass estimates back; genes that are all zero after replacement
    get LFC 0; genes that were not refit keep their first-pass estimates.

    ``MAP_dispersions`` of refit genes is not checked: PyDESeq2 leaves the first-pass
    value there, while DESeq2 copies the refit value back.
    """
    sim, dds = _fitted("TWO_GROUP", fit_type=fit_type)
    _, plain = _fitted("TWO_GROUP", refit_cooks=False, fit_type=fit_type)
    x = sim.design_matrix.to_numpy()
    n, p = x.shape
    groups = sim.metadata["condition"].to_numpy()
    sf = dds.obs["size_factors"].to_numpy()
    y = sim.counts.to_numpy()
    ref = _replacement_reference(sim, dds)

    # Guards on the dataset: no distance within 1% of the cutoff, and no replacement
    # value within 1e-6 of an integer, so neither decision depends on rounding.
    cooks = np.nan_to_num(dds.layers["cooks"], nan=0.0)
    assert (np.abs(cooks / ref["cutoff"] - 1) > 0.01).all()
    product = ref["product"][ref["flagged"] & ref["replaceable"][:, None]]
    frac = product % 1
    assert ((product == 0) | ((frac > 1e-6) & (frac < 1 - 1e-6))).all()

    # The cooks layer still holds the first-pass distances, which come from the same
    # first-pass fit in both runs.
    np.testing.assert_allclose(dds.layers["cooks"], plain.layers["cooks"], rtol=1e-6)

    replaced = ref["flagged"].any(axis=0)
    new_all_zero = replaced & (ref["counts"] == 0).all(axis=0)
    refitted = replaced & ~new_all_zero
    np.testing.assert_array_equal(dds.obs["replaceable"].to_numpy(), ref["replaceable"])
    np.testing.assert_array_equal(dds.var["replaced"].to_numpy(), replaced)
    np.testing.assert_array_equal(dds.var["refitted"].to_numpy(), refitted)
    changed = (ref["counts"] != y).any(axis=0)
    # The dataset exercises every kind of gene.
    assert (refitted & changed).sum() >= 4
    assert (refitted & ~changed).sum() >= 2
    assert list(dds.var_names[new_all_zero]) == ["single_sample1"]

    # Genes that were not refit keep their first-pass estimates.
    keep = ~refitted
    for field in [
        "genewise_dispersions",
        "fitted_dispersions",
        "MAP_dispersions",
        "dispersions",
    ]:
        helpers.assert_fields_close(
            dds.var[field][keep],
            plain.var[field][keep],
            rtol=DISPERSION_RTOL,
            name=field,
        )
    lfc, plain_lfc = dds.varm["LFC"], plain.varm["LFC"]
    keep_lfc = keep & ~new_all_zero
    helpers.assert_fields_close(
        lfc[keep_lfc], plain_lfc[keep_lfc], rtol=0, atol=LFC_ATOL, name="LFC"
    )
    # All zero after replacement: LFC set to 0.
    assert (lfc[new_all_zero].to_numpy() == 0).all()

    # Refit on unchanged counts: the same problem as the first pass.
    same = refitted & ~changed
    for field in ["genewise_dispersions", "fitted_dispersions", "dispersions"]:
        helpers.assert_fields_close(
            dds.var[field][same],
            plain.var[field][same],
            rtol=DISPERSION_RTOL,
            name=field,
        )
    helpers.assert_fields_close(
        lfc[same], plain_lfc[same], rtol=0, atol=LFC_ATOL, name="LFC (unchanged counts)"
    )

    # Refit on replaced counts: against the references. The prior is the first pass's
    # (MAD of log residuals over genes with gene-wise dispersion >= 100 * min_disp).
    nz = plain.var["non_zero"].to_numpy()
    log_res = np.log(plain.var["genewise_dispersions"].to_numpy()[nz]) - np.log(
        plain.var["fitted_dispersions"].to_numpy()[nz]
    )
    use = plain.var["genewise_dispersions"].to_numpy()[nz] >= 100 * 1e-8
    centred = np.abs(log_res[use] - np.median(log_res[use]))
    squared_logres = (np.median(centred) / stats.norm.ppf(0.75)) ** 2
    prior_var = float(np.asarray(dds.uns["prior_disp_var"]))
    max_disp = max(10.0, n)

    def trend(normed_mean):
        if fit_type == "mean":
            return float(np.asarray(dds.uns["mean_disp"]))
        coeffs = dds.uns["trend_coeffs"]
        return coeffs["a0"] + coeffs["a1"] / normed_mean

    rows = {"genewise_dispersions": [], "fitted_dispersions": [], "dispersions": []}
    genes, lfc_genes, lfc_expected = [], [], []
    for j in np.flatnonzero(refitted & changed):
        gene = dds.var_names[j]
        yj = ref["counts"][:, j].astype(float)
        normed = yj / sf
        mu_hat = np.maximum(sf * _group_means(normed, groups), MIN_MU)
        fitted = trend(normed.mean())
        beta = _lfc_reference_or_none(yj, sf, x, dds.var["dispersions"].iloc[j])
        if beta is not None:
            lfc_genes.append(gene)
            lfc_expected.append(beta)
        if _starts_at_min_disp(yj, sf, groups, p):
            continue
        genewise = helpers.reference_dispersion_mle(yj, mu_hat, x, max_disp=max_disp)
        map_disp = helpers.reference_dispersion_mle(
            yj,
            mu_hat,
            x,
            max_disp=max_disp,
            prior_disp_var=prior_var,
            prior_dispersion=fitted,
        )
        outlier = np.log(genewise) > np.log(fitted) + 2 * np.sqrt(squared_logres)
        genes.append(gene)
        rows["genewise_dispersions"].append(genewise)
        rows["fitted_dispersions"].append(fitted)
        rows["dispersions"].append(genewise if outlier else map_disp)
    assert len(genes) >= 4 and len(lfc_genes) >= 4
    for field, values in rows.items():
        # The fitted dispersion is a formula of the counts: rounding only.
        rtol = 1e-10 if field == "fitted_dispersions" else DISPERSION_RTOL
        helpers.assert_fields_close(
            dds.var[field].loc[genes], np.array(values), rtol=rtol, name=f"refit {field}"
        )
    helpers.assert_fields_close(
        lfc.loc[lfc_genes],
        pd.DataFrame(lfc_expected, index=lfc_genes, columns=lfc.columns),
        rtol=0,
        atol=LFC_ATOL,
        name="refit LFC",
    )

    # replace_cooks: the first-pass distances with refit genes' replaceable samples
    # set to 0 (copies and zeros, so exact by construction).
    expected_cooks = dds.layers["cooks"].copy()
    expected_cooks[np.ix_(ref["replaceable"], refitted)] = 0.0
    np.testing.assert_array_equal(dds.layers["replace_cooks"], expected_cooks)


# Two factors in cells of seven, so every sample is replaceable and, with four design
# rows for three coefficients, the gene-wise means of the refit come from IRLS rather
# than a linear model. ``wild1`` and ``wild2`` are drawn with dispersions 1.5 and 2.5
# (the trend is near 0.1) and get one count 100 times their mean: after the refit they
# are dispersion outliers, whose final dispersion is the gene-wise estimate.
MULTI_REFIT = {
    "n_genes": 60,
    "n_samples": 28,
    "seed": 62,
    "factors": {"condition": ["A", "B"], "group": ["X", "Y"]},
    "n_outliers": 8,
    "disp_scatter": 0.2,
    "n_all_zero": 1,
    "n_single_sample": 1,
}


def _multi_refit_counts():
    sim = helpers.simulate_counts(**MULTI_REFIT)
    counts = sim.counts.copy()
    rs = np.random.RandomState([62, 1])
    for k, (mean, disp) in enumerate([(300, 1.5), (800, 2.5)]):
        y = rs.negative_binomial(1 / disp, 1 / (1 + disp * mean), size=28)
        y[rs.randint(28)] = 100 * mean
        counts[f"wild{k + 1}"] = y
    return sim, counts


def test_refit_multi_factor_design():
    """Refit under ``~group + condition``: which genes, and their new estimates.

    As in test_refit_replaces_outliers_and_refits, but the gene-wise means come from
    the GLM: IRLS at DESeq2's starting dispersion (the smaller of the rough estimate,
    from a linear model of the normalized counts, and the moments estimate), here the
    Newton reference at that dispersion. Refit genes that are dispersion outliers
    keep their gene-wise estimate.
    """
    sim, counts = _multi_refit_counts()
    design = "~group + condition"
    dds = helpers.make_dds(counts, sim.metadata, design=design)
    dds.deseq2()
    plain = helpers.make_dds(counts, sim.metadata, design=design, refit_cooks=False)
    plain.deseq2()
    x = dds.obsm["design_matrix"].to_numpy(dtype=float)
    n, p = x.shape
    sf = dds.obs["size_factors"].to_numpy()
    y = counts.to_numpy()
    assert dds.obs["replaceable"].all()

    cutoff = stats.f.ppf(0.99, p, n - p)
    cooks = dds.layers["cooks"]
    flagged = np.nan_to_num(cooks, nan=-np.inf) > cutoff
    trimmed = stats.trim_mean(y / sf[:, None], 0.2, axis=0)[None, :] * sf[:, None]
    replaced_counts = np.where(flagged, np.trunc(trimmed).astype(np.int64), y)
    # Guards: no decision within 1% of the cutoff, no replacement near an integer.
    assert (np.abs(np.nan_to_num(cooks, nan=0.0) / cutoff - 1) > 0.01).all()
    frac = trimmed[flagged] % 1
    assert ((trimmed[flagged] == 0) | ((frac > 1e-6) & (frac < 1 - 1e-6))).all()

    replaced = flagged.any(axis=0)
    new_all_zero = replaced & (replaced_counts == 0).all(axis=0)
    refitted = replaced & ~new_all_zero
    np.testing.assert_array_equal(dds.var["replaced"].to_numpy(), replaced)
    np.testing.assert_array_equal(dds.var["refitted"].to_numpy(), refitted)
    assert refitted.sum() >= 6 and new_all_zero.sum() == 1
    assert (dds.varm["LFC"][new_all_zero].to_numpy() == 0).all()

    # Genes that were not refit keep their first-pass estimates.
    keep = ~replaced
    for field in ["genewise_dispersions", "MAP_dispersions", "dispersions"]:
        helpers.assert_fields_close(
            dds.var[field][keep],
            plain.var[field][keep],
            rtol=DISPERSION_RTOL,
            name=field,
        )
    helpers.assert_fields_close(
        dds.varm["LFC"][keep], plain.varm["LFC"][keep], rtol=0, atol=LFC_ATOL, name="LFC"
    )

    coeffs = dds.uns["trend_coeffs"]
    prior_var = float(np.asarray(dds.uns["prior_disp_var"]))
    threshold = 2 * np.sqrt(float(np.asarray(dds.uns["_squared_logres"])))
    expected = {f: [] for f in ["genewise", "fitted", "dispersions", "lfc"]}
    outlier_genes, lfc_genes = [], []
    genes = dds.var_names[refitted]
    for gene in genes:
        yj = replaced_counts[:, dds.var_names.get_loc(gene)].astype(float)
        normed = yj / sf
        linear = np.maximum(x @ np.linalg.lstsq(x, normed, rcond=None)[0], 1.0)
        rough = max(np.sum(((normed - linear) ** 2 - linear) / linear**2) / (n - p), 0)
        moments = (
            normed.var(ddof=1) - np.mean(1 / sf) * normed.mean()
        ) / normed.mean() ** 2
        start = min(rough, moments)
        assert start > 1e-6, gene  # clear of the min_disp start (a known issue)
        mu_hat = sf * np.exp(x @ helpers.reference_glm_mle(yj, sf, x, start))
        assert mu_hat.min() > 2 * MIN_MU, gene
        genewise = helpers.reference_dispersion_mle(yj, mu_hat, x, max_disp=max(10.0, n))
        fitted = coeffs["a0"] + coeffs["a1"] / normed.mean()
        map_disp = helpers.reference_dispersion_mle(
            yj,
            mu_hat,
            x,
            max_disp=max(10.0, n),
            prior_disp_var=prior_var,
            prior_dispersion=fitted,
        )
        margin = np.log(genewise) - np.log(fitted) - threshold
        assert abs(margin) > 0.05, gene  # the outlier rule is not decided by rounding
        if margin > 0:
            outlier_genes.append(gene)
        expected["genewise"].append(genewise)
        expected["fitted"].append(fitted)
        expected["dispersions"].append(genewise if margin > 0 else map_disp)
        # LFCs: the objective gap at 1e-7 relative for every gene (README; IRLS stops
        # on a relative deviance change of 1e-8), and the coefficients at LFC_ATOL
        # where the dispersion is below 1. On the flatter objectives of the two wild
        # genes IRLS stops up to 4e-4 from the minimizer (gap 1e-9).
        disp = dds.var.loc[gene, "dispersions"]
        beta = helpers.reference_glm_mle(yj, sf, x, disp)
        objective = helpers.glm_objective(yj, sf, x, disp)
        fitted_beta = dds.varm["LFC"].loc[gene].to_numpy()
        gap = (objective(fitted_beta) - objective(beta)) / abs(objective(beta))
        assert gap < 1e-7, (gene, gap)
        if disp < 1:
            expected["lfc"].append(beta)
            lfc_genes.append(gene)
    assert {"wild1", "wild2"} <= set(outlier_genes)

    for field, key, rtol in [
        ("genewise_dispersions", "genewise", DISPERSION_RTOL),
        ("fitted_dispersions", "fitted", 1e-10),  # a formula of the counts
        ("dispersions", "dispersions", DISPERSION_RTOL),
    ]:
        helpers.assert_fields_close(
            dds.var[field][genes],
            np.array(expected[key]),
            rtol=rtol,
            name=f"refit {field}",
        )
    assert len(lfc_genes) >= 6
    helpers.assert_fields_close(
        dds.varm["LFC"].loc[lfc_genes],
        pd.DataFrame(expected["lfc"], index=lfc_genes, columns=dds.varm["LFC"].columns),
        rtol=0,
        atol=LFC_ATOL,
        name="refit LFC",
    )


def test_refit_called_step_by_step():
    """``refit()`` after the individual fitting steps gives what ``deseq2()`` gives."""
    sim, expected = _fitted("TWO_GROUP")
    dds = helpers.make_dds(sim.counts, sim.metadata, design=sim.design)
    dds.fit_size_factors()
    dds.fit_genewise_dispersions()
    dds.fit_dispersion_trend()
    dds.fit_dispersion_prior()
    dds.fit_MAP_dispersions()
    dds.fit_LFC()
    dds.calculate_cooks()
    dds.refit()
    assert dds.var["refitted"].sum() >= 4
    _assert_same_fit(dds, expected, name="step by step")


def test_refit_without_replaceable_samples():
    """No cell reaches min_replicates: nothing is replaced and nothing changes."""
    sim, dds = _fitted("MIXED_CELLS")
    _, plain = _fitted("MIXED_CELLS", refit_cooks=False)
    assert not dds.obs["replaceable"].any()
    assert not dds.var["replaced"].any() and not dds.var["refitted"].any()
    _assert_same_fit(dds, plain, name="no replaceable samples", flags=())


def test_refit_cutoff_is_f_quantile():
    """Replacement uses the 0.99 quantile of F(p, n - p), checked next to it.

    Forty genes get a fivefold outlier, which puts several maximum distances within
    10% of the cutoff on either side (none within 1%, so rounding cannot decide).
    The 0.99 quantile of F(p, n), a plausible slip, is 7% lower.
    """
    sim = helpers.simulate_counts(
        n_genes=80,
        n_samples=13,
        seed=47,
        n_outliers=40,
        outlier_factor=5.0,
        disp_scatter=0.3,
    )
    dds = helpers.make_dds(sim.counts, sim.metadata)
    dds.deseq2()
    n, p = sim.design_matrix.shape
    ratio = dds.layers["cooks"].max(axis=0) / stats.f.ppf(0.99, p, n - p)
    assert ((ratio > 0.9) & (ratio < 0.99)).sum() >= 2
    assert ((ratio > 1.01) & (ratio < 1.1)).sum() >= 1
    assert (np.abs(ratio - 1) > 0.01).all()
    np.testing.assert_array_equal(dds.var["replaced"].to_numpy(), ratio > 1)


# DeseqDataSet.deseq2: invariances and a custom Inference


def test_deseq2_gene_order_invariance():
    """Permuting the genes permutes every per-gene output (refit included)."""
    _, dds = _fitted("TWO_GROUP")
    _, permuted = _fitted("TWO_GROUP", permutation_seed=0)
    assert not permuted.var_names.equals(dds.var_names)
    _assert_same_fit(permuted, dds, name="permuted genes")
    pd.testing.assert_series_equal(
        permuted.obs["size_factors"], dds.obs["size_factors"], rtol=1e-12
    )


def test_deseq2_n_cpus_and_batch_size_invariance(n_cpus):
    """One worker vs several with small batches (two workers when the tests use one).

    Workers start fresh processes (0.7 s measured for two), so one dataset and one
    comparison.
    """
    _, serial = _fitted("TWO_GROUP", inference=DefaultInference(n_cpus=1))
    workers = max(2, n_cpus)
    _, parallel = _fitted(
        "TWO_GROUP", inference=DefaultInference(n_cpus=workers, batch_size=3)
    )
    _assert_same_fit(parallel, serial, name=f"n_cpus={workers}, batch_size=3")


def test_deseq2_low_memory_same_results():
    """``low_memory=True`` drops intermediate layers, not results."""
    _, dds = _fitted("TWO_GROUP")
    _, lean = _fitted("TWO_GROUP", low_memory=True)
    for field in ["genewise_dispersions", "MAP_dispersions", "dispersions"]:
        helpers.assert_fields_close(
            lean.var[field], dds.var[field], rtol=DISPERSION_RTOL, name=field
        )
    helpers.assert_fields_close(
        lean.varm["LFC"], dds.varm["LFC"], rtol=0, atol=LFC_ATOL, name="LFC"
    )
    assert lean.var["refitted"].equals(dds.var["refitted"])
    assert "cooks" not in lean.layers


class _ReferenceInference(DefaultInference):
    """DefaultInference whose ``irls`` and ``lin_reg_mu`` are the test references."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_irls_calls = 0

    def lin_reg_mu(self, counts, size_factors, design_matrix, min_mu):
        coef = np.linalg.lstsq(design_matrix, counts / size_factors[:, None], rcond=None)
        return np.maximum(size_factors[:, None] * (design_matrix @ coef[0]), min_mu)

    def irls(self, counts, size_factors, design_matrix, disp, min_mu, beta_tol, **kw):
        beta = np.array(
            [
                helpers.reference_glm_mle(counts[:, i], size_factors, design_matrix, d)
                for i, d in enumerate(disp)
            ]
        )
        mu = size_factors[:, None] * np.exp(design_matrix @ beta.T)
        hat = np.column_stack(
            [
                _hat_diagonal(design_matrix, mu[:, i], d, min_mu)
                for i, d in enumerate(disp)
            ]
        )
        self.n_irls_calls += 1
        return beta, mu, hat, np.ones(len(disp), dtype=bool)


def test_deseq2_uses_custom_inference_irls(n_cpus):
    """A custom Inference's ``irls`` and ``lin_reg_mu`` results reach the outputs.

    Counts with means above 50 keep every fit clear of ``min_mu``, so the reference
    solvers agree with the default ones within the solvers' precision.
    """
    sim = helpers.simulate_counts(
        n_genes=20, n_samples=8, seed=39, mean_range=(50.0, 5000.0), disp_scatter=0.3
    )
    custom = _ReferenceInference(n_cpus=n_cpus)
    dds = helpers.make_dds(sim.counts, sim.metadata, inference=custom)
    dds.deseq2()
    default = helpers.make_dds(sim.counts, sim.metadata)
    default.deseq2()

    # fit_LFC stores what irls returned, gene by gene. The reference Newton solver
    # is deterministic, so rerunning it on the stored dispersions and size factors
    # reproduces it to rounding; 1e-10 allows for the inputs reaching irls in another
    # layout, and is far below the ~1e-5 by which IRLS's own fit differs.
    assert custom.n_irls_calls > 0
    x = sim.design_matrix.to_numpy()
    sf = dds.obs["size_factors"].to_numpy()
    expected = [
        helpers.reference_glm_mle(sim.counts[g].to_numpy(), sf, x, d)
        for g, d in dds.var["dispersions"].items()
    ]
    helpers.assert_fields_close(
        dds.varm["LFC"],
        pd.DataFrame(expected, index=dds.var_names, columns=dds.varm["LFC"].columns),
        rtol=0,
        atol=1e-10,
        name="LFC from the custom irls",
    )
    np.testing.assert_allclose(
        dds.layers["cooks"], _cooks_reference(dds), rtol=COOKS_RTOL
    )
    helpers.assert_fields_close(
        dds.var["dispersions"], default.var["dispersions"], rtol=DISPERSION_RTOL
    )
    helpers.assert_fields_close(
        dds.varm["LFC"], default.varm["LFC"], rtol=0, atol=LFC_ATOL, name="LFC"
    )
