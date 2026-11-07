"""Area B: dispersions (see tests/TARGETS.md).

Module-level functions and ``DefaultInference`` methods. The ``DeseqDataSet``
dispersion steps are in ``test_unit_dispersions_dds.py``, which imports this module's
tolerances; the reference implementations both use are in ``tests/helpers.py``.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pydeseq2 import inference as inference_module
from pydeseq2.default_inference import DefaultInference
from pydeseq2.dispersions import dispersion_trend
from pydeseq2.dispersions import fit_alpha_mle
from pydeseq2.dispersions import fit_moments_dispersions
from pydeseq2.dispersions import fit_rough_dispersions
from pydeseq2.distributions import dnb_nll
from pydeseq2.grid_search import grid_fit_alpha
from pydeseq2.grid_search import vec_nb_nll
from pydeseq2.stats import mean_absolute_deviation
from tests import helpers

pytestmark = helpers.PANDAS_DEPRECATION_FILTERS

# fit_alpha_mle stops L-BFGS-B at a projected gradient of 1e-5 in log dispersion,
# so its error is about 1e-5 over the curvature of the objective. Against the
# reference minimizer (Cox-Reid on) that left relative errors up to 9.5e-5 on 450
# simulated genes with 6-40 samples (2e-4 on genes with a mean of 1e7); 1e-3 leaves
# a 5x margin.
DISPERSION_RTOL = 1e-3

# Where the objective is flat in log dispersion (near-Poisson genes, whose minimizer
# is at or near the lower bound), L-BFGS-B stops when the gradient in log dispersion
# falls below 1e-5, which leaves an objective gap of about that size (measured: up
# to 9.4e-6). Such genes are compared by objective value with a 10x margin.
OBJECTIVE_ATOL = 1e-4

# Genes whose objective decreases towards a bound. L-BFGS-B projects onto the bound
# and returns it to rounding, but a solver that stops on a projected gradient of 1e-5
# in log dispersion may stop up to 1e-5 inside it, which is up to 1e-5 relative in
# dispersion; 1e-4 leaves a 10x margin and still tells the bounds apart.
BOUND_RTOL = 1e-4

# When L-BFGS-B fails on a MAP objective, fit_alpha_mle's grid fallback drops the
# prior (a known issue) and returns the gene-wise estimate. Which genes fail moves
# with changes at the level of rounding (an equivalent form of the likelihood, a
# solver upstream that stops elsewhere within its tolerance), so a comparison of MAP
# estimates leaves such genes out, and at most this many of them.
MAX_MAP_FALLBACKS = 2


def _grid_log_tolerance(min_disp, max_disp, grid_length=100):
    """Error bound of ``grid_fit_alpha`` in log dispersion, plus a 2x margin.

    The coarse grid has spacing ``delta``; the fine grid spans the two coarse cells
    around the coarse minimum with spacing ``2 delta / (grid_length - 1)``, so for a
    unimodal objective the fine minimum is within half a fine step of the minimizer.
    One full fine step is returned.
    """
    delta = (np.log(max_disp) - np.log(min_disp)) / (grid_length - 1)
    return 2 * delta / (grid_length - 1)


# Pilot tests (kept).


@pytest.mark.parametrize(
    "n_samples, seed, gene",
    [(6, 1, "gene1"), (6, 1, "gene2"), (12, 2, "gene3"), (40, 3, "gene4")],
)
def test_fit_alpha_mle_matches_reference(n_samples, seed, gene):
    """One gene's Cox-Reid adjusted dispersion MLE, given the true means.

    Without Cox-Reid the objective of near-Poisson genes is flat enough that
    L-BFGS-B stops far from the minimizer (relative errors above 100 with objective
    gaps below 1e-5), so such tests need an objective-gap criterion instead.
    """
    sim = helpers.simulate_counts(
        n_genes=4, n_samples=n_samples, seed=seed, disp_scatter=0.5
    )
    x = sim.design_matrix.to_numpy()
    y, mu = sim.counts[gene].to_numpy(), sim.mu[gene].to_numpy()

    alpha, converged = fit_alpha_mle(
        counts=y,
        design_matrix=x,
        mu=mu,
        alpha_hat=0.1,
        min_disp=1e-8,
        max_disp=10.0,
    )
    expected = helpers.reference_dispersion_mle(y, mu, x, min_disp=1e-8, max_disp=10.0)

    assert converged
    assert alpha == pytest.approx(expected, rel=DISPERSION_RTOL)


def test_fit_alpha_mle_with_prior_matches_reference():
    """MAP dispersion: the prior is centred on ``alpha_hat`` (the trend value)."""
    sim = helpers.simulate_counts(n_genes=3, n_samples=8, seed=4, disp_scatter=0.5)
    x = sim.design_matrix.to_numpy()
    for gene in sim.counts.columns:
        y, mu = sim.counts[gene].to_numpy(), sim.mu[gene].to_numpy()
        prior = 0.5 * sim.dispersions[gene]
        alpha, converged = fit_alpha_mle(
            counts=y,
            design_matrix=x,
            mu=mu,
            alpha_hat=prior,
            min_disp=1e-8,
            max_disp=10.0,
            prior_disp_var=0.25,
            cr_reg=True,
            prior_reg=True,
        )
        expected = helpers.reference_dispersion_mle(
            y, mu, x, prior_disp_var=0.25, prior_dispersion=prior
        )
        assert converged
        assert alpha == pytest.approx(expected, rel=DISPERSION_RTOL), gene


def test_genewise_dispersions_match_reference():
    """``var["genewise_dispersions"]`` against the reference, gene by gene.

    With a two-group design PyDESeq2 takes the means from a linear model on the
    normalized counts, which reduces to the group means times the size factor,
    floored at ``min_mu=0.5``; the reference recomputes them that way. Dispersions
    are clipped to ``[min_disp, max(max_disp, n_samples)]``.

    Genes whose starting value, DESeq2's minimum of the rough and moments estimates,
    is at ``min_disp`` are left out: the objective is flat in log dispersion there,
    and PyDESeq2, like DESeq2 1.50.2, stays at ``min_disp`` even when the minimizer
    is elsewhere (for example 1e-8 instead of 0.034 for counts 18, 2, 10, 4, 14, 6,
    19, 7, 5, 5 in two alternating groups).
    """
    sim = helpers.simulate_counts(n_genes=60, n_samples=10, seed=5, disp_scatter=0.3)
    dds = helpers.make_dds(sim.counts, sim.metadata)
    dds.fit_size_factors()
    dds.fit_genewise_dispersions()

    sf = dds.obs["size_factors"].to_numpy()
    normed = sim.counts.to_numpy() / sf[:, None]
    groups = sim.metadata["condition"].to_numpy()
    group_means = np.empty_like(normed)
    for level in np.unique(groups):
        group_means[groups == level] = normed[groups == level].mean(axis=0)
    mu_hat = np.maximum(sf[:, None] * group_means, 0.5)

    # DESeq2's roughDispEstimate and momentsDispEstimate.
    n, p = sim.design_matrix.shape
    fitted = np.maximum(group_means, 1.0)
    rough = (((normed - fitted) ** 2 - fitted) / fitted**2).sum(axis=0) / (n - p)
    means = normed.mean(axis=0)
    moments = (normed.var(axis=0, ddof=1) - (1 / sf).mean() * means) / means**2
    keep = np.minimum(rough, moments) > 1e-8
    assert keep.sum() >= 50

    x = sim.design_matrix.to_numpy()
    max_disp = max(10.0, n)
    expected = np.array(
        [
            helpers.reference_dispersion_mle(
                sim.counts.iloc[:, j].to_numpy(), mu_hat[:, j], x, max_disp=max_disp
            )
            for j in np.flatnonzero(keep)
        ]
    )
    helpers.assert_fields_close(
        dds.var["genewise_dispersions"][keep],
        expected,
        rtol=DISPERSION_RTOL,
        name="genewise_dispersions",
    )


# dispersion_trend


def test_dispersion_trend_hand_worked():
    """``a0 + a1 / mean`` with ``a0 = 0.5``, ``a1 = 2``: 2.5, 1.5 and 1 at 1, 2, 4.

    Every value is exact in binary floating point; 1e-15 allows one rounding.
    """
    means = np.array([1.0, 2.0, 4.0])
    expected = np.array([2.5, 1.5, 1.0])
    series_coeffs = pd.Series([0.5, 2.0], index=["a0", "a1"])
    np.testing.assert_allclose(
        dispersion_trend(means, series_coeffs), expected, rtol=1e-15
    )
    np.testing.assert_allclose(
        dispersion_trend(means, np.array([0.5, 2.0])), expected, rtol=1e-15
    )
    assert dispersion_trend(4.0, series_coeffs) == pytest.approx(1.0, rel=1e-15)
    # A Series is read by label, not position.
    swapped = pd.Series([2.0, 0.5], index=["a1", "a0"])
    np.testing.assert_allclose(dispersion_trend(means, swapped), expected, rtol=1e-15)


# Rough and moments estimates


def _rough_design():
    # Eight samples, intercept + two-level factor + continuous covariate (p = 3).
    condition = np.tile([0.0, 1.0], 4)
    age = np.array([-1.2, 0.3, 0.8, -0.5, 1.5, -0.9, 0.1, 0.6])
    return np.column_stack([np.ones(8), condition, age])


def _rough_counts():
    # Normalized counts for five genes (samples x genes): an overdispersed gene, a
    # near-Poisson one, a constant one (rough estimate negative, floored at 0), one
    # whose fitted values fall below 1 (floored at 1), and a large one.
    return np.array(
        [
            [12.0, 50.0, 7.0, 0.0, 1200.0],
            [80.0, 47.0, 7.0, 1.0, 1500.0],
            [5.0, 55.0, 7.0, 0.0, 900.0],
            [40.0, 52.0, 7.0, 2.0, 2100.0],
            [9.0, 49.0, 7.0, 0.0, 1100.0],
            [150.0, 51.0, 7.0, 0.0, 1300.0],
            [20.0, 44.0, 7.0, 1.0, 1000.0],
            [60.0, 53.0, 7.0, 0.0, 1700.0],
        ]
    )


def test_fit_rough_dispersions_matches_definition():
    """Rough estimates against DESeq2's ``roughDispEstimate`` written out here.

    The reference projects with ``numpy.linalg.lstsq`` instead of scikit-learn;
    both are exact least squares, so they agree to rounding. The rough estimate is
    a sum of differences of comparable terms, so 1e-9 relative (plus 1e-12
    absolute for values near 0) leaves room for cancellation.
    """
    x = _rough_design()
    normed = _rough_counts()
    expected = helpers.rough_dispersions(normed, x)

    # The inputs reach the two floors the definition has.
    assert expected[2] == 0.0
    fitted = x @ np.linalg.lstsq(x, normed, rcond=None)[0]
    assert (fitted[:, 3] < 1).any()

    got = fit_rough_dispersions(normed, x)
    assert got.shape == (5,)
    np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-12)
    # The documented input type is a DataFrame; the result is the same.
    got_df = fit_rough_dispersions(normed, pd.DataFrame(x))
    np.testing.assert_allclose(got_df, expected, rtol=1e-9, atol=1e-12)


def test_fit_rough_dispersions_requires_residual_degrees_of_freedom():
    """With as many samples as coefficients there is nothing to estimate from."""
    x = np.column_stack([np.ones(3), [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="no replicates"):
        fit_rough_dispersions(np.ones((3, 2)), x)


def test_fit_moments_dispersions_hand_worked():
    """Moments estimates ``(var - mean(1/s) mean) / mean^2``, worked by hand.

    Size factors 0.5, 1, 2 give ``mean(1/s) = 7/6``. Gene g1 = (2, 4, 6) has mean 4
    and variance 4: ``(4 - 14/3) / 16 = -1/24``; the estimate is not floored.
    g2 is all zero and is dropped from the output. g3 = (10, 10, 40) has mean 20
    and variance 300: ``(300 - 70/3) / 400 = 83/120``. 1e-13 relative covers the
    rounding of the mean and variance.
    """
    normed = np.array([[2.0, 0.0, 10.0], [4.0, 0.0, 10.0], [6.0, 0.0, 40.0]])
    size_factors = np.array([0.5, 1.0, 2.0])
    got = fit_moments_dispersions(normed, size_factors)
    np.testing.assert_allclose(got, [-1 / 24, 83 / 120], rtol=1e-13)


def test_inference_dispersion_estimators(inference):
    """``DefaultInference`` exposes the rough and moments estimators.

    Checked against the definitions written out in this file, at the tolerances of
    the two tests above.
    """
    x = _rough_design()
    normed = _rough_counts()
    np.testing.assert_allclose(
        inference.fit_rough_dispersions(normed, x),
        helpers.rough_dispersions(normed, x),
        rtol=1e-9,
        atol=1e-12,
    )
    sf = np.array([0.6, 1.3, 0.9, 1.1, 0.8, 1.4, 1.0, 1.2])
    np.testing.assert_allclose(
        inference.fit_moments_dispersions(normed, sf),
        helpers.moments_dispersions(normed, sf),
        rtol=1e-12,
    )


def test_inference_interface_declares_dispersion_methods():
    """The abstract ``Inference`` interface requires the four dispersion methods.

    A custom inference class must implement them; ``DefaultInference`` does.
    """
    required = inference_module.Inference.__abstractmethods__
    for name in (
        "alpha_mle",
        "dispersion_trend_gamma_glm",
        "fit_moments_dispersions",
        "fit_rough_dispersions",
    ):
        assert name in required
    assert isinstance(DefaultInference(n_cpus=1), inference_module.Inference)


# mean_absolute_deviation


def test_mean_absolute_deviation_hand_worked():
    """Median absolute deviation scaled by ``1 / Phi^-1(3/4)``, as R's ``mad``.

    (1, 2, 3, 4, 100): median 3, absolute deviations (2, 1, 0, 1, 97), median 1.
    (1, 2, 4, 8): median 3, deviations (2, 1, 1, 5), median 1.5. The scale
    1.482602218505602 is R's ``1 / qnorm(0.75)``. Also checked against
    ``scipy.stats.median_abs_deviation(scale="normal")``; 1e-13 covers rounding.
    """
    scale = 1.482602218505602
    x1 = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    x2 = np.array([8.0, 1.0, 4.0, 2.0])
    assert mean_absolute_deviation(x1) == pytest.approx(scale, rel=1e-13)
    assert mean_absolute_deviation(x2) == pytest.approx(1.5 * scale, rel=1e-13)
    rs = np.random.RandomState(0)
    x3 = rs.standard_t(3, size=101)
    assert mean_absolute_deviation(x3) == pytest.approx(
        stats.median_abs_deviation(x3, scale="normal"), rel=1e-13
    )


# dnb_nll and vec_nb_nll


@pytest.mark.parametrize(
    "counts, mu, alpha",
    [
        ([0, 1, 3], [2.0, 2.0, 2.0], 1.0),
        ([0, 4, 17, 9, 2, 30], [5.0, 6.0, 12.0, 8.0, 3.0, 15.0], 0.3),
        ([812, 1530, 990, 2210], [1000.0, 1400.0, 1100.0, 1800.0], 0.05),
        ([48, 55, 61, 40, 52], [50.0, 50.0, 55.0, 45.0, 52.0], 1e-3),
        ([0, 0, 1, 0], [0.5, 0.7, 0.6, 0.5], 4.0),
    ],
)
def test_dnb_nll_matches_finite_difference(counts, mu, alpha):
    """``dnb_nll`` is the derivative in ``alpha`` of the NB negative log-likelihood.

    The expected value differentiates ``helpers.nb_loglik`` (which shares no code
    with PyDESeq2) numerically: central differences at steps ``h = 1e-3 alpha`` and
    ``h / 2``, combined by Richardson extrapolation, which cancels the ``h^2`` error
    term and leaves about 1e-12 relative. The log-likelihood itself is accurate to
    about 1e-11 absolute, which perturbs the difference quotient by at most 1e-7
    relative here; 1e-6 relative is the tolerance.
    """
    counts = np.asarray(counts)
    mu = np.asarray(mu)

    def central(h):
        upper = helpers.nb_loglik(counts, mu, alpha + h)
        lower = helpers.nb_loglik(counts, mu, alpha - h)
        return -(upper - lower) / (2 * h)

    h = 1e-3 * alpha
    expected = (4 * central(h / 2) - central(h)) / 3
    assert dnb_nll(counts, mu, alpha) == pytest.approx(expected, rel=1e-6, abs=1e-9)


def test_vec_nb_nll_matches_reference():
    """``vec_nb_nll`` evaluates the NB negative log-likelihood on a grid at once.

    With ``mu`` of shape (samples,) and a vector of dispersions it returns one value
    per dispersion; with ``mu`` of shape (samples, k) and one dispersion it returns
    one value per column. Expected values come from ``helpers.nb_loglik``. For
    dispersions of 1e-3 and above both forms are accurate to about 1e-13 relative;
    1e-10 leaves room for the cancellation in PyDESeq2's form.
    """
    counts = np.array([0, 3, 12, 7, 25, 1, 9])
    mu = np.array([2.0, 4.0, 10.0, 6.0, 20.0, 1.5, 8.0])
    alphas = np.logspace(-3, 1, 9)
    expected = np.array([-helpers.nb_loglik(counts, mu, a) for a in alphas])
    got = vec_nb_nll(counts, mu, alphas)
    assert got.shape == (9,)
    np.testing.assert_allclose(got, expected, rtol=1e-10)

    mu_grid = mu[:, None] * np.array([0.5, 1.0, 2.0, 4.0])[None, :]
    expected = np.array([-helpers.nb_loglik(counts, m, 0.2) for m in mu_grid.T])
    got = vec_nb_nll(counts, mu_grid, 0.2)
    assert got.shape == (4,)
    np.testing.assert_allclose(got, expected, rtol=1e-10)


# grid_fit_alpha


@pytest.mark.parametrize(
    "kwargs, min_disp, max_disp, grid_length",
    [
        ({}, 1e-8, 10.0, 100),
        ({"cr_reg": False}, 1e-8, 10.0, 100),
        ({"prior_reg": True, "prior_disp_var": 0.3}, 1e-8, 10.0, 100),
        ({}, 1e-4, 5.0, 40),
    ],
)
def test_grid_fit_alpha_matches_reference(kwargs, min_disp, max_disp, grid_length):
    """The grid fallback against the reference minimizer of the same objective.

    ``grid_fit_alpha`` returns the log dispersion; its error is bounded by the fine
    grid's step (see ``_grid_log_tolerance``), plus the reference's own 1e-6. The
    genes are overdispersed (true dispersion 0.2 + 4 / mean, scatter 0.5), so the
    objective has an interior minimum also without Cox-Reid. The prior is centred
    on ``alpha_hat``, which differs per gene.
    """
    sim = helpers.simulate_counts(
        n_genes=6, n_samples=10, seed=41, disp_scatter=0.5, trend=(0.2, 4.0)
    )
    x = sim.design_matrix.to_numpy()
    atol = _grid_log_tolerance(min_disp, max_disp, grid_length) + 1e-6
    for gene in sim.counts.columns:
        y, mu = sim.counts[gene].to_numpy(), sim.mu[gene].to_numpy()
        alpha_hat = 0.7 * sim.dispersions[gene]
        got = grid_fit_alpha(
            y, x, mu, alpha_hat, min_disp, max_disp, grid_length=grid_length, **kwargs
        )
        expected = helpers.reference_dispersion_mle(
            y,
            mu,
            x,
            min_disp=min_disp,
            max_disp=max_disp,
            cr_reg=kwargs.get("cr_reg", True),
            prior_disp_var=kwargs.get("prior_disp_var"),
            prior_dispersion=alpha_hat if kwargs.get("prior_reg") else None,
        )
        assert got == pytest.approx(np.log(expected), abs=atol), gene


def test_grid_fit_alpha_finds_boundary_minimum():
    """A gene whose objective decreases up to ``max_disp`` gets the upper edge.

    One positive count among eight samples with a constant mean: the Cox-Reid
    adjusted objective decreases in alpha over the whole range (checked here on the
    reference objective), so the grid minimum is in the last coarse cell. The fine
    grid extends one coarse step past each end of that cell, so the result lies
    within one coarse step of ``log(max_disp)`` on either side.
    """
    x = np.column_stack([np.ones(8), np.tile([0.0, 1.0], 4)])
    y = np.array([0, 0, 0, 40, 0, 0, 0, 0])
    mu = np.full(8, 5.0)
    objective = helpers.dispersion_objective(y, mu, x)
    log_grid = np.linspace(np.log(1e-8), np.log(10.0), 200)
    assert (np.diff(objective(log_grid)) < 0).all()

    got = grid_fit_alpha(y, x, mu, 0.1, 1e-8, 10.0)
    delta = (np.log(10.0) - np.log(1e-8)) / 99
    assert abs(got - np.log(10.0)) <= delta + 1e-12


# fit_alpha_mle: bounds, objectives without Cox-Reid, BFGS, non-convergence


def _two_group_design(n_samples):
    return np.column_stack([np.ones(n_samples), np.tile([0.0, 1.0], n_samples // 2)])


def test_fit_alpha_mle_at_max_disp():
    """The objective decreases up to ``max_disp`` (see the grid test above), so the
    bounded optimizer stops on the bound: the reference minimizer is ``max_disp`` to
    rounding (1e-12), and ``fit_alpha_mle`` is within ``BOUND_RTOL`` of it."""
    x = _two_group_design(8)
    y = np.array([0, 0, 0, 40, 0, 0, 0, 0])
    mu = np.full(8, 5.0)
    assert helpers.reference_dispersion_mle(y, mu, x) == pytest.approx(10.0, rel=1e-12)
    for max_disp in (10.0, 3.0):
        alpha, converged = fit_alpha_mle(y, x, mu, 0.1, 1e-8, max_disp)
        assert converged
        assert alpha == pytest.approx(max_disp, rel=BOUND_RTOL)


def test_fit_alpha_mle_at_min_disp():
    """An underdispersed gene: the objective increases in alpha from the lower bound.

    With ``min_disp=0.01`` the reference minimizer is on the bound (to rounding,
    1e-12) and the optimizer stops there (``BOUND_RTOL``). With ``min_disp=1e-8`` the
    objective is flat in log dispersion near the bound, so L-BFGS-B stops above it
    (near 3e-7 here); the test compares objective values there (``OBJECTIVE_ATOL``).
    """
    x = _two_group_design(8)
    y = np.array([10, 11, 9, 10, 10, 11, 9, 10])
    mu = np.full(8, 10.0)
    objective = helpers.dispersion_objective(y, mu, x)

    expected = helpers.reference_dispersion_mle(y, mu, x, min_disp=0.01)
    assert expected == pytest.approx(0.01, rel=1e-12)
    for alpha_hat in (0.1, 1.0):
        alpha, converged = fit_alpha_mle(y, x, mu, alpha_hat, 0.01, 10.0)
        assert converged
        assert alpha == pytest.approx(0.01, rel=BOUND_RTOL)

        alpha, converged = fit_alpha_mle(y, x, mu, alpha_hat, 1e-8, 10.0)
        expected = helpers.reference_dispersion_mle(y, mu, x)
        assert converged
        assert 1e-8 <= alpha < 1e-3
        assert objective(np.log(alpha)) - objective(np.log(expected)) <= OBJECTIVE_ATOL


def test_fit_alpha_mle_without_cox_reid():
    """``cr_reg=False`` minimizes the plain negative log-likelihood.

    Compared by objective value (``OBJECTIVE_ATOL``) on every gene, near-Poisson
    ones included; and by value (``DISPERSION_RTOL``) on genes whose reference
    dispersion is at least 0.05, where the objective has curvature (measured
    agreement 1.4e-4 relative or better).
    """
    sim = helpers.simulate_counts(n_genes=12, n_samples=10, seed=42, disp_scatter=1.0)
    x = sim.design_matrix.to_numpy()
    n_curved = 0
    for gene in sim.counts.columns:
        y, mu = sim.counts[gene].to_numpy(), sim.mu[gene].to_numpy()
        alpha, converged = fit_alpha_mle(y, x, mu, 0.1, 1e-8, 10.0, cr_reg=False)
        expected = helpers.reference_dispersion_mle(y, mu, x, cr_reg=False)
        objective = helpers.dispersion_objective(y, mu, x, cr_reg=False)
        assert converged, gene
        gap = objective(np.log(alpha)) - objective(np.log(expected))
        assert gap <= OBJECTIVE_ATOL, gene
        if expected >= 0.05:
            n_curved += 1
            assert alpha == pytest.approx(expected, rel=DISPERSION_RTOL), gene
    assert n_curved >= 6


def test_fit_alpha_mle_bfgs():
    """``optimizer="BFGS"`` (unbounded, ``gtol`` 1e-5) reaches the same minimizer
    on genes with interior minima, at the L-BFGS-B tolerance."""
    sim = helpers.simulate_counts(n_genes=6, n_samples=10, seed=43, disp_scatter=0.5)
    x = sim.design_matrix.to_numpy()
    for gene in sim.counts.columns:
        y, mu = sim.counts[gene].to_numpy(), sim.mu[gene].to_numpy()
        alpha, converged = fit_alpha_mle(y, x, mu, 0.1, 1e-8, 10.0, optimizer="BFGS")
        expected = helpers.reference_dispersion_mle(y, mu, x)
        assert converged, gene
        assert alpha == pytest.approx(expected, rel=DISPERSION_RTOL), gene


@pytest.mark.parametrize(
    "n_samples, seed, genes",
    [
        (24, 21, ["gene18", "large2", "gene1", "large1"]),
        (24, 13, ["gene24", "gene2"]),
        (6, 25, ["large2", "large3", "gene3"]),
    ],
)
def test_fit_alpha_mle_converged_or_grid(inference, n_samples, seed, genes):
    """Every gene lands near the minimizer, by L-BFGS-B or by the grid fallback.

    The genes include ones where L-BFGS-B, as PyDESeq2 calls it, stops with an
    abnormal line search (gene18 and large2 of the first case, gene24 of the second,
    large2 and large3 of the third, under scipy 1.15.0 and 1.18.1): their objectives
    are flat to rounding near the minimizer (means up to 1e7). The documented
    contract is checked either way, through ``fit_alpha_mle`` and through
    ``DefaultInference.alpha_mle`` (which reports the flag per gene): a converged
    fit is within ``DISPERSION_RTOL`` of the reference, and a non-converged one is
    the grid estimate, within the grid's resolution of it
    (``_grid_log_tolerance``). The grid estimate of gene24 is 1.7e-3 from the
    reference, outside ``DISPERSION_RTOL``, so a grid estimate reported as converged
    for it fails. Which genes fail is not asserted.
    """
    sim = helpers.simulate_counts(
        n_genes=100, n_samples=n_samples, seed=seed, disp_scatter=0.5, n_large=3
    )
    x = sim.design_matrix.to_numpy()
    grid_atol = _grid_log_tolerance(1e-8, 10.0) + 1e-6
    expected = np.array(
        [
            helpers.reference_dispersion_mle(
                sim.counts[gene].to_numpy(), sim.mu[gene].to_numpy(), x
            )
            for gene in genes
        ]
    )

    def check(alpha, converged, expected, gene):
        if converged:
            assert alpha == pytest.approx(expected, rel=DISPERSION_RTOL), gene
        else:
            assert np.log(alpha) == pytest.approx(np.log(expected), abs=grid_atol), gene

    for gene, value in zip(genes, expected, strict=True):
        y, mu = sim.counts[gene].to_numpy(), sim.mu[gene].to_numpy()
        alpha, converged = fit_alpha_mle(y, x, mu, 0.1, 1e-8, 10.0)
        assert isinstance(converged, (bool, np.bool_))
        check(alpha, converged, value, gene)

    disp, converged = inference.alpha_mle(
        counts=sim.counts[genes].to_numpy(),
        design_matrix=x,
        mu=sim.mu[genes].to_numpy(),
        alpha_hat=np.full(len(genes), 0.1),
        min_disp=1e-8,
        max_disp=10.0,
    )
    for gene, alpha, flag, value in zip(genes, disp, converged, expected, strict=True):
        check(alpha, flag, value, gene)


# DefaultInference.alpha_mle: many genes at once


def _alpha_mle_case(n_genes=60, n_samples=10, seed=44, n_large=2):
    sim = helpers.simulate_counts(
        n_genes=n_genes,
        n_samples=n_samples,
        seed=seed,
        disp_scatter=0.5,
        n_large=n_large,
    )
    # A prior centre that differs from gene to gene, so a gene that is paired with
    # another gene's prior gets a different answer.
    scatter = np.random.RandomState(seed).normal(scale=0.7, size=sim.counts.shape[1])
    prior = sim.dispersions.to_numpy() * np.exp(scatter)
    return sim, prior


def _reference_alpha_mle(sim, alpha_hat=None, prior_disp_var=None):
    x = sim.design_matrix.to_numpy()
    out = []
    for j, gene in enumerate(sim.counts.columns):
        y, mu = sim.counts[gene].to_numpy(), sim.mu[gene].to_numpy()
        out.append(
            helpers.reference_dispersion_mle(
                y,
                mu,
                x,
                prior_disp_var=prior_disp_var,
                prior_dispersion=None if prior_disp_var is None else alpha_hat[j],
            )
        )
    return np.array(out)


def test_alpha_mle_matches_reference(inference):
    """``DefaultInference.alpha_mle`` over 62 genes, given the true means.

    Gene-wise (no prior; ``alpha_hat`` is only the start) and MAP (prior centred on a
    different ``alpha_hat`` per gene, ``prior_disp_var`` 0.3). Converged genes are
    within ``DISPERSION_RTOL`` of the reference. Gene-wise genes that did not
    converge are within the grid fallback's resolution. MAP genes that did not
    converge are left out: ``fit_alpha_mle`` calls the grid fallback without the
    prior (``prior_disp_var``, ``cr_reg`` and ``prior_reg`` are not passed on), so
    those genes get the gene-wise estimate instead of the MAP one, where DESeq2's
    grid fallback keeps the prior. At least 95% must converge.
    """
    sim, prior = _alpha_mle_case()
    y = sim.counts.to_numpy()
    x = sim.design_matrix.to_numpy()
    mu = sim.mu.to_numpy()
    genes = sim.counts.columns

    disp, converged = inference.alpha_mle(
        counts=y, design_matrix=x, mu=mu, alpha_hat=prior, min_disp=1e-8, max_disp=10.0
    )
    assert disp.shape == converged.shape == (len(genes),)
    expected = _reference_alpha_mle(sim)
    conv = converged.astype(bool)
    helpers.assert_fields_close(
        pd.Series(disp[conv], index=genes[conv]),
        pd.Series(expected[conv], index=genes[conv]),
        rtol=DISPERSION_RTOL,
        name="alpha_mle",
    )
    grid_atol = _grid_log_tolerance(1e-8, 10.0) + 1e-6
    np.testing.assert_allclose(
        np.log(disp[~conv]), np.log(expected[~conv]), rtol=0, atol=grid_atol
    )

    disp, converged = inference.alpha_mle(
        counts=y,
        design_matrix=x,
        mu=mu,
        alpha_hat=prior,
        min_disp=1e-8,
        max_disp=10.0,
        prior_disp_var=0.3,
        cr_reg=True,
        prior_reg=True,
    )
    conv = converged.astype(bool)
    assert conv.mean() >= 0.95
    expected = _reference_alpha_mle(sim, alpha_hat=prior, prior_disp_var=0.3)
    helpers.assert_fields_close(
        pd.Series(disp[conv], index=genes[conv]),
        pd.Series(expected[conv], index=genes[conv]),
        rtol=DISPERSION_RTOL,
        name="alpha_mle MAP",
    )


def _run_alpha_mle(inference, sim, prior, order):
    disp, converged = inference.alpha_mle(
        counts=sim.counts.to_numpy()[:, order],
        design_matrix=sim.design_matrix.to_numpy(),
        mu=sim.mu.to_numpy()[:, order],
        alpha_hat=prior[order],
        min_disp=1e-8,
        max_disp=10.0,
        prior_disp_var=0.3,
        cr_reg=True,
        prior_reg=True,
    )
    inverse = np.argsort(order)
    return disp[inverse], converged[inverse]


@pytest.mark.parametrize(
    "variant",
    ["permuted", "batch_size=1", "batch_size=7", "n_cpus=2", "n_cpus=2,batch_size=5"],
)
def test_alpha_mle_invariance(n_cpus, variant):
    """Gene order, ``batch_size`` and ``n_cpus`` do not change the estimates.

    MAP estimates for 40 genes, each with its own prior centre, from a reference
    run (``batch_size`` 128, genes in order) and from the variant; results are
    mapped back to the original order. Genes converged in both runs agree within
    ``DISPERSION_RTOL``, the precision a reorganized solver is held to. Genes that
    did not converge in one of the runs took the grid fallback there, which drops
    the prior; they are left out, at most ``MAX_MAP_FALLBACKS`` of them.
    """
    sim, prior = _alpha_mle_case(n_genes=40, n_large=0, seed=45)
    order = np.arange(40)
    base_disp, base_conv = _run_alpha_mle(
        DefaultInference(n_cpus=n_cpus), sim, prior, order
    )

    settings = dict(item.split("=") for item in variant.split(",") if "=" in item)
    variant_inference = DefaultInference(
        n_cpus=int(settings.get("n_cpus", n_cpus)),
        batch_size=int(settings.get("batch_size", 128)),
    )
    if variant == "permuted":
        order = np.random.RandomState(0).permutation(40)
    disp, conv = _run_alpha_mle(variant_inference, sim, prior, order)

    both = base_conv.astype(bool) & conv.astype(bool)
    assert (~both).sum() <= MAX_MAP_FALLBACKS
    genes = sim.counts.columns
    helpers.assert_fields_close(
        pd.Series(disp[both], index=genes[both]),
        pd.Series(base_disp[both], index=genes[both]),
        rtol=DISPERSION_RTOL,
        name=f"alpha_mle {variant}",
    )


@pytest.mark.parametrize("n_genes", [1, 2])
def test_alpha_mle_one_and_two_genes(inference, n_genes):
    """One and two genes: arrays of length 1 and 2, matching the reference; with
    two genes in reverse order the results come back reversed.

    A converged MAP fit is within ``DISPERSION_RTOL`` of the reference MAP. A fit
    reported as not converged comes from the grid fallback: with the prior it lands
    on the MAP minimizer, without it (as now, a known issue) on the gene-wise one.
    It is held to the interval between the two, widened by the grid's resolution.
    """
    sim, prior = _alpha_mle_case(n_genes=n_genes, n_large=0, seed=46)
    expected = _reference_alpha_mle(sim, alpha_hat=prior, prior_disp_var=0.3)
    genewise = np.log(_reference_alpha_mle(sim))
    grid_atol = _grid_log_tolerance(1e-8, 10.0) + 1e-6
    low = np.minimum(genewise, np.log(expected)) - grid_atol
    high = np.maximum(genewise, np.log(expected)) + grid_atol
    for order in (np.arange(n_genes), np.arange(n_genes)[::-1]):
        disp, converged = _run_alpha_mle(inference, sim, prior, order)
        assert disp.shape == converged.shape == (n_genes,)
        conv = converged.astype(bool)
        np.testing.assert_allclose(disp[conv], expected[conv], rtol=DISPERSION_RTOL)
        assert ((np.log(disp) >= low) & (np.log(disp) <= high))[~conv].all()


# DefaultInference.dispersion_trend_gamma_glm


def test_dispersion_trend_gamma_glm_matches_reference(inference):
    """One gamma GLM fit (identity link) of dispersions on ``1 / mean``.

    Compared with IRLS (``helpers.gamma_identity_glm``, run to 1e-13). The method minimizes
    the same objective with L-BFGS-B (default tolerances), which agreed with IRLS to
    better than 1e-5 relative on the trend fits measured; 1e-4 relative is the
    tolerance. Predictions are ``a0 + a1 / mean`` for the returned coefficients.
    """
    rs = np.random.RandomState(47)
    means = np.exp(rs.uniform(np.log(5), np.log(5000), size=300))
    truth = 0.08 + 3.0 / means
    disps = truth * rs.gamma(shape=4.0, scale=0.25, size=300)
    index = [f"gene{i}" for i in range(300)]
    covariates = pd.Series(1 / means, index=index)
    targets = pd.Series(disps, index=index)

    coeffs, predictions, converged = inference.dispersion_trend_gamma_glm(
        covariates, targets
    )
    x = np.column_stack([np.ones(300), 1 / means])
    expected = helpers.gamma_identity_glm(x, disps, np.array([0.1, 1.0]))
    assert converged
    np.testing.assert_allclose(coeffs, expected, rtol=1e-4)
    np.testing.assert_allclose(predictions, x @ np.asarray(coeffs), rtol=1e-12)
    # The fit recovers the generating curve to sampling precision (300 genes,
    # gamma noise with coefficient of variation 0.5): within 20%.
    np.testing.assert_allclose(coeffs, [0.08, 3.0], rtol=0.2)
