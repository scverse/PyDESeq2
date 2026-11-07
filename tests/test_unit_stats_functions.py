"""Area D: module-level functions for testing and shrinkage (see tests/TARGETS.md).

``glm.wald_test``, ``distributions.nbinomFn``, ``distributions.nbinomGLM``,
``grid_search.grid_fit_shrink_beta`` and ``stats.lowess``, each against a reference
in ``tests/helpers.py``: DESeq2's Wald test (sandwich covariance with a ridge),
apeglm's negative log posterior and its minimizer, and Cleveland's robust lowess.
"""

import numpy as np
import pytest
from scipy import stats

from pydeseq2 import glm
from pydeseq2.distributions import nbinomFn
from pydeseq2.distributions import nbinomGLM
from pydeseq2.grid_search import grid_fit_shrink_beta
from pydeseq2.stats import lowess
from tests import helpers

pytestmark = helpers.PANDAS_DEPRECATION_FILTERS

# nbinomGLM stops L-BFGS-B at ftol=gtol=1e-8 on the objective divided by its value
# at beta = 0. Against helpers.reference_map that left absolute errors up to 3.5e-3 on the
# shrunk natural-log coefficient (6.4e-3 on any coefficient) and relative objective
# gaps up to 8.5e-6, over 820 simulated genes (2 to 5 coefficients, 6 to 18
# samples, adaptive and unit prior scales). The tolerances below leave a 3x margin
# on the coefficients and 10x on the objective. They hold for dispersions well above
# 1e-6; at 1e-6 and below L-BFGS-B stops far from the minimum on the scaled
# objective and still reports success, so no test uses such dispersions.
SHRINK_ATOL = 2e-2
SHRINK_OBJECTIVE_RTOL = 1e-4


def _two_group_design(n_per_group):
    return np.column_stack(
        [np.ones(2 * n_per_group), np.repeat([0.0, 1.0], n_per_group)]
    )


# glm.wald_test


def test_wald_test_two_groups_hand_worked():
    """Two groups of two samples, no ridge, contrast on the group coefficient.

    ``W = mu / (1 + disp mu)`` is 1 in group A (mu 2, disp 0.5) and 1.6 in group B
    (mu 8). Without a ridge the sandwich reduces to ``M^-1``, and the variance of the
    difference of the group log means is ``1 / sum(W_A) + 1 / sum(W_B) =
    1/2 + 1/3.2 = 0.8125``. The statistic is ``log(4) / sqrt(0.8125)``.
    """
    x = _two_group_design(2)
    lfc = np.array([np.log(2.0), np.log(4.0)])
    p, stat, se = glm.wald_test(
        design_matrix=x,
        disp=0.5,
        lfc=lfc,
        mu=np.array([2.0, 2.0, 8.0, 8.0]),
        ridge_factor=np.zeros((2, 2)),
        contrast=np.array([0.0, 1.0]),
        lfc_null=0.0,
        alt_hypothesis=None,
    )
    # A 2 x 2 inverse and a few products: rounding only.
    assert se == pytest.approx(np.sqrt(0.8125), rel=1e-12)
    assert stat == pytest.approx(np.log(4.0) / np.sqrt(0.8125), rel=1e-12)
    assert p == pytest.approx(
        2 * stats.norm.sf(np.log(4.0) / np.sqrt(0.8125)), rel=1e-10
    )


@pytest.mark.parametrize("design", ["two_group", "five_coefficients"])
def test_wald_test_matches_sandwich_formula(design):
    """Random genes against :func:`helpers.wald_reference`, with the default and a large ridge.

    The two computations differ only in rounding (inverse versus solve, order of
    products); 1e-9 relative on SE and statistic covers condition numbers up to 1e6.
    p-values amplify a relative error in the statistic by about ``stat^2``; they are
    compared at 1e-7.
    """
    rs = np.random.RandomState([4, 0 if design == "two_group" else 1])
    if design == "two_group":
        x = _two_group_design(5)
    else:
        sim = helpers.simulate_counts(
            n_genes=2,
            n_samples=18,
            seed=4,
            factors={"condition": ["A", "B"], "group": ["X", "Y", "Z"]},
            continuous=("age",),
        )
        x = sim.design_matrix.to_numpy()
    n, p = x.shape
    for ridge in (np.diag(np.full(p, 1e-6)), np.diag(rs.uniform(0.5, 2.0, size=p))):
        for _ in range(10):
            beta = rs.normal(size=p) + np.r_[rs.uniform(1, 8), np.zeros(p - 1)]
            mu = np.exp(x @ beta) * np.exp(rs.normal(scale=0.3, size=n))
            disp = 10 ** rs.uniform(-3, 0)
            contrast = rs.normal(size=p)
            actual = glm.wald_test(
                design_matrix=x,
                disp=disp,
                lfc=beta,
                mu=mu,
                ridge_factor=ridge,
                contrast=contrast,
                lfc_null=0.0,
                alt_hypothesis=None,
            )
            expected = helpers.wald_reference(x, disp, beta, mu, ridge, contrast)
            assert actual[2] == pytest.approx(expected[2], rel=1e-9)
            assert actual[1] == pytest.approx(expected[1], rel=1e-9)
            assert actual[0] == pytest.approx(expected[0], rel=1e-7)


# DESeq2's results() with lfcThreshold T >= 0, contrast LFC b and standard error s
# (log2 in DESeq2; the scale cancels): greaterAbs: stat sign(b) max((|b| - T)/s, 0),
# p min(1, 2 sf((|b| - T)/s)); lessAbs: stat min((T - b)/s, (b + T)/s), p
# max(cdf((b - T)/s), sf((b + T)/s)); greater: stat max((b - T)/s, 0), p
# sf((b - T)/s); less: stat min((b + T)/s, 0), p cdf((b + T)/s). PyDESeq2 passes the
# null value itself as lfc_null, so DESeq2's "less" with threshold T is lfc_null = -T.
# PyDESeq2 reports lessAbs's statistic with a sign; its magnitude is DESeq2's. Where
# DESeq2's greater, less or lessAbs statistic is clipped to zero, PyDESeq2 computes
# the p-value from the clipped statistic (0.5) while DESeq2 uses the unclipped normal
# tail (between 0.5 and 1), so p-values are compared only where the statistic is not
# clipped.
@pytest.mark.parametrize(
    "alt, threshold, lfc",
    [
        ("greaterAbs", 0.5, 1.2),
        ("greaterAbs", 0.5, -1.4),
        ("greaterAbs", 0.5, 0.3),
        ("lessAbs", 0.5, 0.1),
        ("lessAbs", 0.5, -0.35),
        ("lessAbs", 0.5, 0.9),
        ("greater", 0.5, 1.3),
        ("greater", 0.5, 0.2),
        ("less", 0.5, -1.1),
        ("less", 0.5, -0.2),
    ],
)
def test_wald_test_alternative_hypotheses(alt, threshold, lfc):
    """``alt_hypothesis`` against DESeq2's formulas, on a unit contrast vector."""
    x = _two_group_design(4)
    beta = np.array([3.0, lfc])
    mu = np.exp(x @ beta)
    ridge = np.diag([1e-6, 1e-6])
    contrast = np.array([0.0, 1.0])
    _, _, s = helpers.wald_reference(x, 0.2, beta, mu, ridge, contrast)
    t, b = threshold, lfc
    if alt == "greaterAbs":
        stat = np.sign(b) * max((abs(b) - t) / s, 0.0)
        pval = min(1.0, 2 * stats.norm.sf((abs(b) - t) / s))
    elif alt == "lessAbs":
        stat = min(max((t - b) / s, 0.0), max((b + t) / s, 0.0))
        pval = max(stats.norm.cdf((b - t) / s), stats.norm.sf((b + t) / s))
    elif alt == "greater":
        stat = max((b - t) / s, 0.0)
        pval = stats.norm.sf((b - t) / s)
    else:
        stat = min((b + t) / s, 0.0)
        pval = stats.norm.cdf((b + t) / s)

    p, statistic, se = glm.wald_test(
        design_matrix=x,
        disp=0.2,
        lfc=beta,
        mu=mu,
        ridge_factor=ridge,
        contrast=contrast,
        lfc_null=-t if alt == "less" else t,
        alt_hypothesis=alt,
    )
    # Rounding only (see test_wald_test_matches_sandwich_formula).
    assert se == pytest.approx(s, rel=1e-9)
    if alt == "lessAbs":
        statistic = abs(statistic)
    assert statistic == pytest.approx(stat, rel=1e-9, abs=1e-12)
    if stat != 0 or alt == "greaterAbs":
        assert p == pytest.approx(pval, rel=1e-8)


# distributions.nbinomFn


@pytest.mark.parametrize("shrink_index", [1, 2])
def test_nbinomFn_matches_likelihood_and_prior(shrink_index):
    """Differences of nbinomFn between coefficient vectors equal those of the reference.

    nbinomFn drops the terms of the NB log-likelihood that do not depend on beta, so
    only differences are comparable. The reference is ``helpers.nb_loglik`` (accurate
    to about 1e-12 relative) plus the priors; 1e-9 relative to the objective's size
    covers the cancellation in the difference.
    """
    sim = helpers.simulate_counts(
        n_genes=3, n_samples=12, seed=6, factors={"condition": ["A", "B", "C"]}
    )
    x = sim.design_matrix.to_numpy()
    offset = np.log(sim.size_factors.to_numpy())
    rs = np.random.RandomState(6)
    for gene in sim.counts.columns:
        y = sim.counts[gene].to_numpy()
        size = 1.0 / sim.dispersions[gene]
        betas = [
            sim.beta.loc[gene].to_numpy() + rs.normal(scale=0.5, size=3)
            for _ in range(4)
        ]
        for scale in (0.3, 2.0):
            actual = [
                nbinomFn(b, x, y, size, offset, 15.0, scale, shrink_index) for b in betas
            ]
            expected = [
                helpers.apeglm_objective(b, x, y, size, offset, shrink_index, scale)
                for b in betas
            ]
            size_of_objective = max(abs(v) for v in expected)
            np.testing.assert_allclose(
                np.diff(actual), np.diff(expected), rtol=0, atol=1e-9 * size_of_objective
            )


# distributions.nbinomGLM


def _map_cases():
    """Simulated genes and their designs for the MAP comparisons."""
    two = helpers.simulate_counts(n_genes=25, n_samples=8, seed=7, disp_scatter=0.3)
    multi = helpers.simulate_counts(
        n_genes=20,
        n_samples=12,
        seed=8,
        disp_scatter=0.3,
        factors={"condition": ["A", "B"], "group": ["X", "Y"]},
        continuous=("age",),
    )
    return [(two, 1), (multi, 2), (multi, 3)]


@pytest.mark.parametrize("prior_scale", [0.2, 1.0])
def test_nbinomGLM_matches_reference_map(prior_scale):
    """MAP coefficients against :func:`helpers.reference_map`, gene by gene.

    Uses the true dispersions and size factors, so the inputs are what nbinomGLM
    documents and nothing else is under test. ``converged`` must be True; genes with
    two local minima are left out (at most 2 of 65 per prior scale).
    """
    n_bimodal = 0
    for sim, shrink_index in _map_cases():
        x = sim.design_matrix.to_numpy()
        offset = np.log(sim.size_factors.to_numpy())
        for gene in sim.counts.columns:
            y = sim.counts[gene].to_numpy()
            size = 1.0 / sim.dispersions[gene]
            beta, _, converged = nbinomGLM(
                x, y, size, offset, 15.0, prior_scale, shrink_index=shrink_index
            )
            ref, spread = helpers.reference_map(
                x,
                y,
                size,
                offset,
                shrink_index,
                prior_scale,
                [sim.beta.loc[gene].to_numpy(), np.zeros(x.shape[1])],
            )
            assert converged, gene
            if spread > helpers.BIMODAL_SPREAD:
                n_bimodal += 1
                continue
            args = (x, y, size, offset, shrink_index, prior_scale)
            f_ref = helpers.apeglm_objective(ref, *args)
            gap = (helpers.apeglm_objective(beta, *args) - f_ref) / abs(f_ref)
            assert gap <= SHRINK_OBJECTIVE_RTOL, (gene, shrink_index, gap)
            np.testing.assert_allclose(beta, ref, rtol=0, atol=SHRINK_ATOL, err_msg=gene)
    assert n_bimodal <= 2


def test_nbinomGLM_inverse_hessian_under_a_weak_prior():
    """``inv_hessian`` against the inverse of the reference Hessian at the MAP.

    With prior scale 10 the prior's curvature (at most 2 / 10^2 = 0.02 on the shrunk
    coefficient, 1/225 on the others) is below 1e-3 of the smallest diagonal entry of
    the likelihood's curvature on these genes (checked below), so how the prior terms
    enter the Hessian moves its inverse by less than 1e-3. (Under a strong prior the
    returned matrix differs from the inverse Hessian: ``nbinomGLM`` adds the prior's
    curvature to every row, ``np.diag`` of a diagonal matrix being its diagonal
    vector.)
    The MAP is within SHRINK_ATOL of the reference, which moves the curvature by
    under 2%; 5e-2 relative on the diagonal of the inverse covers both.
    """
    sim = helpers.simulate_counts(
        n_genes=12, n_samples=10, seed=9, mean_range=(100, 5000), disp_scatter=0.3
    )
    x = sim.design_matrix.to_numpy()
    offset = np.log(sim.size_factors.to_numpy())
    for gene in sim.counts.columns:
        y = sim.counts[gene].to_numpy()
        size = 1.0 / sim.dispersions[gene]
        beta, inv_hessian, converged = nbinomGLM(x, y, size, offset, 15.0, 10.0)
        assert converged
        _, hess = helpers.apeglm_grad_hess(beta, x, y, size, offset, 1, 10.0)
        likelihood_curv = hess - np.diag([1 / 225, 0.0])
        likelihood_curv[1, 1] -= 2 * (100 - beta[1] ** 2) / (100 + beta[1] ** 2) ** 2
        assert 0.02 < 1e-3 * np.diag(likelihood_curv).min()
        np.testing.assert_allclose(
            np.diag(inv_hessian), np.diag(np.linalg.inv(hess)), rtol=5e-2, err_msg=gene
        )


def test_nbinomGLM_genes_with_zero_group():
    """Genes with all-zero counts in one group: the prior keeps the MAP finite.

    The MLE of the group coefficient is at minus or plus infinity; the posterior's
    minimum is where the Cauchy prior's pull balances a likelihood that flattens
    out, so it is poorly conditioned along that coefficient. Compared through the
    objective gap (SHRINK_OBJECTIVE_RTOL) and the sign and size of the coefficient
    (5% relative: the largest difference measured was 6e-3). The low-count gene has
    two local minima under the narrow prior and is compared only under the wide one.
    """
    x = _two_group_design(4)
    offset = np.zeros(8)
    n_compared = 0
    for counts, size in [
        ([40, 55, 61, 38, 0, 0, 0, 0], 5.0),
        ([0, 0, 0, 0, 900, 1200, 750, 1010], 20.0),
        ([0, 0, 0, 0, 3, 1, 4, 2], 2.0),
    ]:
        y = np.asarray(counts, dtype=float)
        for scale in (0.1, 1.0):
            beta, _, converged = nbinomGLM(x, y, size, offset, 15.0, scale)
            far = np.array([0.0, 5.0 * np.sign(y[4] - y[0])])
            ref, spread = helpers.reference_map(
                x, y, size, offset, 1, scale, [np.zeros(2), far]
            )
            assert converged
            if spread > helpers.BIMODAL_SPREAD:
                continue
            n_compared += 1
            args = (x, y, size, offset, 1, scale)
            f_ref = helpers.apeglm_objective(ref, *args)
            assert (helpers.apeglm_objective(beta, *args) - f_ref) / abs(
                f_ref
            ) <= SHRINK_OBJECTIVE_RTOL
            assert np.sign(beta[1]) == np.sign(ref[1])
            assert beta[1] == pytest.approx(ref[1], rel=0.05)
    assert n_compared == 5


# grid_search.grid_fit_shrink_beta


def test_grid_fit_shrink_beta_matches_reference():
    """The 2D grid minimum against :func:`helpers.reference_map`.

    The coarse grid has 60 points over [-30, 30] (step 60/59); the fine grid has 60
    points over two coarse steps around the coarse minimum (step 2 (60/59) / 59, about
    0.0345). On these well-conditioned genes the fine-grid minimum is within two fine
    steps of the continuous minimum in each coordinate, and its objective is within
    the objective's change over one fine step.
    """
    fine_step = 2 * (60 / 59) / 59
    sim = helpers.simulate_counts(n_genes=4, n_samples=8, seed=10, frac_de=1.0)
    x = sim.design_matrix.to_numpy()
    offset = np.log(sim.size_factors.to_numpy())
    for gene in sim.counts.columns:
        y = sim.counts[gene].to_numpy()
        size = 1.0 / sim.dispersions[gene]
        scale_cnst = max(nbinomFn(np.zeros(2), x, y, size, offset, 15.0, 0.5), 1.0)
        beta = grid_fit_shrink_beta(y, offset, x, size, 15.0, 0.5, scale_cnst)
        ref, spread = helpers.reference_map(
            x, y, size, offset, 1, 0.5, [sim.beta.loc[gene].to_numpy(), np.zeros(2)]
        )
        assert spread < helpers.BIMODAL_SPREAD
        np.testing.assert_allclose(beta, ref, rtol=0, atol=2 * fine_step, err_msg=gene)
        _, hess = helpers.apeglm_grad_hess(ref, x, y, size, offset, 1, 0.5)
        args = (x, y, size, offset, 1, 0.5)
        step = np.full(2, fine_step)
        assert (
            helpers.apeglm_objective(beta, *args) - helpers.apeglm_objective(ref, *args)
            <= step @ hess @ step
        )


# stats.lowess


def test_lowess_reproduces_a_line():
    """One local linear fit (``iter=1``) of exactly linear data is the line.

    Only ``iter=1``: with robustness iterations the residuals of an exact fit are at
    the rounding level, their bisquare weights are set by rounding noise, and points
    whose neighbourhood gets weight 0 are returned as 0.
    """
    x = np.linspace(0.1, 0.95, 50)
    y = 3.0 - 7.5 * x
    for frac in (0.1, 1 / 5, 2 / 3):
        # Weighted least squares on exact data: rounding only.
        np.testing.assert_allclose(lowess(x, y, frac=frac, iter=1), y, rtol=1e-10)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_lowess_matches_reference(seed):
    """Noisy curves with a few gross outliers against :func:`helpers.reference_lowess`.

    The two solve the same weighted least-squares problems in different
    arrangements (normal equations versus centred sums), which differ by rounding;
    1e-8 relative to the range of y covers the robustness iterations' amplification.
    """
    rs = np.random.RandomState([11, seed])
    x = np.sort(rs.uniform(0, 1, size=50))
    y = 40 * np.sin(3 * x) + rs.normal(scale=2.0, size=50)
    y[rs.choice(50, size=3, replace=False)] += 60.0
    for frac, iterations in [(1 / 5, 3), (0.5, 2)]:
        np.testing.assert_allclose(
            lowess(x, y, frac=frac, iter=iterations),
            helpers.reference_lowess(x, y, frac, iterations),
            rtol=0,
            atol=1e-8 * np.ptp(y),
        )
