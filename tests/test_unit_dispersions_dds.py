"""Area B: the ``DeseqDataSet`` dispersion steps (see tests/TARGETS.md).

``fit_genewise_dispersions``, ``fit_dispersion_trend``, ``disp_function``,
``fit_dispersion_prior`` and ``fit_MAP_dispersions``, checked against reference
implementations (``tests/helpers.py``), hand-worked values and DESeq2 1.50.2
(``tests/data/unit_b``).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.special import polygamma

from pydeseq2.default_inference import DefaultInference
from tests import helpers
from tests.test_unit_dispersions import BOUND_RTOL
from tests.test_unit_dispersions import DISPERSION_RTOL
from tests.test_unit_dispersions import MAX_MAP_FALLBACKS

pytestmark = helpers.PANDAS_DEPRECATION_FILTERS

DATA = Path(__file__).parent / "data" / "unit_b"

# The trend is fitted by L-BFGS-B on the gamma GLM objective (default tolerances);
# against IRLS run to convergence its coefficients agreed to 7.7e-6 relative or
# better on five simulated datasets. 1e-4 leaves a 10x margin.
TREND_RTOL = 1e-4

# R's glm stops the gamma fit at a relative deviance change of 1e-8, which left
# DESeq2's trend coefficients 2.5e-5 from the converged fit on the two_group fixture
# (PyDESeq2's: 2.5e-6); 2.5e-4 leaves a 10x margin. The prior variance subtracts
# trigamma from the squared MAD of the log residuals from that trend, which turned
# the trend's error into 1.2e-4 relative there; 1e-3 leaves an 8x margin.
R_TREND_RTOL = 2.5e-4
R_PRIOR_RTOL = 1e-3

# Below 100 * min_disp the dispersion objective is flat in log dispersion (its
# derivative scales with alpha), so two correct fits of the same gene can stop at
# 1.0e-8 and 1.035e-8. DESeq2 treats such values as on the floor (it leaves them out
# of the prior variance), so comparisons between fits add 100 * min_disp absolute.
FLOOR_ATOL = 100 * 1e-8

DISPERSION_COLUMNS = [
    "genewise_dispersions",
    "fitted_dispersions",
    "MAP_dispersions",
    "dispersions",
]


def _fit_dispersions(dds, through="MAP"):
    dds.fit_size_factors()
    dds.fit_genewise_dispersions()
    if through == "genewise":
        return dds
    dds.fit_dispersion_trend()
    dds.fit_dispersion_prior()
    dds.fit_MAP_dispersions()
    return dds


def _normed_means(dds):
    return dds.layers["normed_counts"].mean(axis=0)


def _squared_log_residual_mad(dds):
    """DESeq2's ``varLogDispEsts``: squared MAD of the log residuals of the gene-wise
    dispersions from the trend, over genes at or above ``100 * min_disp``."""
    genewise = dds.var["genewise_dispersions"].to_numpy()
    fitted = dds.var["fitted_dispersions"].to_numpy()
    use = np.isfinite(genewise) & (genewise >= 100 * dds.min_disp)
    residuals = np.log(genewise[use]) - np.log(fitted[use])
    return stats.median_abs_deviation(residuals, scale="normal") ** 2


def _starting_dispersions(dds, size_factors, design_matrix, max_disp):
    # DESeq2's starting value: min(rough, moments), clipped to [min_disp, max_disp].
    normed = dds.X / size_factors[:, None]
    start = np.minimum(
        helpers.rough_dispersions(normed, design_matrix),
        helpers.moments_dispersions(normed, size_factors),
    )
    return np.clip(start, dds.min_disp, max_disp)


# Gene-wise dispersions


def test_genewise_and_map_dispersions_glm_means():
    """A design with a continuous covariate, where the means come from the GLM.

    ``~condition + age`` has as many distinct design rows as samples, so PyDESeq2
    fits the means by IRLS at DESeq2's starting dispersion (``min(rough,
    moments)``). The reference recomputes that start from its definition, fits the
    GLM with ``helpers.reference_glm_mle`` and minimizes the dispersion objective at
    those means: gene-wise without a prior, MAP with the prior centred on
    ``var["fitted_dispersions"]`` with variance ``uns["prior_disp_var"]``. IRLS stops
    at a relative deviance change of 1e-8, which moved the dispersions by up to
    1.3e-4 relative on the measured cases, inside ``DISPERSION_RTOL``.

    Left out: genes whose start is at ``min_disp`` (gene-wise only: there the
    objective is flat in log dispersion, and PyDESeq2, like DESeq2 1.50.2, stays at
    ``min_disp`` even when the minimizer is elsewhere), genes with a fitted mean
    below ``min_mu`` (IRLS clamps them, the reference does not), and, from the MAP
    comparison, at most ``MAX_MAP_FALLBACKS`` genes that took the grid fallback.
    That fallback drops the prior (a known issue), so such a gene's MAP value is
    within 2% (the grid's spacing) of its gene-wise minimizer and closer to it than
    to the MAP one; whether L-BFGS-B fails on a gene moves with changes of 1e-6 in
    the IRLS means.
    """
    sim = helpers.simulate_counts(
        n_genes=60, n_samples=12, seed=10, continuous=("age",), disp_scatter=0.3
    )
    dds = _fit_dispersions(helpers.make_dds(sim.counts, sim.metadata, design=sim.design))

    sf = dds.obs["size_factors"].to_numpy()
    x = dds.obsm["design_matrix"].to_numpy(dtype=float)
    max_disp = max(10.0, dds.n_obs)
    start = _starting_dispersions(dds, sf, x, max_disp)
    fitted = dds.var["fitted_dispersions"].to_numpy()
    prior_var = float(dds.uns["prior_disp_var"])

    map_genes, map_disp, map_genewise, genewise_genes, genewise = [], [], [], [], []
    for j, gene in enumerate(dds.var_names):
        y = dds.X[:, j]
        beta = helpers.reference_glm_mle(y, sf, x, start[j])
        mu = sf * np.exp(x @ beta)
        if (mu < dds.min_mu).any():
            continue
        minimizer = helpers.reference_dispersion_mle(y, mu, x, max_disp=max_disp)
        if start[j] > dds.min_disp:
            genewise_genes.append(gene)
            genewise.append(minimizer)
        map_genes.append(gene)
        map_genewise.append(minimizer)
        map_disp.append(
            helpers.reference_dispersion_mle(
                y,
                mu,
                x,
                max_disp=max_disp,
                prior_disp_var=prior_var,
                prior_dispersion=fitted[j],
            )
        )
    assert len(map_genes) >= 55
    assert len(genewise_genes) >= 53

    helpers.assert_fields_close(
        dds.var.loc[genewise_genes, "genewise_dispersions"],
        pd.Series(genewise, index=genewise_genes),
        rtol=DISPERSION_RTOL,
        name="genewise_dispersions",
    )
    actual = dds.var.loc[map_genes, "MAP_dispersions"].to_numpy()
    map_disp, map_genewise = np.array(map_disp), np.array(map_genewise)
    fallback = (
        (np.abs(actual - map_disp) > DISPERSION_RTOL * map_disp)
        & (np.abs(actual / map_genewise - 1) < 0.02)
        & (np.abs(actual - map_genewise) < np.abs(actual - map_disp))
    )
    assert fallback.sum() <= MAX_MAP_FALLBACKS
    helpers.assert_fields_close(
        pd.Series(actual[~fallback], index=np.array(map_genes)[~fallback]),
        pd.Series(map_disp[~fallback], index=np.array(map_genes)[~fallback]),
        rtol=DISPERSION_RTOL,
        name="MAP_dispersions",
    )


def test_all_zero_genes_get_nan_and_leave_other_genes_unchanged():
    """All-zero genes, first, in the middle and last, are skipped by every step.

    Their gene-wise, fitted (parametric trend), MAP and final dispersions are NaN
    and ``var["non_zero"]`` is False for them alone. Every other gene's dispersions,
    the trend and the prior variance equal those of the same data without the
    all-zero genes (the ratio size factors ignore genes with a zero), within
    ``DISPERSION_RTOL``: a gene paired with a neighbour's inputs or outputs would be
    off by far more.
    """
    sim = helpers.simulate_counts(
        n_genes=40, n_samples=8, seed=50, disp_scatter=0.5, n_all_zero=3
    )
    regular = [f"gene{i + 1}" for i in range(40)]
    order = ["all_zero1", *regular[:20], "all_zero2", *regular[20:], "all_zero3"]
    counts = sim.counts[order]
    zero = counts.columns.str.startswith("all_zero")

    with_zero = _fit_dispersions(helpers.make_dds(counts, sim.metadata))
    without = _fit_dispersions(helpers.make_dds(sim.counts[regular], sim.metadata))

    np.testing.assert_array_equal(with_zero.var["non_zero"].to_numpy(), ~zero)
    for column in DISPERSION_COLUMNS:
        assert with_zero.var[column].isna().to_numpy().tolist() == zero.tolist(), column
    _assert_same_dispersions(
        with_zero, without, name="non-zero genes", trend_rtol=TREND_RTOL
    )


@pytest.mark.parametrize("n_samples, max_disp", [(6, 10.0), (12, 10.0), (8, 5.0)])
def test_genewise_dispersions_capped_at_max_disp(n_samples, max_disp):
    """Genes with a count in one sample only reach the upper bound.

    The bound is ``max(max_disp, n_samples)``, as the ``DeseqDataSet`` docstring
    says. The reference minimizer of the Cox-Reid objective at the linear-model
    means sits on that bound for these genes (to rounding, 1e-12), so the expected
    value is the bound itself, within ``BOUND_RTOL``.
    """
    sim = helpers.simulate_counts(
        n_genes=20, n_samples=n_samples, seed=51, n_single_sample=2
    )
    dds = _fit_dispersions(
        helpers.make_dds(sim.counts, sim.metadata, max_disp=max_disp),
        through="genewise",
    )
    bound = max(max_disp, n_samples)
    x = sim.design_matrix.to_numpy()
    mu = helpers.lin_reg_mu(dds.X, dds.obs["size_factors"].to_numpy(), x)
    for gene in ["single_sample1", "single_sample2"]:
        j = dds.var_names.get_loc(gene)
        expected = helpers.reference_dispersion_mle(
            dds.X[:, j], mu[:, j], x, max_disp=bound
        )
        assert expected == pytest.approx(bound, rel=1e-12)
        assert dds.var.loc[gene, "genewise_dispersions"] == pytest.approx(
            bound, rel=BOUND_RTOL
        )
    assert (dds.var["genewise_dispersions"] <= bound).all()


def test_genewise_dispersions_with_min_disp():
    """``min_disp=0.01``: underdispersed genes stop on the lower bound.

    Every gene is compared with the reference minimizer over ``[0.01, max_disp]`` at
    the linear-model means. 47 genes start on the bound here (their ``min(rough,
    moments)`` is below 0.01). At ``min_disp=1e-8`` a gene that starts on the bound
    stays there, because the objective is flat in log dispersion so close to zero;
    at 0.01 it is not, and every gene here whose minimizer is inside moves off the
    bound, so no gene is left out. At least 20 of the 150 genes have their minimizer
    on the bound; those must be 0.01 (``BOUND_RTOL``).
    """
    sim = helpers.simulate_counts(
        n_genes=150, n_samples=10, seed=9, disp_scatter=1.0, trend=(0.02, 1.0)
    )
    dds = _fit_dispersions(
        helpers.make_dds(sim.counts, sim.metadata, min_disp=0.01), through="genewise"
    )
    x = sim.design_matrix.to_numpy()
    mu = helpers.lin_reg_mu(dds.X, dds.obs["size_factors"].to_numpy(), x)
    expected = np.array(
        [
            helpers.reference_dispersion_mle(
                dds.X[:, j], mu[:, j], x, min_disp=0.01, max_disp=10.0
            )
            for j in range(dds.n_vars)
        ]
    )
    at_bound = expected <= 0.01 * (1 + 1e-12)
    assert at_bound.sum() >= 20
    helpers.assert_fields_close(
        dds.var["genewise_dispersions"],
        pd.Series(expected, index=dds.var_names),
        rtol=DISPERSION_RTOL,
        name="genewise_dispersions",
    )
    np.testing.assert_allclose(
        dds.var["genewise_dispersions"][at_bound], 0.01, rtol=BOUND_RTOL
    )


def test_dispersions_of_genes_with_an_all_zero_group():
    """Genes with no counts in one of two groups.

    The linear-model mean of the empty group is 0. DESeq2 scales the fitted
    normalized means by the size factors and then floors them at ``min_mu=0.5``
    (``linearModelMuNormalized``, then ``pmax(mu, minmu)``), so every sample of that
    group gets a mean of 0.5 whatever its size factor; ``helpers.lin_reg_mu`` does the
    same. Gene-wise and MAP dispersions of these genes and of the regular ones are
    compared with the reference at those means (``DISPERSION_RTOL``). Flooring before
    the scaling moves these genes' dispersions by 4% to 17%.
    """
    sim = helpers.simulate_counts(
        n_genes=40, n_samples=10, seed=59, disp_scatter=0.3, size_factor_sd=0.5
    )
    in_b = (sim.metadata["condition"] == "B").to_numpy()
    rs = np.random.RandomState(59)
    means = np.exp(rs.uniform(np.log(5), np.log(60), size=8))
    one_group = np.zeros((10, 8), dtype=np.int64)
    for j in range(8):
        mu_b = sim.size_factors.to_numpy()[in_b] * means[j]
        one_group[in_b, j] = rs.negative_binomial(n=1 / 0.4, p=1 / (1 + 0.4 * mu_b))
    names = [f"one_group{j + 1}" for j in range(8)]
    counts = pd.concat(
        [sim.counts, pd.DataFrame(one_group, index=sim.counts.index, columns=names)],
        axis=1,
    )
    dds = _fit_dispersions(helpers.make_dds(counts, sim.metadata))

    sf = dds.obs["size_factors"].to_numpy()
    x = sim.design_matrix.to_numpy()
    mu = helpers.lin_reg_mu(dds.X, sf, x)
    assert (mu[~in_b][:, -8:] == 0.5).all()
    start = _starting_dispersions(dds, sf, x, max(10.0, dds.n_obs))
    fitted = dds.var["fitted_dispersions"].to_numpy()
    prior_var = float(dds.uns["prior_disp_var"])
    genewise = np.full(dds.n_vars, np.nan)
    map_disp = np.empty(dds.n_vars)
    for j in range(dds.n_vars):
        if start[j] > dds.min_disp:
            genewise[j] = helpers.reference_dispersion_mle(dds.X[:, j], mu[:, j], x)
        map_disp[j] = helpers.reference_dispersion_mle(
            dds.X[:, j],
            mu[:, j],
            x,
            prior_disp_var=prior_var,
            prior_dispersion=fitted[j],
        )
    compared = ~np.isnan(genewise)
    assert compared[-8:].all()
    helpers.assert_fields_close(
        dds.var["genewise_dispersions"][compared],
        pd.Series(genewise[compared], index=dds.var_names[compared]),
        rtol=DISPERSION_RTOL,
        name="genewise_dispersions",
    )
    helpers.assert_fields_close(
        dds.var["MAP_dispersions"],
        pd.Series(map_disp, index=dds.var_names),
        rtol=DISPERSION_RTOL,
        name="MAP_dispersions",
    )


# Trend


def test_parametric_trend_matches_reference():
    """``fit_dispersion_trend`` (parametric) against DESeq2's algorithm written out.

    ``helpers.reference_trend`` is DESeq2's ``parametricDispersionFit`` with IRLS for the
    gamma GLM, applied to PyDESeq2's own gene-wise dispersions and the means of the
    normalized counts. The data include genes at ``min_disp``, which the iteration
    leaves out. Coefficients within ``TREND_RTOL``; ``var["fitted_dispersions"]`` is
    ``a0 + a1 / mean`` of the returned coefficients (rounding, 1e-12) and NaN for
    all-zero genes; ``disp_function`` evaluates the same curve.
    """
    sim = helpers.simulate_counts(
        n_genes=300, n_samples=6, seed=3, disp_scatter=0.8, n_all_zero=2
    )
    dds = helpers.make_dds(sim.counts, sim.metadata)
    _fit_dispersions(dds, through="genewise")
    dds.fit_dispersion_trend()

    non_zero = dds.var["non_zero"].to_numpy()
    means = _normed_means(dds)
    genewise = dds.var["genewise_dispersions"].to_numpy()
    expected, used = helpers.reference_trend(means[non_zero], genewise[non_zero])
    assert (~used).sum() >= 5  # the iteration leaves some genes out

    assert dds.uns["disp_function_type"] == "parametric"
    assert list(dds.uns["trend_coeffs"].keys()) == ["a0", "a1"]
    coeffs = helpers.trend_coeffs(dds.uns["trend_coeffs"])
    np.testing.assert_allclose(coeffs.to_numpy(), expected, rtol=TREND_RTOL)

    a0, a1 = coeffs["a0"], coeffs["a1"]
    fitted = dds.var["fitted_dispersions"].to_numpy()
    assert np.isnan(fitted[~non_zero]).all()
    np.testing.assert_allclose(fitted[non_zero], a0 + a1 / means[non_zero], rtol=1e-12)
    x = np.array([0.5, 3.0, 40.0, 2e4])
    np.testing.assert_allclose(dds.disp_function(x), a0 + a1 / x, rtol=1e-12)


def test_parametric_trend_drops_genes_far_from_the_curve():
    """Gene-wise dispersions on an exact curve, plus genes far above and below it.

    150 genes sit on ``0.05 + 2 / mean``; four are 200 times above the curve and
    three are 1e-6 times it. DESeq2's iteration drops genes outside ``(1e-4, 15)``
    times the current curve, so the fit is the gamma GLM on the exact genes, whose
    maximizer is exactly ``(0.05, 2)``; ``TREND_RTOL`` covers the optimizer.
    """
    sim = helpers.simulate_counts(n_genes=157, n_samples=8, seed=52)
    dds = helpers.make_dds(sim.counts, sim.metadata)
    _fit_dispersions(dds, through="genewise")
    means = _normed_means(dds)
    curve = 0.05 + 2.0 / means
    values = curve.copy()
    values[[3, 40, 77, 120]] *= 200
    values[[10, 90, 150]] *= 1e-6
    dds.var["genewise_dispersions"] = values

    dds.fit_dispersion_trend()
    np.testing.assert_allclose(
        helpers.trend_coeffs(dds.uns["trend_coeffs"]), [0.05, 2.0], rtol=TREND_RTOL
    )
    np.testing.assert_allclose(dds.var["fitted_dispersions"], curve, rtol=TREND_RTOL)


def test_parametric_trend_falls_back_to_mean():
    """Gene-wise dispersions that do not depend on the mean: the gamma GLM drives
    ``a1`` to its lower bound, and the documented fallback is the mean trend, with
    a warning. Every gene-wise value is 0.2, so the mean is 0.2 (rounding, 1e-12)."""
    sim = helpers.simulate_counts(n_genes=50, n_samples=8, seed=8)
    dds = helpers.make_dds(sim.counts, sim.metadata)
    _fit_dispersions(dds, through="genewise")
    dds.var["genewise_dispersions"] = 0.2

    with pytest.warns(UserWarning, match="Switching to a mean-based dispersion trend"):
        dds.fit_dispersion_trend()
    assert dds.uns["disp_function_type"] == "mean"
    assert dds.uns["mean_disp"] == pytest.approx(0.2, rel=1e-12)
    np.testing.assert_allclose(dds.var["fitted_dispersions"], 0.2, rtol=1e-12)
    np.testing.assert_allclose(dds.disp_function(np.array([1.0, 50.0])), 0.2, rtol=1e-12)


def test_mean_trend_hand_worked():
    """``fit_type="mean"``: a trimmed mean of the gene-wise dispersions above
    ``10 * min_disp``, as DESeq2's ``mean(x, trim = 0.001)``.

    1,503 genes: three at or below ``10 * min_disp`` (1e-8, 5e-8 and 1e-7), left
    out; 1,500 others, which DESeq2's trim of 0.001 cuts by ``floor(1.5) = 1`` value
    at each end. The values are one huge (1e3), one tiny (1e-6) and 1,498 between
    0.05 and 0.5; the expected value drops the huge and the tiny one and averages
    the rest (rounding, 1e-12). ``var["fitted_dispersions"]`` and ``disp_function``
    return that constant.
    """
    n_genes = 1503
    rs = np.random.RandomState(53)
    counts = pd.DataFrame(
        rs.poisson(20, size=(6, n_genes)),
        index=[f"s{i}" for i in range(6)],
        columns=[f"g{i}" for i in range(n_genes)],
    )
    metadata = pd.DataFrame({"condition": ["A", "B"] * 3}, index=counts.index)
    dds = helpers.make_dds(counts, metadata, fit_type="mean")
    middle = rs.uniform(0.05, 0.5, size=1498)
    values = np.concatenate([[1e-8, 5e-8, 1e-7, 1e3, 1e-6], middle])
    dds.var["genewise_dispersions"] = values

    dds.fit_dispersion_trend()
    expected = middle.mean()
    assert dds.uns["disp_function_type"] == "mean"
    assert dds.uns["mean_disp"] == pytest.approx(expected, rel=1e-12)
    np.testing.assert_allclose(dds.var["fitted_dispersions"], expected, rtol=1e-12)
    np.testing.assert_allclose(
        dds.disp_function(np.array([1.0, 10.0, 1e4])), expected, rtol=1e-12
    )


# Prior variance


@pytest.mark.parametrize("disp_scatter, floored", [(1.0, False), (0.05, True)])
def test_fit_dispersion_prior_formula(disp_scatter, floored):
    """``uns["prior_disp_var"]`` is DESeq2's ``dispPriorVar``.

    ``max(mad(log genewise - log fitted)^2 - trigamma((n - p) / 2), 0.25)`` over genes
    with a gene-wise dispersion of at least ``100 * min_disp``. The MAD comes from
    ``scipy.stats.median_abs_deviation``, from the dispersions in ``var``. Two genes
    are set to ``min_disp`` and ``50 * min_disp`` first, which the rule leaves out.
    ``~condition + group`` has ``p = 3``. With a wide scatter the formula is above
    0.25, with a narrow one it is floored at 0.25. Only rounding differs (1e-10).
    """
    sim = helpers.simulate_counts(
        n_genes=120,
        n_samples=12,
        seed=54,
        factors={"condition": ["A", "B"], "group": ["X", "Y"]},
        disp_scatter=disp_scatter,
    )
    dds = helpers.make_dds(sim.counts, sim.metadata, design=sim.design)
    _fit_dispersions(dds, through="genewise")
    dds.fit_dispersion_trend()
    genewise = dds.var["genewise_dispersions"].to_numpy().copy()
    genewise[[5, 60]] = [1e-8, 5e-7]
    dds.var["genewise_dispersions"] = genewise

    dds.fit_dispersion_prior()
    raw = _squared_log_residual_mad(dds) - polygamma(1, (12 - 3) / 2)
    expected = max(raw, 0.25)
    assert (raw < 0.25) == floored
    assert float(dds.uns["prior_disp_var"]) == pytest.approx(expected, rel=1e-10)


def test_fit_dispersion_prior_warns_with_few_degrees_of_freedom():
    """Five samples and two coefficients leave three residual degrees of freedom,
    which the docstring says makes the estimate imprecise; the method warns."""
    sim = helpers.simulate_counts(n_genes=30, n_samples=5, seed=55, disp_scatter=0.5)
    dds = helpers.make_dds(sim.counts, sim.metadata)
    _fit_dispersions(dds, through="genewise")
    dds.fit_dispersion_trend()
    with pytest.warns(UserWarning, match="residual degrees of freedom is less than 3"):
        dds.fit_dispersion_prior()
    # DESeq2 replaces this formula by a simulation-based estimate at three or fewer
    # residual degrees of freedom; PyDESeq2 keeps the formula and warns.
    raw = _squared_log_residual_mad(dds) - polygamma(1, 3 / 2)
    assert float(dds.uns["prior_disp_var"]) == pytest.approx(max(raw, 0.25), rel=1e-10)


# MAP dispersions and the final dispersions


def test_map_dispersions_and_outlier_rule():
    """``fit_MAP_dispersions`` against the reference, and the choice of final value.

    MAP: the reference minimizes the Cox-Reid objective plus a normal prior on log
    dispersion centred on ``var["fitted_dispersions"]`` with variance
    ``uns["prior_disp_var"]``, at the linear-model means, for every non-zero gene
    (``DISPERSION_RTOL``; every MAP fit here converges, so none reaches the grid
    fallback, which drops the prior).

    Final: DESeq2 keeps the gene-wise estimate for dispersion outliers, genes with
    ``log(genewise) > log(fitted) + 2 sqrt(varLogDispEsts)``, and the MAP estimate
    otherwise. The rule is applied to the values in ``var``; the chosen values are
    copies, so they are compared exactly. All-zero genes are NaN throughout. Three
    genes drawn with dispersion 2 + 4 / mean (the others: 0.1 + 4 / mean, scatter
    0.9) make sure there are outliers.
    """
    sim = helpers.simulate_counts(
        n_genes=80, n_samples=10, seed=60, disp_scatter=0.9, n_all_zero=1
    )
    # Same seed and samples, so the same size factors and metadata.
    high = helpers.simulate_counts(
        n_genes=3, n_samples=10, seed=60, trend=(2.0, 4.0), mean_range=(200, 2000)
    )
    high_counts = high.counts.rename(columns=lambda c: c.replace("gene", "high"))
    counts = pd.concat([sim.counts, high_counts], axis=1)
    dds = _fit_dispersions(helpers.make_dds(counts, sim.metadata))
    non_zero = dds.var["non_zero"].to_numpy()
    x = sim.design_matrix.to_numpy()
    mu = helpers.lin_reg_mu(dds.X, dds.obs["size_factors"].to_numpy(), x)
    fitted = dds.var["fitted_dispersions"].to_numpy()
    prior_var = float(dds.uns["prior_disp_var"])
    expected = np.full(dds.n_vars, np.nan)
    for j in np.flatnonzero(non_zero):
        expected[j] = helpers.reference_dispersion_mle(
            dds.X[:, j],
            mu[:, j],
            x,
            max_disp=10.0,
            prior_disp_var=prior_var,
            prior_dispersion=fitted[j],
        )
    helpers.assert_fields_close(
        dds.var["MAP_dispersions"],
        pd.Series(expected, index=dds.var_names),
        rtol=DISPERSION_RTOL,
        name="MAP_dispersions",
    )

    genewise = dds.var["genewise_dispersions"].to_numpy()
    with np.errstate(invalid="ignore"):
        outlier = np.log(genewise) > np.log(fitted) + 2 * np.sqrt(
            _squared_log_residual_mad(dds)
        )
    assert 2 <= outlier.sum() <= 10
    final = dds.var["dispersions"].to_numpy()
    np.testing.assert_array_equal(final[outlier], genewise[outlier])
    keep = non_zero & ~outlier
    np.testing.assert_array_equal(
        final[keep], dds.var["MAP_dispersions"].to_numpy()[keep]
    )
    assert np.isnan(final[~non_zero]).all()


# Invariance


def _assert_same_dispersions(actual, expected, name, trend_rtol=DISPERSION_RTOL):
    """Two fits of the same genes give the same dispersions, matched by gene name.

    ``DISPERSION_RTOL`` plus ``FLOOR_ATOL``. The MAP and final dispersions of genes
    that took the MAP grid fallback in one fit only (``helpers.map_fallback_flips``)
    are left out, at most ``MAX_MAP_FALLBACKS`` of them. The trend (when parametric)
    and the prior variance move with the gene-wise inputs.
    """
    genes = expected.var_names
    flips = helpers.map_fallback_flips(actual, expected)
    assert len(flips) <= MAX_MAP_FALLBACKS, list(flips)
    got = actual.var.loc[genes, DISPERSION_COLUMNS].copy()
    want = expected.var[DISPERSION_COLUMNS].copy()
    for frame in (got, want):
        frame.loc[flips, ["MAP_dispersions", "dispersions"]] = np.nan
    helpers.assert_fields_close(
        got, want, rtol=DISPERSION_RTOL, atol=FLOOR_ATOL, name=name
    )
    if "trend_coeffs" in expected.uns:
        np.testing.assert_allclose(
            helpers.trend_coeffs(actual.uns["trend_coeffs"]),
            helpers.trend_coeffs(expected.uns["trend_coeffs"]),
            rtol=trend_rtol,
        )
    assert float(actual.uns["prior_disp_var"]) == pytest.approx(
        float(expected.uns["prior_disp_var"]), rel=DISPERSION_RTOL
    )


@pytest.mark.parametrize(
    "variant",
    [
        "permuted",
        "batch_size=3",
        "n_cpus=2",
        "n_cpus=2,batch_size=4",
        "glm_means,permuted,n_cpus=2,batch_size=4",
    ],
)
def test_dispersions_invariant_to_gene_order_batch_size_and_n_cpus(n_cpus, variant):
    """Gene order, ``batch_size`` and ``n_cpus`` do not change any dispersion output.

    The whole chain (size factors to MAP) on 60 genes, two of them all zero, run
    once with genes in order at ``batch_size`` 128 and once with the variant.
    Outputs are matched by gene name and compared within ``DISPERSION_RTOL`` (plus
    ``FLOOR_ATOL``, and without genes whose MAP took the grid fallback in one run
    only; see ``_assert_same_dispersions``), the precision a reorganized solver is
    held to; the trend and prior variance move with the gene-wise inputs, so they
    share that tolerance. ``glm_means`` uses
    ``~condition + age``, so the gene-wise means come from IRLS instead of the
    linear model.
    """
    options = variant.split(",")
    continuous = ("age",) if "glm_means" in options else ()
    sim = helpers.simulate_counts(
        n_genes=58,
        n_samples=8,
        seed=57,
        disp_scatter=0.5,
        n_all_zero=2,
        continuous=continuous,
    )
    base = _fit_dispersions(
        helpers.make_dds(
            sim.counts,
            sim.metadata,
            design=sim.design,
            inference=DefaultInference(n_cpus=n_cpus),
        )
    )

    settings = dict(item.split("=") for item in options if "=" in item)
    variant_inference = DefaultInference(
        n_cpus=int(settings.get("n_cpus", n_cpus)),
        batch_size=int(settings.get("batch_size", 128)),
    )
    counts = sim.counts
    if "permuted" in options:
        counts = counts.iloc[:, np.random.RandomState(1).permutation(counts.shape[1])]
    other = _fit_dispersions(
        helpers.make_dds(
            counts, sim.metadata, design=sim.design, inference=variant_inference
        )
    )
    _assert_same_dispersions(other, base, name=variant)


@pytest.mark.parametrize("n_genes", [1, 2])
def test_one_and_two_gene_datasets(n_genes):
    """One and two genes, with size factors given (with one gene the ratio method
    would normalize every count to the same value).

    Mean trend (a parametric curve has two coefficients). Gene-wise dispersions
    match the reference at the linear-model means; the trend is their mean; the
    prior variance follows DESeq2's formula (with one gene the MAD is 0, so 0.25);
    MAP dispersions match the reference with that prior. With two genes, reversing
    their order reverses every per-gene output.
    """
    sim = helpers.simulate_counts(
        n_genes=n_genes, n_samples=8, seed=58, disp_scatter=0.3, trend=(0.3, 4.0)
    )
    sf = sim.size_factors.to_numpy()
    x = sim.design_matrix.to_numpy()

    def run(counts):
        dds = helpers.make_dds(counts, sim.metadata, fit_type="mean")
        dds.obs["size_factors"] = sf
        dds.layers["normed_counts"] = dds.X / sf[:, None]
        dds.fit_genewise_dispersions()
        dds.fit_dispersion_trend()
        dds.fit_dispersion_prior()
        dds.fit_MAP_dispersions()
        return dds

    dds = run(sim.counts)
    mu = helpers.lin_reg_mu(dds.X, sf, x)
    genewise = np.array(
        [
            helpers.reference_dispersion_mle(dds.X[:, j], mu[:, j], x)
            for j in range(n_genes)
        ]
    )
    np.testing.assert_allclose(
        dds.var["genewise_dispersions"], genewise, rtol=DISPERSION_RTOL
    )
    got_genewise = dds.var["genewise_dispersions"].to_numpy()
    assert dds.uns["mean_disp"] == pytest.approx(got_genewise.mean(), rel=1e-12)
    mad2 = _squared_log_residual_mad(dds)
    expected_var = max(mad2 - polygamma(1, (8 - 2) / 2), 0.25)
    assert float(dds.uns["prior_disp_var"]) == pytest.approx(expected_var, rel=1e-10)
    map_disp = np.array(
        [
            helpers.reference_dispersion_mle(
                dds.X[:, j],
                mu[:, j],
                x,
                prior_disp_var=expected_var,
                prior_dispersion=float(dds.uns["mean_disp"]),
            )
            for j in range(n_genes)
        ]
    )
    np.testing.assert_allclose(
        dds.var["MAP_dispersions"], map_disp, rtol=DISPERSION_RTOL
    )

    if n_genes == 2:
        reversed_dds = run(sim.counts.iloc[:, ::-1])
        _assert_same_dispersions(reversed_dds, dds, name="reversed")


# DESeq2 1.50.2


def _read_r_case(case):
    directory = DATA / case
    counts = pd.read_csv(directory / "counts.csv", index_col=0).T
    metadata = pd.read_csv(directory / "metadata.csv", index_col=0)
    # dispOutlier is TRUE, FALSE or NA; read as text so that every pandas version
    # parses it the same way.
    per_gene = pd.read_csv(
        directory / "r_dispersions.csv",
        index_col=0,
        comment="#",
        dtype={"dispOutlier": str},
    )
    fit = pd.read_csv(directory / "r_dispersion_fit.csv", index_col=0, comment="#")
    return counts, metadata, per_gene, fit["value"]


@pytest.mark.parametrize(
    "case, design",
    [("two_group", "~condition"), ("continuous", "~condition + age")],
    ids=["two_group", "continuous"],
)
def test_trend_prior_and_outliers_match_deseq2(case, design):
    """Trend, prior variance and dispersion outliers against DESeq2 1.50.2.

    DESeq2's gene-wise values are not at the minimizer of the Cox-Reid adjusted
    objective on every gene (on ``two_group``, 49 of 153 genes differ from it by more
    than 1e-3 and one by 12%, 0.048 above the minimum; PyDESeq2's are within 7.5e-6),
    and its MAP values differ from PyDESeq2's by up to 22%, so neither is compared.
    DESeq2's gene-wise values are put into ``var["genewise_dispersions"]`` instead,
    and every later step is compared with DESeq2's on those same inputs:

    - size factors (median of ratios; rounding, 1e-12), which set the means the
      trend is fitted against;
    - trend coefficients and ``fitted_dispersions`` (``R_TREND_RTOL``);
    - ``prior_disp_var`` (``R_PRIOR_RTOL``);
    - the dispersion outliers: genes DESeq2 flags keep the gene-wise value in
      ``var["dispersions"]``, the others take the MAP value. No gene is within 0.01
      in log dispersion of the outlier threshold, so the flags do not depend on the
      trend's last digits.

    The fixtures and the script that made them are in ``tests/data/unit_b``.
    """
    counts, metadata, r_genes, r_fit = _read_r_case(case)
    dds = helpers.make_dds(counts, metadata, design=design)
    _fit_dispersions(dds, through="genewise")
    np.testing.assert_allclose(
        dds.obs["size_factors"],
        r_fit[[f"size_factor_{s}" for s in counts.index]].to_numpy(),
        rtol=1e-12,
    )

    r_genewise = r_genes["dispGeneEst"].to_numpy()
    dds.var["genewise_dispersions"] = r_genewise
    dds.fit_dispersion_trend()
    np.testing.assert_allclose(
        helpers.trend_coeffs(dds.uns["trend_coeffs"]).to_numpy(),
        r_fit[["a0", "a1"]].to_numpy(),
        rtol=R_TREND_RTOL,
    )
    helpers.assert_fields_close(
        dds.var["fitted_dispersions"],
        pd.Series(r_genes["dispFit"].to_numpy(), index=dds.var_names),
        rtol=R_TREND_RTOL,
        name="fitted_dispersions",
    )

    dds.fit_dispersion_prior()
    assert float(dds.uns["prior_disp_var"]) == pytest.approx(
        r_fit["dispPriorVar"], rel=R_PRIOR_RTOL
    )

    dds.fit_MAP_dispersions()
    r_outlier = (r_genes["dispOutlier"] == "TRUE").to_numpy(dtype=bool)
    with np.errstate(invalid="ignore"):
        margin = (
            np.log(r_genewise)
            - np.log(r_genes["dispFit"].to_numpy())
            - 2 * np.sqrt(r_fit["varLogDispEsts"])
        )
    finite = np.isfinite(margin)
    assert (np.abs(margin[finite]) > 0.01).all()
    assert r_outlier.sum() >= 1
    final = dds.var["dispersions"].to_numpy()
    np.testing.assert_array_equal(final[r_outlier], r_genewise[r_outlier])
    keep = finite & ~r_outlier
    np.testing.assert_array_equal(
        final[keep], dds.var["MAP_dispersions"].to_numpy()[keep]
    )
