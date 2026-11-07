"""Area D: Wald tests, Cook's filtering, p-value adjustment and LFC shrinkage.

``DeseqStats`` (constructor, ``run_wald_test``, ``summary``, ``lfc_shrink``,
``variables``), ``DeseqDataSet.cond``, ``contrast`` and ``cooks_outlier``, and the
``DefaultInference``/``Inference`` methods behind them (see tests/TARGETS.md). Every
expected value is computed here from the fitted ``DeseqDataSet``'s public fields
(``obs``, ``var``, ``varm``, ``obsm``, ``layers``) with the references in
``tests/helpers.py``, so these tests check the testing and shrinkage steps whatever
the precision of the fits before them.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
from pydeseq2.inference import Inference
from tests import helpers
from tests.test_unit_stats_functions import SHRINK_ATOL

pytestmark = helpers.PANDAS_DEPRECATION_FILTERS

LOG2 = np.log(2)

# The Wald quantities are a few products and a p x p inverse per gene, computed from
# the same inputs here and in PyDESeq2; they differ by rounding, amplified by the
# condition number of X^T W X + ridge (up to about 1e6 for genes with a near-zero
# group). A p-value's relative error is about stat^2 times the statistic's.
WALD_RTOL = 1e-8
PVALUE_RTOL = 1e-6

TWO_GROUP_COEFF = "condition[T.B]"


# Datasets, fitted once per module; tests do not modify them.


def _fit(counts, metadata, design="~condition", **kwargs):
    dds = helpers.make_dds(counts, metadata, design=design, **kwargs)
    dds.deseq2()
    return dds


def _two_group_counts():
    """16 samples in two groups of 8: Cook's outliers are replaced and refitted.

    Regular genes, 6 outlier counts, 2 all-zero genes, 2 genes positive in a single
    sample (all zero after replacement), and two genes with all-zero counts in one
    group (LFCs far from zero).
    """
    sim = helpers.simulate_counts(
        n_genes=150,
        n_samples=16,
        seed=11,
        mean_range=(2, 5000),
        disp_scatter=0.3,
        n_outliers=6,
        n_all_zero=2,
        n_single_sample=2,
    )
    counts = sim.counts.copy()
    in_b = (sim.metadata["condition"] == "B").to_numpy()
    rs = np.random.RandomState(11)
    counts["zero_in_a"] = np.where(in_b, rs.poisson(300, size=16), 0)
    counts["zero_in_b"] = np.where(in_b, 0, rs.poisson(40, size=16))
    return counts, sim.metadata


@pytest.fixture(scope="module")
def two_group():
    return _fit(*_two_group_counts())


def _few_replicates_sim():
    return helpers.simulate_counts(
        n_genes=120, n_samples=12, seed=12, disp_scatter=0.3, n_outliers=8
    )


@pytest.fixture(scope="module")
def few_replicates():
    """12 samples in two groups of 6: Cook's outliers are flagged but not replaced.

    ``two_larger`` has a high count (600) in group A whose Cook's distance is above
    the cutoff and exactly two larger counts, in group B.
    """
    sim = _few_replicates_sim()
    counts = sim.counts.copy()
    in_a = (sim.metadata["condition"] == "A").to_numpy()
    two_larger = np.zeros(12, dtype=np.int64)
    two_larger[in_a] = [600, 50, 52, 48, 51, 49]
    two_larger[~in_a] = [700, 710, 400, 410, 405, 395]
    counts["two_larger"] = two_larger
    return _fit(counts, sim.metadata)


@pytest.fixture(scope="module")
def unbalanced():
    """Two groups of 2 and 8 samples: only the group of 8 counts for Cook's maxima.

    The counts are drawn for alternating groups and then relabelled, so they do not
    follow this design; only the Cook's filtering of the result is of interest.
    """
    sim = helpers.simulate_counts(
        n_genes=80, n_samples=10, seed=13, disp_scatter=0.3, n_outliers=10
    )
    metadata = sim.metadata.copy()
    metadata["condition"] = ["A", "A"] + ["B"] * 8
    return _fit(sim.counts, metadata, fit_type="mean")


def _multi_factor_sim():
    return helpers.simulate_counts(
        n_genes=120,
        n_samples=24,
        seed=14,
        disp_scatter=0.3,
        factors={"condition": ["A", "B"], "group": ["X", "Y", "Z"]},
        continuous=("age",),
        n_all_zero=1,
    )


MULTI_DESIGN = "~group + condition + age"


@pytest.fixture(scope="module")
def multi_factor():
    sim = _multi_factor_sim()
    return _fit(sim.counts, sim.metadata, design=MULTI_DESIGN)


@pytest.fixture(scope="module")
def filtering():
    """Many low-count null genes and some high-count DE genes, so that independent
    filtering removes the lowest-mean genes."""
    sim = helpers.simulate_counts(
        n_genes=300, n_samples=8, seed=15, mean_range=(0.5, 2000), frac_de=0.4
    )
    return _fit(sim.counts, sim.metadata)


DATASETS = ["two_group", "few_replicates", "unbalanced", "multi_factor", "filtering"]


def _stats(dds, contrast=None, **kwargs):
    if contrast is None:
        contrast = ["condition", "B", "A"]
    kwargs.setdefault("quiet", True)
    if kwargs.get("inference") is None:
        kwargs.setdefault("n_cpus", helpers.get_test_cpus())
    return DeseqStats(dds, contrast=contrast, **kwargs)


# References computed from a fitted DeseqDataSet's public fields.


def _replicates(dds):
    """Per sample, how many samples share its row of the design matrix."""
    rows = dds.obsm["design_matrix"].to_numpy()
    _, inverse, n = np.unique(rows, axis=0, return_inverse=True, return_counts=True)
    return n[inverse.ravel()]


def _cooks_cutoff(dds):
    n, p = dds.obsm["design_matrix"].shape
    return stats.f.ppf(0.99, p, n - p)


def reference_replaced_counts(dds):
    """Counts after DESeq2's outlier replacement (``replaceOutliers``).

    Counts with a Cook's distance above the 0.99 quantile of F(p, n - p), in samples
    with at least ``min_replicates`` replicates, become the 20% trimmed mean of the
    gene's normalized counts times the sample's size factor, truncated to an integer.
    """
    counts = np.asarray(dds.X, dtype=float).copy()
    sf = dds.obs["size_factors"].to_numpy()
    replaceable = _replicates(dds) >= dds.min_replicates
    with np.errstate(invalid="ignore"):
        outlier = (dds.layers["cooks"] > _cooks_cutoff(dds)) & replaceable[:, None]
    trimmed = stats.trim_mean(counts / sf[:, None], 0.2, axis=0)
    replacement = np.trunc(trimmed[None, :] * sf[:, None])
    counts[outlier] = replacement[outlier]
    return counts


def reference_cooks_outlier(dds):
    """DESeq2's Cook's outlier flag for p-value filtering (``results``), two-group
    single-factor designs.

    A gene is flagged when its maximum Cook's distance over the samples with three or
    more replicates exceeds the 0.99 quantile of F(p, n - p), unless three or more
    samples have a larger count than the sample with the gene's largest Cook's
    distance. Refitted genes do not count their replaced samples.

    DESeq2 applies the "three larger counts" exception only to designs with a single
    two-level factor, and PyDESeq2 to every design, so the comparisons use two-group
    designs. DESeq2 never flags genes that are all zero after replacement (p-value 1)
    and PyDESeq2 does (p-value NaN); the comparisons leave them out
    (:func:`_cooks_comparable`).
    """
    cooks = dds.layers["cooks"].copy()
    if dds.refit_cooks and "refitted" in dds.var and dds.var["refitted"].any():
        replaceable = _replicates(dds) >= dds.min_replicates
        refitted = dds.var["refitted"].to_numpy(dtype=bool)
        cooks[np.ix_(replaceable, refitted)] = 0.0
    use_for_max = _replicates(dds) >= 3
    with np.errstate(invalid="ignore"):
        flagged = (cooks[use_for_max] > _cooks_cutoff(dds)).any(axis=0)
    counts = np.asarray(dds.X)
    for g in np.flatnonzero(flagged):
        at_max = np.nanargmax(dds.layers["cooks"][:, g])
        flagged[g] = np.sum(counts[:, g] > counts[at_max, g]) < 3
    return flagged


def reference_wald(dds, contrast, ridge=None):
    """Per-gene Wald p-values, statistics and SEs from the fitted fields."""
    x = dds.obsm["design_matrix"].to_numpy(dtype=float)
    lfc = dds.varm["LFC"].to_numpy()
    sf = dds.obs["size_factors"].to_numpy()
    disp = dds.var["dispersions"].to_numpy()
    if ridge is None:
        ridge = np.diag(np.full(x.shape[1], 1e-6))
    out = np.full((dds.n_vars, 3), np.nan)
    for g in range(dds.n_vars):
        if np.isfinite(lfc[g]).all():
            mu = sf * np.exp(x @ lfc[g])
            out[g] = helpers.wald_reference(
                x, disp[g], lfc[g], mu, ridge, np.asarray(contrast)
            )
    return pd.DataFrame(out, index=dds.var_names, columns=["pvalue", "stat", "se"])


def _new_all_zero(dds):
    replaced = dds.var["replaced"].to_numpy(dtype=bool)
    refitted = dds.var["refitted"].to_numpy(dtype=bool)
    return replaced & ~refitted


def _cooks_comparable(dds):
    """Genes whose Cook's outlier flag is compared: all but those all zero after
    outlier replacement (see :func:`reference_cooks_outlier`)."""
    if dds.refit_cooks and "refitted" in dds.var:
        return ~_new_all_zero(dds)
    return np.ones(dds.n_vars, dtype=bool)


# DeseqDataSet.cond and DeseqDataSet.contrast


@pytest.mark.parametrize(
    "design",
    [MULTI_DESIGN, "~condition * group", "~condition + C(group)"],
    ids=["additive", "interaction", "categorical"],
)
def test_cond_is_the_design_row(design):
    """``cond(**levels)`` is the design-matrix row of a sample with those levels.

    Exact: the rows are 0/1 indicators and copied covariate values.
    """
    sim = helpers.simulate_counts(
        n_genes=4,
        n_samples=12,
        seed=16,
        factors={"condition": ["A", "B"], "group": ["X", "Y", "Z"]},
        continuous=("age",),
    )
    dds = helpers.make_dds(sim.counts, sim.metadata, design=design)
    design_matrix = dds.obsm["design_matrix"]
    for sample in sim.metadata.index:
        levels = sim.metadata.loc[sample, list(dds.variables)].to_dict()
        np.testing.assert_array_equal(
            np.asarray(dds.cond(**levels), dtype=float),
            design_matrix.loc[sample].to_numpy(dtype=float),
        )


def test_contrast_is_a_difference_of_conds():
    """``contrast(column, baseline, group_to_compare)`` is ``cond(compared) -
    cond(baseline)``, other factors at their reference level; exact."""
    sim = _multi_factor_sim()
    dds = helpers.make_dds(sim.counts, sim.metadata, design=MULTI_DESIGN)
    columns = list(dds.obsm["design_matrix"].columns)
    assert columns == ["Intercept", "group[T.Y]", "group[T.Z]", "condition[T.B]", "age"]
    cases = {
        ("condition", "A", "B"): [0, 0, 0, 1, 0],
        ("condition", "B", "A"): [0, 0, 0, -1, 0],
        ("group", "X", "Z"): [0, 0, 1, 0, 0],
        ("group", "Y", "Z"): [0, -1, 1, 0, 0],
        ("group", "Z", "Y"): [0, 1, -1, 0, 0],
    }
    for (column, baseline, compared), expected in cases.items():
        actual = np.asarray(dds.contrast(column, baseline, compared), dtype=float)
        np.testing.assert_array_equal(actual, expected)
        difference = np.asarray(
            dds.cond(**{column: compared}), dtype=float
        ) - np.asarray(dds.cond(**{column: baseline}), dtype=float)
        np.testing.assert_array_equal(actual, difference)


def test_cond_contrast_and_variables_need_a_formula():
    """A design given as a DataFrame has no factor levels to name."""
    sim = helpers.simulate_counts(n_genes=30, n_samples=6, seed=17)
    dds = helpers.make_dds(
        sim.counts, sim.metadata, design=sim.design_matrix, fit_type="mean"
    )
    with pytest.raises(AttributeError):
        dds.cond(condition="B")
    with pytest.raises(AttributeError):
        dds.contrast("condition", "A", "B")
    dds.deseq2()
    ds = _stats(dds, contrast=np.array([0.0, 1.0]))
    with pytest.raises(ValueError):
        ds.variables  # noqa: B018


# DeseqStats constructor and variables


def test_deseqstats_variables(multi_factor):
    """``variables`` names the variables of the design formula."""
    ds = _stats(multi_factor)
    assert set(ds.variables) == {"group", "condition", "age"}
    assert ds.variables == multi_factor.variables


def test_deseqstats_constructor_validates_inputs(two_group):
    """Input checks: a contrast is required, a contrast vector has one entry per
    design column, a negative null is invalid for the absolute hypotheses, and
    ``refit_cooks`` needs ``refit`` to have run."""
    with pytest.raises(ValueError):
        DeseqStats(two_group, contrast=None, quiet=True, n_cpus=helpers.get_test_cpus())
    with pytest.raises(ValueError):
        _stats(two_group, contrast=np.array([0.0, 1.0, 0.0]))
    for alt in ("greaterAbs", "lessAbs"):
        with pytest.raises(ValueError):
            _stats(two_group, lfc_null=-0.5, alt_hypothesis=alt)

    sim = helpers.simulate_counts(n_genes=10, n_samples=6, seed=18)
    dds = helpers.make_dds(sim.counts, sim.metadata)
    dds.fit_size_factors()
    dds.fit_genewise_dispersions()
    dds.fit_dispersion_trend()
    dds.fit_dispersion_prior()
    dds.fit_MAP_dispersions()
    dds.fit_LFC()
    dds.calculate_cooks()
    with pytest.raises(AttributeError):
        _stats(dds)


def test_deseqstats_requires_fitted_lfcs():
    sim = helpers.simulate_counts(n_genes=10, n_samples=6, seed=18)
    dds = helpers.make_dds(sim.counts, sim.metadata)
    with pytest.raises(AssertionError):
        _stats(dds)


# run_wald_test and DefaultInference.wald_test


@pytest.mark.parametrize("dataset", DATASETS)
def test_run_wald_test_matches_reference(dataset, request):
    """``p_values``, ``statistics`` and ``SE`` against the Wald test computed from the
    fitted dispersions, LFCs and size factors.

    All-zero genes have no LFC and get NaN. Genes that become all zero after outlier
    replacement get SE 0, statistic 0 and p-value 1, as in DESeq2 1.50.2 (checked on
    simulated counts with genes positive in a single sample).
    """
    dds = request.getfixturevalue(dataset)
    ds = _stats(dds)
    ds.run_wald_test()
    expected = reference_wald(dds, ds.contrast_vector)
    if "replaced" in dds.var and dds.refit_cooks:
        new_all_zero = _new_all_zero(dds)
        expected.loc[new_all_zero, ["pvalue", "stat", "se"]] = [1.0, 0.0, 0.0]
        if dataset == "two_group":
            assert new_all_zero.sum() == 2
    helpers.assert_fields_close(ds.SE, expected["se"], rtol=WALD_RTOL, name="SE")
    helpers.assert_fields_close(
        ds.statistics, expected["stat"], rtol=WALD_RTOL, atol=1e-12, name="stat"
    )
    helpers.assert_fields_close(
        ds.p_values, expected["pvalue"], rtol=PVALUE_RTOL, name="pvalue"
    )


def test_run_wald_test_with_unit_prior_LFC_var(two_group):
    """``prior_LFC_var`` replaces the 1e-6 ridge; with unit prior variances the ridge
    is the identity whichever power of the variance it uses (PyDESeq2 uses
    ``1 / prior_LFC_var**2``, DESeq2 ``1 / betaPriorVar``)."""
    ds = _stats(two_group, prior_LFC_var=np.ones(2))
    ds.run_wald_test()
    expected = reference_wald(two_group, ds.contrast_vector, ridge=np.eye(2))
    refitted_or_regular = ~_new_all_zero(two_group)
    helpers.assert_fields_close(
        ds.SE[refitted_or_regular], expected["se"][refitted_or_regular], rtol=WALD_RTOL
    )
    # The ridge visibly changes the SEs of genes with a near-zero group.
    default = _stats(two_group)
    default.run_wald_test()
    assert ds.SE["zero_in_a"] < 0.5 * default.SE["zero_in_a"]


def test_inference_wald_test_gene_subsets_and_batches(few_replicates, n_cpus):
    """``DefaultInference.wald_test`` on one gene, two genes and all genes, with
    batch sizes 1 and 128, gives each gene the same result: genes are independent."""
    dds = few_replicates
    x = dds.obsm["design_matrix"].to_numpy(dtype=float)
    lfc = dds.varm["LFC"].to_numpy()
    mu = dds.obs["size_factors"].to_numpy()[:, None] * np.exp(x @ lfc.T)
    disp = dds.var["dispersions"].to_numpy()
    kwargs = {
        "design_matrix": x,
        "ridge_factor": np.diag([1e-6, 1e-6]),
        "contrast": np.array([0.0, 1.0]),
        "lfc_null": 0.0,
        "alt_hypothesis": None,
    }
    full = DefaultInference(n_cpus=n_cpus, batch_size=128).wald_test(
        disp=disp, lfc=lfc, mu=mu, **kwargs
    )
    expected = reference_wald(dds, kwargs["contrast"])
    np.testing.assert_allclose(full[2], expected["se"], rtol=WALD_RTOL)
    for genes in ([7], [3, 50], list(range(dds.n_vars))[::-1]):
        for batch_size in (1, 128):
            part = DefaultInference(n_cpus=n_cpus, batch_size=batch_size).wald_test(
                disp=disp[genes], lfc=lfc[genes], mu=mu[:, genes], **kwargs
            )
            for actual, whole in zip(part, full, strict=True):
                assert np.asarray(actual).shape == (len(genes),)
                np.testing.assert_allclose(actual, whole[genes], rtol=WALD_RTOL)


# summary


@pytest.mark.parametrize("dataset", DATASETS)
def test_summary_results_table(dataset, request):
    """``results_df`` column by column against references from the fitted fields.

    ``baseMean`` is the mean of normalized counts after outlier replacement;
    ``log2FoldChange`` and ``lfcSE`` are the contrast of the LFCs and the Wald SE in
    log2; ``pvalue`` is NaN for Cook's outliers; ``padj`` is DESeq2's independent
    filtering applied to those p-values.
    """
    dds = request.getfixturevalue(dataset)
    ds = _stats(dds)
    ds.summary()
    res = ds.results_df
    assert list(res.columns) == [
        "baseMean",
        "log2FoldChange",
        "lfcSE",
        "stat",
        "pvalue",
        "padj",
    ]
    assert res.index.equals(dds.var_names)

    sf = dds.obs["size_factors"].to_numpy()
    base_mean = (reference_replaced_counts(dds) / sf[:, None]).mean(axis=0)
    # Sums of a few dozen normalized counts: rounding only.
    helpers.assert_fields_close(res["baseMean"], base_mean, rtol=1e-12, name="baseMean")

    lfc = dds.varm["LFC"].to_numpy() @ ds.contrast_vector / LOG2
    helpers.assert_fields_close(
        res["log2FoldChange"], lfc, rtol=1e-12, atol=1e-15, name="log2FoldChange"
    )
    wald = reference_wald(dds, ds.contrast_vector)
    if dds.refit_cooks and "replaced" in dds.var:
        wald.loc[_new_all_zero(dds), ["pvalue", "stat", "se"]] = [1.0, 0.0, 0.0]
    helpers.assert_fields_close(
        res["lfcSE"], wald["se"] / LOG2, rtol=WALD_RTOL, name="lfcSE"
    )
    helpers.assert_fields_close(
        res["stat"], wald["stat"], rtol=WALD_RTOL, atol=1e-12, name="stat"
    )

    outliers = reference_cooks_outlier(dds) if dataset != "multi_factor" else None
    pvalues = wald["pvalue"].copy()
    if outliers is not None:
        pvalues[outliers] = np.nan
    else:
        # Every row of this design is unique (continuous covariate), so no sample
        # has the three replicates needed to count towards a Cook's maximum.
        assert not dds.cooks_outlier().any()
    keep = _cooks_comparable(dds)
    helpers.assert_fields_close(
        res["pvalue"][keep], pvalues[keep], rtol=PVALUE_RTOL, name="pvalue"
    )

    padj, _ = helpers.reference_independent_filtering(
        res["baseMean"], res["pvalue"], 0.05
    )
    # BH multiplies p-values by at most m / rank and takes running minima.
    helpers.assert_fields_close(res["padj"], padj, rtol=1e-10, name="padj")


def test_summary_independent_filtering_removes_low_counts(filtering):
    """On data with many low-count null genes, the chosen quantile is above the
    first, so the lowest-mean genes get NaN ``padj`` while keeping a ``pvalue``."""
    ds = _stats(filtering)
    ds.summary()
    res = ds.results_df
    _, chosen = helpers.reference_independent_filtering(
        res["baseMean"], res["pvalue"], 0.05
    )
    assert chosen > 0
    filtered = res["padj"].isna() & res["pvalue"].notna()
    assert filtered.sum() >= 10
    kept = res["padj"].notna()
    assert res.loc[filtered, "baseMean"].max() <= res.loc[kept, "baseMean"].min()


@pytest.mark.parametrize("alpha", [0.01, 0.1])
def test_summary_alpha_changes_the_filter(filtering, alpha):
    """``alpha`` is the rejection threshold of independent filtering."""
    ds = _stats(filtering, alpha=alpha)
    ds.summary()
    res = ds.results_df
    padj, _ = helpers.reference_independent_filtering(
        res["baseMean"], res["pvalue"], alpha
    )
    helpers.assert_fields_close(res["padj"], padj, rtol=1e-10, name="padj")


@pytest.mark.parametrize("dataset", ["two_group", "few_replicates"])
def test_summary_without_independent_filtering_is_bh(dataset, request):
    """``independent_filter=False``: BH over every gene with a p-value."""
    ds = _stats(request.getfixturevalue(dataset), independent_filter=False)
    ds.summary()
    p = ds.results_df["pvalue"]
    expected = pd.Series(np.nan, index=p.index)
    expected[p.notna()] = helpers.bh_adjust(p[p.notna()].to_numpy())
    helpers.assert_fields_close(ds.results_df["padj"], expected, rtol=1e-12, name="padj")


@pytest.mark.parametrize("dataset", ["few_replicates", "unbalanced", "two_group"])
def test_summary_cooks_filter(dataset, request):
    """``cooks_filter`` sets the p-values of Cook's outliers to NaN, and only those."""
    dds = request.getfixturevalue(dataset)
    outliers = reference_cooks_outlier(dds)
    keep = _cooks_comparable(dds)
    filtered, unfiltered = _stats(dds), _stats(dds, cooks_filter=False)
    filtered.summary()
    unfiltered.summary()
    no_pvalue = unfiltered.results_df["pvalue"].isna().to_numpy()
    np.testing.assert_array_equal(
        filtered.results_df["pvalue"].isna().to_numpy()[keep],
        (no_pvalue | outliers)[keep],
    )
    if dataset == "few_replicates":
        # Outlier counts 100 times the mean, in groups too small to replace them.
        assert outliers.sum() >= 5
    helpers.assert_fields_close(
        filtered.results_df["pvalue"][~outliers],
        unfiltered.results_df["pvalue"][~outliers],
        rtol=0,
        name="pvalue",
    )


@pytest.mark.parametrize(
    "dataset", ["few_replicates", "unbalanced", "two_group", "filtering"]
)
def test_cooks_outlier_matches_rule(dataset, request):
    """``DeseqDataSet.cooks_outlier`` against DESeq2's rule (two-group designs)."""
    dds = request.getfixturevalue(dataset)
    keep = _cooks_comparable(dds)
    np.testing.assert_array_equal(
        dds.cooks_outlier().to_numpy(dtype=bool)[keep],
        reference_cooks_outlier(dds)[keep],
    )


def test_cooks_outlier_two_larger_counts(few_replicates):
    """A gene stays a Cook's outlier when fewer than three samples have a larger count
    than its sample with the largest Cook's distance (here two do)."""
    dds = few_replicates
    g = list(dds.var_names).index("two_larger")
    cooks = dds.layers["cooks"][:, g]
    assert np.nanmax(cooks) > _cooks_cutoff(dds)
    at_max = np.nanargmax(cooks)
    assert np.sum(np.asarray(dds.X)[:, g] > dds.X[at_max, g]) == 2
    assert dds.cooks_outlier()["two_larger"]


def test_cooks_outliers_include_injected_outliers(few_replicates):
    """Counts 100 times their mean (simulator truth) are flagged when the mean is at
    least 20; a lower one need not be, as a count of 100 x mu may have three larger
    counts in the gene."""
    sim = _few_replicates_sim()
    flagged = few_replicates.cooks_outlier()
    for gene in sim.counts.columns[sim.outliers.any()]:
        sample = sim.outliers[gene].to_numpy().argmax()
        if sim.mu[gene].iloc[sample] >= 20:
            assert flagged[gene], gene


def test_summary_lfc_null_and_alt_hypothesis_kwargs(two_group):
    """``summary(lfc_null=..., alt_hypothesis=...)`` reruns the Wald test with the new
    values (log2 null, converted to natural log) and keeps them for later calls."""
    ds = _stats(two_group)
    ds.summary()
    before = ds.results_df.copy()
    ds.summary(lfc_null=0.5, alt_hypothesis="greaterAbs")
    assert (ds.lfc_null, ds.alt_hypothesis) == (0.5, "greaterAbs")
    fresh = _stats(two_group, lfc_null=0.5, alt_hypothesis="greaterAbs")
    fresh.summary()
    pd.testing.assert_frame_equal(ds.results_df, fresh.results_df, rtol=1e-12)
    ds.summary()
    pd.testing.assert_frame_equal(ds.results_df, fresh.results_df, rtol=1e-12)
    # The fold changes and their SEs do not depend on the hypothesis.
    pd.testing.assert_series_equal(ds.results_df["lfcSE"], before["lfcSE"], rtol=1e-12)
    with pytest.raises(ValueError):
        ds.summary(lfc_null=-1.0)


@pytest.mark.parametrize(
    "alt, lfc_null",
    [
        (None, 0.4),
        ("greaterAbs", 0.5),
        ("lessAbs", 0.5),
        ("greater", 0.3),
        ("less", -0.3),
    ],
)
def test_summary_alternative_hypotheses(two_group, alt, lfc_null):
    """Statistics and p-values for a log2 null value and alternative, against
    DESeq2's formulas (see test_wald_test_alternative_hypotheses), contrast B vs A.

    ``lfc_null`` is log2 and the Wald test works in natural log; the statistic does
    not depend on the scale. p-values are compared where DESeq2's statistic is not
    clipped to zero (where it is, PyDESeq2 reports 0.5 and DESeq2 the unclipped
    tail). The contrast is a unit vector: for other contrasts PyDESeq2 applies the
    null value and the alternative to each coefficient rather than to the
    contrast.
    """
    ds = _stats(two_group, lfc_null=lfc_null, alt_hypothesis=alt, cooks_filter=False)
    ds.summary()
    res = ds.results_df
    b, s = res["log2FoldChange"].to_numpy(), res["lfcSE"].to_numpy()
    t = abs(lfc_null)
    with np.errstate(divide="ignore", invalid="ignore"):
        if alt is None:
            stat = (b - lfc_null) / s
            pval = 2 * stats.norm.sf(np.abs(stat))
        elif alt == "greaterAbs":
            stat = np.sign(b) * np.fmax((np.abs(b) - t) / s, 0)
            pval = np.minimum(1, 2 * stats.norm.sf((np.abs(b) - t) / s))
        elif alt == "lessAbs":
            stat = np.fmin(np.fmax((t - b) / s, 0), np.fmax((b + t) / s, 0))
            pval = np.fmax(stats.norm.cdf((b - t) / s), stats.norm.sf((b + t) / s))
        elif alt == "greater":
            stat = np.fmax((b - t) / s, 0)
            pval = stats.norm.sf((b - t) / s)
        else:
            stat = np.fmin((b + t) / s, 0)
            pval = stats.norm.cdf((b + t) / s)
    regular = s > 0  # leaves out all-zero genes and genes all zero after replacement
    actual_stat = res["stat"].to_numpy()
    if alt == "lessAbs":
        actual_stat = np.abs(actual_stat)
    np.testing.assert_allclose(
        actual_stat[regular], stat[regular], rtol=WALD_RTOL, atol=1e-10
    )
    compare = regular & ((stat != 0) | (alt in (None, "greaterAbs")))
    assert compare.sum() >= 20
    np.testing.assert_allclose(
        res["pvalue"].to_numpy()[compare], pval[compare], rtol=PVALUE_RTOL
    )


# Contrasts through DeseqStats


def test_contrast_between_non_reference_levels_matches_releveled_fit(multi_factor):
    """Z vs Y from the fit with reference X equals the Z coefficient of a fit with
    reference Y.

    The two parametrizations describe the same model: the dispersion objective
    changes by a constant (the Cox-Reid term's log det of a fixed transform) and the
    fitted means are the same, so dispersions, fitted means and the Wald test of the
    contrast agree up to the optimizers' precision and the 1e-6 ridge (which is not
    invariant to the parametrization). Tolerances: IRLS coefficients within 1e-3
    (natural log, README), SEs and statistics within 1e-3 relative, p-values within
    1e-2 relative. Genes whose MAP fit takes the grid fallback in one fit only are
    left out (``helpers.map_fallback_flips``): on this design each numpy and scipy
    build's L-BFGS-B fails on a few genes, a different few under rounding-level
    changes.
    """
    ds = _stats(multi_factor, contrast=["group", "Z", "Y"])
    ds.summary()

    sim = _multi_factor_sim()
    metadata = sim.metadata.copy()
    metadata["group"] = metadata["group"].replace({"Y": "W"})  # W sorts first
    releveled = _fit(sim.counts, metadata, design=MULTI_DESIGN)
    assert releveled.obsm["design_matrix"].columns[1] == "group[T.X]"
    ref = _stats(releveled, contrast=["group", "Z", "W"])
    ref.summary()

    flips = helpers.map_fallback_flips(releveled, multi_factor)
    assert len(flips) <= 3, list(flips)
    genes = multi_factor.var_names.difference(flips, sort=False)
    a, e = ds.results_df.loc[genes], ref.results_df.loc[genes]
    helpers.assert_fields_close(
        a["log2FoldChange"], e["log2FoldChange"], rtol=0, atol=1e-3 / LOG2, name="lfc"
    )
    helpers.assert_fields_close(
        a[["lfcSE", "stat"]], e[["lfcSE", "stat"]], rtol=1e-3, atol=1e-6
    )
    helpers.assert_fields_close(a["pvalue"], e["pvalue"], rtol=1e-2, atol=1e-12)
    # The direct coefficient against the reference level is the same as before.
    np.testing.assert_allclose(
        ds.contrast_vector, multi_factor.contrast("group", "Y", "Z"), rtol=0
    )


def test_contrast_vector_matches_named_contrast(multi_factor):
    """A contrast given as a vector gives the same results as the named contrast
    that builds it; a covariate's coefficient is tested with a unit vector."""
    named = _stats(multi_factor, contrast=["condition", "B", "A"])
    named.summary()
    vector = _stats(multi_factor, contrast=np.array([0.0, 0.0, 0.0, 1.0, 0.0]))
    vector.summary()
    pd.testing.assert_frame_equal(named.results_df, vector.results_df, rtol=1e-12)

    age = _stats(multi_factor, contrast=np.array([0.0, 0.0, 0.0, 0.0, 1.0]))
    age.summary()
    lfc = multi_factor.varm["LFC"]["age"] / LOG2
    helpers.assert_fields_close(age.results_df["log2FoldChange"], lfc, rtol=1e-12)
    expected = reference_wald(multi_factor, np.array([0.0, 0.0, 0.0, 0.0, 1.0]))
    helpers.assert_fields_close(
        age.results_df["pvalue"], expected["pvalue"], rtol=PVALUE_RTOL, name="pvalue"
    )


def test_reversed_contrast_flips_signs(two_group):
    """A vs B is minus B vs A in fold change and statistic; p-values are equal."""
    forward, reverse = (
        _stats(two_group),
        _stats(two_group, contrast=["condition", "A", "B"]),
    )
    forward.summary()
    reverse.summary()
    f, r = forward.results_df, reverse.results_df
    helpers.assert_fields_close(r["log2FoldChange"], -f["log2FoldChange"], rtol=1e-12)
    helpers.assert_fields_close(r["stat"], -f["stat"], rtol=1e-12)
    helpers.assert_fields_close(
        r[["lfcSE", "pvalue", "padj"]], f[["lfcSE", "pvalue", "padj"]], rtol=1e-12
    )


def test_two_gene_dataset(n_cpus):
    """A dataset of two genes (mean trend) goes through summary and shrinkage, and
    its results match the references."""
    sim = helpers.simulate_counts(n_genes=2, n_samples=8, seed=19, frac_de=1.0)
    dds = _fit(sim.counts, sim.metadata, fit_type="mean")
    ds = _stats(dds)
    ds.summary()
    expected = reference_wald(dds, ds.contrast_vector)
    helpers.assert_fields_close(
        ds.results_df["pvalue"], expected["pvalue"], rtol=PVALUE_RTOL
    )
    np.testing.assert_allclose(
        ds.results_df["padj"],
        helpers.bh_adjust(ds.results_df["pvalue"].to_numpy()),
        rtol=1e-12,
    )
    _assert_shrinkage_matches_reference(dds, ds, TWO_GROUP_COEFF, adapt=True)


# lfc_shrink and DefaultInference.lfc_shrink_nbinom_glm


def _starts(mle, k):
    """Starts for helpers.reference_map: the MLE, and the MLE with the shrunk coefficient
    at zero, inside the two basins a Cauchy prior can create."""
    at_zero = np.array(mle, dtype=float)
    at_zero[k] = 0.0
    return [np.array(mle, dtype=float), at_zero]


def _assert_shrinkage_matches_reference(dds, ds, coeff, adapt, max_bimodal=2):
    """Shrink ``coeff`` and compare every gene with the reference MAP.

    The prior scale is apeglm's, from the unshrunk LFCs and SEs (``ds.LFC``,
    ``ds.SE``), capped at 1; 1 without ``adapt``. Only the shrunk coefficient is
    reported, so it is compared at SHRINK_ATOL. Genes whose posterior has two local
    minima are left out (at most ``max_bimodal``).
    """
    k = list(ds.LFC.columns).index(coeff)
    mle = ds.LFC.to_numpy().copy()
    finite = np.isfinite(mle[:, k])
    if adapt:
        prior_var = helpers.apeglm_prior_var(mle[finite, k], ds.SE.to_numpy()[finite])
        scale = min(np.sqrt(prior_var), 1.0)
    else:
        scale = 1.0
    other = [c for c in ds.LFC.columns if c != coeff]
    before = ds.LFC[other].copy()

    ds.lfc_shrink(coeff=coeff, adapt=adapt)

    pd.testing.assert_frame_equal(ds.LFC[other], before)
    shrunk = ds.LFC[coeff].to_numpy()
    np.testing.assert_array_equal(np.isnan(shrunk), ~finite)
    x = dds.obsm["design_matrix"].to_numpy(dtype=float)
    offset = np.log(dds.obs["size_factors"].to_numpy())
    size = 1.0 / dds.var["dispersions"].to_numpy()
    n_bimodal = 0
    for g in np.flatnonzero(finite):
        y = np.asarray(dds.X[:, g], dtype=float)
        ref, spread = helpers.reference_map(
            x, y, size[g], offset, k, scale, _starts(mle[g], k)
        )
        if spread > helpers.BIMODAL_SPREAD:
            n_bimodal += 1
            continue
        assert abs(shrunk[g] - ref[k]) <= SHRINK_ATOL, (
            dds.var_names[g],
            shrunk[g],
            ref[k],
        )
    assert n_bimodal <= max_bimodal
    return scale


@pytest.mark.parametrize(
    "dataset, adapt",
    [("few_replicates", True), ("few_replicates", False), ("two_group", False)],
)
def test_lfc_shrink_two_group_matches_reference(dataset, adapt, request):
    """Shrunk LFCs of every non-zero gene against the reference MAP; the other
    columns, p-values and adjusted p-values do not change.

    ``two_group`` is shrunk only with the unit prior: its genes that are all zero
    after outlier replacement have LFC 0 and SE 0, which puts apeglm's adaptive prior
    variance at its minimum (as in DESeq2 1.50.2 and apeglm 1.32.0 on the same
    counts), and the resulting posteriors have two modes for many genes.
    """
    dds = request.getfixturevalue(dataset)
    ds = _stats(dds)
    ds.summary()
    before = ds.results_df.copy()
    scale = _assert_shrinkage_matches_reference(dds, ds, TWO_GROUP_COEFF, adapt)
    assert 0.01 < scale < 1.0 if adapt else scale == 1.0
    res = ds.results_df
    pd.testing.assert_frame_equal(
        res[["baseMean", "stat", "pvalue", "padj"]],
        before[["baseMean", "stat", "pvalue", "padj"]],
    )
    np.testing.assert_allclose(
        res["log2FoldChange"], ds.LFC[TWO_GROUP_COEFF] / LOG2, rtol=1e-12
    )


@pytest.mark.parametrize("coeff", ["condition[T.B]", "group[T.Z]", "age"])
def test_lfc_shrink_multi_factor_matches_reference(multi_factor, coeff):
    """Shrinking a factor level or a covariate in a five-coefficient design."""
    contrast = (multi_factor.obsm["design_matrix"].columns == coeff).astype(float)
    ds = _stats(multi_factor, contrast=contrast)
    ds.summary()
    scale = _assert_shrinkage_matches_reference(multi_factor, ds, coeff, adapt=True)
    assert scale > 0.01


def test_lfc_shrink_null_data_uses_minimum_prior_variance(n_cpus):
    """Without differential expression apeglm's prior variance is at its minimum
    (1e-6, scale 1e-3), so every shrunk LFC is within a few thousandths of zero."""
    sim = helpers.simulate_counts(n_genes=100, n_samples=8, seed=5, frac_de=0.0)
    dds = _fit(sim.counts, sim.metadata)
    ds = _stats(dds)
    ds.summary()
    scale = _assert_shrinkage_matches_reference(dds, ds, TWO_GROUP_COEFF, adapt=True)
    assert scale == pytest.approx(1e-3, rel=1e-12)
    assert ds.LFC[TWO_GROUP_COEFF].abs().max() < 5e-3


def test_lfc_shrink_unknown_coefficient(two_group):
    ds = _stats(two_group)
    with pytest.raises(KeyError):
        ds.lfc_shrink(coeff="condition[T.C]")


def test_lfc_shrink_before_summary(two_group):
    """Shrinking before ``summary`` gives the same shrunk LFCs as after it."""
    first = _stats(two_group)
    first.run_wald_test()
    first.lfc_shrink(coeff=TWO_GROUP_COEFF)
    second = _stats(two_group)
    second.summary()
    second.lfc_shrink(coeff=TWO_GROUP_COEFF)
    helpers.assert_fields_close(first.LFC, second.LFC, rtol=1e-12, name="LFC")


def test_inference_lfc_shrink_gene_subsets(two_group, n_cpus):
    """``DefaultInference.lfc_shrink_nbinom_glm`` on one gene, two genes and all genes
    gives each gene the result of the reference MAP; ``converged`` is per gene."""
    dds = two_group
    nz = dds.var["non_zero"].to_numpy(dtype=bool)
    x = dds.obsm["design_matrix"].to_numpy(dtype=float)
    counts = np.asarray(dds.X[:, nz], dtype=float)
    size = 1.0 / dds.var["dispersions"].to_numpy()[nz]
    offset = np.log(dds.obs["size_factors"].to_numpy())
    kwargs = {
        "design_matrix": x,
        "offset": offset,
        "prior_no_shrink_scale": 15.0,
        "prior_scale": 0.5,
        "optimizer": "L-BFGS-B",
        "shrink_index": 1,
    }
    inference = DefaultInference(n_cpus=n_cpus, batch_size=16)
    lfcs, inv_hessians, converged = inference.lfc_shrink_nbinom_glm(
        counts=counts, size=size, **kwargs
    )
    n = counts.shape[1]
    assert np.asarray(lfcs).shape == (n, 2)
    assert np.asarray(inv_hessians).shape == (n, 2, 2)
    assert np.asarray(converged).shape == (n,)
    assert np.asarray(converged).all()
    for genes in ([0], [5, 140]):
        part = DefaultInference(n_cpus=n_cpus, batch_size=1).lfc_shrink_nbinom_glm(
            counts=counts[:, genes], size=size[genes], **kwargs
        )
        np.testing.assert_allclose(
            part[0], np.asarray(lfcs)[genes], rtol=0, atol=SHRINK_ATOL
        )
        for g, beta in zip(genes, part[0], strict=True):
            start = [np.log(counts[:, g].mean() + 0.5), 0.0]
            ref, spread = helpers.reference_map(
                x, counts[:, g], size[g], offset, 1, 0.5, _starts(start, 1)
            )
            assert spread < helpers.BIMODAL_SPREAD
            np.testing.assert_allclose(beta, ref, rtol=0, atol=SHRINK_ATOL)


# Invariance


def test_results_do_not_depend_on_n_cpus_or_batch_size(two_group):
    """Wald results and shrunk LFCs with one process and batches of one equal those
    with two processes and batches of seven (each gene is solved on its own)."""
    out = []
    for n_cpus, batch_size in [(1, 1), (2, 7)]:
        ds = _stats(
            two_group, inference=DefaultInference(n_cpus=n_cpus, batch_size=batch_size)
        )
        ds.summary()
        ds.lfc_shrink(coeff=TWO_GROUP_COEFF)
        out.append(ds.results_df)
    helpers.assert_fields_close(
        out[1][["baseMean", "stat", "pvalue", "padj"]],
        out[0][["baseMean", "stat", "pvalue", "padj"]],
        rtol=WALD_RTOL,
        atol=1e-12,
    )
    helpers.assert_fields_close(
        out[1]["log2FoldChange"],
        out[0]["log2FoldChange"],
        rtol=0,
        atol=SHRINK_ATOL / LOG2,
    )


def test_results_permute_with_genes(n_cpus):
    """Permuting the genes of the count matrix permutes every column of the results,
    unshrunk and shrunk.

    The two fits are separate ``deseq2()`` runs, so they agree to the solvers'
    precision, not to rounding: a solver whose stopping point depends on which genes
    share a batch is a correct reorganization. Dispersions within 1e-3 relative and
    LFCs within 1e-3 (natural log) move the statistic by about 1e-3 relative plus
    1e-3 / SE; SEs of these genes are above 0.05, so 2e-2 absolute. A two-sided
    p-value moves by about ``stat^2`` times the statistic's relative change (and
    BH-adjusted values with it), so 5e-2 relative with a 1e-6 floor. A gene whose
    MAP fit takes the grid fallback in one run only is left out
    (``helpers.map_fallback_flips``).
    """
    counts, metadata = _two_group_counts()
    order = np.random.RandomState(20).permutation(counts.shape[1])
    fits, results = [], []
    for c in (counts, counts.iloc[:, order]):
        dds = _fit(c, metadata)
        ds = _stats(dds)
        ds.summary()
        unshrunk = ds.results_df.copy()
        ds.lfc_shrink(coeff=TWO_GROUP_COEFF)
        fits.append(dds)
        results.append((unshrunk, ds.results_df))
    flips = helpers.map_fallback_flips(fits[1], fits[0])
    assert len(flips) <= 1, list(flips)
    genes = fits[0].var_names.difference(flips, sort=False)
    se = results[0][0]["lfcSE"].loc[genes] * LOG2  # natural log
    assert se[se > 0].min() > 0.05
    for a, b in zip(results[0], results[1], strict=True):
        a, b = a.loc[genes], b.loc[genes]
        # The counts after outlier replacement: rounding only.
        helpers.assert_fields_close(b["baseMean"], a["baseMean"], rtol=1e-12)
        helpers.assert_fields_close(b["lfcSE"], a["lfcSE"], rtol=2e-3)
        helpers.assert_fields_close(b["stat"], a["stat"], rtol=2e-3, atol=2e-2)
        helpers.assert_fields_close(
            b[["pvalue", "padj"]], a[["pvalue", "padj"]], rtol=5e-2, atol=1e-6
        )
        helpers.assert_fields_close(
            b["log2FoldChange"], a["log2FoldChange"], rtol=0, atol=SHRINK_ATOL / LOG2
        )


# The Inference interface


def test_inference_interface_requires_testing_methods():
    """``Inference`` is abstract: a subclass must implement ``wald_test`` and
    ``lfc_shrink_nbinom_glm`` to be instantiated."""

    class WithoutTesting(Inference):
        lin_reg_mu = DefaultInference.lin_reg_mu
        irls = DefaultInference.irls
        alpha_mle = DefaultInference.alpha_mle
        fit_rough_dispersions = DefaultInference.fit_rough_dispersions
        fit_moments_dispersions = DefaultInference.fit_moments_dispersions
        dispersion_trend_gamma_glm = DefaultInference.dispersion_trend_gamma_glm

    with pytest.raises(TypeError):
        WithoutTesting()

    class WithTesting(WithoutTesting):
        wald_test = DefaultInference.wald_test
        lfc_shrink_nbinom_glm = DefaultInference.lfc_shrink_nbinom_glm

    WithTesting()


def test_deseqstats_uses_its_inference_object(two_group, n_cpus):
    """``DeseqStats`` takes its Wald test and shrinkage from the ``inference`` object
    it is given (the documented way to plug in other solvers)."""

    class Shifted(DefaultInference):
        def wald_test(self, *args, **kwargs):
            pvals, stats_, se = super().wald_test(*args, **kwargs)
            return pvals / 2, stats_, se

        def lfc_shrink_nbinom_glm(self, *args, **kwargs):
            lfcs, inv_hessians, converged = super().lfc_shrink_nbinom_glm(
                *args, **kwargs
            )
            return np.asarray(lfcs) + 1.0, inv_hessians, converged

    plain = _stats(two_group, cooks_filter=False)
    plain.run_wald_test()
    shifted = _stats(two_group, cooks_filter=False, inference=Shifted(n_cpus=n_cpus))
    shifted.run_wald_test()
    # Genes all zero after replacement get p-value 1 whatever the test returns.
    tested = ~_new_all_zero(two_group)
    np.testing.assert_allclose(
        shifted.p_values[tested], plain.p_values[tested] / 2, rtol=1e-12
    )
    plain.lfc_shrink(coeff=TWO_GROUP_COEFF)
    shifted.lfc_shrink(coeff=TWO_GROUP_COEFF)
    np.testing.assert_allclose(
        shifted.LFC[TWO_GROUP_COEFF], plain.LFC[TWO_GROUP_COEFF] + 1.0, rtol=1e-12
    )
