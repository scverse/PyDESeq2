"""Area A: data handling and normalization (see tests/TARGETS.md).

Size factors (``DeseqDataSet.fit_size_factors`` with the ratio, poscounts and
iterative methods), the median-of-ratios functions of ``pydeseq2.preprocessing``, and
the helpers of ``pydeseq2.misc``. The constructor, ``Inference`` and
``DefaultInference`` are in ``test_unit_data_dataset.py``; the variance stabilizing
transformation is in ``test_unit_data_vst.py``.
"""

import multiprocessing
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from pydeseq2 import misc
from pydeseq2 import preprocessing
from tests import helpers

pytestmark = helpers.PANDAS_DEPRECATION_FILTERS

DATA = Path(__file__).parent / "data" / "unit_a"

# Size factors are a median of log ratios followed by exp: a handful of roundings.
EXACT_RTOL = 1e-12


def poscounts_size_factors(counts, control=None):
    """DESeq2's ``estimateSizeFactors(type = "poscounts")``, written out.

    ``geoMeanNZ = exp(sum(log(y[y > 0])) / n_samples)`` per gene (zero for an all-zero
    gene); per sample, the median of ``log(y) - log(geoMeanNZ)`` over genes with
    ``y > 0`` and a positive ``geoMeanNZ``; then rescaled to a geometric mean of 1.
    """
    y = np.asarray(counts, dtype=float)
    positive = y > 0
    log_y = np.log(np.where(positive, y, 1.0))
    log_geo = log_y.sum(axis=0) / y.shape[0]
    usable = positive.any(axis=0)
    if control is not None:
        usable &= np.asarray(control)
    log_sf = np.array(
        [
            np.median((log_y[j] - log_geo)[usable & positive[j]])
            for j in range(y.shape[0])
        ]
    )
    return np.exp(log_sf - log_sf.mean())


def read_fixture(prefix):
    counts = pd.read_csv(DATA / f"{prefix}_counts.csv", index_col=0)
    metadata = pd.read_csv(DATA / f"{prefix}_metadata.csv", index_col=0)
    return counts, metadata


def test_ratio_size_factors_hand_worked():
    """Median-of-ratios on the matrix worked in ``helpers.size_factor_example``.

    The expected values are exact powers of two; 1e-12 relative covers the rounding
    of log, mean, median and exp.
    """
    example = helpers.size_factor_example()
    dds = helpers.make_dds(example["counts"], example["metadata"])
    dds.fit_size_factors(fit_type="ratio")

    np.testing.assert_allclose(
        dds.obs["size_factors"], example["ratio_size_factors"], rtol=1e-12
    )
    np.testing.assert_allclose(
        dds.layers["normed_counts"],
        example["counts"].to_numpy() / example["ratio_size_factors"][:, None],
        rtol=1e-12,
    )


def test_ratio_size_factors_control_genes_hand_worked():
    """Control genes g1-g3 give medians -1, -1, 1 (``helpers.size_factor_example``)."""
    example = helpers.size_factor_example()
    dds = helpers.make_dds(
        example["counts"],
        example["metadata"],
        control_genes=example["control_genes"],
    )
    dds.fit_size_factors(fit_type="ratio")

    np.testing.assert_allclose(
        dds.obs["size_factors"], example["control_size_factors"], rtol=1e-12
    )


@pytest.mark.parametrize(
    "indexer",
    [
        ["g1", "g2", "g3"],
        pd.Index(["g1", "g2", "g3"]),
        np.array([True, True, True, False, False, False, False]),
        [0, 1, 2],
    ],
    ids=["names", "index", "mask", "positions"],
)
def test_control_genes_accept_any_indexer(indexer):
    """Names, an Index, a boolean mask and positions select the same control genes.

    Passed to ``fit_size_factors`` they override the genes given at initialization
    (g4 and g5 alone would give log2 medians 1, -0.5 and -0.5).
    """
    example = helpers.size_factor_example()
    expected = example["control_size_factors"]

    dds = helpers.make_dds(example["counts"], example["metadata"], control_genes=indexer)
    dds.fit_size_factors()
    np.testing.assert_allclose(dds.obs["size_factors"], expected, rtol=EXACT_RTOL)

    dds = helpers.make_dds(
        example["counts"], example["metadata"], control_genes=["g4", "g5"]
    )
    dds.fit_size_factors(control_genes=indexer)
    np.testing.assert_allclose(dds.obs["size_factors"], expected, rtol=EXACT_RTOL)


def _ratio_dataset():
    """Simulated counts with large, all-zero, single-sample and outlier genes.

    Columns are shuffled so that the genes the ratio method ignores sit between the
    others.
    """
    sim = helpers.simulate_counts(
        n_genes=300,
        n_samples=9,
        seed=101,
        n_large=3,
        n_all_zero=2,
        n_single_sample=3,
        n_outliers=6,
        disp_scatter=0.3,
    )
    order = np.random.RandomState(101).permutation(sim.counts.shape[1])
    return sim, sim.counts.iloc[:, order]


def test_ratio_size_factors_match_reference():
    """Size factors and normalized counts against the median-of-ratios written out."""
    sim, counts = _ratio_dataset()
    dds = helpers.make_dds(counts, sim.metadata)
    dds.fit_size_factors()

    expected = helpers.median_of_ratios(counts)
    helpers.assert_fields_close(
        dds.obs["size_factors"].to_numpy(), expected, rtol=EXACT_RTOL, name="sf"
    )
    helpers.assert_fields_close(
        pd.DataFrame(
            dds.layers["normed_counts"], index=counts.index, columns=counts.columns
        ),
        counts / expected[:, None],
        rtol=EXACT_RTOL,
        name="normed_counts",
    )


def test_ratio_size_factors_recover_truth():
    """Size factors estimate the simulated ones within a 6 standard error bound.

    The median of G log ratios with spread s has a standard error of about
    1.2533 s / sqrt(G); s is the per-sample MAD of the log ratios, estimated from the
    same counts. Both sides are divided by their geometric mean (the ratio method
    does not rescale).
    """
    sim = helpers.simulate_counts(
        n_genes=2000, n_samples=8, seed=102, frac_de=0.0, size_factor_sd=0.5
    )
    dds = helpers.make_dds(sim.counts, sim.metadata)
    dds.fit_size_factors()

    log_sf = np.log(dds.obs["size_factors"].to_numpy())
    log_sf -= log_sf.mean()
    log_true = np.log(sim.size_factors.to_numpy())

    y = sim.counts.to_numpy(dtype=float)
    usable = (y > 0).all(axis=0)
    log_y = np.log(y[:, usable])
    ratios = log_y - log_y.mean(axis=0)
    spread = 1.4826 * np.median(
        np.abs(ratios - np.median(ratios, axis=1, keepdims=True)), axis=1
    )
    se = 1.2533 * spread / np.sqrt(usable.sum())
    assert np.all(np.abs(log_sf - log_true) < 6 * se.max())


@pytest.mark.parametrize("fit_type", ["ratio", "poscounts"])
def test_size_factors_follow_gene_and_sample_order(fit_type):
    """Permuting genes leaves size factors unchanged; permuting samples permutes them.

    The median and the per-gene sums are the same numbers in another order, so the
    results agree to the rounding of a reordered sum (1e-12).
    """
    sim, counts = _ratio_dataset()
    dds = helpers.make_dds(counts, sim.metadata)
    dds.fit_size_factors(fit_type=fit_type)
    sf = dds.obs["size_factors"]

    genes = np.random.RandomState(1).permutation(counts.shape[1])
    dds_genes = helpers.make_dds(counts.iloc[:, genes], sim.metadata)
    dds_genes.fit_size_factors(fit_type=fit_type)
    helpers.assert_fields_close(
        dds_genes.obs["size_factors"], sf, rtol=EXACT_RTOL, name="sf"
    )

    samples = np.random.RandomState(2).permutation(counts.shape[0])
    dds_samples = helpers.make_dds(counts.iloc[samples], sim.metadata.iloc[samples])
    dds_samples.fit_size_factors(fit_type=fit_type)
    helpers.assert_fields_close(
        dds_samples.obs["size_factors"], sf.iloc[samples], rtol=EXACT_RTOL, name="sf"
    )


def test_ratio_size_factors_scale_with_a_sample():
    """Multiplying one sample's counts by 3 multiplies its size factor by 3 relative
    to every other sample's (the gene means of log counts shift by the same amount
    for every gene, so the medians shift exactly)."""
    sim, counts = _ratio_dataset()
    dds = helpers.make_dds(counts, sim.metadata)
    dds.fit_size_factors()
    sf = dds.obs["size_factors"].to_numpy()

    scaled = counts.copy()
    scaled.iloc[4] *= 3
    dds_scaled = helpers.make_dds(scaled, sim.metadata)
    dds_scaled.fit_size_factors()
    sf_scaled = dds_scaled.obs["size_factors"].to_numpy()

    expected = sf / sf[0]
    expected[4] *= 3
    np.testing.assert_allclose(sf_scaled / sf_scaled[0], expected, rtol=EXACT_RTOL)


def test_size_factors_of_one_and_two_genes():
    """With one gene the size factor is the count over its geometric mean; with two,
    the median of two log ratios is their mean."""
    y = np.array([[3, 40], [12, 10], [6, 20], [24, 5]])
    metadata = pd.DataFrame(
        {"condition": ["A", "A", "B", "B"]}, index=[f"s{i}" for i in range(4)]
    )
    one = pd.DataFrame(y[:, :1], index=metadata.index, columns=["g1"])
    two = pd.DataFrame(y, index=metadata.index, columns=["g1", "g2"])

    log_y = np.log(y.astype(float))
    ratios = log_y - log_y.mean(axis=0)
    expected_one = np.exp(ratios[:, 0])
    expected_two = np.exp(ratios.mean(axis=1))

    for counts, expected in [(one, expected_one), (two, expected_two)]:
        dds = helpers.make_dds(counts, metadata)
        dds.fit_size_factors()
        np.testing.assert_allclose(dds.obs["size_factors"], expected, rtol=EXACT_RTOL)
        np.testing.assert_allclose(
            preprocessing.deseq2_norm(counts)[1], expected, rtol=EXACT_RTOL
        )


def _poscounts_example():
    """Counts ``2**e`` (``-`` is a zero count) worked by hand::

             g1  g2  g3  g4  g5
        s1    0   3   2   -   -
        s2    1   -   2   3   -
        s3    2   1   -   3   -
        s4    1   0   4   2   -

    Sums of exponents over the 4 samples are 4, 4, 8, 8, so the log2 geometric means
    of the positive counts (zeros count as 1) are 1, 1, 2, 2; g5 is all zero. The log2
    ratios over each sample's positive counts are s1: (-1, 2, 0), s2: (0, 0, 1),
    s3: (1, 0, 1), s4: (0, -1, 2, 0), with medians 0, 0, 1, 0. Rescaled to a geometric
    mean of 1 they are 2**-0.25, 2**-0.25, 2**0.75, 2**-0.25. With control genes g1
    and g2 the medians are 0.5, 0, 0.5, -0.5, giving 2**(0.375, -0.125, 0.375,
    -0.625). No gene has positive counts that are all 1, where PyDESeq2 and DESeq2
    differ.
    """
    exponents = np.array(
        [[0, 3, 2, -1, -1], [1, -1, 2, 3, -1], [2, 1, -1, 3, -1], [1, 0, 4, 2, -1]]
    )
    counts = np.where(exponents >= 0, 2.0 ** np.maximum(exponents, 0), 0).astype(int)
    samples = ["s1", "s2", "s3", "s4"]
    return {
        "counts": pd.DataFrame(
            counts, index=samples, columns=[f"g{i}" for i in range(1, 6)]
        ),
        "metadata": pd.DataFrame({"condition": ["A", "A", "B", "B"]}, index=samples),
        "size_factors": 2.0 ** np.array([-0.25, -0.25, 0.75, -0.25]),
        "control_size_factors": 2.0 ** np.array([0.375, -0.125, 0.375, -0.625]),
    }


def test_poscounts_size_factors_hand_worked():
    example = _poscounts_example()
    dds = helpers.make_dds(example["counts"], example["metadata"])
    dds.fit_size_factors(fit_type="poscounts")
    np.testing.assert_allclose(
        dds.obs["size_factors"], example["size_factors"], rtol=EXACT_RTOL
    )
    np.testing.assert_allclose(
        dds.layers["normed_counts"],
        example["counts"].to_numpy() / example["size_factors"][:, None],
        rtol=EXACT_RTOL,
    )
    # The reference implementation reproduces the hand-worked values.
    np.testing.assert_allclose(
        poscounts_size_factors(example["counts"]), example["size_factors"], rtol=1e-14
    )

    dds = helpers.make_dds(
        example["counts"], example["metadata"], control_genes=["g1", "g2"]
    )
    dds.fit_size_factors(fit_type="poscounts")
    np.testing.assert_allclose(
        dds.obs["size_factors"], example["control_size_factors"], rtol=EXACT_RTOL
    )


def test_poscounts_size_factors_match_deseq2():
    """Sparse counts: against the reference and DESeq2 1.50.2's poscounts.

    DESeq2's output is written with 17 significant digits; both sides compute the
    same medians, so 1e-10 covers rounding.
    """
    counts, metadata = read_fixture("sparse")
    r_size_factors = pd.read_csv(DATA / "sparse_size_factors.csv", index_col=0)
    dds = helpers.make_dds(counts, metadata)
    dds.fit_size_factors(fit_type="poscounts")

    helpers.assert_fields_close(
        dds.obs["size_factors"], r_size_factors["poscounts"], rtol=1e-10, name="sf"
    )
    helpers.assert_fields_close(
        dds.obs["size_factors"].to_numpy(),
        poscounts_size_factors(counts),
        rtol=EXACT_RTOL,
        name="sf",
    )
    control = np.arange(counts.shape[1]) % 3 == 0
    dds = helpers.make_dds(counts, metadata, control_genes=control)
    dds.fit_size_factors(fit_type="poscounts")
    helpers.assert_fields_close(
        dds.obs["size_factors"].to_numpy(),
        poscounts_size_factors(counts, control),
        rtol=EXACT_RTOL,
        name="sf",
    )


# PyDESeq2 minimizes the trimmed NB likelihood over log size factors with Powell's
# method, DESeq2 with Nelder-Mead (reltol 1e-4); both stop when the size factors change
# by less than 1e-4 in summed squared log. On the bundled synthetic data and on the
# sparse fixture the two agree to 1.1e-3 and 4.6e-4; 5e-3 leaves a margin of 4.5x.
ITERATIVE_RTOL = 5e-3


@pytest.mark.parametrize("fit_type", [None, "iterative"])
def test_iterative_size_factors_match_deseq2(fit_type):
    """Every gene has a zero: the ratio method switches to the iterative one.

    Against DESeq2 1.50.2's ``estimateSizeFactors(type = "iterate")``, through the
    switch (``fit_type=None`` is the default ratio method) and called directly.
    """
    counts, metadata = read_fixture("sparse")
    r_size_factors = pd.read_csv(DATA / "sparse_size_factors.csv", index_col=0)
    dds = helpers.make_dds(counts, metadata)
    design = dds.obsm["design_matrix"].copy()

    if fit_type is None:
        with pytest.warns(UserWarning, match="Every gene contains at least one zero"):
            dds.fit_size_factors()
    else:
        dds.fit_size_factors(fit_type=fit_type)

    sf = dds.obs["size_factors"]
    helpers.assert_fields_close(
        sf, r_size_factors["iterate"], rtol=ITERATIVE_RTOL, name="sf"
    )
    # Rescaled to a geometric mean of 1 (rounding of log, mean and exp).
    assert np.exp(np.log(sf).mean()) == pytest.approx(1.0, rel=EXACT_RTOL)
    np.testing.assert_allclose(
        dds.layers["normed_counts"],
        counts.to_numpy() / sf.to_numpy()[:, None],
        rtol=EXACT_RTOL,
    )
    # The intercept-only design used inside the fit is replaced by the original.
    pd.testing.assert_frame_equal(
        pd.DataFrame(dds.obsm["design_matrix"]), pd.DataFrame(design)
    )


# preprocessing


def test_deseq2_norm_fit_hand_worked():
    """Log geometric means of ``helpers.size_factor_example``: exponent means 1, 2, 2,
    2, 2 times log 2; g6 has a zero and g7 is all zero, so both are -inf and filtered
    out."""
    example = helpers.size_factor_example()
    expected = np.array([1, 2, 2, 2, 2, -np.inf, -np.inf]) * np.log(2)
    for counts in [example["counts"], example["counts"].to_numpy()]:
        logmeans, filtered_genes = preprocessing.deseq2_norm_fit(counts)
        np.testing.assert_allclose(np.asarray(logmeans), expected, rtol=1e-14)
        np.testing.assert_array_equal(
            np.asarray(filtered_genes), [True] * 5 + [False] * 2
        )


def test_deseq2_norm_transform_applies_fitted_logmeans():
    """New samples normalized with the log means fitted on ``size_factor_example``.

    The fitted log2 means of g1-g5 are 1, 2, 2, 2, 2. New samples, as exponents of 2
    for g1-g5: t1 = (2, 3, 5, 4, 6), log2 ratios (1, 1, 3, 2, 4), median 2, size
    factor 4; t2 = (1, 3, 2, 1, 0), ratios (0, 1, 0, -1, -2), median 0, size factor 1;
    t3 = (3, 3, 3, 3, 3), ratios (2, 1, 1, 1, 1), median 1, size factor 2. g6 and g7
    are filtered out at the fit, so a zero count there changes nothing.
    """
    example = helpers.size_factor_example()
    new_exponents = np.array(
        [[2, 3, 5, 4, 6], [1, 3, 2, 1, 0], [3, 3, 3, 3, 3]], dtype=float
    )
    new = np.hstack([2**new_exponents, [[0, 3], [7, 0], [1, 1]]]).astype(int)
    expected_sf = np.array([4.0, 1.0, 2.0])

    for counts, new_counts in [
        (example["counts"], pd.DataFrame(new, columns=example["counts"].columns)),
        (example["counts"].to_numpy(), new),
    ]:
        logmeans, filtered_genes = preprocessing.deseq2_norm_fit(counts)
        normed, sf = preprocessing.deseq2_norm_transform(
            new_counts, logmeans, filtered_genes
        )
        np.testing.assert_allclose(np.asarray(sf), expected_sf, rtol=1e-14)
        np.testing.assert_allclose(
            np.asarray(normed), new / expected_sf[:, None], rtol=1e-14
        )
        if isinstance(new_counts, pd.DataFrame):
            assert isinstance(normed, pd.DataFrame)
            assert normed.columns.equals(new_counts.columns)
            assert normed.index.equals(new_counts.index)


@pytest.mark.parametrize("as_frame", [True, False])
def test_deseq2_norm_hand_worked(as_frame):
    example = helpers.size_factor_example()
    counts = example["counts"] if as_frame else example["counts"].to_numpy()
    normed, sf = preprocessing.deseq2_norm(counts)
    np.testing.assert_allclose(np.asarray(sf), example["ratio_size_factors"], rtol=1e-14)
    np.testing.assert_allclose(
        np.asarray(normed),
        example["counts"].to_numpy() / example["ratio_size_factors"][:, None],
        rtol=1e-14,
    )
    assert isinstance(normed, pd.DataFrame) == as_frame


def test_deseq2_norm_matches_reference_simulated():
    sim, counts = _ratio_dataset()
    normed, sf = preprocessing.deseq2_norm(counts)
    expected = helpers.median_of_ratios(counts)
    np.testing.assert_allclose(np.asarray(sf), expected, rtol=EXACT_RTOL)
    np.testing.assert_allclose(
        np.asarray(normed), counts.to_numpy() / expected[:, None], rtol=EXACT_RTOL
    )


# misc


@pytest.mark.parametrize(
    "counts",
    [
        pd.DataFrame([[0, 3], [5, 10]]),
        np.array([[0.0, 3.0], [5.0, 10.0]]),
        np.zeros((2, 2), dtype=np.int64),
    ],
    ids=["int_frame", "integral_float_array", "zeros"],
)
def test_valid_counts_accepts_non_negative_integers(counts):
    """Accepted counts return None, and a dataset built from them holds them unchanged,
    as integers (integral floats included)."""
    assert misc.test_valid_counts(counts) is None

    samples = ["s1", "s2"]
    metadata = pd.DataFrame({"condition": ["A", "B"]}, index=samples)
    frame = pd.DataFrame(np.asarray(counts), index=samples, columns=["g1", "g2"])
    dds = helpers.make_dds(frame, metadata)
    assert np.issubdtype(dds.X.dtype, np.integer)
    np.testing.assert_array_equal(dds.X, np.asarray(counts))


@pytest.mark.parametrize(
    "counts, message",
    [
        (pd.DataFrame([[0.0, np.nan], [5.0, 10.0]]), "NaNs are not allowed"),
        (np.array([[0.0, np.nan], [5.0, 10.0]]), "NaNs are not allowed"),
        (pd.DataFrame([[0, "a"], [5, 10]]), "only contain numbers"),
        (pd.DataFrame([[0.0, 1.5], [5.0, 10.0]]), "only contain integers"),
        (np.array([[0.0, 1.5], [5.0, 10.0]]), "only contain integers"),
        (pd.DataFrame([[0, -1], [5, 10]]), "non-negative"),
        (np.array([[0, -1], [5, 10]]), "non-negative"),
    ],
    ids=[
        "nan_frame",
        "nan_array",
        "string_frame",
        "fraction_frame",
        "fraction_array",
        "negative_frame",
        "negative_array",
    ],
)
def test_valid_counts_rejects_invalid_counts(counts, message):
    """Rejected by ``test_valid_counts`` and by both ways of building a dataset."""
    with pytest.raises(ValueError, match=message):
        misc.test_valid_counts(counts)

    samples = ["s1", "s2"]
    metadata = pd.DataFrame({"condition": ["A", "B"]}, index=samples)
    frame = pd.DataFrame(np.asarray(counts), index=samples, columns=["g1", "g2"])
    with pytest.raises(ValueError, match=message):
        helpers.make_dds(frame, metadata)
    if frame.dtypes.map(pd.api.types.is_numeric_dtype).all():
        adata = ad.AnnData(X=frame.to_numpy(dtype=float), obs=metadata)
        with pytest.raises(ValueError, match=message):
            helpers.make_dds(None, None, adata=adata)


def test_n_or_more_replicates_hand_worked():
    """Groups by the rows of the design matrix: (A, X) has 3 samples, (A, Y) 1 and
    (B, X) 3."""
    samples = [f"s{i}" for i in range(1, 8)]
    condition_b = [0, 0, 0, 1, 1, 1, 0]
    group_y = [0, 0, 1, 0, 0, 0, 0]
    design = pd.DataFrame(
        {"Intercept": 1.0, "condition[T.B]": condition_b, "group[T.Y]": group_y},
        index=samples,
    )
    for min_replicates, expected in [
        (1, [True] * 7),
        (2, [True, True, False, True, True, True, True]),
        (3, [True, True, False, True, True, True, True]),
        (4, [False] * 7),
    ]:
        result = misc.n_or_more_replicates(design, min_replicates)
        assert list(result.index) == samples
        np.testing.assert_array_equal(result.to_numpy(dtype=bool), expected)


def test_get_num_processes():
    """An integer is used as given; None means every CPU the machine reports."""
    assert misc.get_num_processes(3) == 3
    assert misc.get_num_processes(1) == 1
    n = misc.get_num_processes(None)
    assert isinstance(n, int) and n >= 1
    assert n == multiprocessing.cpu_count()
