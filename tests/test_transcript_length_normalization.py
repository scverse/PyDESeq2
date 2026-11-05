import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from pydeseq2.datasets import load_example_data
from pydeseq2.dds import DeseqDataSet
from pydeseq2.dispersions import fit_moments_dispersions
from pydeseq2.ds import DeseqStats


def small_tximport_data():
    samples = pd.Index([f"sample{i}" for i in range(1, 7)])
    genes = pd.Index(["gene1", "gene2", "gene3"])
    counts = pd.DataFrame(
        [
            [100.2, 201.7, 302.1],
            [121.8, 181.2, 330.9],
            [89.6, 240.4, 281.2],
            [170.1, 260.8, 410.3],
            [160.7, 290.2, 389.9],
            [190.4, 250.6, 430.8],
        ],
        index=samples,
        columns=genes,
    )
    metadata = pd.DataFrame(
        {"condition": ["A", "A", "A", "B", "B", "B"]},
        index=samples,
    )
    transcript_lengths = pd.DataFrame(
        [
            [900.0, 1400.0, 2100.0],
            [950.0, 1350.0, 2050.0],
            [1000.0, 1300.0, 2000.0],
            [1250.0, 1100.0, 1850.0],
            [1300.0, 1050.0, 1800.0],
            [1350.0, 1000.0, 1750.0],
        ],
        index=samples,
        columns=genes,
    )
    return counts, metadata, transcript_lengths


def small_pytximport_adata():
    counts, metadata, transcript_lengths = small_tximport_data()
    adata = ad.AnnData(
        X=counts.to_numpy(),
        obs=metadata.copy(),
        var=pd.DataFrame(index=counts.columns),
    )
    adata.obsm["length"] = transcript_lengths.to_numpy()
    adata.uns["counts_from_abundance"] = None
    return adata, counts, metadata, transcript_lengths


def varying_transcript_lengths(counts):
    sample_effect = np.linspace(-1.0, 1.0, counts.shape[0])
    gene_effect = np.linspace(-1.0, 1.0, counts.shape[1])
    lengths = (800.0 + 100.0 * np.arange(counts.shape[1]))[None, :]
    lengths = lengths * (1.0 + 0.2 * np.outer(sample_effect, gene_effect))
    return pd.DataFrame(lengths, index=counts.index, columns=counts.columns)


@pytest.mark.parametrize("method", ["ratio", "poscounts"])
def test_transcript_length_normalization_matches_deseq2_reference(method):
    counts = load_example_data("raw_counts", "synthetic", debug=False)
    metadata = load_example_data("metadata", "synthetic", debug=False)
    dds = DeseqDataSet(
        counts=counts,
        metadata=metadata,
        transcript_lengths=varying_transcript_lengths(counts),
        quiet=True,
    )
    dds.fit_size_factors(method)

    directory = Path(__file__).parent / "data" / "transcript_length"
    for suffix, layer in (
        ("factors", "normalization_factors"),
        ("counts", "normed_counts"),
    ):
        expected = pd.read_csv(directory / f"r_{method}_{suffix}.csv", index_col=0).T
        pd.testing.assert_index_equal(expected.index, dds.obs_names, check_names=False)
        pd.testing.assert_index_equal(expected.columns, dds.var_names, check_names=False)
        np.testing.assert_allclose(
            dds.layers[layer], expected.to_numpy(), rtol=1e-8, atol=1e-8
        )


@pytest.mark.parametrize("bad_value", [0.0, np.nan, np.inf])
def test_transcript_lengths_must_be_positive_and_finite(bad_value):
    counts, metadata, transcript_lengths = small_tximport_data()
    transcript_lengths.iloc[0, 0] = bad_value
    with pytest.raises(ValueError, match="transcript_lengths"):
        DeseqDataSet(
            counts=counts, metadata=metadata, transcript_lengths=transcript_lengths
        )


@pytest.mark.parametrize(
    ("zero_rows", "message"),
    [
        (slice(None), "No genes are available"),
        (0, "At least one sample has no positive counts"),
    ],
)
def test_transcript_length_normalization_rejects_unusable_counts(zero_rows, message):
    counts, metadata, transcript_lengths = small_tximport_data()
    counts.iloc[zero_rows, :] = 0
    dds = DeseqDataSet(
        counts=counts,
        metadata=metadata,
        transcript_lengths=transcript_lengths,
        quiet=True,
    )

    with pytest.raises(ValueError, match=message):
        dds.fit_size_factors("poscounts")


@pytest.mark.parametrize(
    "axis,message,use_anndata",
    [(0, "same sample index", False), (1, "same gene columns", True)],
)
def test_transcript_length_labels_must_match_counts(axis, message, use_anndata):
    adata, counts, metadata, lengths = small_pytximport_adata()
    lengths = lengths.rename(lambda label: f"wrong_{label}", axis=axis)
    if use_anndata:
        adata.obsm["length"] = lengths
        inputs = {"adata": adata}
    else:
        inputs = {"counts": counts, "metadata": metadata}
    with pytest.raises(ValueError, match=message):
        DeseqDataSet(**inputs, transcript_lengths=lengths)


def test_transcript_lengths_reject_iterative_size_factors():
    counts, metadata, transcript_lengths = small_tximport_data()
    dds = DeseqDataSet(
        counts=counts,
        metadata=metadata,
        transcript_lengths=transcript_lengths,
    )

    with pytest.raises(ValueError, match="iterative"):
        dds.fit_size_factors("iterative")


def test_external_vst_transform_requires_matching_transcript_lengths():
    counts, metadata, transcript_lengths = small_tximport_data()
    dds = DeseqDataSet(
        counts=counts,
        metadata=metadata,
        transcript_lengths=transcript_lengths,
    )
    dds.fit_size_factors()

    with pytest.raises(ValueError, match="matching transcript lengths"):
        dds.vst_transform(dds.X)


@pytest.mark.parametrize("refit_cooks", [False, True])
def test_transcript_length_factors_are_used_throughout_pipeline(refit_cooks):
    counts = load_example_data("raw_counts", "synthetic", debug=False)
    metadata = load_example_data("metadata", "synthetic", debug=False)
    if refit_cooks:
        counts.iloc[0, 0] *= 10_000
    lengths = varying_transcript_lengths(counts)
    adata = ad.AnnData(
        X=counts.to_numpy(), obs=metadata, var=pd.DataFrame(index=counts.columns)
    )
    adata.obsm["length"] = lengths.to_numpy()
    adata.uns["counts_from_abundance"] = None
    settings = {
        "design": "~0 + condition",
        "fit_type": "mean",
        "refit_cooks": refit_cooks,
        "min_replicates": 3 if refit_cooks else 7,
        "n_cpus": 1,
        "quiet": True,
    }
    dds = DeseqDataSet(adata=adata, transcript_lengths=adata.obsm["length"], **settings)
    reference_dds = DeseqDataSet(
        counts=counts, metadata=metadata, transcript_lengths=lengths, **settings
    )
    dds.deseq2()
    reference_dds.deseq2()

    np.testing.assert_allclose(
        dds.layers["normalization_factors"],
        reference_dds.layers["normalization_factors"],
    )
    dispersion_columns = ["_MoM_dispersions", "genewise_dispersions", "dispersions"]
    np.testing.assert_allclose(
        dds.var[dispersion_columns].to_numpy(),
        reference_dds.var[dispersion_columns].to_numpy(),
    )
    np.testing.assert_allclose(dds.varm["LFC"], reference_dds.varm["LFC"])

    if refit_cooks:
        assert dds.var["replaced"].sum() == 1
        assert dds.var["refitted"].sum() == 1
        np.testing.assert_array_equal(
            dds.var[["replaced", "refitted"]],
            reference_dds.var[["replaced", "refitted"]],
        )
        np.testing.assert_allclose(
            dds.layers["cooks"],
            reference_dds.layers["cooks"],
            rtol=1e-7,
            atol=1e-10,
            equal_nan=True,
        )
        np.testing.assert_array_equal(
            dds.counts_to_refit.X, reference_dds.counts_to_refit.X
        )
        refitted = dds.var["refitted"].to_numpy()
        np.testing.assert_allclose(
            dds.counts_to_refit.layers["normalization_factors"],
            dds.layers["normalization_factors"][:, refitted],
        )
    else:
        non_zero = dds.var["non_zero"].to_numpy()
        normalization_factors = dds.layers["normalization_factors"][:, non_zero]
        design_matrix = dds.obsm["design_matrix"].to_numpy()
        coefficients = dds.varm["LFC"].loc[dds.var_names[non_zero]].to_numpy()
        expected_mu = np.maximum(
            normalization_factors * np.exp(design_matrix @ coefficients.T),
            dds.min_mu,
        )
        np.testing.assert_allclose(dds.obsm["_mu_LFC"], expected_mu)

        stats = DeseqStats(dds, contrast=["condition", "B", "A"], quiet=True)
        reference_stats = DeseqStats(
            reference_dds, contrast=["condition", "B", "A"], quiet=True
        )
        stats.summary()
        reference_stats.summary()
        assert stats.results_df["pvalue"].notna().any()
        np.testing.assert_allclose(
            stats.results_df[["stat", "pvalue", "lfcSE"]],
            reference_stats.results_df[["stat", "pvalue", "lfcSE"]],
            equal_nan=True,
        )

        coefficient = stats.LFC.columns[-1]
        stats.lfc_shrink(coeff=coefficient, adapt=False)
        reference_stats.lfc_shrink(coeff=coefficient, adapt=False)
        assert stats.shrunk_LFCs
        np.testing.assert_allclose(
            stats.LFC[coefficient], reference_stats.LFC[coefficient], equal_nan=True
        )
        dds.vst(fit_type="mean")
        reference_dds.vst(fit_type="mean")
        np.testing.assert_allclose(
            dds.layers["vst_counts"], reference_dds.layers["vst_counts"]
        )


def test_moments_dispersions_with_matrix_factors():
    normed_counts = np.array([[10.0, 20.0], [14.0, 28.0], [18.0, 36.0]])
    normalization_factors = np.array([[1.0, 3.0], [2.0, 6.0], [4.0, 12.0]])
    # Gene means: 14, 28; sample variances: 16, 64.
    # Row-mean factors: 2, 4, 8; mean inverse factor: 7/24.
    np.testing.assert_allclose(
        fit_moments_dispersions(normed_counts, normalization_factors),
        [143 / 2352, 335 / 4704],
    )


def test_moments_dispersions_align_matrix_factors_after_zero_gene_filtering():
    normed_counts = np.array(
        [
            [0.0, 10.0],
            [0.0, 14.0],
            [0.0, 18.0],
        ]
    )
    normalization_factors = np.array(
        [
            [100.0, 1.0],
            [100.0, 1.0],
            [100.0, 1.0],
        ]
    )

    np.testing.assert_allclose(
        fit_moments_dispersions(normed_counts, normalization_factors),
        np.array([2.0 / 196.0]),
    )


def test_matching_integer_transcript_length_labels_are_accepted():
    counts, metadata, transcript_lengths = small_tximport_data()
    integer_samples = pd.Index(range(1, len(counts) + 1))
    integer_genes = pd.Index(range(101, 101 + counts.shape[1]))
    counts.index = integer_samples
    counts.columns = integer_genes
    metadata.index = integer_samples
    transcript_lengths.index = integer_samples
    transcript_lengths.columns = integer_genes

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ad.ImplicitModificationWarning)
        dds = DeseqDataSet(
            counts=counts,
            metadata=metadata,
            transcript_lengths=transcript_lengths,
            quiet=True,
        )

    np.testing.assert_allclose(
        dds.layers["avg_tx_length"], transcript_lengths.to_numpy()
    )


def test_scalar_size_factor_refit_clears_stale_matrix_factors():
    counts, metadata, _ = small_tximport_data()
    counts = counts.round().astype(int)
    dds = DeseqDataSet(counts=counts, metadata=metadata, quiet=True)
    dds.layers["normalization_factors"] = np.full(dds.shape, 2.0)

    dds.fit_size_factors()

    assert "normalization_factors" not in dds.layers
    np.testing.assert_allclose(
        dds.layers["normed_counts"],
        dds.X / dds.obs["size_factors"].to_numpy()[:, None],
    )


@pytest.mark.parametrize(
    "source_lengths,opt_in", [(False, True), (True, False), (True, True)]
)
@pytest.mark.parametrize("serialized", [False, True])
def test_inherited_length_normalization_starts_a_fresh_fit(
    source_lengths, opt_in, serialized
):
    counts = load_example_data("raw_counts", "synthetic", debug=False)
    metadata = load_example_data("metadata", "synthetic", debug=False)
    lengths = varying_transcript_lengths(counts)
    settings = {"fit_type": "mean", "refit_cooks": False, "n_cpus": 1, "quiet": True}
    source = DeseqDataSet(
        counts=counts,
        metadata=metadata,
        transcript_lengths=lengths if source_lengths else None,
        **settings,
    )
    source.deseq2()
    original_counts = source.layers["normed_counts"].copy()
    original_lfc = source.varm["LFC"].copy()
    adata = source.to_picklable_anndata() if serialized else source
    # Different lengths must replace the source normalization, not reuse its fit.
    replacement = lengths.copy()
    replacement.iloc[:, 0] *= np.linspace(1.0, 2.0, len(lengths))
    kwargs = dict(transcript_lengths=replacement if opt_in else None, **settings)
    dds = DeseqDataSet(adata=adata, **kwargs)
    reference = DeseqDataSet(counts=counts, metadata=metadata, **kwargs)
    assert "size_factors" not in dds.obs
    assert "LFC" not in dds.varm
    # These public methods would reuse stale fitted state if it survived construction.
    for dataset in (dds, reference):
        dataset.fit_dispersion_trend()
        dataset.fit_LFC()
        dataset.vst(fit_type="mean")
    np.testing.assert_allclose(
        dds.layers["normed_counts"], reference.layers["normed_counts"]
    )
    np.testing.assert_allclose(dds.var["dispersions"], reference.var["dispersions"])
    np.testing.assert_allclose(dds.varm["LFC"], reference.varm["LFC"])
    np.testing.assert_allclose(dds.layers["vst_counts"], reference.layers["vst_counts"])
    if opt_in:
        np.testing.assert_allclose(
            dds.layers["normalization_factors"],
            reference.layers["normalization_factors"],
        )
    else:
        assert "normalization_factors" not in dds.layers
        np.testing.assert_allclose(
            dds.vst_transform(dds.X), reference.vst_transform(reference.X)
        )
    np.testing.assert_array_equal(source.layers["normed_counts"], original_counts)
    pd.testing.assert_frame_equal(source.varm["LFC"], original_lfc)
    if source_lengths:
        np.testing.assert_array_equal(source.layers["avg_tx_length"], lengths.to_numpy())


def test_pytximport_dataframe_length_labels_are_preserved():
    adata, _, _, transcript_lengths = small_pytximport_adata()
    adata.obsm["length"] = transcript_lengths.copy()

    dds = DeseqDataSet(adata=adata, transcript_lengths=adata.obsm["length"], quiet=True)

    assert dds.obs_names.equals(transcript_lengths.index)
    assert dds.var_names.equals(transcript_lengths.columns)
    np.testing.assert_allclose(
        dds.layers["avg_tx_length"], transcript_lengths.to_numpy()
    )


@pytest.mark.parametrize("mode", ["scaled_tpm", "length_scaled_tpm"])
@pytest.mark.parametrize("stored_lengths", [False, True])
def test_scaled_pytximport_rejects_explicit_length_correction(mode, stored_lengths):
    adata, _, _, lengths = small_pytximport_adata()
    adata.uns["counts_from_abundance"] = mode
    del adata.obsm["length"]
    if stored_lengths:
        adata.layers["avg_tx_length"] = lengths.to_numpy()
    with pytest.raises(ValueError, match="must not be combined"):
        DeseqDataSet(adata=adata, transcript_lengths=lengths, quiet=True)


def test_pytximport_lengths_reject_unsynchronized_gene_subsetting():
    adata, _, _, _ = small_pytximport_adata()
    subset = adata[:, :2].copy()

    with pytest.raises(ValueError, match="same shape"):
        DeseqDataSet(adata=subset, transcript_lengths=subset.obsm["length"], quiet=True)


def test_explicit_lengths_override_unused_stored_lengths():
    adata, _, _, lengths = small_pytximport_adata()
    adata.layers["avg_tx_length"] = np.full(adata.shape, np.nan)
    adata.obsm["length"] = np.zeros(adata.shape)
    dds = DeseqDataSet(adata=adata, transcript_lengths=lengths, quiet=True)
    np.testing.assert_array_equal(dds.layers["avg_tx_length"], lengths.to_numpy())
    assert np.isnan(adata.layers["avg_tx_length"]).all()


@pytest.mark.parametrize("mode", [None, "scaled_tpm", "length_scaled_tpm", "missing"])
@pytest.mark.parametrize(
    "stored_layer", [None, "avg_tx_length", "normalization_factors"]
)
def test_stored_metadata_does_not_enable_length_normalization(
    mode, stored_layer, recwarn
):
    adata, counts, metadata, _ = small_pytximport_adata()
    adata.uns["counts_from_abundance"] = mode
    if mode == "missing":
        del adata.uns["counts_from_abundance"]
    # Unused lengths must neither be validated nor cause estimated counts to be rounded.
    adata.obsm["length"] = np.full(adata.shape, np.nan)
    if stored_layer is not None:
        del adata.obsm["length"]
        adata.layers[stored_layer] = np.full(adata.shape, np.nan)
    with pytest.raises(ValueError, match="only contain integers"):
        DeseqDataSet(adata=adata, quiet=True)
    adata.X = counts.round().to_numpy().astype(int)
    dds = DeseqDataSet(adata=adata, quiet=True)
    reference = DeseqDataSet(
        counts=counts.round().astype(int), metadata=metadata, quiet=True
    )
    dds.fit_size_factors()
    reference.fit_size_factors()
    assert "avg_tx_length" not in dds.layers
    assert "normalization_factors" not in dds.layers
    np.testing.assert_allclose(dds.obs["size_factors"], reference.obs["size_factors"])
    np.testing.assert_allclose(
        dds.layers["normed_counts"], reference.layers["normed_counts"]
    )
    if stored_layer is not None:
        assert np.isnan(adata.layers[stored_layer]).all()
    assert sum("Unscaled pytximport" in str(w.message) for w in recwarn) == (
        2 if mode is None else 0
    )


def test_pytximport_input_anndata_is_not_mutated(recwarn):
    adata, counts, _, transcript_lengths = small_pytximport_adata()
    original_obs = adata.obs.copy()
    connectivities = np.eye(adata.n_obs)
    adata.uns["neighbors"] = {"connectivities": connectivities}

    dds = DeseqDataSet(adata=adata, transcript_lengths=adata.obsm["length"], quiet=True)
    dds.fit_size_factors()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        picklable_adata = dds.to_picklable_anndata()

    np.testing.assert_array_equal(dds.X, np.rint(counts.to_numpy()))
    np.testing.assert_array_equal(adata.X, counts.to_numpy())
    pd.testing.assert_frame_equal(adata.obs, original_obs)
    assert dds.var is not adata.var
    assert adata.uns["neighbors"]["connectivities"] is connectivities
    assert "connectivities" in picklable_adata.obsp
    np.testing.assert_array_equal(adata.obsm["length"], transcript_lengths.to_numpy())
    assert "design_matrix" not in adata.obsm
    assert not adata.layers
    assert not any("Unscaled pytximport" in str(w.message) for w in recwarn)
