"""GPU concordance tests for PyDESeq2.

Mirrors the structure of test_pydeseq2.py but runs all pipelines with
``inference_type="gpu"``, validating against the same R DESeq2 reference
outputs. Requires CUDA to be available.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import tests
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from pydeseq2.utils import load_example_data

# Skip entire module if CUDA is not available
torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

GPU_KWARGS = {"inference_type": "gpu"}


# ---- Fixtures ----


@pytest.fixture
def counts_df():
    return load_example_data(
        modality="raw_counts",
        dataset="synthetic",
        debug=False,
    )


@pytest.fixture
def metadata():
    return load_example_data(
        modality="metadata",
        dataset="synthetic",
        debug=False,
    )


def _test_path():
    return str(Path(os.path.realpath(tests.__file__)).parent.resolve())


def assert_res_almost_equal(py_res, r_res, tol=0.02):
    """Assert that PyDESeq2 results match R DESeq2 results.

    For p-values near machine epsilon, GPU torch.special.ndtr may
    underflow to exactly 0 where scipy gives ~1e-16. We skip those
    from the relative error check and instead verify they are both
    extremely small.
    """
    assert (py_res.pvalue.isna() == r_res.pvalue.isna()).all()
    assert (py_res.padj.isna() == r_res.padj.isna()).all()

    assert (
        abs(r_res.log2FoldChange - py_res.log2FoldChange) / abs(r_res.log2FoldChange)
    ).max() < tol

    # For p-values, skip genes where both values are < 1e-14
    # (underflow region for ndtr vs sf)
    pval_mask = ~(
        r_res.pvalue.isna() | ((r_res.pvalue < 1e-14) & (py_res.pvalue < 1e-14))
    )
    if pval_mask.any():
        assert (
            abs(r_res.pvalue[pval_mask] - py_res.pvalue[pval_mask])
            / r_res.pvalue[pval_mask]
        ).max() < tol

    padj_mask = ~(r_res.padj.isna() | ((r_res.padj < 1e-14) & (py_res.padj < 1e-14)))
    if padj_mask.any():
        assert (
            abs(r_res.padj[padj_mask] - py_res.padj[padj_mask]) / r_res.padj[padj_mask]
        ).max() < tol


# ---- Single-factor pipeline tests ----


def test_gpu_deseq_parametric_fit(counts_df, metadata, tol=0.02):
    """GPU pipeline with parametric fit matches R reference."""
    r_res = pd.read_csv(
        os.path.join(_test_path(), "data/single_factor/r_test_res.csv"),
        index_col=0,
    )

    dds = DeseqDataSet(
        counts=counts_df,
        metadata=metadata,
        design="~condition",
        fit_type="parametric",
        **GPU_KWARGS,
    )
    dds.deseq2()

    ds = DeseqStats(dds, contrast=["condition", "B", "A"])
    ds.summary()

    assert_res_almost_equal(ds.results_df, r_res, tol)


def test_gpu_deseq_mean_fit(counts_df, metadata, tol=0.02):
    """GPU pipeline with mean fit matches R reference."""
    r_res = pd.read_csv(
        os.path.join(
            _test_path(),
            "data/single_factor/r_test_res_mean_curve.csv",
        ),
        index_col=0,
    )

    dds = DeseqDataSet(
        counts=counts_df,
        metadata=metadata,
        design="~condition",
        fit_type="mean",
        **GPU_KWARGS,
    )
    dds.deseq2()

    ds = DeseqStats(dds, contrast=["condition", "B", "A"])
    ds.summary()

    assert_res_almost_equal(ds.results_df, r_res, tol)


def test_gpu_no_independent_filtering(counts_df, metadata, tol=0.02):
    """GPU pipeline without independent filtering matches R."""
    r_res = pd.read_csv(
        os.path.join(
            _test_path(),
            "data/single_factor/r_test_res_no_independent_filtering.csv",
        ),
        index_col=0,
    )

    dds = DeseqDataSet(
        counts=counts_df,
        metadata=metadata,
        design="~condition",
        fit_type="parametric",
        **GPU_KWARGS,
    )
    dds.deseq2()

    ds = DeseqStats(
        dds,
        contrast=["condition", "B", "A"],
        independent_filter=False,
    )
    ds.summary()

    assert_res_almost_equal(ds.results_df, r_res, tol)


@pytest.mark.parametrize(
    "alt_hypothesis",
    ["lessAbs", "greaterAbs", "less", "greater"],
)
def test_gpu_alt_hypothesis(alt_hypothesis, counts_df, metadata, tol=0.02):
    """GPU pipeline with alternative hypotheses matches R."""
    r_res = pd.read_csv(
        os.path.join(
            _test_path(),
            f"data/single_factor/r_test_res_{alt_hypothesis}.csv",
        ),
        index_col=0,
    )

    dds = DeseqDataSet(
        counts=counts_df,
        metadata=metadata,
        design="~condition",
        **GPU_KWARGS,
    )
    dds.deseq2()

    ds = DeseqStats(
        dds,
        contrast=["condition", "B", "A"],
        alt_hypothesis=alt_hypothesis,
        lfc_null=-0.5 if alt_hypothesis == "less" else 0.5,
    )
    ds.summary()

    res = ds.results_df

    # Same NaN pattern
    assert (res.pvalue.isna() == r_res.pvalue.isna()).all()
    assert (res.padj.isna() == r_res.padj.isna()).all()

    # LFC matches
    assert (
        abs(r_res.log2FoldChange - res.log2FoldChange) / abs(r_res.log2FoldChange)
    ).max() < tol

    # Stat matches (abs for lessAbs, as in upstream test)
    if alt_hypothesis == "lessAbs":
        res.stat = res.stat.abs()
    assert (abs(r_res.stat - res.stat) / abs(r_res.stat)).max() < tol

    # P-values match only where stat != 0
    assert (
        abs(r_res.pvalue[r_res.stat != 0] - res.pvalue[res.stat != 0])
        / r_res.pvalue[r_res.stat != 0]
    ).max() < tol


def test_gpu_no_refit_cooks(counts_df, metadata, tol=0.02):
    """GPU pipeline without Cook's refit matches R dispersions."""
    r_dispersions = pd.read_csv(
        os.path.join(
            _test_path(),
            "data/single_factor/r_test_dispersions.csv",
        ),
        index_col=0,
    ).squeeze()

    dds = DeseqDataSet(
        counts=counts_df,
        metadata=metadata,
        design="~condition",
        refit_cooks=False,
        **GPU_KWARGS,
    )
    dds.deseq2()

    np.testing.assert_array_almost_equal(
        dds.var["dispersions"],
        r_dispersions,
        decimal=1,
    )


# ---- LFC Shrinkage tests ----


def test_gpu_lfc_shrinkage(counts_df, metadata, tol=0.02):
    """GPU LFC shrinkage matches R reference."""
    r_res = pd.read_csv(
        os.path.join(_test_path(), "data/single_factor/r_test_res.csv"),
        index_col=0,
    )
    r_shrunk_res = pd.read_csv(
        os.path.join(
            _test_path(),
            "data/single_factor/r_test_lfc_shrink_res.csv",
        ),
        index_col=0,
    )
    r_size_factors = pd.read_csv(
        os.path.join(
            _test_path(),
            "data/single_factor/r_test_size_factors.csv",
        ),
        index_col=0,
    )["x"].values
    r_dispersions = pd.read_csv(
        os.path.join(
            _test_path(),
            "data/single_factor/r_test_dispersions.csv",
        ),
        index_col=0,
    ).squeeze()

    dds = DeseqDataSet(
        counts=counts_df,
        metadata=metadata,
        design="~condition",
        **GPU_KWARGS,
    )
    dds.deseq2()

    # Override with R values for controlled shrinkage test
    dds.obs["size_factors"] = r_size_factors
    dds.var["dispersions"] = r_dispersions.values
    dds.varm["LFC"].iloc[:, 1] = r_res.log2FoldChange.values * np.log(2)

    res = DeseqStats(dds, contrast=["condition", "B", "A"])
    res.summary()
    res.SE = r_res.lfcSE * np.log(2)
    res.lfc_shrink(coeff="condition[T.B]")
    shrunk_res = res.results_df

    assert (
        abs(r_shrunk_res.log2FoldChange - shrunk_res.log2FoldChange)
        / abs(r_shrunk_res.log2FoldChange)
    ).max() < tol


# ---- Multi-factor tests ----


@pytest.mark.parametrize("with_outliers", [True, False])
def test_gpu_multifactor_deseq(counts_df, metadata, with_outliers, tol=0.04):
    """GPU multi-factor pipeline matches R reference."""
    if with_outliers:
        r_res = pd.read_csv(
            os.path.join(
                _test_path(),
                "data/multi_factor/r_test_res_outliers.csv",
            ),
            index_col=0,
        )
    else:
        r_res = pd.read_csv(
            os.path.join(
                _test_path(),
                "data/multi_factor/r_test_res.csv",
            ),
            index_col=0,
        )

    if with_outliers:
        counts_df.loc["sample1", "gene1"] = 2000
        counts_df.loc["sample11", "gene7"] = 1000
        metadata.loc["sample1", "condition"] = "C"

    dds = DeseqDataSet(
        counts=counts_df,
        metadata=metadata,
        design="~group + condition",
        **GPU_KWARGS,
    )
    dds.deseq2()

    res = DeseqStats(dds, contrast=["condition", "B", "A"])
    res.summary()

    assert_res_almost_equal(res.results_df, r_res, tol)


# ---- Continuous factor tests ----


@pytest.mark.parametrize("with_outliers", [True, False])
def test_gpu_continuous_deseq(with_outliers, tol=0.04):
    """GPU continuous-factor pipeline matches R reference."""
    counts_df = pd.read_csv(
        os.path.join(_test_path(), "data/continuous/test_counts.csv"),
        index_col=0,
    ).T

    metadata = pd.read_csv(
        os.path.join(_test_path(), "data/continuous/test_metadata.csv"),
        index_col=0,
    )

    if with_outliers:
        r_res = pd.read_csv(
            os.path.join(
                _test_path(),
                "data/continuous/r_test_res_outliers.csv",
            ),
            index_col=0,
        )
        counts_df.loc["sample1", "gene1"] = 2000
        counts_df.loc["sample11", "gene7"] = 1000
        metadata.loc["sample1", "condition"] = "C"
    else:
        r_res = pd.read_csv(
            os.path.join(
                _test_path(),
                "data/continuous/r_test_res.csv",
            ),
            index_col=0,
        )

    dds = DeseqDataSet(
        counts=counts_df,
        metadata=metadata,
        design="~group + condition + measurement",
        **GPU_KWARGS,
    )
    dds.deseq2()

    contrast_vector = np.zeros(dds.obsm["design_matrix"].shape[1])
    contrast_vector[-1] = 1

    ds = DeseqStats(dds, contrast=contrast_vector)
    ds.summary()

    assert_res_almost_equal(ds.results_df, r_res, tol)


# ---- Wide data test ----


def test_gpu_wide_deseq(tol=0.02):
    """GPU wide dataset (more genes than samples) matches R."""
    r_res = pd.read_csv(
        os.path.join(_test_path(), "data/wide/r_test_res.csv"),
        index_col=0,
    )

    counts_df = pd.read_csv(
        os.path.join(_test_path(), "data/wide/test_counts.csv"),
        index_col=0,
    ).T

    metadata = pd.read_csv(
        os.path.join(_test_path(), "data/wide/test_metadata.csv"),
        index_col=0,
    )

    dds = DeseqDataSet(
        counts=counts_df,
        metadata=metadata,
        design="~group + condition",
        **GPU_KWARGS,
    )
    dds.deseq2()

    ds = DeseqStats(dds, contrast=["condition", "B", "A"])
    ds.summary()

    assert_res_almost_equal(ds.results_df, r_res, tol)


# ---- VST test ----


def test_gpu_vst(counts_df, metadata, tol=0.02):
    """GPU variance stabilizing transformation produces valid results."""
    dds = DeseqDataSet(
        counts=counts_df,
        metadata=metadata,
        design="~condition",
        **GPU_KWARGS,
    )
    dds.vst()

    vst_counts = dds.layers["vst_counts"]
    assert not np.isnan(vst_counts).any()
    assert vst_counts.shape == counts_df.shape


# ---- Inference inheritance test ----


def test_gpu_inference_inherited_by_stats(counts_df, metadata):
    """DeseqStats inherits GPU inference from DeseqDataSet."""
    dds = DeseqDataSet(
        counts=counts_df,
        metadata=metadata,
        design="~condition",
        **GPU_KWARGS,
    )
    dds.deseq2()

    ds = DeseqStats(dds, contrast=["condition", "B", "A"])

    from pydeseq2.torch_inference import TorchInference

    assert isinstance(ds.inference, TorchInference)
