"""Compare PyDESeq2 with DESeq2, which is run through rpy2.

These tests require R and DESeq2, and are skipped if rpy2 is not installed.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import tests
from pydeseq2.datasets import load_example_data
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from tests.test_pydeseq2 import assert_res_almost_equal

ro = pytest.importorskip("rpy2.robjects")
r_deseq2 = pytest.importorskip("rpy2.robjects.packages").importr("DESeq2")


def load_data(dataset):
    """Load the counts and metadata of the synthetic or of the continuous dataset."""
    if dataset == "synthetic":
        return (
            load_example_data(modality="raw_counts", dataset="synthetic", debug=False),
            load_example_data(modality="metadata", dataset="synthetic", debug=False),
        )
    data_path = Path(tests.__file__).parent / "data" / dataset
    return (
        pd.read_csv(data_path / "test_counts.csv", index_col=0).T,
        pd.read_csv(data_path / "test_metadata.csv", index_col=0),
    )


def run_pydeseq2(counts_df, metadata, design, contrast, dds_kwargs, ds_kwargs):
    """Run PyDESeq2 and return its results.

    The contrast is either a list of a factor and two of its levels, or the name of a coefficient.
    """
    dds = DeseqDataSet(counts=counts_df, metadata=metadata, design=design, **dds_kwargs)
    dds.deseq2()
    if isinstance(contrast, str):
        contrast = (dds.obsm["design_matrix"].columns == contrast).astype(float)
    ds = DeseqStats(dds, contrast=contrast, **ds_kwargs)
    ds.summary()
    return ds.results_df


def run_r_deseq2(counts_df, metadata, design, contrast, deseq_kwargs, results_kwargs):
    """Run DESeq2 and return its results.

    The contrast is either a list of a factor and two of its levels, or the name of a coefficient.
    """
    col_data = ro.DataFrame(
        {
            name: ro.FloatVector(col) if pd.api.types.is_numeric_dtype(col) else ro.FactorVector(ro.StrVector(col))
            for name, col in metadata.items()
        }
    )
    col_data.rownames = ro.StrVector(metadata.index)
    counts = ro.r["matrix"](
        ro.IntVector(counts_df.to_numpy().ravel()),
        nrow=counts_df.shape[1],
        dimnames=ro.r["list"](ro.StrVector(counts_df.columns), ro.StrVector(counts_df.index)),
    )
    dds = r_deseq2.DESeqDataSetFromMatrix(countData=counts, colData=col_data, design=ro.Formula(design))
    dds = r_deseq2.DESeq(dds, quiet=True, **deseq_kwargs)
    if isinstance(contrast, str):
        res = r_deseq2.results(dds, name=contrast, **results_kwargs)
    else:
        res = r_deseq2.results(dds, contrast=ro.StrVector(contrast), **results_kwargs)
    res = ro.r["as.data.frame"](res)
    return pd.DataFrame({name: np.asarray(col) for name, col in res.items()}, index=list(res.rownames))


@pytest.mark.parametrize(
    ("dataset", "design", "contrast"),
    [
        ("synthetic", "~condition", ["condition", "B", "A"]),
        ("synthetic", "~group + condition", ["condition", "B", "A"]),
        ("synthetic", "~group + condition", ["group", "Y", "X"]),
        ("synthetic", "~group * condition", ["condition", "B", "A"]),
        ("continuous", "~measurement", "measurement"),
        ("continuous", "~condition + measurement", "measurement"),
        ("continuous", "~condition + measurement", ["condition", "B", "A"]),
        ("continuous", "~group + condition + measurement", ["group", "Y", "X"]),
        ("continuous", "~condition * measurement", ["condition", "B", "A"]),
    ],
)
def test_design(dataset, design, contrast):
    """Test that PyDESeq2 matches DESeq2 for different designs and contrasts."""
    counts_df, metadata = load_data(dataset)

    res = run_pydeseq2(counts_df, metadata, design, contrast, {}, {})
    r_res = run_r_deseq2(counts_df, metadata, design, contrast, {}, {})

    assert_res_almost_equal(res, r_res, tol=0.04 if dataset == "continuous" else 0.02)


def test_single_sample_level():
    """Test that PyDESeq2 matches DESeq2 when zero counts in a single-sample level or in a whole condition make coefficients diverge."""
    counts_df, metadata = load_data("synthetic")
    metadata = metadata.assign(batch=["b2"] + ["b1"] * (len(metadata) - 1))
    counts_df = counts_df.copy()
    counts_df.iloc[0, :5] = 0
    counts_df.loc[metadata.condition == "B", "gene6"] = 0
    design, contrast = "~batch + group + condition", ["batch", "b2", "b1"]

    res = run_pydeseq2(counts_df, metadata, design, contrast, {}, {})
    r_res = run_r_deseq2(counts_df, metadata, design, contrast, {}, {})

    assert_res_almost_equal(res, r_res, tol=0.005)
    assert (abs(r_res.lfcSE - res.lfcSE) / r_res.lfcSE).max() < 0.005


@pytest.mark.parametrize(
    ("dataset", "design", "reduced", "contrast"),
    [
        ("synthetic", "~group + condition", "~group", ["condition", "B", "A"]),
        ("synthetic", "~group * condition", "~group + condition", ["condition", "B", "A"]),
        ("synthetic", "~group + condition", "~1", ["condition", "B", "A"]),
        ("continuous", "~group + condition + measurement", "~condition", ["group", "Y", "X"]),
    ],
)
def test_lrt(dataset, design, reduced, contrast):
    """Test that the likelihood ratio test of PyDESeq2 matches DESeq2."""
    counts_df, metadata = load_data(dataset)

    res = run_pydeseq2(counts_df, metadata, design, contrast, {}, {"reduced": reduced})
    r_res = run_r_deseq2(counts_df, metadata, design, contrast, {"test": "LRT", "reduced": ro.Formula(reduced)}, {})

    assert_res_almost_equal(res, r_res)
    assert (abs(r_res.stat - res.stat) / r_res.stat).max() < 2e-3


@pytest.mark.parametrize(
    ("dds_kwargs", "deseq_kwargs"),
    [({}, {}), ({"refit_cooks": False}, {"minReplicatesForReplace": np.inf})],
)
def test_lrt_outlier(dds_kwargs, deseq_kwargs):
    """Test that the likelihood ratio test of PyDESeq2 matches DESeq2 when a count is replaced or filtered as an outlier."""
    counts_df, metadata = load_data("synthetic")
    counts_df = counts_df.copy()
    counts_df.iloc[3, 1] = 2000
    design, reduced, contrast = "~group + condition", "~group", ["condition", "B", "A"]

    res = run_pydeseq2(counts_df, metadata, design, contrast, dds_kwargs, {"reduced": reduced})
    r_res = run_r_deseq2(
        counts_df, metadata, design, contrast, {"test": "LRT", "reduced": ro.Formula(reduced), **deseq_kwargs}, {}
    )

    assert_res_almost_equal(res, r_res)
    assert (abs(r_res.stat - res.stat) / r_res.stat).max() < 2e-3


@pytest.mark.parametrize(
    ("dds_kwargs", "ds_kwargs", "deseq_kwargs", "results_kwargs"),
    [
        ({"fit_type": "mean"}, {}, {"fitType": "mean"}, {}),
        ({"fit_type": "local"}, {}, {"fitType": "local"}, {}),
        ({"size_factors_fit_type": "poscounts"}, {}, {"sfType": "poscounts"}, {}),
        ({}, {"independent_filter": False}, {}, {"independentFiltering": False}),
        ({}, {"alpha": 0.01}, {}, {"alpha": 0.01}),
    ],
)
def test_options(dds_kwargs, ds_kwargs, deseq_kwargs, results_kwargs):
    """Test that PyDESeq2 matches DESeq2 when setting the corresponding options."""
    counts_df, metadata = load_data("synthetic")
    design, contrast = "~condition", ["condition", "B", "A"]

    res = run_pydeseq2(counts_df, metadata, design, contrast, dds_kwargs, ds_kwargs)
    r_res = run_r_deseq2(counts_df, metadata, design, contrast, deseq_kwargs, results_kwargs)

    assert_res_almost_equal(res, r_res)
