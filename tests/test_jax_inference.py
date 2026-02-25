import copy
import dataclasses
import os
from pathlib import Path

import jax
import numpy as np
import pandas as pd
import pytest

import pydeseq2
import tests
from pydeseq2 import dds
from pydeseq2 import ds
from pydeseq2 import jax_inference

# This is necessary for reproducibility
jax.config.update("jax_enable_x64", True)

TEST_PATH = Path(os.path.realpath(tests.__file__)).parent.resolve()


@dataclasses.dataclass
class _TestData:
    counts: pd.DataFrame
    metadata: pd.DataFrame


def _load_synthetic_data(with_outliers: bool = False) -> _TestData:
    counts_df = pydeseq2.utils.load_example_data(
        modality="raw_counts", dataset="synthetic", debug=False
    )
    metadata = pydeseq2.utils.load_example_data(
        modality="metadata", dataset="synthetic", debug=False
    )
    if with_outliers:
        counts_df.loc["sample1", "gene1"] = 2000
        counts_df.loc["sample11", "gene7"] = 1000
        metadata.loc["sample1", "condition"] = "C"
    return _TestData(counts=counts_df, metadata=metadata)


def _load_continuous_data() -> _TestData:
    counts_df = pd.read_csv(TEST_PATH / "data/continuous/test_counts.csv", index_col=0).T
    metadata = pd.read_csv(TEST_PATH / "data/continuous/test_metadata.csv", index_col=0)
    return _TestData(counts=counts_df, metadata=metadata)


@pytest.fixture
def synthetic_data() -> _TestData:
    return _load_synthetic_data()


@pytest.mark.parametrize("jointly_fit_genes", [True, False])
@pytest.mark.parametrize("lbfgs_after_irls", [True, False])
@pytest.mark.parametrize(
    "data_loader,design",
    [
        pytest.param(_load_synthetic_data, "~condition", id="single_factor"),
        pytest.param(_load_synthetic_data, "~group + condition", id="multi_factor"),
        pytest.param(
            lambda: _load_synthetic_data(with_outliers=True),
            "~group + condition",
            id="with_outliers",
        ),
        pytest.param(
            _load_continuous_data,
            "~group + condition + measurement",
            id="continuous",
        ),
    ],
)
def test_pipeline(data_loader, design, jointly_fit_genes, lbfgs_after_irls):
    data = data_loader()

    jax_dds = dds.DeseqDataSet(
        counts=data.counts,
        metadata=data.metadata,
        design=design,
        inference=jax_inference.JaxInference(
            jointly_fit_genes=jointly_fit_genes,
            lbfgs_after_irls=lbfgs_after_irls,
        ),
    )
    jax_dds.fit_size_factors()

    orig = dds.DeseqDataSet(
        counts=data.counts,
        metadata=data.metadata,
        design=design,
    )
    orig.fit_size_factors()

    # Genewise dispersions
    jax_dds.fit_genewise_dispersions()
    orig.fit_genewise_dispersions()
    np.testing.assert_allclose(
        jax_dds.var["genewise_dispersions"],
        orig.var["genewise_dispersions"],
        rtol=2e-4,
    )
    np.testing.assert_allclose(
        jax_dds.layers["_mu_hat"], orig.layers["_mu_hat"], rtol=1e-5
    )

    # Dispersion trend
    jax_dds.fit_dispersion_trend()
    orig.fit_dispersion_trend()
    np.testing.assert_allclose(
        jax_dds.var["fitted_dispersions"],
        orig.var["fitted_dispersions"],
        rtol=2e-4,
    )

    jax_dds.fit_dispersion_prior()
    orig.fit_dispersion_prior()

    # MAP dispersions
    jax_dds.fit_MAP_dispersions()
    orig.fit_MAP_dispersions()
    np.testing.assert_allclose(
        jax_dds.var["MAP_dispersions"], orig.var["MAP_dispersions"], rtol=2e-4
    )
    np.testing.assert_allclose(
        jax_dds.var["dispersions"], orig.var["dispersions"], rtol=2e-4
    )
    np.testing.assert_allclose(
        jax_dds.var["fitted_dispersions"],
        orig.var["fitted_dispersions"],
        rtol=2e-4,
    )

    # LFC
    jax_dds.fit_LFC()
    orig.fit_LFC()
    np.testing.assert_allclose(jax_dds.varm["LFC"], orig.varm["LFC"], rtol=2e-4)


@pytest.mark.parametrize("lbfgs_after_irls", [True, False])
@pytest.mark.parametrize("jointly_fit_genes", [True, False])
@pytest.mark.parametrize("continuous_factors", [True, False])
def test_fit_lfc(lbfgs_after_irls, jointly_fit_genes, continuous_factors):
    data = _load_continuous_data() if continuous_factors else _load_synthetic_data()

    orig = dds.DeseqDataSet(
        counts=data.counts,
        metadata=data.metadata,
        design="~group + condition",
    )
    orig.fit_size_factors()
    orig.fit_genewise_dispersions()
    orig.fit_MAP_dispersions()
    orig.fit_LFC()
    orig_lfc = orig.varm["LFC"].copy()
    orig.inference = jax_inference.JaxInference(
        lbfgs_after_irls=lbfgs_after_irls,
        jointly_fit_genes=jointly_fit_genes,
    )
    orig.fit_LFC()
    np.testing.assert_allclose(orig.varm["LFC"], orig_lfc, rtol=5e-6)


@pytest.mark.parametrize(
    "design",
    [
        pytest.param("~condition", id="single_factor"),
        pytest.param("~group + condition", id="multi_factor"),
    ],
)
def test_stats(synthetic_data, design):
    orig = dds.DeseqDataSet(
        counts=synthetic_data.counts,
        metadata=synthetic_data.metadata,
        design=design,
    )
    orig.deseq2()

    res_orig = ds.DeseqStats(copy.deepcopy(orig), contrast=["condition", "B", "A"])
    res_orig.summary()
    res_orig_df = res_orig.results_df

    res_jax = ds.DeseqStats(
        orig,
        inference=jax_inference.JaxInference(),
        contrast=["condition", "B", "A"],
    )
    res_jax.summary()
    res_jax_df = res_jax.results_df
    np.testing.assert_allclose(res_orig_df, res_jax_df, rtol=1e-5)


@pytest.mark.parametrize(
    "design",
    [
        pytest.param("~condition", id="single_factor"),
        pytest.param("~group + condition", id="multi_factor"),
    ],
)
@pytest.mark.parametrize("adapt", [True, False])
def test_lfc_shrinkage(synthetic_data, design, adapt):
    orig = dds.DeseqDataSet(
        counts=synthetic_data.counts,
        metadata=synthetic_data.metadata,
        design=design,
    )
    orig.deseq2()

    res_orig = ds.DeseqStats(copy.deepcopy(orig), contrast=["condition", "B", "A"])
    res_orig.summary()
    res_orig.lfc_shrink(coeff="condition[T.B]", adapt=adapt)
    shrunk_res_orig = res_orig.results_df

    res_jax = ds.DeseqStats(
        orig,
        inference=jax_inference.JaxInference(),
        contrast=["condition", "B", "A"],
    )
    res_jax.summary()
    res_jax.lfc_shrink(coeff="condition[T.B]", adapt=adapt)
    shrunk_res_jax = res_jax.results_df
    np.testing.assert_allclose(
        shrunk_res_jax.log2FoldChange,
        shrunk_res_orig.log2FoldChange,
        rtol=5e-6,
        atol=7e-4,
    )
    np.testing.assert_allclose(
        shrunk_res_jax.lfcSE, shrunk_res_orig.lfcSE, rtol=5e-6, atol=7e-4
    )
