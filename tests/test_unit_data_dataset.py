"""Area A: the ``DeseqDataSet`` constructor, ``variables``, ``to_picklable_anndata``,
and the inference interface it consumes (``Inference``, ``DefaultInference``)."""

import pickle

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from pydeseq2 import misc
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
from pydeseq2.inference import Inference
from tests import helpers

pytestmark = helpers.PANDAS_DEPRECATION_FILTERS


def _small_dataset(n_genes=60, n_samples=8, seed=201, **kwargs):
    kwargs.setdefault("factors", {"condition": ["A", "B"], "group": ["X", "Y"]})
    kwargs.setdefault("continuous", ("age",))
    return helpers.simulate_counts(
        n_genes=n_genes, n_samples=n_samples, seed=seed, **kwargs
    )


def test_init_from_counts_and_metadata():
    """Integral float counts are accepted and stored as integers, unchanged."""
    sim = _small_dataset(n_genes=20, n_samples=6)
    dds = helpers.make_dds(
        sim.counts.astype(float), sim.metadata, design="~condition", max_disp=4.0
    )

    assert np.issubdtype(dds.X.dtype, np.integer)
    np.testing.assert_array_equal(dds.X, sim.counts.to_numpy())
    assert list(dds.obs_names) == list(sim.counts.index)
    assert list(dds.var_names) == list(sim.counts.columns)
    assert list(dds.obs["condition"].astype(str)) == list(sim.metadata["condition"])
    # Documented: the enforced upper bound is max(max_disp, number of samples).
    assert dds.max_disp == 6
    assert helpers.make_dds(sim.counts, sim.metadata, max_disp=20.0).max_disp == 20


def test_init_from_anndata():
    """An AnnData gives the same dataset as its counts and metadata; counts or metadata
    passed beside it are ignored with a warning."""
    sim = _small_dataset(n_genes=20, n_samples=6)
    adata = ad.AnnData(X=sim.counts.to_numpy(dtype=float), obs=sim.metadata)
    adata.var_names = sim.counts.columns

    dds = helpers.make_dds(None, None, adata=adata)
    assert np.issubdtype(dds.X.dtype, np.integer)
    np.testing.assert_array_equal(dds.X, sim.counts.to_numpy())
    assert list(dds.var_names) == list(sim.counts.columns)

    reference = helpers.make_dds(sim.counts, sim.metadata)
    dds.fit_size_factors()
    reference.fit_size_factors()
    np.testing.assert_allclose(
        dds.obs["size_factors"], reference.obs["size_factors"], rtol=1e-12
    )

    ignored = sim.counts * 2
    with pytest.warns(UserWarning, match="adata was provided; ignoring counts"):
        dds = helpers.make_dds(ignored, None, adata=adata)
    np.testing.assert_array_equal(dds.X, sim.counts.to_numpy())
    with pytest.warns(UserWarning, match="adata was provided; ignoring metadata"):
        helpers.make_dds(None, sim.metadata, adata=adata)


def test_init_errors():
    sim = _small_dataset(n_genes=5, n_samples=6)
    with pytest.raises(ValueError, match="Either adata or both counts and metadata"):
        helpers.make_dds(sim.counts, None)
    with pytest.raises(ValueError, match="Either adata or both counts and metadata"):
        helpers.make_dds(None, sim.metadata)

    design = sim.design_matrix.copy()
    design.iloc[2, 1] = np.nan
    with pytest.raises(ValueError, match="NaNs are not allowed in the design"):
        helpers.make_dds(sim.counts, sim.metadata, design=design)


def test_deprecated_arguments_warn():
    sim = _small_dataset(n_genes=5, n_samples=8)
    with pytest.warns(DeprecationWarning, match="design_factors is deprecated"):
        dds = helpers.make_dds(
            sim.counts, sim.metadata, design="~1", design_factors=["condition", "group"]
        )
    expected = helpers.make_dds(sim.counts, sim.metadata, design="~condition + group")
    # 0/1 indicator columns: exact by construction.
    pd.testing.assert_frame_equal(
        pd.DataFrame(dds.obsm["design_matrix"]),
        pd.DataFrame(expected.obsm["design_matrix"]),
    )
    with pytest.warns(DeprecationWarning, match="continuous_factors is deprecated"):
        helpers.make_dds(
            sim.counts, sim.metadata, design="~age", continuous_factors=["age"]
        )
    with pytest.warns(DeprecationWarning, match="ref_level is deprecated"):
        helpers.make_dds(sim.counts, sim.metadata, ref_level=["condition", "A"])


@pytest.mark.parametrize(
    "design, expected",
    [
        ("~condition", {"condition"}),
        ("~condition + group + age", {"condition", "group", "age"}),
        ("~group + condition:group", {"condition", "group"}),
        ("condition + age", {"condition", "age"}),
    ],
    ids=["one_factor", "additive", "interaction", "no_tilde"],
)
def test_variables(design, expected):
    sim = _small_dataset(n_genes=5, n_samples=8)
    dds = helpers.make_dds(sim.counts, sim.metadata, design=design)
    assert set(dds.variables) == expected


def test_variables_need_a_formula():
    sim = _small_dataset(n_genes=5, n_samples=8)
    dds = helpers.make_dds(sim.counts, sim.metadata, design=sim.design_matrix)
    with pytest.raises(ValueError, match="only possible if the model was initialized"):
        _ = dds.variables


def test_design_matrix_dataframe_matches_formula():
    """A design given as a DataFrame is used as is, and fits like the same formula.

    Both datasets run the same computation on the same numbers, so their outputs
    agree to rounding (1e-10).
    """
    sim = _small_dataset()
    formula = helpers.make_dds(sim.counts, sim.metadata, design=sim.design)
    frame = helpers.make_dds(sim.counts, sim.metadata, design=sim.design_matrix)
    np.testing.assert_array_equal(
        frame.obsm["design_matrix"].to_numpy(), sim.design_matrix.to_numpy()
    )
    for dds in (formula, frame):
        dds.vst(use_design=True)
    np.testing.assert_allclose(
        frame.layers["vst_counts"], formula.layers["vst_counts"], rtol=1e-10
    )


def test_rank_deficient_design_warns_and_design_free_vst_works():
    """The warning says design-free VST remains possible; it gives the same values
    as with a full-rank design (the blind VST does not read the design)."""
    sim = helpers.simulate_counts(n_genes=60, n_samples=8, seed=202)
    metadata = sim.metadata.assign(copy=sim.metadata["condition"])
    with pytest.warns(UserWarning, match="not full rank"):
        deficient = helpers.make_dds(sim.counts, metadata, design="~condition + copy")
    full = helpers.make_dds(sim.counts, metadata, design="~condition")
    for dds in (deficient, full):
        dds.vst(use_design=False)
    np.testing.assert_allclose(
        deficient.layers["vst_counts"], full.layers["vst_counts"], rtol=1e-10
    )


def test_to_picklable_anndata():
    """A plain AnnData with the same fields survives a pickle round trip.

    Fields are copied, not recomputed, so the comparisons are exact.
    """
    sim = _small_dataset()
    dds = helpers.make_dds(sim.counts, sim.metadata, design=sim.design)
    dds.vst(use_design=True)

    adata = dds.to_picklable_anndata()
    assert type(adata) is ad.AnnData
    restored = pickle.loads(pickle.dumps(adata))

    np.testing.assert_array_equal(restored.X, dds.X)
    pd.testing.assert_frame_equal(restored.obs, dds.obs)
    pd.testing.assert_frame_equal(restored.var, dds.var)
    assert set(restored.layers) == set(dds.layers)
    for key in dds.layers:
        np.testing.assert_array_equal(restored.layers[key], dds.layers[key])
    assert set(restored.uns) == set(dds.uns)
    pd.testing.assert_series_equal(
        helpers.trend_coeffs(restored.uns["vst_trend_coeffs"]),
        helpers.trend_coeffs(dds.uns["vst_trend_coeffs"]),
    )
    design = restored.obsm["design_matrix"]
    assert type(design) is pd.DataFrame
    np.testing.assert_array_equal(
        design.to_numpy(), dds.obsm["design_matrix"].to_numpy()
    )


# Inference


def test_default_inference_n_cpus():
    inference = DefaultInference(n_cpus=3)
    assert inference.n_cpus == 3
    inference.n_cpus = 2
    assert inference.n_cpus == 2
    assert DefaultInference().n_cpus == misc.get_num_processes(None)


def test_dataset_n_cpus_reaches_inference():
    """``n_cpus`` builds a DefaultInference with that many CPUs, or overrides the
    ``n_cpus`` of a given inference object."""
    sim = _small_dataset(n_genes=5, n_samples=6)
    dds = DeseqDataSet(counts=sim.counts, metadata=sim.metadata, n_cpus=2, quiet=True)
    assert isinstance(dds.inference, DefaultInference)
    assert dds.inference.n_cpus == 2

    inference = DefaultInference(n_cpus=1)
    dds = DeseqDataSet(
        counts=sim.counts,
        metadata=sim.metadata,
        inference=inference,
        n_cpus=3,
        quiet=True,
    )
    assert dds.inference is inference
    assert inference.n_cpus == 3

    with pytest.warns(UserWarning, match="cannot override `n_cpus`"):
        dds = DeseqDataSet(
            counts=sim.counts,
            metadata=sim.metadata,
            inference=DelegatingInference(),
            n_cpus=3,
            quiet=True,
        )


class DelegatingInference(Inference):
    """A user-defined inference: implements exactly the abstract methods of
    ``Inference`` by calling a ``DefaultInference``, and has no ``n_cpus``."""

    def __init__(self):
        self._default = DefaultInference(n_cpus=helpers.get_test_cpus())

    def lin_reg_mu(self, *args, **kwargs):
        return self._default.lin_reg_mu(*args, **kwargs)

    def irls(self, *args, **kwargs):
        return self._default.irls(*args, **kwargs)

    def alpha_mle(self, *args, **kwargs):
        return self._default.alpha_mle(*args, **kwargs)

    def wald_test(self, *args, **kwargs):
        return self._default.wald_test(*args, **kwargs)

    def fit_rough_dispersions(self, *args, **kwargs):
        return self._default.fit_rough_dispersions(*args, **kwargs)

    def fit_moments_dispersions(self, *args, **kwargs):
        return self._default.fit_moments_dispersions(*args, **kwargs)

    def dispersion_trend_gamma_glm(self, *args, **kwargs):
        return self._default.dispersion_trend_gamma_glm(*args, **kwargs)

    def lfc_shrink_nbinom_glm(self, *args, **kwargs):
        return self._default.lfc_shrink_nbinom_glm(*args, **kwargs)


def test_inference_is_abstract():
    with pytest.raises(TypeError):
        Inference()

    class Partial(Inference):
        def lin_reg_mu(self, *args, **kwargs):
            return None

    with pytest.raises(TypeError):
        Partial()


def test_custom_inference_runs_the_pipeline_like_default():
    """A subclass of ``Inference`` that implements only its abstract methods runs
    ``deseq2``, ``summary`` and ``lfc_shrink`` and gives DefaultInference's results.

    The pipeline may take a different route for a custom inference object, so the
    comparison is at solver precision: dispersions 1e-3 relative (L-BFGS-B on log
    dispersion, README), natural-log LFCs 1e-3 absolute (IRLS), standard errors and
    shrunk LFCs 1e-3, p-values 1e-2 relative (a z of 5 turns an LFC error of 1e-4
    into a p-value error of about 5e-4 relative).
    """
    sim = helpers.simulate_counts(n_genes=40, n_samples=10, seed=203, lfc_sd=1.5)
    results = {}
    for name, inference in [
        ("default", DefaultInference(n_cpus=helpers.get_test_cpus())),
        ("custom", DelegatingInference()),
    ]:
        dds = helpers.make_dds(sim.counts, sim.metadata, inference=inference)
        dds.deseq2()
        stats = DeseqStats(
            dds, contrast=["condition", "B", "A"], inference=inference, quiet=True
        )
        stats.summary()
        unshrunk = stats.results_df.copy()
        stats.lfc_shrink(coeff="condition[T.B]")
        results[name] = (dds, unshrunk, stats.results_df.copy())

    (dds, unshrunk, shrunk), (dds_c, unshrunk_c, shrunk_c) = (
        results["default"],
        results["custom"],
    )
    helpers.assert_fields_close(
        dds_c.var[["genewise_dispersions", "MAP_dispersions", "dispersions"]],
        dds.var[["genewise_dispersions", "MAP_dispersions", "dispersions"]],
        rtol=1e-3,
        name="var",
    )
    helpers.assert_fields_close(
        dds_c.varm["LFC"], dds.varm["LFC"], rtol=0, atol=1e-3, name="LFC"
    )
    helpers.assert_fields_close(
        unshrunk_c[["baseMean", "lfcSE"]],
        unshrunk[["baseMean", "lfcSE"]],
        rtol=1e-3,
        name="results",
    )
    helpers.assert_fields_close(
        unshrunk_c[["pvalue", "padj"]],
        unshrunk[["pvalue", "padj"]],
        rtol=1e-2,
        name="results",
    )
    helpers.assert_fields_close(
        shrunk_c[["log2FoldChange", "lfcSE"]],
        shrunk[["log2FoldChange", "lfcSE"]],
        rtol=1e-3,
        atol=1e-3,
        name="shrunk",
    )
