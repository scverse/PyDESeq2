"""GPU-specific tests for PyDESeq2.

Tests device placement, fallback behavior, numerical precision,
memory management, and edge cases specific to the GPU inference path.
"""

import numpy as np
import pandas as pd
import pytest

from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from pydeseq2.utils import load_example_data

torch = pytest.importorskip("torch")
pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    ),
    pytest.mark.filterwarnings(
        "ignore::UserWarning"
    ),
]


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


def _generate_synthetic(
    n_samples=20, n_genes=100, seed=42
):
    """Generate synthetic count data for testing."""
    rng = np.random.default_rng(seed)
    counts = rng.integers(0, 500, size=(n_samples, n_genes)).astype(
        float
    )
    counts[: n_samples // 2, : n_genes // 2] += 50

    counts_df = pd.DataFrame(
        counts,
        index=[f"sample_{i}" for i in range(n_samples)],
        columns=[f"gene_{i}" for i in range(n_genes)],
    )

    conditions = ["A"] * (n_samples // 2) + ["B"] * (
        n_samples - n_samples // 2
    )
    metadata = pd.DataFrame(
        {"condition": conditions}, index=counts_df.index
    )
    return counts_df, metadata


# ---- Device placement tests ----


class TestDevicePlacement:
    def test_explicit_device_cuda(self, counts_df, metadata):
        """TorchInference uses the specified CUDA device."""
        dds = DeseqDataSet(
            counts=counts_df,
            metadata=metadata,
            design="~condition",
            inference_type="gpu",
            device="cuda:0",
        )
        assert str(dds.inference.device) == "cuda:0"

    def test_auto_device_detection(self, counts_df, metadata):
        """TorchInference auto-detects CUDA when available."""
        dds = DeseqDataSet(
            counts=counts_df,
            metadata=metadata,
            design="~condition",
            inference_type="gpu",
        )
        assert "cuda" in str(dds.inference.device)

    def test_cpu_torch_inference(self, counts_df, metadata):
        """TorchInference works on CPU when explicitly set."""
        dds = DeseqDataSet(
            counts=counts_df,
            metadata=metadata,
            design="~condition",
            inference_type="gpu",
            device="cpu",
        )
        dds.deseq2()
        ds = DeseqStats(dds, contrast=["condition", "B", "A"])
        ds.summary()
        assert ds.results_df is not None
        assert not ds.results_df.empty


# ---- Precision tests ----


class TestPrecision:
    def test_cpu_gpu_concordance_tight_tol(self):
        """GPU and CPU produce nearly identical results."""
        counts_df, metadata = _generate_synthetic(
            n_samples=20, n_genes=50
        )

        # CPU run
        dds_cpu = DeseqDataSet(
            counts=counts_df,
            metadata=metadata,
            design="~condition",
            quiet=True,
        )
        dds_cpu.deseq2()
        ds_cpu = DeseqStats(
            dds_cpu, contrast=["condition", "B", "A"]
        )
        ds_cpu.summary()

        # GPU run
        dds_gpu = DeseqDataSet(
            counts=counts_df.copy(),
            metadata=metadata.copy(),
            design="~condition",
            inference_type="gpu",
            quiet=True,
        )
        dds_gpu.deseq2()
        ds_gpu = DeseqStats(
            dds_gpu, contrast=["condition", "B", "A"]
        )
        ds_gpu.summary()

        # Compare LFCs
        cpu_lfc = ds_cpu.results_df["log2FoldChange"].values
        gpu_lfc = ds_gpu.results_df["log2FoldChange"].values

        # Filter out NaN and zero values
        valid = ~(
            np.isnan(cpu_lfc)
            | np.isnan(gpu_lfc)
            | (cpu_lfc == 0)
        )
        if valid.sum() > 0:
            rel_err = np.abs(cpu_lfc[valid] - gpu_lfc[valid]) / (
                np.abs(cpu_lfc[valid]) + 1e-10
            )
            assert rel_err.max() < 0.01, (
                f"Max LFC relative error {rel_err.max():.6f} "
                f"exceeds 1% tolerance"
            )

    def test_float64_used(self, counts_df, metadata):
        """Verify TorchInference uses float64 tensors."""
        from pydeseq2.torch_inference import TorchInference

        ti = TorchInference(device="cuda")
        design = np.column_stack(
            [
                np.ones(len(counts_df)),
                (metadata["condition"] == "B").astype(float),
            ]
        )
        mu = ti.lin_reg_mu(
            counts_df.values,
            np.ones(len(counts_df)),
            design,
            0.5,
        )
        assert mu.dtype == np.float64


# ---- Memory tests ----


class TestMemory:
    def test_gpu_memory_released_after_pipeline(self):
        """GPU memory is released after pipeline completes."""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()

        counts_df, metadata = _generate_synthetic(
            n_samples=20, n_genes=100
        )
        dds = DeseqDataSet(
            counts=counts_df,
            metadata=metadata,
            design="~condition",
            inference_type="gpu",
            quiet=True,
        )
        dds.deseq2()

        ds = DeseqStats(dds, contrast=["condition", "B", "A"])
        ds.summary()

        # Force cleanup
        del dds, ds
        torch.cuda.empty_cache()

        # Memory should return close to baseline
        after = torch.cuda.memory_allocated()
        assert after <= baseline + 1024 * 1024, (
            f"GPU memory not released: baseline={baseline}, "
            f"after={after}"
        )


# ---- Edge case tests ----


class TestEdgeCases:
    def test_gpu_all_zero_genes(self, metadata):
        """Genes with all-zero counts produce NaN results."""
        counts_df = load_example_data(
            modality="raw_counts",
            dataset="synthetic",
            debug=False,
        )
        counts_df["zero_gene"] = 0

        dds = DeseqDataSet(
            counts=counts_df,
            metadata=metadata,
            design="~condition",
            inference_type="gpu",
        )
        dds.deseq2()

        ds = DeseqStats(dds, contrast=["condition", "B", "A"])
        ds.summary()

        assert np.isnan(
            ds.results_df.loc["zero_gene", "pvalue"]
        )

    def test_gpu_large_counts(self):
        """GPU handles genes with very large count values."""
        counts_data = pd.DataFrame(
            data=[
                [25, 405, 489843],
                [28, 480, 514571],
                [12, 690, 564106],
                [31, 420, 556380],
                [34, 278, 295565],
                [19, 249, 280945],
                [17, 491, 214062],
                [15, 251, 312551],
            ],
            index=[f"s{i}" for i in range(8)],
            columns=["g1", "g2", "g3"],
        )
        metadata = pd.DataFrame(
            {"condition": ["A"] * 4 + ["B"] * 4},
            index=counts_data.index,
        )

        dds = DeseqDataSet(
            counts=counts_data,
            metadata=metadata,
            design="~condition",
            inference_type="gpu",
        )
        dds.deseq2()
        ds = DeseqStats(dds, contrast=["condition", "B", "A"])
        ds.summary()

        # Should produce finite results
        assert not np.isnan(
            ds.results_df["log2FoldChange"].values
        ).all()

    def test_gpu_many_genes(self):
        """GPU handles datasets with many genes efficiently."""
        counts_df, metadata = _generate_synthetic(
            n_samples=20, n_genes=1000
        )

        dds = DeseqDataSet(
            counts=counts_df,
            metadata=metadata,
            design="~condition",
            inference_type="gpu",
            quiet=True,
        )
        dds.deseq2()

        ds = DeseqStats(dds, contrast=["condition", "B", "A"])
        ds.summary()

        assert len(ds.results_df) == 1000

    def test_gpu_multifactor_design(self):
        """GPU handles multi-factor designs (n_coeffs > 2)."""
        counts_df, metadata = _generate_synthetic(
            n_samples=30, n_genes=50
        )
        metadata["group"] = (
            ["X", "Y", "Z"] * 10
        )[:30]

        dds = DeseqDataSet(
            counts=counts_df,
            metadata=metadata,
            design="~group + condition",
            inference_type="gpu",
            quiet=True,
        )
        dds.deseq2()

        ds = DeseqStats(dds, contrast=["condition", "B", "A"])
        ds.summary()

        assert ds.results_df is not None
        assert not ds.results_df.empty
