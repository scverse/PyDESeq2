"""
CPU vs GPU Concordance Benchmark
================================

Verifies that GPU results are concordant with CPU results across
dataset sizes. Reports LFC correlation, max absolute difference,
p-value rank correlation, and significant gene overlap.

Usage::

    python benchmark_concordance.py
"""

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

warnings.filterwarnings("ignore", category=UserWarning)


def generate_synthetic_data(num_samples, num_genes, seed=42):
    """Generate synthetic count matrix and metadata."""
    rng = np.random.default_rng(seed)
    counts = rng.integers(0, 500, size=(num_samples, num_genes)).astype(float)
    counts[: num_samples // 2, : num_genes // 2] += 50

    counts_df = pd.DataFrame(
        counts,
        index=[f"sample_{i}" for i in range(num_samples)],
        columns=[f"gene_{i}" for i in range(num_genes)],
    )

    conditions = ["A"] * (num_samples // 2) + ["B"] * (num_samples - num_samples // 2)
    metadata = pd.DataFrame({"condition": conditions}, index=counts_df.index)
    return counts_df, metadata


def run_pipeline(counts_df, metadata, inference_type, device=None):
    """Run the full DESeq2 pipeline and return results."""
    kwargs = {"inference_type": inference_type, "quiet": True}
    if device:
        kwargs["device"] = device

    dds = DeseqDataSet(
        counts=counts_df.copy(),
        metadata=metadata.copy(),
        design="~condition",
        **kwargs,
    )
    dds.deseq2()

    ds = DeseqStats(dds, contrast=["condition", "B", "A"])
    ds.summary()
    return ds.results_df


def compute_concordance(cpu_res, gpu_res):
    """Compute concordance metrics between CPU and GPU results."""
    # Filter to common non-NaN genes
    valid_lfc = ~(cpu_res["log2FoldChange"].isna() | gpu_res["log2FoldChange"].isna())
    valid_pval = ~(cpu_res["pvalue"].isna() | gpu_res["pvalue"].isna())
    valid_padj = ~(cpu_res["padj"].isna() | gpu_res["padj"].isna())

    metrics = {}

    # LFC metrics
    cpu_lfc = cpu_res.loc[valid_lfc, "log2FoldChange"]
    gpu_lfc = gpu_res.loc[valid_lfc, "log2FoldChange"]
    if len(cpu_lfc) > 1:
        metrics["lfc_pearson_r"] = np.corrcoef(cpu_lfc, gpu_lfc)[0, 1]
        metrics["lfc_max_abs_diff"] = np.abs(cpu_lfc.values - gpu_lfc.values).max()
        nonzero = cpu_lfc.values != 0
        if nonzero.sum() > 0:
            metrics["lfc_max_rel_err"] = (
                np.abs(cpu_lfc.values[nonzero] - gpu_lfc.values[nonzero])
                / np.abs(cpu_lfc.values[nonzero])
            ).max()
    else:
        metrics["lfc_pearson_r"] = np.nan
        metrics["lfc_max_abs_diff"] = np.nan
        metrics["lfc_max_rel_err"] = np.nan

    # P-value rank correlation
    cpu_pval = cpu_res.loc[valid_pval, "pvalue"]
    gpu_pval = gpu_res.loc[valid_pval, "pvalue"]
    if len(cpu_pval) > 1:
        metrics["pval_spearman_r"] = stats.spearmanr(cpu_pval, gpu_pval).statistic
    else:
        metrics["pval_spearman_r"] = np.nan

    # Significant gene overlap (padj < 0.05)
    cpu_sig = set(cpu_res.index[valid_padj & (cpu_res["padj"] < 0.05)])
    gpu_sig = set(gpu_res.index[valid_padj & (gpu_res["padj"] < 0.05)])

    if len(cpu_sig | gpu_sig) > 0:
        metrics["jaccard_index"] = len(cpu_sig & gpu_sig) / len(cpu_sig | gpu_sig)
    else:
        metrics["jaccard_index"] = 1.0

    metrics["n_sig_cpu"] = len(cpu_sig)
    metrics["n_sig_gpu"] = len(gpu_sig)
    metrics["n_sig_both"] = len(cpu_sig & gpu_sig)

    return metrics


def main():
    """Run concordance benchmarks across dataset sizes."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    scenarios = [
        (20, 1_000),
        (50, 5_000),
        (100, 10_000),
    ]

    results = []
    for n_samples, n_genes in scenarios:
        print(f"\n--- {n_samples} samples x {n_genes} genes ---")
        counts_df, metadata = generate_synthetic_data(n_samples, n_genes)

        cpu_res = run_pipeline(counts_df, metadata, "default")
        gpu_res = run_pipeline(counts_df, metadata, "gpu", device)

        metrics = compute_concordance(cpu_res, gpu_res)
        metrics["Samples"] = n_samples
        metrics["Genes"] = n_genes
        results.append(metrics)

        print(f"  LFC Pearson r:      {metrics['lfc_pearson_r']:.8f}")
        print(f"  LFC max rel error:  {metrics['lfc_max_rel_err']:.2e}")
        print(f"  P-val Spearman r:   {metrics['pval_spearman_r']:.8f}")
        print(f"  Jaccard (padj<.05): {metrics['jaccard_index']:.4f}")
        print(
            f"  Significant: CPU={metrics['n_sig_cpu']}, "
            f"GPU={metrics['n_sig_gpu']}, "
            f"Both={metrics['n_sig_both']}"
        )

    # Summary table
    print("\n\n=== Concordance Summary ===")
    df = pd.DataFrame(results)
    cols = [
        "Samples",
        "Genes",
        "lfc_pearson_r",
        "lfc_max_rel_err",
        "pval_spearman_r",
        "jaccard_index",
    ]
    print(df[cols].to_markdown(index=False, floatfmt=".6f"))
    print("============================")


if __name__ == "__main__":
    main()
