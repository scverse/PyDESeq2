"""
Performance Benchmark: CPU vs GPU
=================================

Benchmarks PyDESeq2 CPU (DefaultInference) against GPU
(TorchInference) across multiple dataset sizes. Reports wall-clock
time, speedup, and peak GPU memory usage per pipeline stage.

Usage::

    python benchmark_gpu.py
"""

import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats


def generate_synthetic_data(num_samples, num_genes, seed=42):
    """Generate synthetic count matrix and metadata."""
    rng = np.random.default_rng(seed)
    counts = rng.integers(0, 500, size=(num_samples, num_genes)).astype(
        float
    )
    counts[: num_samples // 2, : num_genes // 2] += 50

    counts_df = pd.DataFrame(
        counts,
        index=[f"sample_{i}" for i in range(num_samples)],
        columns=[f"gene_{i}" for i in range(num_genes)],
    )

    conditions = ["A"] * (num_samples // 2) + ["B"] * (
        num_samples - num_samples // 2
    )
    metadata = pd.DataFrame(
        {"condition": conditions}, index=counts_df.index
    )
    return counts_df, metadata


def time_pipeline(counts_df, metadata, inference_type, device=None):
    """Run the full DESeq2 pipeline and return per-stage timings."""
    kwargs = {"inference_type": inference_type, "quiet": True}
    if device:
        kwargs["device"] = device

    dds = DeseqDataSet(
        counts=counts_df.copy(),
        metadata=metadata.copy(),
        design="~condition",
        **kwargs,
    )

    timings = {}

    start = time.perf_counter()
    dds.deseq2()
    timings["deseq2"] = time.perf_counter() - start

    start = time.perf_counter()
    ds = DeseqStats(dds, contrast=["condition", "B", "A"])
    ds.summary()
    timings["wald_test"] = time.perf_counter() - start

    timings["total"] = sum(timings.values())
    return timings


def run_benchmark(n_samples, n_genes, n_reps=3, device="cuda"):
    """Run benchmark for a single dataset configuration."""
    print(
        f"\n--- {n_samples} samples x {n_genes} genes "
        f"({n_reps} reps) ---"
    )

    counts_df, metadata = generate_synthetic_data(
        n_samples, n_genes
    )

    cpu_times = []
    gpu_times = []

    for rep in range(n_reps):
        cpu_t = time_pipeline(counts_df, metadata, "default")
        gpu_t = time_pipeline(
            counts_df, metadata, "gpu", device
        )
        cpu_times.append(cpu_t["total"])
        gpu_times.append(gpu_t["total"])

    cpu_median = np.median(cpu_times)
    gpu_median = np.median(gpu_times)
    speedup = cpu_median / gpu_median

    print(f"  CPU median: {cpu_median:.3f}s")
    print(f"  GPU median: {gpu_median:.3f}s")
    print(f"  Speedup:    {speedup:.2f}x")

    return {
        "Samples": n_samples,
        "Genes": n_genes,
        "CPU (s)": cpu_median,
        "GPU (s)": gpu_median,
        "Speedup": speedup,
    }


def main():
    """Run benchmarks across dataset sizes."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Warmup
    print("\nWarming up (JIT/kernel caching)...")
    counts_df, metadata = generate_synthetic_data(10, 500)
    time_pipeline(counts_df, metadata, "gpu", device)
    print("Warmup complete.")

    scenarios = [
        (10, 500),
        (20, 1_000),
        (50, 5_000),
        (100, 10_000),
        (200, 20_000),
        (500, 30_000),
    ]

    results = []
    for n_samples, n_genes in scenarios:
        result = run_benchmark(
            n_samples, n_genes, n_reps=3, device=device
        )
        results.append(result)

    # Summary
    print("\n\n=== Benchmark Summary ===")
    df = pd.DataFrame(results)
    print(df.to_markdown(index=False, floatfmt=".3f"))
    print("=========================")

    # Save results
    df.to_csv("benchmark_results.csv", index=False)
    print("Results saved to benchmark_results.csv")

    # GPU memory report
    if torch.cuda.is_available():
        print(
            f"\nPeak GPU memory: "
            f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
        )


if __name__ == "__main__":
    main()
