import warnings

import numpy as np
import pandas as pd


def deseq2_norm(
    counts: pd.DataFrame | np.ndarray,
) -> tuple[pd.DataFrame | np.ndarray, pd.DataFrame | np.ndarray]:
    """Return normalized counts and size_factors.

    Uses the median of ratios method.

    Parameters
    ----------
    counts
        Raw counts.
        One column per gene, one row per sample.

    Returns
    -------
    deseq2_counts
        DESeq2 normalized counts.
        One column per gene, rows are indexed by sample barcodes.
    size_factors
        DESeq2 normalization factors.
    """
    logmeans, filtered_genes = deseq2_norm_fit(counts)
    deseq2_counts, size_factors = deseq2_norm_transform(counts, logmeans, filtered_genes)
    return deseq2_counts, size_factors


def deseq2_norm_fit(counts: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``logmeans`` and ``filtered_genes``, needed in the median of ratios method.

    ``Logmeans`` and ``filtered_genes`` can then be used to normalize external datasets.

    Parameters
    ----------
    counts
        Raw counts.
        One column per gene, one row per sample.

    Returns
    -------
    logmeans
        Gene-wise mean log counts.
    filtered_genes
        Genes whose log means are finite.
    """
    # Compute gene-wise mean log counts
    with np.errstate(divide="ignore"):  # ignore division by zero warnings
        log_counts = np.log(counts)
    logmeans = log_counts.mean(axis=0)
    # Filter out genes with -∞ or NaN log means
    filtered_genes = np.isfinite(logmeans)

    return logmeans, filtered_genes


def deseq2_norm_transform(
    counts: pd.DataFrame | np.ndarray,
    logmeans: np.ndarray,
    filtered_genes: np.ndarray,
) -> tuple[pd.DataFrame | np.ndarray, pd.DataFrame | np.ndarray]:
    """Return normalized counts and size factors from the median of ratios method.

    Can be applied on external dataset, using the ``logmeans`` and ``filtered_genes`` previously computed in the ``fit`` function.
    Zero counts are ignored in each sample's median.
    A sample with no positive count or with a NaN count gets a NaN size factor.

    Parameters
    ----------
    counts
        Raw counts.
        One column per gene, one row per sample.
    logmeans
        Gene-wise mean log counts.
    filtered_genes
        Genes whose log means are finite.

    Returns
    -------
    deseq2_counts
        DESeq2 normalized counts.
        One column per gene, rows are indexed by sample barcodes.
    size_factors
        DESeq2 normalization factors.
    """
    with np.errstate(divide="ignore"):  # ignore division by zero warnings
        log_counts = np.log(counts)
    # Subtract filtered log means from log counts
    if isinstance(log_counts, pd.DataFrame):
        log_ratios = log_counts.loc[:, filtered_genes] - logmeans[filtered_genes]
    else:
        log_ratios = log_counts[:, filtered_genes] - logmeans[filtered_genes]
    # Compute sample-wise median of log ratios
    finite_log_ratios = np.where(np.isfinite(log_ratios), log_ratios, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        log_medians = np.nanmedian(finite_log_ratios, axis=1)
    log_medians[np.isnan(np.asarray(log_ratios, dtype=float)).any(axis=1)] = np.nan
    # Return raw counts divided by size factors (exponential of log ratios)
    # and size factors
    size_factors = np.exp(log_medians)
    deseq2_counts = counts / size_factors[:, None]
    return deseq2_counts, size_factors
