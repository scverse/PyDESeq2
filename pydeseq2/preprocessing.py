import numpy as np
import pandas as pd


def deseq2_norm(
    counts: pd.DataFrame | np.ndarray,
) -> tuple[pd.DataFrame | np.ndarray, pd.DataFrame | np.ndarray]:
    """Return normalized counts and size_factors.

    Uses the median of ratios method.

    Parameters
    ----------
    counts : pandas.DataFrame or ndarray
            Raw counts. One column per gene, one row per sample.

    Returns
    -------
    deseq2_counts : pandas.DataFrame or ndarray
        DESeq2 normalized counts.
        One column per gene, rows are indexed by sample barcodes.

    size_factors : pandas.DataFrame or ndarray
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
    counts : pandas.DataFrame or ndarray
            Raw counts. One column per gene, one row per sample.

    Returns
    -------
    logmeans : ndarray
        Gene-wise mean log counts.

    filtered_genes : ndarray
        Genes whose log means are different from -∞.
    """
    # Compute gene-wise mean log counts
    with np.errstate(divide="ignore"):  # ignore division by zero warnings
        log_counts = np.log(counts)
    logmeans = log_counts.mean(axis=0)
    # Filter out genes with -∞ log means
    filtered_genes = ~np.isinf(logmeans)

    return logmeans, filtered_genes


def deseq2_norm_transform(
    counts: pd.DataFrame | np.ndarray,
    logmeans: np.ndarray,
    filtered_genes: np.ndarray,
) -> tuple[pd.DataFrame | np.ndarray, pd.DataFrame | np.ndarray]:
    """Return normalized counts and size factors from the median of ratios method.

    Can be applied on external dataset, using the ``logmeans`` and ``filtered_genes``
    previously computed in the ``fit`` function.

    Parameters
    ----------
    counts : pandas.DataFrame or ndarray
            Raw counts. One column per gene, one row per sample.

    logmeans : ndarray
        Gene-wise mean log counts.

    filtered_genes : ndarray
        Genes whose log means are different from -∞.

    Returns
    -------
    deseq2_counts : pandas.DataFrame or ndarray
        DESeq2 normalized counts.
        One column per gene, rows are indexed by sample barcodes.

    size_factors : pandas.DataFrame or ndarray
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
    log_medians = np.median(log_ratios, axis=1)
    # Return raw counts divided by size factors (exponential of log ratios)
    # and size factors
    size_factors = np.exp(log_medians)
    deseq2_counts = counts / size_factors[:, None]
    return deseq2_counts, size_factors
