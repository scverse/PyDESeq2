from math import ceil

import numpy as np
import pandas as pd

from pydeseq2.misc import _CHUNK_COLS
from pydeseq2.misc import _MIN_THREADED_SIZE
from pydeseq2.misc import _balanced_width
from pydeseq2.misc import _chunk_bounds
from pydeseq2.misc import _FPFlag
from pydeseq2.misc import _map_threads
from pydeseq2.misc import _thread_pool


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

    Can be applied on external dataset, using the ``logmeans`` and ``filtered_genes`` previously computed in the ``fit`` function.

    Parameters
    ----------
    counts
        Raw counts.
        One column per gene, one row per sample.
    logmeans
        Gene-wise mean log counts.
    filtered_genes
        Genes whose log means are different from -∞.

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
    log_medians = np.median(log_ratios, axis=1)
    # Return raw counts divided by size factors (exponential of log ratios)
    # and size factors
    size_factors = np.exp(log_medians)
    deseq2_counts = counts / size_factors[:, None]
    return deseq2_counts, size_factors


def _threadable_counts(counts) -> bool:
    """Whether the threaded size-factor passes take ``counts``."""
    return (
        isinstance(counts, np.ndarray)
        and counts.ndim == 2
        and counts.dtype in (np.dtype(np.int64), np.dtype(np.int32))
        and (counts.flags.c_contiguous or counts.flags.f_contiguous)
        and counts.size >= _MIN_THREADED_SIZE
        and counts.shape[1] > _CHUNK_COLS
    )


def _column_bounds(n_genes: int, n_threads: int) -> list[tuple[int, int]]:
    return _chunk_bounds(n_genes, _balanced_width(n_genes, n_threads, _CHUNK_COLS))


def _deseq2_norm_fit_threads(
    counts: np.ndarray, n_threads: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Compute the size-factor fit's inputs on threads.

    Returns whether each gene has a zero count, and :func:`deseq2_norm_fit`'s
    ``logmeans``, computed on column chunks on ``n_threads`` threads.

    Returns ``None``, having changed nothing, when the serial code must run:
    ``counts`` not taken by the threaded passes, a floating-point error or an
    exception in any chunk.
    """
    if not _threadable_counts(counts):
        return None

    def chunk(start, stop):
        flag = _FPFlag()
        block = counts[:, start:stop]
        zero = (block == 0).any(0)
        with np.errstate(all="call", call=flag):
            with np.errstate(divide="ignore"):
                log_counts = np.log(block)
            logmeans = log_counts.mean(axis=0)
        return zero, logmeans, flag.raised

    try:
        with _thread_pool(n_threads) as pool:
            parts = _map_threads(chunk, _column_bounds(counts.shape[1], n_threads), pool)
    except Exception:  # noqa: BLE001
        return None
    if any(part[2] for part in parts):
        return None
    return (
        np.concatenate([part[0] for part in parts]),
        np.concatenate([part[1] for part in parts]),
    )


def _deseq2_norm_transform_threads(
    counts: np.ndarray, logmeans: np.ndarray, filtered_genes: np.ndarray, n_threads: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Normalize the counts on threads.

    Returns :func:`deseq2_norm_transform`'s normalized counts and size factors,
    and the per-gene means of the normalized counts. The log ratios and the
    normalization run on column chunks and the per-sample medians on row chunks,
    on ``n_threads`` threads.

    Returns ``None``, having changed nothing, when the serial code must run.
    """
    if not (
        _threadable_counts(counts)
        and isinstance(logmeans, np.ndarray)
        and logmeans.dtype == np.float64
        and logmeans.shape == (counts.shape[1],)
        and isinstance(filtered_genes, np.ndarray)
        and filtered_genes.dtype == bool
        and filtered_genes.shape == (counts.shape[1],)
    ):
        return None
    n_samples, n_genes = counts.shape
    offsets = np.zeros(n_genes + 1, dtype=np.intp)
    np.cumsum(filtered_genes, out=offsets[1:])
    if offsets[-1] == 0:
        return None
    # The memory orders of counts / size_factors[:, None] and of the serial
    # log_ratios.
    order = "F" if counts.flags.f_contiguous and not counts.flags.c_contiguous else "C"
    log_ratios = np.empty((n_samples, offsets[-1]), order="F")
    col_bounds = _column_bounds(n_genes, n_threads)

    def ratios(start, stop):
        flag = _FPFlag()
        keep = filtered_genes[start:stop]
        with np.errstate(all="call", call=flag):
            with np.errstate(divide="ignore"):
                log_counts = np.log(counts[:, start:stop])
            np.subtract(
                log_counts[:, keep],
                logmeans[start:stop][keep],
                out=log_ratios[:, offsets[start] : offsets[stop]],
            )
        return flag.raised

    def medians(start, stop):
        flag = _FPFlag()
        with np.errstate(all="call", call=flag):
            out = np.median(log_ratios[start:stop], axis=1)
        return out, flag.raised

    normed = np.empty(counts.shape, order=order)  # type: ignore[call-overload]
    normed_means = np.empty(n_genes)

    def normalize(start, stop):
        flag = _FPFlag()
        with np.errstate(all="call", call=flag):
            block = normed[:, start:stop]
            np.divide(counts[:, start:stop], size_factors[:, None], out=block)
            normed_means[start:stop] = block.mean(axis=0)
        return flag.raised

    try:
        with _thread_pool(n_threads) as pool:
            if any(_map_threads(ratios, col_bounds, pool)):
                return None
            parts = _map_threads(
                medians,
                _chunk_bounds(n_samples, max(2, ceil(n_samples / n_threads))),
                pool,
            )
            if any(part[1] for part in parts):
                return None
            flag = _FPFlag()
            with np.errstate(all="call", call=flag):
                size_factors = np.exp(np.concatenate([part[0] for part in parts]))
            if flag.raised:
                return None
            if any(_map_threads(normalize, col_bounds, pool)):
                return None
    except Exception:  # noqa: BLE001
        return None
    return normed, size_factors, normed_means
