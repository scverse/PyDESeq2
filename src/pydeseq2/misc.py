"""Helpers that do not belong to any single step of the pipeline."""

import multiprocessing
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from joblib import cpu_count  # type: ignore

# Columns per task in the main process's threaded passes over the count matrix,
# and the matrix size (samples x genes) below which they run serially.
_CHUNK_COLS = 1024
_MIN_THREADED_SIZE = 1 << 18


def test_valid_counts(counts: pd.DataFrame | np.ndarray) -> None:
    """Test that the count matrix contains valid inputs.

    More precisely, test that inputs are non-negative integers.

    Parameters
    ----------
    counts
        Raw counts.
        One column per gene, rows are indexed by sample barcodes.
    """
    if isinstance(counts, pd.DataFrame):
        if counts.isna().any().any():
            raise ValueError("NaNs are not allowed in the count matrix.")
        if not np.issubdtype(counts.to_numpy().dtype, np.number):
            raise ValueError("The count matrix should only contain numbers.")
    else:
        if np.isnan(counts).any().any():
            raise ValueError("NaNs are not allowed in the count matrix.")
        if not np.issubdtype(counts.dtype, np.number):
            raise ValueError("The count matrix should only contain numbers.")
    if (counts % 1 != 0).any().any():
        raise ValueError("The count matrix should only contain integers.")
    if (counts < 0).any().any():
        raise ValueError("The count matrix should only contain non-negative values.")


def n_or_more_replicates(design_matrix: pd.DataFrame, min_replicates: int) -> pd.Series:
    """
    Return a  series indicating whether samples have a minimum number of replicates.

    Checks whether each sample has at least ``min_replicates`` replicates, based on its combination of design factors.

    Parameters
    ----------
    design_matrix
        A DataFrame with experiment design information (to split cohorts).
    min_replicates
        The minimum number of replicates to have to pass the threshold.

    Returns
    -------
    A boolean series indicating whether each sample has at least ``min_replicates`` replicates.
    """
    n_or_more = design_matrix.value_counts() >= min_replicates
    replaceable = n_or_more[pd.MultiIndex.from_frame(design_matrix)]
    replaceable.index = design_matrix.index
    return replaceable


def get_num_processes(n_cpus: int | None) -> int:
    """Return the number of processes to use for multiprocessing.

    Returns the maximum number of available cpus by default.

    Parameters
    ----------
    n_cpus
        Desired number of cpus.
        If ``None``, will return the number of available cpus. (default: ``None``).

    Returns
    -------
    Number of processes to spawn.
    """
    if n_cpus is None:
        try:
            n_processes = multiprocessing.cpu_count()
        except NotImplementedError:
            n_processes = 5  # arbitrary default
    else:
        n_processes = n_cpus

    return n_processes


def _thread_count(n_cpus) -> int:
    """Return the thread count for the main process's numpy passes.

    That is ``n_cpus`` read as joblib's ``n_jobs``, capped at the usable cores; 1 for
    anything else.
    """
    if isinstance(n_cpus, bool) or not isinstance(n_cpus, int | np.integer):
        return 1
    n_cpus = int(n_cpus)
    if n_cpus == 0:
        return 1
    usable = cpu_count()
    if n_cpus < 0:
        n_cpus = usable + 1 + n_cpus
    return max(1, min(n_cpus, usable))


def _chunk_bounds(n: int, width: int) -> list[tuple[int, int]]:
    """Split ``range(n)`` into ``(start, stop)`` chunks of ``width``.

    A one-item tail is merged into the chunk before it.

    numpy reduces along axis 0 of a C-ordered block row by row only when the
    block has two or more columns; one column is summed pairwise, which changes
    the low bits.
    """
    starts = list(range(0, n, width))
    if len(starts) > 1 and n - starts[-1] < 2:
        starts.pop()
    return list(zip(starts, starts[1:] + [n], strict=True))


def _thread_pool(n_threads: int) -> ThreadPoolExecutor:
    """Return the executor of one threaded pass."""
    return ThreadPoolExecutor(max_workers=n_threads)


def _balanced_width(n: int, n_threads: int, target: int) -> int:
    """Pick a chunk width that keeps every thread busy.

    The width is near ``target`` and splits ``n`` items into a multiple of
    ``n_threads`` chunks, so no thread idles through the last round.
    """
    n_chunks = n_threads * -(-n // (n_threads * target))
    return max(2, -(-n // n_chunks))


def _map_threads(func, bounds: list[tuple[int, int]], pool: ThreadPoolExecutor) -> list:
    """``func(start, stop)`` for every bound on ``pool``, results in order."""
    return list(pool.map(lambda bound: func(*bound), bounds))


class _FPFlag:
    """``np.errstate(call=...)`` target that records any floating-point error.

    numpy's error state is per thread, so the threaded passes run under
    ``np.errstate(all="call", call=flag)`` and hand over to the serial code,
    which reports the error the caller's own settings ask for, when it is set.
    """

    def __init__(self) -> None:
        self.raised = False

    def __call__(self, kind, flag) -> None:
        self.raised = True
