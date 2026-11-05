"""Helpers that do not belong to any single step of the pipeline."""

import multiprocessing

import numpy as np
import pandas as pd


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
