"""Loaders for the example data shipped with PyDESeq2."""

from pathlib import Path
from typing import Literal

import pandas as pd

import pydeseq2


def load_example_data(
    modality: Literal["raw_counts", "metadata"] = "raw_counts",
    dataset: Literal["synthetic"] = "synthetic",
    debug: bool = False,
    debug_seed: int = 42,
) -> pd.DataFrame:
    """Load synthetic example data.

    May load either metadata or rna-seq data.
    For now, this function may only return the synthetic data provided as part of this repo, but new datasets might be added in the future.

    Parameters
    ----------
    modality
        Data modality. "raw_counts" or "metadata".
    dataset
        The dataset for which to return gene expression data.
        If "synthetic", will return the synthetic data that is used for CI unit tests. (default: ``"synthetic"``).
    debug
        If true, subsample 10 samples and 100 genes at random.
        (Note that the "synthetic" dataset is already 10 features 100.) (default: ``False``).
    debug_seed
        Seed for the debug mode. (default: ``42``).

    Returns
    -------
    Requested data modality.
    """
    assert modality in [
        "raw_counts",
        "metadata",
    ], "The modality argument must be one of the following: raw_counts, metadata"

    assert dataset in ["synthetic"], "The dataset argument must be one of the following: synthetic."

    # Load data
    datasets_path = Path(pydeseq2.__file__).parent / "datasets"
    if not datasets_path.exists():
        datasets_path = Path(pydeseq2.__file__).parents[2] / "datasets"

    if dataset == "synthetic":
        path_to_data = datasets_path / "synthetic"
        if Path(path_to_data).is_dir():
            # Cast the Paths to strings to have coherent types wrt to the url case (that
            # does not handle Paths), else mypy throws an error.
            path_to_data_counts = str(path_to_data / "test_counts.csv")
            path_to_data_metadata = str(path_to_data / "test_metadata.csv")
        else:
            # if the path does not exist (as is the case in RDT) load it from github
            url_to_data = "https://raw.githubusercontent.com/scverse/PyDESeq2/main/datasets/synthetic/"
            path_to_data_counts = url_to_data + "/test_counts.csv"
            path_to_data_metadata = url_to_data + "/test_metadata.csv"

        if modality == "raw_counts":
            df = pd.read_csv(
                path_to_data_counts,
                sep=",",
                index_col=0,
            ).T
        elif modality == "metadata":
            df = pd.read_csv(
                path_to_data_metadata,
                sep=",",
                index_col=0,
            )

    if debug:
        # TODO: until we provide a larger dataset, this option is useless
        # subsample 10 samples and 100 genes
        df = df.sample(n=10, axis=0, random_state=debug_seed)
        if modality == "raw_counts":
            df = df.sample(n=100, axis="index", random_state=debug_seed)

    return df
