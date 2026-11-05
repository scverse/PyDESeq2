import pathlib
from unittest import mock

import numpy as np
import pytest

from pydeseq2.datasets import load_example_data
from pydeseq2.distributions import nb_nll


@pytest.mark.parametrize("mu, alpha", [(10, 0.5), (10, 0.1), (3, 0.5), (9, 0.05)])
def test_nb_nll_moments(mu, alpha):
    # get the probability of many points
    y = np.arange(int(10 * (mu + mu**2 / alpha)))
    probas = np.zeros(y.shape)
    for i in range(y.size):
        # crude trapezoidal interpolation
        probas[i] = np.exp(-nb_nll(np.array([y[i]]), mu, alpha))
    # check that probas sums very close to 1
    assert np.allclose(probas.sum(), 1.0)
    # Re-sample according to probas
    n_montecarlo = int(1e6)
    rng = np.random.default_rng(42)
    sample = rng.choice(y, size=(n_montecarlo,), p=probas)
    # Get the theoretical values
    mean_th = mu
    var_th = (mu * alpha + 1) * mu
    # Check that the mean is in an acceptable range, up to stochasticity
    diff = sample.mean() - mean_th
    deviation = var_th / np.sqrt(n_montecarlo)
    assert np.abs(diff) < 0.2 * deviation
    error_var = np.abs(sample.var() - var_th) / var_th
    assert error_var < 1 / np.sqrt(n_montecarlo)


# Test data loading from outside the package (e.g. on RTF)
@pytest.mark.parametrize("modality", ["raw_counts", "metadata"])
@pytest.mark.parametrize("mocked_dir_flag", [True, False])
@mock.patch("pathlib.Path.is_dir")
def test_rtd_example_data_loading(mocked_function, modality, mocked_dir_flag):
    """
    Test that load_example_data still works when run from a place where the ``datasets`` directory is not accessible, as is when the documentation is built on readthedocs.
    """

    # Mock the output of is_dir() as False to emulate not having access to the
    # ``datasets`` directory
    pathlib.Path.is_dir.return_value = mocked_dir_flag

    # Try loading data.
    load_example_data(
        modality=modality,
        dataset="synthetic",
        debug=False,
    )
