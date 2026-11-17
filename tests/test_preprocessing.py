"""Tests for pydeseq2.preprocessing."""

import numpy as np

from pydeseq2.preprocessing import deseq2_norm_fit
from pydeseq2.preprocessing import deseq2_norm_transform

TRAIN = np.array([[10, 20, 30, 40, 50], [12, 18, 33, 41, 45], [9, 25, 28, 39, 55]])


def test_zero_counts_are_ignored():
    new = np.array([[11, 21, 29, 42, 48], [0, 0, 0, 40, 50], [0, 20, 30, 40, 50]])
    logmeans, filtered = deseq2_norm_fit(TRAIN)
    normed, size_factors = deseq2_norm_transform(new, logmeans, filtered)
    np.testing.assert_allclose(size_factors, [1.0096, 1.0018, 0.9957], atol=1e-4)
    assert np.isfinite(normed).all()


def test_all_zero_or_nan_sample_gives_nan_size_factor():
    new = np.array([[11, 21, 29, 42, 48], [0, 0, 0, 0, 0], [np.nan, 20, 30, 40, 50]])
    logmeans, filtered = deseq2_norm_fit(TRAIN)
    _, size_factors = deseq2_norm_transform(new, logmeans, filtered)
    assert np.isfinite(size_factors[0])
    assert np.isnan(size_factors[1:]).all()
