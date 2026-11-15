"""Robust summary statistics and smoothing."""

from math import ceil
from math import floor

import numpy as np
from scipy.stats import norm  # type: ignore


def trimmed_mean(x: np.ndarray, trim: float = 0.1, **kwargs) -> float | np.ndarray:
    """Return trimmed mean.

    Compute the mean after trimming data of its smallest and largest quantiles.

    Parameters
    ----------
    x
        Data whose mean to compute.
    trim
        Fraction of data to trim at each end. (default: ``0.1``).
    **kwargs
        Keyword arguments, useful to pass axis.

    Returns
    -------
    Trimmed mean.
    """
    assert trim <= 0.5
    if "axis" in kwargs:
        axis = kwargs["axis"]
        s = np.sort(x, axis=axis)
        n = x.shape[axis]
        ntrim = floor(n * trim)
        return np.take(s, np.arange(ntrim, n - ntrim), axis).mean(axis)
    else:
        n = len(x)
        s = np.sort(x)
        ntrim = floor(n * trim)
        return s[ntrim : n - ntrim].mean()


def mean_absolute_deviation(x: np.ndarray) -> float:
    """
    Compute a scaled estimator of the mean absolute deviation.

    Used in :meth:`pydeseq2.dds.DeseqDataSet.fit_dispersion_prior()`.

    Parameters
    ----------
    features
        1D array whose MAD to compute.

    Returns
    -------
    Mean absolute deviation estimator.
    """
    center = np.median(x)
    return np.median(np.abs(x - center)) / norm.ppf(0.75)


def lowess(features: np.ndarray, targets: np.ndarray, frac: float = 2.0 / 3.0, iter: int = 3):
    """Run lowess smoothing: Robust locally weighted regression.

    The lowess function fits a nonparametric regression curve to a scatterplot.
    The arrays features and targets contain an equal number of elements; each pair (features[i], targets[i]) defines a data point in the scatterplot.
    The function returns the estimated (smooth) values of targets.
    The smoothing span is given by frac.
    A larger value for frac will result in a smoother curve.
    The number of robustifying iterations is given by iter.
    The function will run faster with a smaller number of iterations.

    Parameters
    ----------
    features
        A 1D array of data points.
    targets
        A 1D array of target values (with the same shape as features).
    frac
        The fraction of the data used when estimating each y-value. (default: ``2/3``).
    iter
        The number of robustifying iterations. (default: ``3``).

    Returns
    -------
    Estimated (smooth) values of targets.
    """
    n = len(features)
    r = int(ceil(frac * n))
    h = np.maximum(np.array([np.sort(np.abs(features - features[i]))[r] for i in range(n)]), 1e-12)
    w = np.clip(np.abs(np.nan_to_num((features[:, None] - features[None, :]) / h)), 0.0, 1.0)
    w = (1 - w**3) ** 3
    yest = np.zeros(n)
    delta = np.ones(n)
    for _ in range(iter):
        for i in range(n):
            weights = delta * w[:, i]
            b = np.array([np.sum(weights * targets), np.sum(weights * targets * features)])
            A = np.array(
                [
                    [np.sum(weights), np.sum(weights * features)],
                    [np.sum(weights * features), np.sum(weights * features * features)],
                ]
            )
            beta = np.linalg.lstsq(A, b, rcond=None)[0]
            yest[i] = beta[0] + beta[1] * features[i]

        residuals = targets - yest
        s = np.median(np.abs(residuals))
        if s == 0:
            delta = (np.abs(residuals) > 0).astype(float)
        else:
            delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta**2) ** 2

    return yest
