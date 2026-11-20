"""Robust summary statistics and smoothing."""

from collections.abc import Mapping
from math import ceil
from math import floor

import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.stats import norm  # type: ignore


def trimmed_mean(x: np.ndarray, trim: float = 0.1, **kwargs) -> float | np.ndarray:
    """Return trimmed mean.

    Compute the mean after trimming data of its smallest and largest quantiles.

    Parameters
    ----------
    x
        Data whose mean to compute.
    trim
        Fraction of data to trim at each end.
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
        The fraction of the data used when estimating each y-value.
    iter
        The number of robustifying iterations.

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


def fit_local_regression(
    features: np.ndarray, targets: np.ndarray, weights: np.ndarray, frac: float = 0.7, cut: float = 0.8
) -> dict[str, np.ndarray]:
    """Fit a local quadratic regression curve, as the locfit R package does by default.

    The curve is fitted at vertices, each time with a weighted quadratic regression centered on the vertex.
    The regression uses tricube kernel weights whose bandwidth is the distance of the vertex to its ``frac * n``-th nearest data point.
    Vertices are added by bisecting the range of the features, until neighbouring vertices are closer than ``cut`` times their smallest bandwidth.

    Parameters
    ----------
    features
        A 1D array of data points.
    targets
        A 1D array of target values (with the same shape as features).
    weights
        A 1D array of prior weights (with the same shape as features).
    frac
        The fraction of the data used when fitting at each vertex.
    cut
        The largest distance between neighbouring vertices, relative to their smallest bandwidth.

    Returns
    -------
    The ``vertices``, the ``values`` and ``slopes`` of the fit at the vertices, and the ``curvature`` of a quadratic fitted to all data points, which :func:`pydeseq2.stats.predict_local_regression` uses.
    """
    n_neighbors = int(frac * len(features))
    fits: dict[float, tuple[float, float, float]] = {}

    def bandwidth(vertex: float) -> float:
        if vertex not in fits:
            distances = np.abs(features - vertex)
            width = np.partition(distances, n_neighbors - 1)[n_neighbors - 1]
            kernel = np.sqrt(weights * np.clip(1 - (distances / width) ** 3, 0, None) ** 3)
            design = np.vander(features - vertex, 3, increasing=True)
            beta = np.linalg.lstsq(design * kernel[:, None], targets * kernel, rcond=None)[0]
            fits[vertex] = (beta[0], beta[1], width)
        return fits[vertex][2]

    def grow(left: float, right: float) -> None:
        if right - left > cut * min(bandwidth(left), bandwidth(right)):
            middle = (left + right) / 2
            grow(left, middle)
            grow(middle, right)

    grow(features.min(), features.max())
    vertices = np.array(sorted(fits))
    values, slopes, _ = np.array([fits[vertex] for vertex in vertices]).T
    curvature = np.polynomial.polynomial.polyfit(features, targets, 2, w=np.sqrt(weights))[2]
    return {"vertices": vertices, "values": values, "slopes": slopes, "curvature": curvature}


def predict_local_regression(fit: Mapping[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    """Predict targets from a local regression curve fitted with :func:`pydeseq2.stats.fit_local_regression`.

    Between the vertices, predictions are interpolated with cubic Hermite polynomials.
    Outside of the vertices, the deviation from the quadratic fitted to all data points is extrapolated linearly, as in locfit.

    Parameters
    ----------
    fit
        The output of :func:`pydeseq2.stats.fit_local_regression`.
    features
        A 1D array of data points to predict the targets of.

    Returns
    -------
    Predicted values of targets.
    """
    vertices, values, slopes = fit["vertices"], fit["values"], fit["slopes"]
    predictions = CubicHermiteSpline(vertices, values, slopes)(features)
    for edge, outside in ((0, features < vertices[0]), (-1, features > vertices[-1])):
        distances = features[outside] - vertices[edge]
        predictions[outside] = values[edge] + slopes[edge] * distances + fit["curvature"] * distances**2
    return predictions
