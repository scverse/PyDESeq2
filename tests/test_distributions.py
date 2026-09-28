import numpy as np
import pytest

from pydeseq2.distributions import nbinomFn
from pydeseq2.distributions import nbinomGLM


@pytest.mark.parametrize("width, shrink_index", [(2, 0), (2, 1), (4, 0), (4, 3)])
def test_shrinkage_covariance_matches_objective_curvature(width, shrink_index):
    rng = np.random.default_rng(42)
    design = np.column_stack((np.ones(24), rng.normal(size=(24, width - 1))))
    offset = rng.normal(scale=0.2, size=24)
    mean = np.exp(design @ np.linspace(1.0, 0.2, width) + offset)
    size = 4.0
    counts = rng.negative_binomial(size, size / (size + mean))
    prior_scale = 0.7
    beta, covariance, converged = nbinomGLM(
        design, counts, size, offset, 15.0, prior_scale, shrink_index=shrink_index
    )
    assert converged

    def objective(value):
        return nbinomFn(
            value, design, counts, size, offset, 15.0, prior_scale, shrink_index
        )

    step = 1e-3
    directions = np.eye(width) * step
    hessian = np.empty((width, width))
    for i, first in enumerate(directions):
        for j, second in enumerate(directions):
            hessian[i, j] = (
                objective(beta + first + second)
                - objective(beta + first - second)
                - objective(beta - first + second)
                + objective(beta - first - second)
            ) / (4 * step**2)

    np.testing.assert_allclose(np.linalg.inv(covariance), hessian, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(covariance, covariance.T, rtol=1e-12, atol=1e-12)
