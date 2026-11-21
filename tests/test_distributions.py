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
    beta, covariance, converged = nbinomGLM(design, counts, size, offset, 15.0, prior_scale, shrink_index=shrink_index)
    assert converged

    def objective(value):
        return nbinomFn(value, design, counts, size, offset, 15.0, prior_scale, shrink_index)

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


def test_nbinom_glm_large_covariate():
    """Test that the shrinkage fit does not overflow when a covariate takes large values."""
    rng = np.random.default_rng(0)
    design = np.column_stack([np.ones(20), rng.normal(1000, 300, 20)])
    counts = rng.poisson(50, 20)

    beta, inv_hessian, _ = nbinomGLM(design, counts, 10.0, np.zeros(20), 15.0, 1.0)

    assert np.isfinite(beta).all()
    assert np.isfinite(inv_hessian).all()


@pytest.mark.parametrize(
    ("counts", "dispersion", "prior_scale", "expected"),
    [
        ([10, 12, 9, 11, 30, 28, 33, 31], 1e-1, 1.0, [2.3997, 0.9851]),
        ([10, 12, 9, 11, 30, 28, 33, 31], 1e-6, 1.0, [2.3747, 1.0348]),
        ([10, 12, 9, 11, 30, 28, 33, 31], 1e-8, 1.0, [2.3747, 1.0349]),
        ([413, 177, 16, 5, 7789, 11230, 12823, 8858], 1.42, 1e-3, [5.2244, 3.8334]),
    ],
)
def test_nbinom_glm_apeglm(counts, dispersion, prior_scale, expected):
    """Test that the shrinkage fit finds the MAP of apeglm 1.32.0 at low dispersions and with a narrow prior."""
    design = np.column_stack([np.ones(8), np.repeat([0.0, 1.0], 4)])

    beta, _, converged = nbinomGLM(
        design, np.array(counts, dtype=float), 1 / dispersion, np.zeros(8), 15.0, prior_scale
    )

    assert converged
    np.testing.assert_allclose(beta, expected, atol=1e-3)
