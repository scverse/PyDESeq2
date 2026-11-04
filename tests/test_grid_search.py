import numpy as np
import pytest
from scipy.optimize import OptimizeResult
from scipy.optimize import minimize
from scipy.stats import nbinom

from pydeseq2.distributions import nbinomGLM


@pytest.mark.parametrize("shrink_index", [0, 1])
def test_shrink_grid_fallback_preserves_requested_prior(monkeypatch, shrink_index):
    design = np.column_stack([np.ones(6), [0, 0, 0, 1, 1, 1]])
    counts = np.array([12, 18, 21, 60, 55, 90])
    size = np.full(6, 5.0)
    offset = np.zeros(6)
    prior_no_shrink_scale = 15.0
    prior_scale = 0.3

    def objective(beta):
        mu = np.exp(design @ beta + offset)
        prior = beta[1 - shrink_index] ** 2 / (2 * prior_no_shrink_scale**2)
        prior += np.log1p((beta[shrink_index] / prior_scale) ** 2)
        return -nbinom.logpmf(counts, size, size / (size + mu)).sum() + prior

    optimum = minimize(
        objective,
        np.array([2.0, 1.0]),
        method="L-BFGS-B",
        options={"ftol": 1e-12, "gtol": 1e-6},
    )
    assert optimum.success

    monkeypatch.setattr(
        "pydeseq2.distributions.minimize",
        lambda *args, **kwargs: OptimizeResult(x=np.zeros(2), success=False),
    )
    beta, _, converged = nbinomGLM(
        design,
        counts,
        size,
        offset,
        prior_no_shrink_scale,
        prior_scale,
        shrink_index=shrink_index,
    )
    assert not converged
    # The refined grid has spacing approximately 0.0345 in each coordinate.
    np.testing.assert_allclose(beta, optimum.x, atol=0.035, rtol=0)
    assert objective(beta) - optimum.fun < 0.01

    permuted_beta, _, _ = nbinomGLM(
        design[:, ::-1],
        counts,
        size,
        offset,
        prior_no_shrink_scale,
        prior_scale,
        shrink_index=1 - shrink_index,
    )
    np.testing.assert_allclose(permuted_beta[::-1], beta, atol=1e-12, rtol=0)
