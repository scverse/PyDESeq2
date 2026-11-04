import numpy as np
import pytest
from scipy.optimize import OptimizeResult
from scipy.optimize import minimize_scalar
from scipy.stats import nbinom

from pydeseq2 import dispersions


@pytest.mark.parametrize("optimizer", ["BFGS", "L-BFGS-B"])
@pytest.mark.parametrize("cr_reg", [False, True])
@pytest.mark.parametrize("prior_reg", [False, True])
def test_dispersion_fallback_preserves_objective(
    monkeypatch, optimizer, cr_reg, prior_reg
):
    counts = np.array([1, 2, 4, 8, 2, 10, 15, 30])
    design = np.column_stack([np.ones(8), np.repeat([0, 1], 4)])
    mu = np.repeat([5.0, 15.0], 4)
    alpha_hat, prior_var = 0.05, 0.03
    bounds = np.log([1e-4, 10.0])
    monkeypatch.setattr(
        dispersions, "minimize", lambda *args, **kwargs: OptimizeResult(success=False)
    )

    def objective(log_alpha):
        alpha = np.exp(log_alpha)
        loss = -nbinom.logpmf(counts, 1 / alpha, 1 / (1 + mu * alpha)).sum()
        if cr_reg:
            weights = mu / (1 + mu * alpha)
            loss += 0.5 * np.linalg.slogdet((design.T * weights) @ design)[1]
        if prior_reg:
            loss += (log_alpha - np.log(alpha_hat)) ** 2 / (2 * prior_var)
        return loss

    expected = minimize_scalar(objective, bounds=bounds, method="bounded")
    actual, converged = dispersions.fit_alpha_mle(
        counts,
        design,
        mu,
        alpha_hat,
        1e-4,
        10.0,
        prior_disp_var=prior_var,
        cr_reg=cr_reg,
        prior_reg=prior_reg,
        optimizer=optimizer,
    )

    assert expected.success and not converged
    assert np.log(actual) == pytest.approx(expected.x, abs=0.002)
    assert objective(np.log(actual)) <= expected.fun + 1e-4
