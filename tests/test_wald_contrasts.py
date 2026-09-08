import numpy as np
import pytest
from scipy.stats import norm

from pydeseq2.glm import wald_test


@pytest.mark.parametrize(
    "alternative", [None, "greater", "less", "greaterAbs", "lessAbs"]
)
@pytest.mark.parametrize("null", [0.0, 1.5])
@pytest.mark.parametrize(
    "contrast",
    [[0.0, -1.0, 1.0], [0.0, 1.0, -1.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
)
def test_wald_tests_the_scalar_contrast(alternative, null, contrast):
    # All tests concern the scalar effect c @ beta, including multi-coefficient
    # and reversed contrasts. Thresholding individual coefficients is not invariant
    # to the chosen model parameterization.
    design = np.column_stack([np.ones(12), np.tile([0, 1, 0], 4), np.tile([0, 0, 1], 4)])
    beta = np.array([4.0, 2.0, 3.0])
    contrast = np.asarray(contrast)
    mu = np.exp(design @ beta)
    dispersion = 0.1
    ridge = np.eye(3) * 1e-6
    pvalue, statistic, se = wald_test(
        design, dispersion, beta, mu, ridge, contrast, null, alternative
    )
    weight = mu / (1 + mu * dispersion)
    information = design.T @ (weight[:, None] * design)
    inverse = np.linalg.inv(information + ridge)
    expected_se = np.sqrt(contrast @ inverse @ information @ inverse @ contrast)
    effect = contrast @ beta
    if alternative is None:
        expected_stat = (effect - null) / expected_se
        expected_p = 2 * norm.sf(abs(expected_stat))
    elif alternative == "greater":
        expected_stat = max((effect - null) / expected_se, 0)
        expected_p = norm.sf(expected_stat)
    elif alternative == "less":
        expected_stat = min((effect - null) / expected_se, 0)
        expected_p = norm.sf(abs(expected_stat))
    elif alternative == "greaterAbs":
        expected_stat = np.sign(effect) * max((abs(effect) - null) / expected_se, 0)
        expected_p = 2 * norm.sf(abs(expected_stat))
    else:
        above = max((effect + abs(null)) / expected_se, 0)
        below = min((effect - abs(null)) / expected_se, 0)
        expected_stat = min(above, below, key=abs)
        expected_p = max(norm.sf(abs(above)), norm.sf(abs(below)))
    np.testing.assert_allclose(se, expected_se)
    np.testing.assert_allclose(statistic, expected_stat)
    np.testing.assert_allclose(pvalue, expected_p, atol=0)
