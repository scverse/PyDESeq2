"""Fitting and testing of the negative binomial GLM."""

import re
from functools import cache
from typing import Literal
from typing import cast

import numpy as np
import pandas as pd
import sklearn  # type: ignore
from scipy.linalg import lstsq  # type: ignore
from scipy.linalg import solve  # type: ignore
from scipy.optimize import minimize  # type: ignore
from scipy.special import gammaln  # type: ignore
from scipy.special import ndtr  # type: ignore
from scipy.stats import norm  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore

from pydeseq2.distributions import nb_nll
from pydeseq2.grid_search import grid_fit_beta


def irls_solver(
    counts: np.ndarray,
    size_factors: np.ndarray,
    design_matrix: np.ndarray,
    disp: float,
    min_mu: float = 0.5,
    beta_tol: float = 1e-8,
    min_beta: float = -30,
    max_beta: float = 30,
    optimizer: Literal["BFGS", "L-BFGS-B"] = "L-BFGS-B",
    maxiter: int = 250,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    r"""Fit a NB GLM wit log-link to predict counts from the design matrix.

    See equations (1-2) in the DESeq2 paper.

    Parameters
    ----------
    counts
        Raw counts for a given gene.
    size_factors
        Sample-wise scaling factors (obtained from median-of-ratios).
    design_matrix
        Design matrix.
    disp
        Gene-wise dispersion prior.
    min_mu
        Lower bound on estimated means, to ensure numerical stability. (default: ``0.5``).
    beta_tol
        Stopping criterion for IRWLS: :math:`\vert dev - dev_{old}\vert / \vert dev + 0.1 \vert < \beta_{tol}`. (default: ``1e-8``).
    min_beta
        Lower-bound on LFC. (default: ``-30``).
    max_beta
        Upper-bound on LFC. (default: ``-30``).
    optimizer
        Optimizing method to use in case IRLS starts diverging.
        Accepted values: 'BFGS' or 'L-BFGS-B'.
        NB: only 'L-BFGS-B' ensures that LFCS will lay in the [min_beta, max_beta] range. (default: ``'L-BFGS-B'``).
    maxiter
        Maximum number of IRLS iterations to perform before switching to L-BFGS-B. (default: ``250``).

    Returns
    -------
    beta
        Fitted (basemean, lfc) coefficients of negative binomial GLM.
    mu
        Means estimated from size factors and beta: :math:`\mu = s_{ij} \exp(\beta^t X)`.
    H
        Diagonal of the :math:`W^{1/2} X (X^t W X)^-1 X^t W^{1/2}` covariance matrix.
    converged
        Whether IRLS or the optimizer converged.
        If not and if dimension allows it, perform grid search.
    """
    assert optimizer in ["BFGS", "L-BFGS-B"]

    num_vars = design_matrix.shape[1]
    X = design_matrix

    # if full rank, estimate initial betas for IRLS below
    if np.linalg.matrix_rank(X) == num_vars:
        Q, R = np.linalg.qr(X)
        y = np.log(counts / size_factors + 0.1)
        beta_init = solve(R, Q.T @ y)
        beta = beta_init
    else:  # Initialise intercept with log base mean
        beta_init = np.zeros(num_vars)
        beta_init[0] = np.log(counts / size_factors).mean()
        beta = beta_init

    dev = 1000.0
    dev_ratio = 1.0

    ridge_factor = np.diag(np.repeat(1e-6, num_vars))
    mu = np.maximum(size_factors * np.exp(X @ beta), min_mu)

    converged = True
    i = 0
    while dev_ratio > beta_tol:
        W = mu / (1.0 + mu * disp)
        z = np.log(mu / size_factors) + (counts - mu) / mu
        H = (X.T * W) @ X + ridge_factor
        beta_hat = solve(H, X.T @ (W * z), assume_a="pos")
        i += 1

        if sum(np.abs(beta_hat) > max_beta) > 0 or i >= maxiter:
            # If IRLS starts diverging, use L-BFGS-B
            def f(beta: np.ndarray) -> float:
                # closure to minimize
                mu_ = np.maximum(size_factors * np.exp(X @ beta), min_mu)
                return nb_nll(counts, mu_, disp) + 0.5 * (ridge_factor @ beta**2).sum()

            def df(beta: np.ndarray) -> np.ndarray:
                mu_ = np.maximum(size_factors * np.exp(X @ beta), min_mu)
                return (
                    -X.T @ counts
                    + ((1 / disp + counts) * mu_ / (1 / disp + mu_)) @ X
                    + ridge_factor @ beta
                )

            res = minimize(
                f,
                beta_init,
                jac=df,
                method=optimizer,
                bounds=(
                    [(min_beta, max_beta)] * num_vars
                    if optimizer == "L-BFGS-B"
                    else None
                ),
            )

            beta = res.x
            mu = np.maximum(size_factors * np.exp(X @ beta), min_mu)
            converged = res.success

            if not res.success and num_vars <= 2:
                beta = grid_fit_beta(
                    counts,
                    size_factors,
                    X,
                    disp,
                )
                mu = np.maximum(size_factors * np.exp(X @ beta), min_mu)
            break

        beta = beta_hat
        mu = np.maximum(size_factors * np.exp(X @ beta), min_mu)
        # Compute deviation
        old_dev = dev
        # Replaced deviation with -2 * nll, as in the R code
        dev = -2 * cast(float, nb_nll(counts, mu, disp))
        dev_ratio = np.abs(dev - old_dev) / (np.abs(dev) + 0.1)

    # Compute H diagonal (useful for Cook distance outlier filtering)
    # Calculate only the diagonal for X(XTWX)-1XT using einsum
    # This is numerically equivalent to the more expensive calculation
    # np.diag(X @ (X^T @ np.inv(X^T @ np.diag(W) @ X + lambda) @ X^T)
    W = mu / (1.0 + mu * disp)
    H = np.einsum(
        "ij,jk,ki->i", X, np.linalg.inv((X.T * W[None, :]) @ X + ridge_factor), X.T
    )

    W_sq = np.sqrt(W)
    H = W_sq * H * W_sq

    # Return an UNthresholded mu (as in the R code)
    # Previous quantities are estimated with a threshold though
    mu = size_factors * np.exp(X @ beta)
    return beta, mu, H, converged


@cache
def _solve_takes_stacks() -> bool:
    """Whether ``scipy.linalg.solve`` solves a stack of matrices in one call."""
    try:
        x = solve(np.ones((2, 1, 1)), np.ones((2, 1, 1)), assume_a="pos")
    except ValueError:  # scipy < 1.16 accepts 2-D matrices only
        return False
    return np.shape(x) == (2, 1, 1)


def _solve_rows(
    a: np.ndarray, b: np.ndarray, assume_a: Literal["pos"] | None = None
) -> np.ndarray:
    """``solve(a[g], b[g], assume_a)`` for every row ``g`` of ``b``, stacked.

    ``a`` is one matrix shared by every row or a stack of one matrix per row.
    """
    # solve divides by a 1 x 1 matrix rather than factor it, unless it is stacked.
    if a.shape[-1] > 1 and _solve_takes_stacks():
        return solve(a, b[:, :, None], assume_a=assume_a)[..., 0]
    if a.ndim == 2:
        return np.array([solve(a, bg, assume_a=assume_a) for bg in b])
    return np.array(
        [solve(ag, bg, assume_a=assume_a) for ag, bg in zip(a, b, strict=False)]
    )


def _irls_rows(
    counts: np.ndarray,
    size_factors: np.ndarray,
    design_matrix: np.ndarray,
    disp: np.ndarray,
    min_mu: float = 0.5,
    beta_tol: float = 1e-8,
    min_beta: float = -30,
    max_beta: float = 30,
    optimizer: Literal["BFGS", "L-BFGS-B"] = "L-BFGS-B",
    maxiter: int = 250,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """:func:`irls_solver` on every row of a block of genes.

    The IRLS steps of all genes still iterating are taken together, as products
    and solves over stacks with one matrix per gene, laid out as the per-gene
    computation lays out its operands. A gene whose step leaves ``[-max_beta,
    max_beta]`` or reaches ``maxiter`` runs through :func:`irls_solver` from the
    start, for its optimizer and grid search.

    Parameters
    ----------
    counts
        Raw counts, one gene per row.
    size_factors, design_matrix, min_mu, beta_tol, min_beta, max_beta, optimizer, maxiter
        As in :func:`irls_solver`.
    disp
        Dispersions, one per gene.

    Returns
    -------
    beta, mu, H and converged, as :func:`irls_solver` returns them for each gene
    (stacked); None when the inputs are not ones handled here, or the block raised
    or warned under an error filter, and :func:`irls_solver` must run per gene
    instead.
    """
    from pydeseq2.distributions import _is_real_scalar

    X = design_matrix
    if (
        not all(isinstance(a, np.ndarray) for a in (counts, size_factors, X, disp))
        or X.ndim != 2
        or X.dtype != np.float64
        or not (X.flags.c_contiguous or X.flags.f_contiguous)
        or counts.ndim != 2
        or not (counts.dtype.kind in "iu" or counts.dtype == np.float64)
        or counts.shape[0] == 0
        or counts.shape[1] != X.shape[0]
        or size_factors.shape != (X.shape[0],)
        or size_factors.dtype != np.float64
        or disp.shape != (counts.shape[0],)
        or disp.dtype != np.float64
        or not all(_is_real_scalar(v) for v in (min_mu, beta_tol, max_beta))
        or not isinstance(maxiter, int | np.integer)
        or optimizer not in ["BFGS", "L-BFGS-B"]
    ):
        return None
    try:
        res = _irls_rows_steps(
            np.ascontiguousarray(counts),
            size_factors,
            X,
            disp,
            min_mu,
            beta_tol,
            max_beta,
            maxiter,
        )
    except Exception:  # irls_solver then raises or warns per gene  # noqa: BLE001
        return None
    beta, mu, H, left = res
    converged = np.ones(len(beta), dtype=bool)
    for g in np.flatnonzero(left):
        beta[g], mu[g], H[g], converged[g] = irls_solver(
            counts=counts[g].copy(),
            size_factors=size_factors,
            design_matrix=design_matrix,
            disp=disp[g],
            min_mu=min_mu,
            beta_tol=beta_tol,
            min_beta=min_beta,
            max_beta=max_beta,
            optimizer=optimizer,
            maxiter=maxiter,
        )
    return beta, mu, H, converged


def _irls_rows_steps(counts, size_factors, X, disp, min_mu, beta_tol, max_beta, maxiter):
    """Run :func:`_irls_rows`' IRLS loop; ``left`` flags the genes it hands back."""
    n_genes, n = counts.shape
    num_vars = X.shape[1]
    XT = X.T
    ridge_factor = np.diag(np.repeat(1e-6, num_vars))
    # Each (num_vars, n) slice of X.T * W gets the layout X.T * W has for one gene.
    xtw_c_order = bool((XT * np.ones(n)).flags.c_contiguous)

    def xtwx(W):
        if xtw_c_order:
            xtw = np.empty((len(W), num_vars, n))
        else:
            xtw = np.empty((len(W), n, num_vars)).transpose(0, 2, 1)
        np.multiply(XT, W[:, None, :], out=xtw)
        return xtw @ X

    def x_at(B):
        return (X @ B[:, :, None])[..., 0]

    if np.linalg.matrix_rank(X) == num_vars:
        Q, R = np.linalg.qr(X)
        y = np.log(counts / size_factors + 0.1)
        beta = _solve_rows(R, (Q.T @ y[:, :, None])[..., 0])
    else:
        beta = np.zeros((n_genes, num_vars))
        beta[:, 0] = np.log(counts / size_factors).mean(axis=1)
    mu = np.maximum(size_factors * np.exp(x_at(beta)), min_mu)

    # nb_nll's terms that do not depend on mu, in its order of operations.
    alpha_neg1 = 1 / disp
    nll_const = n * alpha_neg1 * np.log(disp)
    counts_alpha = counts + alpha_neg1[:, None]
    neg_logbinom = -(
        gammaln(counts_alpha) - gammaln(counts + 1) - gammaln(alpha_neg1)[:, None]
    )

    dev = np.full(n_genes, 1000.0)
    left = np.zeros(n_genes, dtype=bool)
    active = np.flatnonzero(np.full(n_genes, 1.0 > beta_tol))
    i = 0
    while active.size:
        mu_a = mu[active]
        W = mu_a / (1.0 + mu_a * disp[active, None])
        z = np.log(mu_a / size_factors) + (counts[active] - mu_a) / mu_a
        H = xtwx(W) + ridge_factor
        beta_hat = _solve_rows(H, (XT @ (W * z)[:, :, None])[..., 0], assume_a="pos")
        i += 1

        out = (np.abs(beta_hat) > max_beta).any(axis=1)
        if i >= maxiter:
            out[:] = True
        left[active[out]] = True
        active, beta_hat = active[~out], beta_hat[~out]

        beta[active] = beta_hat
        mu_a = np.maximum(size_factors * np.exp(x_at(beta_hat)), min_mu)
        mu[active] = mu_a
        old_dev = dev[active]
        nll = nll_const[active] + (
            neg_logbinom[active]
            + counts_alpha[active] * np.log(alpha_neg1[active, None] + mu_a)
            - counts[active] * np.log(mu_a)
        ).sum(axis=1)
        dev[active] = -2 * nll
        dev_ratio = np.abs(dev[active] - old_dev) / (np.abs(dev[active]) + 0.1)
        active = active[dev_ratio > beta_tol]

    kept = ~left
    mu_k = mu[kept]
    W = mu_k / (1.0 + mu_k * disp[kept, None])
    H_inv = np.linalg.inv(xtwx(W) + ridge_factor)
    W_sq = np.sqrt(W)
    H = np.zeros((n_genes, n))
    H[kept] = W_sq * np.einsum("ij,gjk,ki->gi", X, H_inv, XT) * W_sq
    mu = np.zeros((n_genes, n))
    mu[kept] = size_factors * np.exp(x_at(beta[kept]))
    return beta, mu, H, left


def fit_lin_mu(
    counts: np.ndarray,
    size_factors: np.ndarray,
    design_matrix: np.ndarray,
    min_mu: float = 0.5,
) -> np.ndarray:
    """Estimate mean of negative binomial model using a linear regression.

    Used to initialize genewise dispersion models.

    Parameters
    ----------
    counts
        Raw counts for a given gene.
    size_factors
        Sample-wise scaling factors (obtained from median-of-ratios).
    design_matrix
        Design matrix.
    min_mu
        Lower threshold for fitted means, for numerical stability. (default: ``0.5``).

    Returns
    -------
    Estimated mean.
    """
    y = counts / size_factors
    X = design_matrix
    cond = None
    if (
        isinstance(X, np.ndarray)
        and X.dtype == np.float64
        and X.ndim == 2
        and X.size > 0
        and isinstance(y, np.ndarray)
        and y.dtype == np.float64
        and y.shape == X.shape[:1]
        and np.isfinite(X).all()
        and np.isfinite(y).all()
    ):
        cond = _linear_regression_cond(*X.shape)
    if cond is not None:
        # The lstsq call and the product LinearRegression(fit_intercept=False)
        # makes in fit and predict, without its per-call input validation.
        coef = np.ravel(lstsq(X, y, cond=cond, check_finite=False)[0])
        mu_hat = size_factors * (X @ coef + 0.0)
    else:
        reg = LinearRegression(fit_intercept=False)
        reg.fit(design_matrix, y)
        mu_hat = size_factors * reg.predict(design_matrix)
    # Threshold mu_hat as 1/mu_hat will be used later on.
    return np.maximum(mu_hat, min_mu)


@cache
def _lstsq_takes_stacks() -> bool:
    """Whether ``scipy.linalg.lstsq`` solves a stack of right-hand sides in one call."""
    try:
        x = lstsq(np.ones((3, 1)), np.ones((2, 3, 1)), check_finite=False)[0]
    except ValueError:  # scipy < 1.16 takes a 1-D or 2-D b only
        return False
    return np.shape(x) == (2, 1, 1)


def _fit_lin_mu_rows(
    counts: np.ndarray,
    size_factors: np.ndarray,
    design_matrix: np.ndarray,
    min_mu: float = 0.5,
) -> np.ndarray | None:
    """``fit_lin_mu`` for every row of a (genes, samples) block, in one lstsq call.

    None when the installed scipy does not take stacks, the block would leave
    ``fit_lin_mu``'s lstsq path, or a floating-point flag would warn otherwise than
    the per-gene calls do; the caller then fits gene by gene.
    """
    X = design_matrix
    if not (
        isinstance(X, np.ndarray)
        and X.dtype == np.float64
        and X.ndim == 2
        and X.size > 0
        and isinstance(counts, np.ndarray)
        and counts.ndim == 2
        and counts.shape[0] > 0
        and np.ndim(size_factors) == 1
        and np.geterr()["under"] == "ignore"
        and _lstsq_takes_stacks()
    ):
        return None
    # A division or product that sets divide, overflow or invalid leaves a non-finite
    # value; the per-gene loop then runs and warns as fit_lin_mu does, gene by gene.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        Y = counts / size_factors
    if not (
        Y.dtype == np.float64
        and Y.shape == (counts.shape[0], X.shape[0])
        and np.isfinite(X).all()
        and np.isfinite(Y).all()
    ):
        return None
    cond = _linear_regression_cond(*X.shape)
    if cond is None:
        return None
    try:
        coef = lstsq(X, Y[:, :, None], cond=cond, check_finite=False)[0]
    except (np.linalg.LinAlgError, ValueError):  # fit_lin_mu then raises per gene
        return None
    # One gemv per gene, as X @ coef makes in fit_lin_mu.
    coef = np.ascontiguousarray(coef[:, :, 0])
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        mu_hat = size_factors * (np.matmul(X, coef[:, :, None])[:, :, 0] + 0.0)
    if not np.isfinite(mu_hat).all():
        return None
    return np.maximum(mu_hat, min_mu)


@cache
def _linear_regression_cond(n_samples: int, n_features: int) -> float | None:
    """Return LinearRegression's ``cond`` for ``scipy.linalg.lstsq`` on dense X.

    scikit-learn 1.9 passes its ``tol``; 1.6 to 1.8 pass ``max(X.shape) * eps``.
    None when the installed version cannot be read.
    """
    match = re.match(r"(\d+)\.(\d+)", sklearn.__version__)
    if match is None:
        return None
    if (int(match[1]), int(match[2])) >= (1, 9):
        return float(LinearRegression().tol)
    return float(max(n_samples, n_features) * np.finfo(np.float64).eps)


def wald_test(
    design_matrix: np.ndarray,
    disp: float,
    lfc: np.ndarray,
    mu: np.ndarray,
    ridge_factor: np.ndarray,
    contrast: np.ndarray,
    lfc_null: float,
    alt_hypothesis: Literal["greaterAbs", "lessAbs", "greater", "less"] | None,
) -> tuple[float, float, float]:
    """Run Wald test for differential expression.

    Computes Wald statistics, standard error and p-values from dispersion and LFC estimates.

    Parameters
    ----------
    design_matrix
        Design matrix.
    disp
        Dispersion estimate.
    lfc
        Log-fold change estimate (in natural log scale).
    mu
        Mean estimation for the NB model.
    ridge_factor
        Regularization factors.
    contrast
        Vector encoding the contrast that is being tested.
    lfc_null
        The (log2) log fold change under the null hypothesis.
    alt_hypothesis
        The alternative hypothesis for computing wald p-values.

    Returns
    -------
    wald_p_value
        Estimated p-value.
    wald_statistic
        Wald statistic.
    wald_se
        Standard error of the Wald statistic.
    """
    # Build covariance matrix estimator
    W = mu / (1 + mu * disp)
    M = (design_matrix.T * W[None, :]) @ design_matrix
    H = np.linalg.inv(M + ridge_factor)
    Hc = H @ contrast
    # Evaluate standard error and Wald statistic
    wald_se: float = np.sqrt(Hc.T @ M @ Hc)

    def greater(lfc_null):
        stat = contrast @ np.fmax((lfc - lfc_null) / wald_se, 0)
        pval = norm.sf(stat)
        return stat, pval

    def less(lfc_null):
        stat = contrast @ np.fmin((lfc - lfc_null) / wald_se, 0)
        pval = norm.sf(np.abs(stat))
        return stat, pval

    def greater_abs(lfc_null):
        stat = contrast @ (np.sign(lfc) * np.fmax((np.abs(lfc) - lfc_null) / wald_se, 0))
        pval = 2 * norm.sf(np.abs(stat))  # Only case where the test is two-tailed
        return stat, pval

    def less_abs(lfc_null):
        stat_above, pval_above = greater(-abs(lfc_null))
        stat_below, pval_below = less(abs(lfc_null))
        return min(stat_above, stat_below, key=abs), max(pval_above, pval_below)

    wald_statistic: float
    wald_p_value: float
    if alt_hypothesis is not None:
        wald_statistic, wald_p_value = {
            "greaterAbs": greater_abs(lfc_null),
            "lessAbs": less_abs(lfc_null),
            "greater": greater(lfc_null),
            "less": less(lfc_null),
        }[alt_hypothesis]
    else:
        wald_statistic = float(contrast @ (lfc - lfc_null) / wald_se)
        wald_p_value = 2 * norm.sf(np.abs(wald_statistic))

    return wald_p_value, wald_statistic, wald_se


def _wald_test_rows(
    design_matrix: np.ndarray,
    disp: np.ndarray,
    lfc: np.ndarray,
    mu: np.ndarray,
    ridge_factor: np.ndarray,
    contrast: np.ndarray,
    lfc_null: float,
    alt_hypothesis: Literal["greaterAbs", "lessAbs", "greater", "less"] | None,
    block_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """:func:`wald_test` on every gene, as arrays.

    Products and inverses run over stacks with one matrix per gene, laid out as the
    per-gene computation lays out its operands, ``block_size`` genes at a time;
    p-values come from ``ndtr``, which ``norm.sf`` calls.

    Parameters
    ----------
    disp
        Dispersions, one per gene.
    lfc
        Log-fold changes (natural log), one gene per row.
    mu
        Means, one gene per column.
    design_matrix, ridge_factor, contrast, lfc_null, alt_hypothesis
        As in :func:`wald_test`.
    block_size
        Number of genes whose stacks are built at once.

    Returns
    -------
    p-values, statistics and standard errors, one per gene; None when the inputs are
    not ones handled here, or the computation raised or warned under an error filter,
    and :func:`wald_test` must run per gene instead.
    """
    from pydeseq2.distributions import _is_real_scalar

    X = design_matrix
    # DeseqStats passes a Series, whose @ is np.dot on its values.
    from_series = isinstance(contrast, pd.Series)
    if from_series:
        contrast = contrast.to_numpy()  # type: ignore[attr-defined]
    if (
        not all(
            isinstance(a, np.ndarray) and a.dtype == np.float64
            for a in (X, disp, lfc, mu, ridge_factor, contrast)
        )
        or X.ndim != 2
        or not (X.flags.c_contiguous or X.flags.f_contiguous)
        or mu.ndim != 2
        or mu.shape[0] != X.shape[0]
        or mu.shape[1] == 0
        or disp.shape != (mu.shape[1],)
        or lfc.shape != (mu.shape[1], X.shape[1])
        or ridge_factor.shape != (X.shape[1], X.shape[1])
        or contrast.shape != (X.shape[1],)
        or not _is_real_scalar(lfc_null)
        or alt_hypothesis not in (None, "greaterAbs", "lessAbs", "greater", "less")
    ):
        return None
    n_genes = mu.shape[1]
    out = np.empty((3, n_genes))
    try:
        for start in range(0, n_genes, block_size):
            g = slice(start, min(start + block_size, n_genes))
            out[:, g] = _wald_test_block(
                X,
                disp[g],
                lfc[g],
                mu[:, g],
                ridge_factor,
                contrast,
                lfc_null,
                alt_hypothesis,
                from_series,
            )
    except Exception:  # noqa: BLE001
        return None
    return out[0], out[1], out[2]


def _wald_test_block(
    X, disp, lfc, mu, ridge_factor, contrast, lfc_null, alt_hypothesis, from_series
):
    """Compute the arrays of :func:`_wald_test_rows` for one block of genes."""
    n, num_vars = X.shape
    XT = X.T
    W = (mu / (1 + mu * disp)).T
    # Each (num_vars, n) slice of X.T * W gets the layout X.T * W has for one gene.
    if (XT * np.ones(n)).flags.c_contiguous:
        xtw = np.empty((len(W), num_vars, n))
    else:
        xtw = np.empty((len(W), n, num_vars)).transpose(0, 2, 1)
    np.multiply(XT, W[:, None, :], out=xtw)
    M = xtw @ X
    H = np.linalg.inv(M + ridge_factor)
    if from_series and num_vars == 1:
        # np.dot multiplies one-element operands; matmul's dot adds the product to
        # +0.0, which turns -0.0 into +0.0.
        def dot(v):
            return contrast[0] * v[:, 0]

        Hc = contrast[0] * H[:, 0, :]
    else:
        # contrast @ v for each row v, one dot product per gene.
        def dot(v):
            return (contrast @ v[:, :, None])[:, 0]

        Hc = H @ contrast
    wald_se = np.sqrt((Hc[:, None, :] @ M) @ Hc[:, :, None])[:, 0, 0]

    def greater(lfc_null):
        stat = dot(np.fmax((lfc - lfc_null) / wald_se[:, None], 0))
        return stat, ndtr(-stat)

    def less(lfc_null):
        stat = dot(np.fmin((lfc - lfc_null) / wald_se[:, None], 0))
        return stat, ndtr(-np.abs(stat))

    def greater_abs(lfc_null):
        stat = dot(
            np.sign(lfc) * np.fmax((np.abs(lfc) - lfc_null) / wald_se[:, None], 0)
        )
        return stat, 2 * ndtr(-np.abs(stat))

    def less_abs(lfc_null):
        stat_above, pval_above = greater(-abs(lfc_null))
        stat_below, pval_below = less(abs(lfc_null))
        # min(..., key=abs) and max(...) of wald_test, NaN included.
        stat = np.where(np.abs(stat_below) < np.abs(stat_above), stat_below, stat_above)
        pval = np.where(pval_below > pval_above, pval_below, pval_above)
        return stat, pval

    if alt_hypothesis is not None:
        # All four, as wald_test evaluates them, so the same warnings are raised.
        wald_statistic, wald_p_value = {
            "greaterAbs": greater_abs(lfc_null),
            "lessAbs": less_abs(lfc_null),
            "greater": greater(lfc_null),
            "less": less(lfc_null),
        }[alt_hypothesis]
    else:
        wald_statistic = dot(lfc - lfc_null) / wald_se
        wald_p_value = 2 * ndtr(-np.abs(wald_statistic))

    return wald_p_value, wald_statistic, wald_se
