"""Negative binomial log-likelihoods and their optimization."""

import numpy as np
from scipy.optimize import minimize  # type: ignore
from scipy.special import gammaln  # type: ignore
from scipy.special import polygamma  # type: ignore

from pydeseq2.grid_search import grid_fit_shrink_beta


def nb_nll(
    counts: np.ndarray, mu: np.ndarray, alpha: float | np.ndarray
) -> float | np.ndarray:
    r"""Neg log-likelihood of a negative binomial of parameters ``mu`` and ``alpha``.

    Mathematically, if ``counts`` is a vector of counting entries :math:`y_i` then the likelihood of each entry :math:`y_i` to be drawn from a negative binomial :math:`NB(\mu, \alpha)` is [1]

    .. math::
        p(y_i | \mu, \alpha) = \frac{\Gamma(y_i + \alpha^{-1})}{
            \Gamma(y_i + 1)\Gamma(\alpha^{-1})
        }
        \left(\frac{1}{1 + \alpha \mu} \right)^{1/\alpha}
        \left(\frac{\mu}{\alpha^{-1} + \mu} \right)^{y_i}

    As a consequence, assuming there are :math:`n` entries, the total negative log-likelihood for ``counts`` is

    .. math::
        \ell(\mu, \alpha) = \frac{n}{\alpha} \log(\alpha) +
            \sum_i \left \lbrace
            - \log \left( \frac{\Gamma(y_i + \alpha^{-1})}{
            \Gamma(y_i + 1)\Gamma(\alpha^{-1})
        } \right)
        + (\alpha^{-1} + y_i) \log (\alpha^{-1} + \mu)
        - y_i \log \mu
            \right \rbrace

    This is implemented in this function.

    Parameters
    ----------
    counts
        Observations.
    mu
        Mean of the distribution :math:`\mu`.
    alpha
        Dispersion of the distribution :math:`\alpha`, s.t. the variance is :math:`\mu + \alpha \mu^2`.

    Returns
    -------
    Negative log likelihood of the observations counts following :math:`NB(\mu, \alpha)`.

    Notes
    -----
    [1] https://en.wikipedia.org/wiki/Negative_binomial_distribution
    """
    n = len(counts)
    alpha_neg1 = 1 / alpha
    logbinom = gammaln(counts + alpha_neg1) - gammaln(counts + 1) - gammaln(alpha_neg1)
    if hasattr(alpha, "__len__") and len(alpha) > 1:
        return (
            alpha_neg1 * np.log(alpha)
            - logbinom
            + (counts + alpha_neg1) * np.log(mu + alpha_neg1)
            - (counts * np.log(mu))
        ).sum(axis=0)
    else:
        return (
            n * alpha_neg1 * np.log(alpha)
            + (
                -logbinom
                + (counts + alpha_neg1) * np.log(alpha_neg1 + mu)
                - counts * np.log(mu)
            ).sum()
        )


def dnb_nll(counts: np.ndarray, mu: np.ndarray, alpha: float) -> float:
    r"""Gradient of the negative log-likelihood of a negative binomial.

    Unvectorized.

    Parameters
    ----------
    counts
        Observations.
    mu
        Mean of the distribution.
    alpha
        Dispersion of the distribution, s.t. the variance is :math:`\mu + \alpha\mu^2`.

    Returns
    -------
    Derivative of negative log likelihood of NB w.r.t. :math:`\alpha`.
    """
    alpha_neg1 = 1 / alpha
    ll_part = (
        alpha_neg1**2
        * (
            polygamma(0, alpha_neg1)
            - polygamma(0, counts + alpha_neg1)
            + np.log(1 + mu * alpha)
            + (counts - mu) / (mu + alpha_neg1)
        ).sum()
    )

    return -ll_part


def nbinomFn(
    beta: np.ndarray,
    design_matrix: np.ndarray,
    counts: np.ndarray,
    size: np.ndarray,
    offset: np.ndarray,
    prior_no_shrink_scale: float,
    prior_scale: float,
    shrink_index: int = 1,
) -> float:
    """Return the NB negative likelihood with apeGLM prior.

    Use for LFC shrinkage.

    Parameters
    ----------
    beta
        2-element array: intercept and LFC coefficients.
    design_matrix
        Design matrix.
    counts
        Raw counts.
    size
        Size parameter of NB family (inverse of dispersion).
    offset
        Natural logarithm of size factor.
    prior_no_shrink_scale
        Prior variance for the intercept.
    prior_scale
        Prior variance for the intercept.
    shrink_index
        Index of the LFC coordinate to shrink. (default: ``1``).

    Returns
    -------
    Sum of the NB negative likelihood and apeGLM prior.
    """
    num_vars = design_matrix.shape[-1]

    shrink_mask = np.zeros(num_vars)
    shrink_mask[shrink_index] = 1
    no_shrink_mask = np.ones(num_vars) - shrink_mask

    xbeta = design_matrix @ beta
    prior = (
        (beta * no_shrink_mask) ** 2 / (2 * prior_no_shrink_scale**2)
    ).sum() + np.log1p((beta[shrink_index] / prior_scale) ** 2)

    nll = (
        counts * xbeta - (counts + size) * np.logaddexp(xbeta + offset, np.log(size))
    ).sum(axis=0)

    return prior - nll


def nbinomGLM(
    design_matrix: np.ndarray,
    counts: np.ndarray,
    size: np.ndarray,
    offset: np.ndarray,
    prior_no_shrink_scale: float,
    prior_scale: float,
    optimizer="L-BFGS-B",
    shrink_index: int = 1,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Fit a negative binomial MAP LFC using an apeGLM prior.

    Only the LFC is shrinked, and not the intercept.

    Parameters
    ----------
    design_matrix
        Design matrix.
    counts
        Raw counts.
    size
        Size parameter of NB family (inverse of dispersion).
    offset
        Natural logarithm of size factor.
    prior_no_shrink_scale
        Prior variance for the intercept.
    prior_scale
        Prior variance for the LFC parameter.
    optimizer
        Optimizing method to use in case IRLS starts diverging.
        Accepted values: 'L-BFGS-B', 'BFGS' or 'Newton-CG'. (default: ``'Newton-CG'``).
    shrink_index
        Index of the LFC coordinate to shrink. (default: ``1``).

    Returns
    -------
    beta
        2-element array, containing the intercept (first) and the LFC (second).
    inv_hessian
        Inverse of the Hessian of the objective at the estimated MAP LFC.
    converged
        Whether L-BFGS-B converged.
    """
    num_vars = design_matrix.shape[-1]

    shrink_mask = np.zeros(num_vars)
    shrink_mask[shrink_index] = 1
    no_shrink_mask = np.ones(num_vars) - shrink_mask

    beta_init = np.ones(num_vars) * 0.1 * (-1) ** (np.arange(num_vars))

    # Set optimization scale
    scale_cnst = nbinomFn(
        np.zeros(num_vars),
        design_matrix,
        counts,
        size,
        offset,
        prior_no_shrink_scale,
        prior_scale,
        shrink_index,
    )
    scale_cnst = np.maximum(scale_cnst, 1)

    def f(beta: np.ndarray, cnst: float = scale_cnst) -> float:
        # Function to optimize
        return (
            nbinomFn(
                beta,
                design_matrix,
                counts,
                size,
                offset,
                prior_no_shrink_scale,
                prior_scale,
                shrink_index,
            )
            / cnst
        )

    def df(beta: np.ndarray, cnst: float = scale_cnst) -> np.ndarray:
        # Gradient of the function to optimize
        xbeta = design_matrix @ beta
        d_neg_prior = (
            beta * no_shrink_mask / prior_no_shrink_scale** 2
            + 2 * beta * shrink_mask / (prior_scale**2 + beta[shrink_index] ** 2)
        )

        d_nll = (
            counts - (counts + size) / (1 + size * np.exp(-xbeta - offset))
        ) @ design_matrix

        return (d_neg_prior - d_nll) / cnst

    def ddf(beta: np.ndarray, cnst: float = scale_cnst) -> np.ndarray:
        # Hessian of the function to optimize
        # Note: will only work if there is a single shrink index
        xbeta = design_matrix @ beta
        exp_xbeta_off = np.exp(xbeta + offset)
        frac = (counts + size) * size * exp_xbeta_off / (size + exp_xbeta_off) ** 2
        # Build diagonal
        h11 = 1 / prior_no_shrink_scale**2
        h22 = (
            2
            * (prior_scale**2 - beta[shrink_index] ** 2)
            / (prior_scale**2 + beta[shrink_index] ** 2) ** 2
        )

        h = np.diag(no_shrink_mask * h11 + shrink_mask * h22)

        return 1 / cnst * ((design_matrix.T * frac) @ design_matrix + np.diag(h))

    res = minimize(
        f,
        beta_init,
        jac=df,
        hess=ddf if optimizer == "Newton-CG" else None,
        method=optimizer,
        options={
            "ftol": 1e-8,
            "gtol": 1e-8,
        },
    )

    beta = res.x
    converged = res.success

    if not converged and num_vars == 2:
        # If the solver failed, fit using grid search (slow)
        # Only for single-factor analysis
        beta = grid_fit_shrink_beta(
            counts,
            offset,
            design_matrix,
            size,
            prior_no_shrink_scale,
            prior_scale,
            scale_cnst,
            grid_length=60,
            min_beta=-30,
            max_beta=30,
        )

    inv_hessian = np.linalg.inv(ddf(beta, 1))

    return beta, inv_hessian, converged


def _is_real_scalar(value) -> bool:
    dtype = np.asarray(value).dtype
    return np.ndim(value) == 0 and (dtype.kind in "iu" or dtype == np.float64)


def _nbinom_glm_rows(
    beta: np.ndarray,
    design_matrix: np.ndarray,
    counts: np.ndarray,
    counts_size: np.ndarray,
    size: np.ndarray,
    log_size: np.ndarray,
    offset: np.ndarray,
    prior_no_shrink_scale: float,
    prior_scale: float,
    shrink_index: int,
    hessian: bool = False,
    xtw_c_order: bool = True,
    offset_finite: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """:func:`nbinomFn` and the gradient of :func:`nbinomGLM`, one gene per row.

    Unscaled (``cnst`` of 1). With ``hessian``, the matrix ``ddf(beta, 1)`` of
    :func:`nbinomGLM` instead, one per row. ``counts_size`` holds the rows of
    ``counts + size[:, None]``. Every value equals the per-gene one bitwise: the
    operations and their order are nbinomGLM's, ``X @ beta`` and ``r @ X`` are
    stacks of the per-gene matrix-vector products, row sums run along the
    contiguous axis, ``** 2`` on a scalar is libm's pow (``np.float_power``),
    and each ``(X.T * frac)`` slice is laid out as the per-gene product is.
    """
    num_vars = design_matrix.shape[-1]
    shrink_mask = np.zeros(num_vars)
    shrink_mask[shrink_index] = 1
    no_shrink_mask = np.ones(num_vars) - shrink_mask
    size_col = size[:, None]
    beta_shrink = beta[:, shrink_index]
    xbeta = np.matmul(design_matrix, beta[:, :, None])[:, :, 0]

    if hessian:
        exp_xbeta_off = np.exp(xbeta + offset)
        frac = counts_size * size_col * exp_xbeta_off / (size_col + exp_xbeta_off) ** 2
        beta_shrink_sq = np.float_power(beta_shrink, 2.0)
        h11 = 1 / prior_no_shrink_scale**2
        h22 = (
            2
            * (prior_scale**2 - beta_shrink_sq)
            / np.float_power(prior_scale**2 + beta_shrink_sq, 2.0)
        )
        diag = no_shrink_mask * h11 + shrink_mask * h22[:, None]
        n_genes, (n_samples, _) = len(beta), design_matrix.shape
        if xtw_c_order:
            xtw = np.empty((n_genes, num_vars, n_samples))
        else:
            xtw = np.empty((n_genes, n_samples, num_vars)).transpose(0, 2, 1)
        np.multiply(design_matrix.T, frac[:, None, :], out=xtw)
        return xtw @ design_matrix + diag[:, None, :], diag

    prior = ((beta * no_shrink_mask) ** 2 / (2 * prior_no_shrink_scale**2)).sum(
        axis=1
    ) + np.log1p(np.float_power(beta_shrink / prior_scale, 2.0))
    xbo = xbeta + offset
    nll = (counts * xbeta - counts_size * np.logaddexp(xbo, log_size[:, None])).sum(
        axis=1
    )

    d_neg_prior = (
        beta * no_shrink_mask / prior_no_shrink_scale**2
        + 2
        * beta
        * shrink_mask
        / (prior_scale**2 + np.float_power(beta_shrink, 2.0))[:, None]
    )
    # -(a + b) equals (-a) - b but for the sign of a zero or a NaN, which a
    # non-finite offset can produce.
    neg_xbo = -xbo if offset_finite else -xbeta - offset
    resid = counts - counts_size / (1 + size_col * np.exp(neg_xbo))
    d_nll = np.matmul(resid[:, None, :], design_matrix)[:, 0, :]
    return prior - nll, d_neg_prior - d_nll


# The L-BFGS-B loop of _nbinom_glm_lockstep below is adapted from SciPy's
# scipy.optimize._lbfgsb_py._minimize_lbfgsb under these two licenses:
#
# Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# License for the Python wrapper
# ==============================
#
# Copyright (c) 2004 David M. Cooke <cookedm@physics.mcmaster.ca>
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# Modifications by Travis Oliphant and Enthought, Inc. for inclusion in SciPy
def _nbinom_glm_lockstep(
    design_matrix: np.ndarray,
    counts: np.ndarray,
    size: np.ndarray,
    offset: np.ndarray,
    prior_no_shrink_scale: float,
    prior_scale: float,
    shrink_index: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """:func:`nbinomGLM` with L-BFGS-B on every row of a block of genes.

    Each gene keeps its own L-BFGS-B state and takes the same steps as
    ``scipy.optimize.minimize`` would, but the genes advance together, so the
    objective and gradient of all genes waiting for one are evaluated in one
    vectorized pass.

    Parameters
    ----------
    design_matrix
        Design matrix.
    counts
        Raw counts, one gene per row.
    size
        Size parameter of NB family (inverse of dispersion), one per gene.
    offset, prior_no_shrink_scale, prior_scale, shrink_index
        As in :func:`nbinomGLM`.

    Returns
    -------
    MAP coefficients, inverse Hessians and whether L-BFGS-B converged, as
    :func:`nbinomGLM` returns them for each gene (stacked); None when the inputs or
    scipy's L-BFGS-B are not ones this driver handles, and :func:`nbinomGLM` must
    run per gene instead.
    """
    from pydeseq2.dispersions import _lbfgsb_step

    step = _lbfgsb_step()
    if (
        step is None
        or not all(
            isinstance(a, np.ndarray) for a in (design_matrix, counts, size, offset)
        )
        or design_matrix.ndim != 2
        or design_matrix.dtype != np.float64
        or not (design_matrix.flags.c_contiguous or design_matrix.flags.f_contiguous)
        or counts.ndim != 2
        or not (counts.dtype.kind in "iu" or counts.dtype == np.float64)
        or counts.shape[0] == 0
        or counts.shape[1] != design_matrix.shape[0]
        or size.shape != (counts.shape[0],)
        or size.dtype != np.float64
        or offset.shape != (counts.shape[1],)
        or offset.dtype != np.float64
        or not _is_real_scalar(prior_no_shrink_scale)
        or not _is_real_scalar(prior_scale)
        or not isinstance(shrink_index, int | np.integer)
        or not -design_matrix.shape[1] <= shrink_index < design_matrix.shape[1]
    ):
        return None
    setulb, int_dtype, options = step
    m = options["maxcor"]
    factr = 1e-8 / np.finfo(float).eps
    pgtol = 1e-8
    maxls, maxiter, maxfun = options["maxls"], options["maxiter"], options["maxfun"]

    counts = np.ascontiguousarray(counts)
    n_genes, num_vars = counts.shape[0], design_matrix.shape[1]
    log_size = np.log(size)
    counts_size = counts + size[:, None]
    offset_finite = bool(np.isfinite(offset).all())
    xtw_c_order = bool((design_matrix.T * offset).flags.c_contiguous)

    def rows(beta, idx, hessian=False):
        return _nbinom_glm_rows(
            beta,
            design_matrix,
            counts[idx],
            counts_size[idx],
            size[idx],
            log_size[idx],
            offset,
            prior_no_shrink_scale,
            prior_scale,
            shrink_index,
            hessian,
            xtw_c_order,
            offset_finite,
        )

    all_genes = np.arange(n_genes)
    scale_cnst = np.maximum(rows(np.zeros((n_genes, num_vars)), all_genes)[0], 1)

    def loss_grad(idx):
        f, g = rows(x[idx], idx)
        cnst = scale_cnst[idx]
        return f / cnst, g / cnst[:, None]

    # State per gene as _minimize_lbfgsb allocates it without bounds, one row each.
    beta_init = np.ones(num_vars) * 0.1 * (-1) ** (np.arange(num_vars))
    x = np.empty((n_genes, num_vars))
    x[:] = beta_init
    low = np.zeros((n_genes, num_vars))
    up = np.zeros((n_genes, num_vars))
    nbd = np.zeros((n_genes, num_vars), dtype=int_dtype)
    g = np.zeros((n_genes, num_vars))
    wa = np.zeros((n_genes, 2 * m * num_vars + 5 * num_vars + 11 * m * m + 8 * m))
    iwa = np.zeros((n_genes, 3 * num_vars), dtype=int_dtype)
    task = np.zeros((n_genes, 2), dtype=int_dtype)
    ln_task = np.zeros((n_genes, 2), dtype=int_dtype)
    lsave = np.zeros((n_genes, 4), dtype=int_dtype)
    isave = np.zeros((n_genes, 44), dtype=int_dtype)
    dsave = np.zeros((n_genes, 29))
    state = list(
        zip(x, low, up, nbd, g, wa, iwa, task, lsave, isave, dsave, ln_task, strict=True)
    )
    f = [0.0] * n_genes
    n_iterations = [0] * n_genes

    # Evaluated at x0 first, as in dispersions._fit_alpha_mle_lockstep.
    f_last, g_last = loss_grad(all_genes)
    x_last = x.copy()
    nfev = [1] * n_genes

    active = list(range(n_genes))
    while active:
        waiting = []
        for i in active:
            xi, li, ui, nbdi, gi, wai, iwai, ti, lsi, isi, dsi, lni = state[i]
            while True:
                setulb(
                    m, xi, li, ui, nbdi, f[i], gi, factr, pgtol,
                    wai, iwai, ti, lsi, isi, dsi, maxls, lni,
                )  # fmt: skip
                t = ti.item(0)
                if t == 3:
                    waiting.append(i)
                    break
                if t != 1:
                    break
                n_iterations[i] += 1
                if n_iterations[i] >= maxiter:
                    ti[0], ti[1] = 5, 504
                elif nfev[i] > maxfun:
                    ti[0], ti[1] = 5, 502
        if not waiting:
            break
        idx = np.array(waiting)
        moved = idx[(x[idx] != x_last[idx]).any(axis=1)]
        if len(moved):
            f_last[moved], g_last[moved] = loss_grad(moved)
            x_last[moved] = x[moved]
            for i in moved.tolist():
                nfev[i] += 1
        for i, f_i in zip(waiting, f_last[idx].tolist(), strict=True):
            f[i] = f_i
        g[idx] = g_last[idx]
        active = waiting

    converged = task[:, 0] == 4
    beta = x.copy()
    if num_vars == 2:
        for i in np.flatnonzero(~converged):  # type: ignore[assignment]
            beta[i] = grid_fit_shrink_beta(
                counts[i],
                offset,
                design_matrix,
                size[i],
                prior_no_shrink_scale,
                prior_scale,
                scale_cnst[i],
                grid_length=60,
                min_beta=-30,
                max_beta=30,
            )
    inv_hessians = np.linalg.inv(rows(beta, all_genes, hessian=True)[0])
    return beta, inv_hessians, converged
