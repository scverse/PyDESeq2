"""Estimation of gene-wise and trended dispersions."""

import inspect
from functools import lru_cache
from typing import Literal
from typing import cast

import numpy as np
import pandas as pd
from scipy.optimize import minimize  # type: ignore
from scipy.special import gammaln  # type: ignore
from scipy.special import psi  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore

from pydeseq2.distributions import dnb_nll
from pydeseq2.distributions import nb_nll
from pydeseq2.grid_search import grid_fit_alpha
from pydeseq2.misc import n_or_more_replicates
from pydeseq2.stats import trimmed_mean


def dispersion_trend(
    normed_mean: float | np.ndarray,
    coeffs: pd.Series | np.ndarray,
) -> float | np.ndarray:
    r"""Return dispersion trend from normalized counts.

    :math:`a_1/ \mu + a_0`.

    Parameters
    ----------
    normed_mean
        Mean of normalized counts for a given gene or set of genes.
    coeffs
        Fitted dispersion trend coefficients :math:`a_0` and :math:`a_1`.

    Returns
    -------
    Dispersion trend :math:`a_1/ \mu + a_0`.
    """
    if isinstance(coeffs, pd.Series):
        return coeffs["a0"] + coeffs["a1"] / normed_mean
    else:
        return coeffs[0] + coeffs[1] / normed_mean


def trimmed_cell_variance(counts: np.ndarray, cells: pd.Series) -> np.ndarray:
    """Return trimmed variance of counts according to condition.

    Compute the variance after trimming data of its smallest and largest elements, grouped by cohorts, and return the max across cohorts.
    The trim factor is a function of data size.

    Parameters
    ----------
    counts
        Sample-wise gene counts.
    cells
        Cohort affiliation of each sample.

    Returns
    -------
    Gene-wise trimmed variance estimate.
    """
    # how much to trim at different n
    trimratio = (1 / 3, 1 / 4, 1 / 8)
    # returns an index for the vector above for three sample size bins

    def trimfn(x: float) -> int:
        return 2 if x >= 23.5 else 1 if x >= 3.5 else 0

    ns = cells.value_counts()
    sqerror = np.zeros_like(counts)

    for lvl in cells.unique():
        cell_means = cast(
            np.ndarray,
            trimmed_mean(
                counts[cells == lvl, :], trim=trimratio[trimfn(ns[lvl])], axis=0
            ),
        )
        sqerror[cells == lvl, :] = counts[cells == lvl, :] - cell_means[None, :]

    sqerror **= 2

    varEst = np.zeros((len(ns), counts.shape[1]), dtype=float)
    for i, lvl in enumerate(cells.unique()):
        scale = [2.04, 1.86, 1.51][trimfn(ns[lvl])]
        varEst[i, :] = scale * trimmed_mean(
            sqerror[cells == lvl, :], trim=trimratio[trimfn(ns[lvl])], axis=0
        )

    return varEst.max(axis=0)


def trimmed_variance(
    x: np.ndarray, trim: float = 0.125, axis: int = 0
) -> float | np.ndarray:
    """Return trimmed variance.

    Compute the variance after trimming data of its smallest and largest quantiles.

    Parameters
    ----------
    features
        Data whose trimmed variance to compute.
    trim
        Fraction of data to trim at each end. (default: ``0.125``).
    axis
        Dimension along which to compute variance. (default: ``0``).

    Returns
    -------
    Trimmed variances.
    """
    rm = trimmed_mean(x, trim=trim, axis=axis)
    sqerror = (x - rm) ** 2
    # scale due to trimming of large squares
    return 1.51 * trimmed_mean(sqerror, trim=trim, axis=axis)


def fit_alpha_mle(
    counts: np.ndarray,
    design_matrix: np.ndarray,
    mu: np.ndarray,
    alpha_hat: float,
    min_disp: float,
    max_disp: float,
    prior_disp_var: float | None = None,
    cr_reg: bool = True,
    prior_reg: bool = False,
    optimizer: Literal["BFGS", "L-BFGS-B"] = "L-BFGS-B",
) -> tuple[float, bool]:
    """Estimate the dispersion parameter of a negative binomial GLM.

    Note: it is possible to pass counts, design_matrix and mu arguments in the form of pandas Series, but using numpy arrays makes the code significantly faster.

    Parameters
    ----------
    counts
        Raw counts for a given gene.
    design_matrix
        Design matrix.
    mu
        Mean estimation for the NB model.
    alpha_hat
        Initial dispersion estimate.
    min_disp
        Lower threshold for dispersion parameters.
    max_disp
        Upper threshold for dispersion parameters.
    prior_disp_var
        Prior dispersion variance.
    cr_reg
        Whether to use Cox-Reid regularization. (default: ``True``).
    prior_reg
        Whether to use prior log-residual regularization. (default: ``False``).
    optimizer
        Optimizing method to use.
        Accepted values: 'BFGS' or 'L-BFGS-B'. (default: ``'L-BFGS-B'``).

    Returns
    -------
    Dispersion estimate.

    Whether L-BFGS-B converged.
    If not, dispersion is estimated using grid search.
    """
    assert optimizer in ["BFGS", "L-BFGS-B"]

    if prior_reg:
        # Note: assertion is not working when using numpy
        assert prior_disp_var is not None, (
            "Sigma_prior is required for prior regularization"
        )

    log_alpha_hat = np.log(alpha_hat)

    def loss(log_alpha: float) -> float:
        # closure to be minimized
        alpha = np.exp(log_alpha)
        reg = 0
        if cr_reg:
            W = mu / (1 + mu * alpha)
            reg += 0.5 * np.linalg.slogdet((design_matrix.T * W) @ design_matrix)[1]
        if prior_reg:
            if prior_disp_var is None:
                raise ValueError("Sigma_prior is required for prior regularization")
            reg += (log_alpha - log_alpha_hat) ** 2 / (2 * prior_disp_var)
        return cast(float, nb_nll(counts, mu, alpha)) + reg

    def dloss(log_alpha: float) -> float:
        # gradient closure
        alpha = np.exp(log_alpha)
        reg_grad = 0
        if cr_reg:
            W = mu / (1 + mu * alpha)
            dW = -(W**2)
            reg_grad += (
                0.5
                * (
                    np.linalg.inv((design_matrix.T * W) @ design_matrix)
                    * ((design_matrix.T * dW) @ design_matrix)
                ).sum()
            ) * alpha  # since we want the gradient wrt log_alpha,
            # we need to multiply by alpha
        if prior_reg:
            if prior_disp_var is None:
                raise ValueError("Sigma_prior is required for prior regularization")

            reg_grad += (log_alpha - log_alpha_hat) / prior_disp_var
        # dnb_nll is the gradient wrt alpha, we need to multiply by alpha to get the
        # gradient wrt log_alpha
        return alpha * dnb_nll(counts, mu, alpha) + reg_grad

    res = minimize(
        lambda x: loss(x[0]),
        x0=np.asarray([np.log(alpha_hat)]),
        jac=lambda x: np.asarray([dloss(x[0])]),
        method=optimizer,
        bounds=(
            [(np.log(min_disp), np.log(max_disp))] if optimizer == "L-BFGS-B" else None
        ),
    )

    if res.success:
        return np.exp(res.x[0]), res.success
    else:
        return (
            np.exp(
                grid_fit_alpha(counts, design_matrix, mu, alpha_hat, min_disp, max_disp)
            ),
            res.success,
        )


# The L-BFGS-B loop of _fit_alpha_mle_lockstep below (with _lbfgsb_step) is adapted
# from SciPy's scipy.optimize._lbfgsb_py._minimize_lbfgsb under these two licenses:
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
_SETULB_SIGNATURE = (
    "setulb(m, x, l, u, nbd, f, g, factr, pgtol, wa, iwa, task, lsave, isave, dsave, "
    "maxls, ln_task)"
)


@lru_cache(maxsize=1)
def _lbfgsb_step():
    """Return scipy's L-BFGS-B step, its integer dtype and minimize's defaults.

    None when ``setulb`` or ``_minimize_lbfgsb`` is not the one the lockstep driver
    mirrors, so that callers fall back to ``scipy.optimize.minimize``.
    """
    try:
        from scipy.optimize import _lbfgsb  # type: ignore
        from scipy.optimize import _lbfgsb_py  # type: ignore
    except ImportError:
        return None
    setulb = getattr(_lbfgsb, "setulb", None)
    # scipy 1.15 to 1.17 print the same signature without spaces.
    doc = "".join((getattr(setulb, "__doc__", None) or "").split())
    if not doc.startswith("".join(_SETULB_SIGNATURE.split())):
        return None
    try:
        params = inspect.signature(_lbfgsb_py._minimize_lbfgsb).parameters
    except (AttributeError, TypeError, ValueError):
        return None
    names = ("maxcor", "ftol", "gtol", "maxfun", "maxiter", "maxls")
    if any(name not in params for name in names):
        return None
    options = {name: params[name].default for name in names}
    int_dtype = np.int64 if getattr(_lbfgsb_py, "HAS_ILP64", False) else np.int32
    return setulb, int_dtype, options


def _alpha_mle_loss_grad(
    log_alpha: np.ndarray,
    counts: np.ndarray,
    design_matrix: np.ndarray,
    mu: np.ndarray,
    log_alpha_hat: np.ndarray,
    prior_disp_var: float | None,
    cr_reg: bool,
    prior_reg: bool,
    xtw_c_order: bool,
    lgamma_counts1: np.ndarray,
    counts_log_mu: np.ndarray,
    counts_minus_mu: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Loss and gradient of :func:`fit_alpha_mle` for one gene per row.

    Every value equals the per-gene closures' bitwise: the operations and their
    order are theirs, row sums run along the contiguous axis, ``** 2`` on a scalar
    is libm's pow (``np.float_power``), and each ``(X.T * W) @ X`` slice is laid out
    as the per-gene product is, so that matmul makes the same BLAS call.

    ``lgamma_counts1``, ``counts_log_mu`` and ``counts_minus_mu`` are
    ``gammaln(counts + 1)``, ``counts * np.log(mu)`` and ``counts - mu``, which do
    not depend on alpha.
    """
    alpha = np.exp(log_alpha)
    alpha_col = alpha[:, None]
    one_mu_alpha = 1 + mu * alpha_col
    W = mu / one_mu_alpha
    reg: float | np.ndarray = 0
    reg_grad: float | np.ndarray = 0
    if cr_reg:
        n_genes, (n_samples, n_coefs) = len(log_alpha), design_matrix.shape
        if xtw_c_order:
            xtw = np.empty((n_genes, n_coefs, n_samples))
        else:
            xtw = np.empty((n_genes, n_samples, n_coefs)).transpose(0, 2, 1)
        np.multiply(design_matrix.T, W[:, None, :], out=xtw)
        M = xtw @ design_matrix
        reg = reg + 0.5 * np.linalg.slogdet(M)[1]
        np.multiply(design_matrix.T, (-(W**2))[:, None, :], out=xtw)
        dM = xtw @ design_matrix
        reg_grad = reg_grad + (0.5 * (np.linalg.inv(M) * dM).sum(axis=(1, 2))) * alpha
    if prior_reg:
        diff = log_alpha - log_alpha_hat
        reg = reg + np.float_power(diff, 2.0) / (2 * prior_disp_var)  # type: ignore[operator]
        reg_grad = reg_grad + diff / prior_disp_var

    # nb_nll and dnb_nll with a scalar alpha, one gene per row
    n = counts.shape[1]
    alpha_neg1 = 1 / alpha
    alpha_neg1_col = alpha_neg1[:, None]
    counts_alpha_neg1 = counts + alpha_neg1_col
    # dnb_nll's mu + alpha_neg1: addition is commutative bitwise
    alpha_neg1_mu = alpha_neg1_col + mu
    logbinom = gammaln(counts_alpha_neg1) - lgamma_counts1 - gammaln(alpha_neg1)[:, None]
    nll = n * alpha_neg1 * np.log(alpha) + (
        -logbinom + counts_alpha_neg1 * np.log(alpha_neg1_mu) - counts_log_mu
    ).sum(axis=1)
    ll_part = np.float_power(alpha_neg1, 2.0) * (
        psi(alpha_neg1)[:, None]
        - psi(counts_alpha_neg1)
        + np.log(one_mu_alpha)
        + counts_minus_mu / alpha_neg1_mu
    ).sum(axis=1)
    return nll + reg, alpha * -ll_part + reg_grad


def _fit_alpha_mle_lockstep(
    counts: np.ndarray,
    design_matrix: np.ndarray,
    mu: np.ndarray,
    alpha_hat: np.ndarray,
    min_disp: float,
    max_disp: float,
    prior_disp_var: float | None = None,
    cr_reg: bool = True,
    prior_reg: bool = False,
) -> tuple[np.ndarray, np.ndarray] | None:
    """:func:`fit_alpha_mle` with L-BFGS-B on every row of a block of genes.

    Each gene keeps its own L-BFGS-B state and takes the same steps as
    ``scipy.optimize.minimize`` would, but the genes advance together, so the loss
    and gradient of all genes waiting for one are evaluated in one vectorized pass.

    Parameters
    ----------
    counts
        Raw counts, one gene per row.
    design_matrix
        Design matrix.
    mu
        Mean estimations, one gene per row.
    alpha_hat
        Initial dispersion estimates, one per gene.
    min_disp, max_disp, prior_disp_var, cr_reg, prior_reg
        As in :func:`fit_alpha_mle`.

    Returns
    -------
    Dispersion estimates and whether L-BFGS-B converged, as :func:`fit_alpha_mle`
    returns them for each gene; None when the inputs or scipy's L-BFGS-B are not
    ones this driver handles, and :func:`fit_alpha_mle` must run per gene instead.
    """
    step = _lbfgsb_step()
    if (
        step is None
        or not all(isinstance(a, np.ndarray) for a in (counts, design_matrix, mu))
        or design_matrix.ndim != 2
        or design_matrix.dtype != np.float64
        or counts.ndim != 2
        or mu.shape != counts.shape
        or mu.shape[1] != design_matrix.shape[0]
        or (prior_reg and prior_disp_var is None)
    ):
        return None
    alpha_hat = np.asarray(alpha_hat)
    if alpha_hat.dtype != np.float64 or alpha_hat.shape != (counts.shape[0],):
        return None
    # minimize rounds the bounds to float64 before clipping x0 to them.
    lower, upper = float(np.log(min_disp)), float(np.log(max_disp))
    if not (np.isfinite(lower) and np.isfinite(upper) and lower < upper):
        return None
    setulb, int_dtype, options = step
    m = options["maxcor"]
    factr = options["ftol"] / np.finfo(float).eps
    pgtol = options["gtol"]
    maxls, maxiter, maxfun = options["maxls"], options["maxiter"], options["maxfun"]

    counts = np.ascontiguousarray(counts)
    mu = np.ascontiguousarray(mu)
    n_genes = counts.shape[0]
    if n_genes == 0:
        return np.zeros(0), np.zeros(0, dtype=bool)
    xtw_c_order = bool((design_matrix.T * mu[0]).flags.c_contiguous)
    lgamma_counts1 = gammaln(counts + 1)
    counts_log_mu = counts * np.log(mu)
    counts_minus_mu = counts - mu

    def loss_grad(idx):
        return _alpha_mle_loss_grad(
            x[idx, 0],
            counts[idx],
            design_matrix,
            mu[idx],
            log_alpha_hat[idx],
            prior_disp_var,
            cr_reg,
            prior_reg,
            xtw_c_order,
            lgamma_counts1[idx],
            counts_log_mu[idx],
            counts_minus_mu[idx],
        )

    # State per gene as _minimize_lbfgsb allocates it, one row each (n = 1).
    log_alpha_hat = np.log(alpha_hat)
    x = np.clip(log_alpha_hat, lower, upper).reshape(n_genes, 1)
    low = np.full((n_genes, 1), float(lower))
    up = np.full((n_genes, 1), float(upper))
    nbd = np.full((n_genes, 1), 2, dtype=int_dtype)
    g = np.zeros((n_genes, 1))
    wa = np.zeros((n_genes, 2 * m + 5 + 11 * m * m + 8 * m))
    iwa = np.zeros((n_genes, 3), dtype=int_dtype)
    task = np.zeros((n_genes, 2), dtype=int_dtype)
    ln_task = np.zeros((n_genes, 2), dtype=int_dtype)
    lsave = np.zeros((n_genes, 4), dtype=int_dtype)
    isave = np.zeros((n_genes, 44), dtype=int_dtype)
    dsave = np.zeros((n_genes, 29))
    rows = list(
        zip(x, low, up, nbd, g, wa, iwa, task, lsave, isave, dsave, ln_task, strict=True)
    )
    f = [0.0] * n_genes
    n_iterations = [0] * n_genes

    # ScalarFunction evaluates at x0 when built; later points reuse the last value
    # when x has not moved (np.array_equal), as its fun_and_grad does.
    all_genes = np.arange(n_genes)
    f_last, g_last = loss_grad(all_genes)
    x_last = x[:, 0].copy()
    nfev = [1] * n_genes

    active = list(range(n_genes))
    while active:
        waiting = []
        for i in active:
            xi, li, ui, nbdi, gi, wai, iwai, ti, lsi, isi, dsi, lni = rows[i]
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
        moved = idx[x[idx, 0] != x_last[idx]]
        if len(moved):
            f_last[moved], g_last[moved] = loss_grad(moved)
            x_last[moved] = x[moved, 0]
            for i in moved.tolist():
                nfev[i] += 1
        for i, f_i in zip(waiting, f_last[idx].tolist(), strict=True):
            f[i] = f_i
        g[idx, 0] = g_last[idx]
        active = waiting

    converged = task[:, 0] == 4
    dispersions = np.exp(x[:, 0])
    for i in np.flatnonzero(~converged):  # type: ignore[assignment]
        dispersions[i] = np.exp(
            grid_fit_alpha(
                counts[i], design_matrix, mu[i], alpha_hat[i], min_disp, max_disp
            )
        )
    return dispersions, converged


def fit_rough_dispersions(
    normed_counts: np.ndarray, design_matrix: pd.DataFrame
) -> np.ndarray:
    """Rough dispersion estimates from linear model, as per the R code.

    Used as initial estimates in :meth:`DeseqDataSet.fit_genewise_dispersions() <pydeseq2.dds.DeseqDataSet.fit_genewise_dispersions>`.

    Parameters
    ----------
    normed_counts
        Array of deseq2-normalized read counts.
        Rows: samples, columns: genes.
    design_matrix
        A DataFrame with experiment design information (to split cohorts).
        Indexed by sample barcodes.
        Unexpanded, *with* intercept.

    Returns
    -------
    Estimated dispersion parameter for each gene.
    """
    num_samples, num_vars = design_matrix.shape
    # This method is only possible when num_samples > num_vars.
    # If this is not the case, throw an error.
    if num_samples == num_vars:
        raise ValueError(
            "The number of samples and the number of design variables are "
            "equal, i.e., there are no replicates to estimate the "
            "dispersion. Please use a design with fewer variables."
        )

    reg = LinearRegression(fit_intercept=False)
    reg.fit(design_matrix, normed_counts)
    y_hat = reg.predict(design_matrix)
    y_hat = np.maximum(y_hat, 1)
    alpha_rde = (
        ((normed_counts - y_hat) ** 2 - y_hat) / ((num_samples - num_vars) * y_hat**2)
    ).sum(axis=0)
    return np.maximum(alpha_rde, 0)


def fit_moments_dispersions(
    normed_counts: np.ndarray, size_factors: np.ndarray
) -> np.ndarray:
    """Dispersion estimates based on moments, as per the R code.

    Used as initial estimates in :meth:`DeseqDataSet.fit_genewise_dispersions() <pydeseq2.dds.DeseqDataSet.fit_genewise_dispersions>`.

    Parameters
    ----------
    normed_counts
        Array of deseq2-normalized read counts.
        Rows: samples, columns: genes.
    size_factors
        DESeq2 normalization factors.

    Returns
    -------
    Estimated dispersion parameter for each gene.
    """
    # Exclude genes with all zeroes
    normed_counts = normed_counts[:, ~(normed_counts == 0).all(axis=0)]
    # mean inverse size factor
    s_mean_inv = (1 / size_factors).mean(axis=0)
    mu = normed_counts.mean(0)
    sigma = normed_counts.var(0, ddof=1)
    # ddof=1 is to use an unbiased estimator, as in R
    # NaN (variance = 0) are replaced with 0s
    return np.nan_to_num((sigma - s_mean_inv * mu) / mu**2)


def robust_method_of_moments_disp(
    normed_counts: np.ndarray, design_matrix: pd.DataFrame
) -> np.ndarray:
    """Perform dispersion estimation using a method of trimmed moments.

    Used for outlier detection based on Cook's distance.

    Parameters
    ----------
    normed_counts
        Array of deseq2-normalized read counts.
        Rows: samples, columns: genes.
    design_matrix
        A DataFrame with experiment design information (to split cohorts).
        Indexed by sample barcodes.
        Unexpanded, *with* intercept.

    Returns
    -------
    Trimmed method of moment dispersion estimates.
    Used for outlier detection based on Cook's distance.
    """
    # if there are 3 or more replicates in any cell
    three_or_more = n_or_more_replicates(design_matrix, 3)
    if three_or_more.any():
        # 1 - group rows by unique combinations of design factors
        # 2 - keep only groups with 3 or more replicates
        # 3 - filter the counts matrix to only keep rows in those groups
        filtered_counts = normed_counts[three_or_more.to_numpy(), :]
        filtered_design = design_matrix.loc[three_or_more, :]
        cell_id = pd.Series(
            filtered_design.groupby(filtered_design.columns.values.tolist()).ngroup(),
            index=filtered_design.index,
        )
        v = trimmed_cell_variance(filtered_counts, cell_id)
    else:
        v = cast(
            np.ndarray, trimmed_variance(normed_counts)
        )  # Since normed_counts is always 2D, trimmed_variance returns ndarray

    m = normed_counts.mean(axis=0)
    alpha = (v - m) / m**2
    # cannot use the typical min_disp = 1e-8 here or else all counts in the same
    # group as the outlier count will get an extreme Cook's distance
    minDisp = 0.04
    np.maximum(alpha, minDisp, out=alpha)
    return alpha


def _robust_moments_plan(design_matrix: pd.DataFrame):
    """Precompute the design-only part of :func:`robust_method_of_moments_disp`.

    Returns ``None`` when no sample has 3 or more replicates, else the
    replicated-sample mask, one ``(mask, trim, scale)`` per cohort as
    :func:`trimmed_cell_variance` sets them, and the cohort count.
    """
    three_or_more = n_or_more_replicates(design_matrix, 3)
    if not three_or_more.any():
        return None
    filtered_design = design_matrix.loc[three_or_more, :]
    cells = pd.Series(
        filtered_design.groupby(filtered_design.columns.values.tolist()).ngroup(),
        index=filtered_design.index,
    )
    trimratio = (1 / 3, 1 / 4, 1 / 8)
    scales = (2.04, 1.86, 1.51)
    ns = cells.value_counts()
    groups = []
    for lvl in cells.unique():
        k = 2 if ns[lvl] >= 23.5 else 1 if ns[lvl] >= 3.5 else 0
        groups.append(((cells == lvl).to_numpy(), trimratio[k], scales[k]))
    return three_or_more.to_numpy(), groups, len(ns)


def _robust_moments_disp_columns(normed_counts: np.ndarray, plan) -> np.ndarray:
    """Compute :func:`robust_method_of_moments_disp` on a block of genes.

    The design part is precomputed by :func:`_robust_moments_plan`; every gene's
    estimate depends on its own column only.
    """
    if plan is None:
        v = cast(np.ndarray, trimmed_variance(normed_counts))
    else:
        rows, groups, n_cells = plan
        counts = normed_counts[rows, :]
        sqerror = np.zeros_like(counts)
        for mask, trim, _ in groups:
            cell_means = cast(
                np.ndarray, trimmed_mean(counts[mask, :], trim=trim, axis=0)
            )
            sqerror[mask, :] = counts[mask, :] - cell_means[None, :]
        sqerror **= 2
        var_est = np.zeros((n_cells, counts.shape[1]), dtype=float)
        for i, (mask, trim, scale) in enumerate(groups):
            var_est[i, :] = scale * trimmed_mean(sqerror[mask, :], trim=trim, axis=0)
        v = var_est.max(axis=0)
    m = normed_counts.mean(axis=0)
    alpha = (v - m) / m**2
    np.maximum(alpha, 0.04, out=alpha)
    return alpha
