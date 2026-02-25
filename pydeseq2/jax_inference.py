"""PyDESeq2 inference with a jax backend."""

import functools
from collections.abc import Callable
from typing import Any
from typing import Literal

import chex
import jax
import jax.numpy as jnp
import jax.scipy
import numpy as np
import optax
import pandas as pd

from pydeseq2 import inference


# Jax lstsq can be inaccurate https://github.com/google/jax/issues/11433
def _lstsq(a, b):
    return jnp.linalg.solve(a.T @ a, a.T @ b)


def _maybe_cast_to_float64(x):
    return x.astype(jnp.float64) if jax.config.x64_enabled else x


def _optax_solver(
    init_params: chex.Array,
    fun: Callable[..., chex.Array],
    opt: optax.GradientTransformationExtraArgs,
    max_iter: int,
    g_tol: float = 1e-5,
) -> tuple[chex.Array, chex.Array]:
    """Interface to minimize using optax."""
    value_and_grad_fun = optax.value_and_grad_from_state(fun)

    def step(carry):
        params, state = carry
        value, grad = value_and_grad_fun(params, state=state)
        updates, state = opt.update(
            _maybe_cast_to_float64(grad),
            state,
            params,
            value=value,
            grad=grad,
            value_fn=fun,
        )
        params = optax.apply_updates(params, updates)
        return params, state

    def continuing_criterion(carry):
        _, state = carry
        iter_num = optax.tree_utils.tree_get(state, "count")
        grad = optax.tree_utils.tree_get(state, "grad")
        grad_error = optax.tree_utils.tree_norm(grad, ord=jnp.inf)
        return (iter_num == 0) | ((iter_num < max_iter) & (grad_error >= g_tol))

    init_carry = (init_params, opt.init(init_params))
    final_params, final_state = jax.lax.while_loop(
        continuing_criterion, step, init_carry
    )
    converged = jnp.asarray(
        optax.tree_utils.tree_norm(
            optax.tree_utils.tree_get(final_state, "grad"), ord=jnp.inf
        )
        < g_tol
    )
    return final_params, converged


def _minimize(
    fun: Callable[..., chex.Array],
    x0: chex.Array,
    g_tol: float = 1e-5,
    max_iter: int = 1_000,
) -> tuple[chex.Array, chex.Array]:
    """Interface to minimize using optax."""
    x0_hat, success = _optax_solver(
        init_params=x0,
        fun=fun,
        opt=optax.lbfgs(),
        max_iter=max_iter,
        g_tol=g_tol,
    )
    return x0_hat, success


def _nb_loss_constant_terms(
    counts: chex.Array, dispersion: float | chex.Array
) -> float | chex.Array:
    """Compute the constant part of the NB NLL (w.r.t mean)."""
    n = len(counts)
    dispersion_neg1 = 1 / dispersion
    logbinom = (
        jax.scipy.special.gammaln(counts + dispersion_neg1)
        - jax.scipy.special.gammaln(counts + 1)
        - jax.scipy.special.gammaln(dispersion_neg1)
    )
    return n * dispersion_neg1 * jnp.log(dispersion) - logbinom.sum()


def _nb_loss_variable_terms(
    counts: chex.Array, mean: chex.Array, dispersion: float | chex.Array
) -> float | chex.Array:
    """Compute the variable part of the NB NLL (w.r.t mean)."""
    dispersion_neg1 = 1 / dispersion
    return (
        (counts + dispersion_neg1) * jnp.log(dispersion_neg1 + mean)
        - counts * jnp.log(mean)
    ).sum()


@jax.jit
def _negative_binomial_loss(
    counts: chex.Array, mean: chex.Array, dispersion: float | chex.Array
) -> float | chex.Array:
    """Negative log-likelihood of a negative binomial for one gene.

    Matches the implementation in pydeseq2.utils.nb_nll.

    Parameters
    ----------
    counts
        Observations.
    mean
        Mean of the distribution.
    dispersion
        Dispersion of the distribution (alpha).

    Returns
    -------
    Negative log likelihood of the observations counts.
    """
    return _nb_loss_constant_terms(counts, dispersion) + _nb_loss_variable_terms(
        counts, mean, dispersion
    )


def _compute_mu(
    beta: chex.Array,
    x: chex.Array,
    size_factors: chex.Array,
    min_mu: float,
) -> chex.Array:
    """Compute mu from beta, design matrix, size factors and min_mu."""
    return jnp.maximum(size_factors * jnp.exp(x @ beta), min_mu)


@chex.dataclass(frozen=True, kw_only=True)
class _IRLSGeneArgs:
    """Arguments to _irls_loop that vary per gene.

    Attributes
    ----------
    beta
        Initial estimate of GLM coefficients.
    counts
        Observations.
    mu
        Mean of the distribution.
    disp
        Dispersion of the distribution (alpha).
    """

    beta: chex.Array
    counts: chex.Array
    mu: chex.Array
    disp: chex.Array


@chex.dataclass(frozen=True, kw_only=True)
class _IRLSGlobalArgs:
    """Arguments to _irls_loop that are global across genes.

    Attributes
    ----------
    x
        Design matrix.
    ridge_factor
        Regularization factor.
    size_factors
        Sample-wise scaling factors.
    min_mu
        Lower threshold for fitted means, for numerical stability.
    beta_tol
        Tolerance for beta convergence.
    max_beta
        Maximum beta value.
    maxiter
        Maximum number of iterations.
    lbfgs_after_irls
        Whether to use L-BFGS after IRLS.
    """

    x: chex.Array
    ridge_factor: chex.Array
    size_factors: chex.Array
    min_mu: float
    beta_tol: float
    max_beta: float
    maxiter: int
    lbfgs_after_irls: bool


@chex.dataclass(frozen=True, kw_only=True)
class _IRLSLoopState:
    """State of the IRLS loop.

    Attributes
    ----------
    beta_hat
        Current estimate of GLM coefficients.
    old_dev
        Previous deviation from target.
    dev
        Current deviation from target.
    i
        Current iteration number.
    counts
        Observations.
    mu
        Mean of the distribution.
    disp
        Dispersion of the distribution (alpha).
    x
        Design matrix.
    """

    beta_hat: chex.Array
    old_dev: chex.Array
    dev: chex.Array
    i: chex.Array
    counts: chex.Array
    mu: chex.Array
    disp: chex.Array
    x: chex.Array


@chex.dataclass(frozen=True, kw_only=True)
class _BacktrackState:
    """State of the backtracking line search in the IRLS loop.

    Attributes
    ----------
    beta
        Current estimate of GLM coefficients.
    mu
        Mean of the distribution.
    loss
        Current loss.
    step_mult
        Current step size multiplier.
    done
        Whether the backtracking line search is done.
    """

    beta: chex.Array
    mu: chex.Array
    loss: chex.Array
    step_mult: float
    done: bool


def _irls_loop(
    gene_args: _IRLSGeneArgs,
    global_args: _IRLSGlobalArgs,
) -> tuple[chex.Array, chex.Array, chex.Array]:
    """Run the internal reweighted least squares loop with Backtracking Line Search.

    This uses a step-halving approach, reducing the step size by half each
    iteration if the loss function does not decrease. The goal is to improve
    convergence of the IRLS loop to avoid needing LBFGS after IRLS.

    Parameters
    ----------
    gene_args
        Arguments that vary per gene.
    global_args
        Arguments that are global across genes.

    Returns
    -------
    beta_hat
        Estimated GLM coefficients.
    mu
        Estimated means.
    success
        Whether the optimization converged.
    """
    size_factors = global_args.size_factors.ravel()
    chex.assert_equal_shape([gene_args.mu, gene_args.counts, size_factors])
    chex.assert_rank(gene_args.mu, 1)

    nll_const = _nb_loss_constant_terms(gene_args.counts, gene_args.disp[0])

    def _compute_loss(beta, mu):
        deviance = 2 * (
            nll_const + _nb_loss_variable_terms(gene_args.counts, mu, gene_args.disp[0])
        )
        ridge_loss = (beta**2 * jnp.diag(global_args.ridge_factor)).sum()
        return deviance + ridge_loss

    def cond_fun(val: _IRLSLoopState):
        dev, old_dev = val.dev, val.old_dev
        dev_ratio = jnp.abs(dev - old_dev) / (jnp.abs(dev) + 0.1)

        cond1 = dev_ratio > global_args.beta_tol
        cond2 = jnp.sum(jnp.abs(val.beta_hat) > global_args.max_beta) == 0
        cond3 = val.i < global_args.maxiter
        return jnp.logical_and(jnp.logical_and(cond1, cond2), cond3)

    def body_fun(val: _IRLSLoopState) -> _IRLSLoopState:
        w = val.mu / (1.0 + val.mu * val.disp)
        z = jnp.log(val.mu) - jnp.log(size_factors) + (val.counts - val.mu) / val.mu

        h = (val.x.T * w) @ val.x + global_args.ridge_factor

        target_beta = jax.scipy.linalg.solve(h, val.x.T @ (w * z), assume_a="pos")
        step_direction = target_beta - val.beta_hat

        # Backtracking line search
        init_mu = _compute_mu(target_beta, val.x, size_factors, global_args.min_mu)
        init_loss = _compute_loss(target_beta, init_mu)

        # Continue stepping back if loss increased AND step size is not tiny
        def backtrack_cond(bs: _BacktrackState):
            return jnp.logical_and(bs.loss >= val.dev, bs.step_mult > 1e-10)

        # Halve the step size and re-evaluate
        def backtrack_body(bs: _BacktrackState):
            new_mult = bs.step_mult * 0.5
            new_beta = val.beta_hat + new_mult * step_direction
            new_mu = _compute_mu(new_beta, val.x, size_factors, global_args.min_mu)
            new_loss = _compute_loss(new_beta, new_mu)
            return _BacktrackState(  # type: ignore[call-arg]
                beta=new_beta,
                mu=new_mu,
                loss=new_loss,
                step_mult=new_mult,
                done=False,
            )

        init_bs = _BacktrackState(  # type: ignore[call-arg]
            beta=target_beta, mu=init_mu, loss=init_loss, step_mult=1.0, done=False
        )

        final_bs = jax.lax.while_loop(backtrack_cond, backtrack_body, init_bs)

        return _IRLSLoopState(  # type: ignore[call-arg]
            beta_hat=final_bs.beta,
            old_dev=val.dev,
            dev=final_bs.loss,
            i=val.i + 1,
            counts=val.counts,
            mu=final_bs.mu,
            disp=val.disp,
            x=val.x,
        )

    # Compute initial mu and loss for the starting beta
    init_mu_val = _compute_mu(
        gene_args.beta, global_args.x, size_factors, global_args.min_mu
    )
    init_loss_val = _compute_loss(gene_args.beta, init_mu_val)

    init_val = _IRLSLoopState(  # type: ignore[call-arg]
        beta_hat=gene_args.beta,
        old_dev=init_loss_val + 1e9,  # Force at least one iteration
        dev=init_loss_val,
        i=jnp.array(0),
        counts=gene_args.counts,
        mu=init_mu_val,
        disp=gene_args.disp,
        x=global_args.x,
    )

    # Run the main IRLS loop
    val = jax.lax.while_loop(cond_fun, body_fun, init_val)

    # L-BFGS fallback logic
    continue_cond = jnp.logical_or(
        jnp.sum(jnp.abs(val.beta_hat) > global_args.max_beta) > 0,
        val.i >= global_args.maxiter,
    )

    def loss_fn(beta):
        mu_ = _compute_mu(beta, global_args.x, size_factors, global_args.min_mu)
        return _compute_loss(beta, mu_)

    def continue_opt():
        return _minimize(loss_fn, x0=gene_args.beta)

    def extract_from_irls():
        return val.beta_hat, jnp.logical_not(continue_cond)

    if global_args.lbfgs_after_irls:
        beta_hat, success = jax.lax.cond(continue_cond, continue_opt, extract_from_irls)
    else:
        beta_hat, success = extract_from_irls()

    # Final mu calculation
    mu = _compute_mu(beta_hat, global_args.x, size_factors, global_args.min_mu)

    return beta_hat, mu, success


def _vmapped_irls(
    beta: chex.Array,
    counts: chex.Array,
    x: chex.Array,
    mu: chex.Array,
    disp: chex.Array,
    ridge_factor: chex.Array,
    size_factors: chex.Array,
    min_mu: float,
    beta_tol: float,
    max_beta: float,
    maxiter: int,
    lbfgs_after_irls: bool,
) -> tuple[chex.Array, chex.Array, chex.Array]:
    """Vmapped reweighted least squares."""
    gene_args = _IRLSGeneArgs(beta=beta, counts=counts, mu=mu, disp=disp[None, :])  # type: ignore[call-arg]
    global_args = _IRLSGlobalArgs(  # type: ignore[call-arg]
        x=x,
        ridge_factor=ridge_factor,
        size_factors=size_factors,
        min_mu=min_mu,
        beta_tol=beta_tol,
        max_beta=max_beta,
        maxiter=maxiter,
        lbfgs_after_irls=lbfgs_after_irls,
    )
    beta_, mu_, converged_ = jax.vmap(
        _irls_loop,
        in_axes=(1, None),
        out_axes=(1, 1, 0),
    )(gene_args, global_args)
    return beta_, mu_, converged_


def _compute_hat_matrix_diagonal(
    mu: chex.Array,
    disp: chex.Array,
    x: chex.Array,
    ridge_factor: chex.Array,
) -> chex.Array:
    """Compute H diagonal (useful for Cook distance outlier filtering)."""
    w = mu / (1.0 + mu * disp)
    h = jnp.einsum(
        "ij,jk,ki->i",
        x,
        jnp.linalg.inv((x.T * w[None, :]) @ x + ridge_factor),
        x.T,
    )
    w_sq = jnp.sqrt(w)
    return w_sq * h * w_sq


@functools.partial(jax.jit, static_argnames=["lbfgs_after_irls"])
def _irls_solver(
    counts: chex.Array,
    size_factors: chex.Array,
    design_matrix: chex.Array,
    disp: chex.Array,
    min_mu: float = 0.5,
    beta_tol: float = 1e-8,
    max_beta: float = 30.0,
    maxiter: int = 250,
    lbfgs_after_irls: bool = True,
) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    """Jax implementation of the IRLS solver."""
    num_vars = design_matrix.shape[1]
    num_samples = design_matrix.shape[0]
    num_genes = counts.shape[1]
    x = design_matrix
    size_factors = size_factors[:, None]
    chex.assert_shape(disp, (num_genes,))

    rank_diff = jnp.linalg.matrix_rank(x) - num_vars

    def full_rank_init():
        q, r = jax.numpy.linalg.qr(x)
        y = jnp.log(counts / size_factors + 0.1)
        return jax.scipy.linalg.solve(r, q.T @ y)

    def not_full_rank_init():
        beta = jnp.zeros((num_vars, num_genes))
        return beta.at[0].set(jnp.log(counts / size_factors).mean(0))

    beta = jax.lax.cond(rank_diff == 0.0, full_rank_init, not_full_rank_init)
    chex.assert_shape(beta, (num_vars, num_genes))

    ridge_factor = jnp.diag(jnp.full_like(x[0], 1e-6))
    mu = _compute_mu(beta, x, size_factors, min_mu)
    chex.assert_shape(mu, (num_samples, num_genes))

    beta, mu, converged = _vmapped_irls(
        beta=beta,
        counts=counts,
        x=x,
        mu=mu,
        disp=disp,
        ridge_factor=ridge_factor,
        size_factors=size_factors,
        min_mu=min_mu,
        beta_tol=beta_tol,
        max_beta=max_beta,
        maxiter=maxiter,
        lbfgs_after_irls=lbfgs_after_irls,
    )
    chex.assert_shape(beta, (num_vars, num_genes))
    chex.assert_shape(mu, (num_samples, num_genes))

    h = jax.vmap(_compute_hat_matrix_diagonal, in_axes=(1, 0, None, None))(
        mu, disp, x, ridge_factor
    )
    mu = size_factors * jnp.exp(x @ beta)

    return (
        beta.T,
        mu,
        h.T,
        converged,
    )


@jax.jit
def _alpha_mle_loss(
    log_alpha: jnp.ndarray,
    log_alpha_hat: jnp.ndarray,
    counts: jnp.ndarray,
    design_matrix: jnp.ndarray,
    mu: jnp.ndarray,
    cr_reg: bool,
    prior_reg: bool,
    prior_disp_var: float,
) -> jnp.ndarray:
    """Loss to minimize for fit alpha mle."""
    # A leading dimension is necessary for optimization.
    log_alpha = log_alpha[0]
    alpha = jnp.exp(log_alpha)

    w = mu / (1 + mu * alpha)
    mat = (design_matrix.T * w) @ design_matrix

    # There are a few ways to compute the log determinant. Here we use an approach
    # that leverages the SPD nature of the matrix and is numerically more stable.
    def cr_reg_fn_cholesky():
        # https://github.com/pytorch/pytorch/issues/22848#issuecomment-1032737956
        return jnp.log(
            jnp.diagonal(
                jax.scipy.linalg.cholesky(mat + 1e-10, lower=True),
                axis1=-2,
                axis2=-1,
            )
        ).sum()

    def zero_array():
        return jnp.array(0.0)

    reg = jax.lax.cond(cr_reg, cr_reg_fn_cholesky, zero_array)

    def prior_reg_fn():
        reg = (log_alpha - log_alpha_hat) ** 2 / (2 * prior_disp_var)
        return reg

    reg = reg + jax.lax.cond(prior_reg, prior_reg_fn, zero_array)

    return _negative_binomial_loss(counts, mu, alpha) + reg


@functools.partial(jax.jit, static_argnames=["jointly_fit_genes"])
def _fit_alpha_mle(
    counts: chex.Array,
    design_matrix: chex.Array,
    mu: chex.Array,
    alpha_hat: chex.Array,
    prior_disp_var: float = 1.0,
    cr_reg: bool = True,
    prior_reg: bool = False,
    jointly_fit_genes: bool = True,
) -> tuple[chex.Array, chex.Array]:
    """Estimate the dispersion parameter of a negative binomial GLM.

    Notes
    -----
    This jointly optimizes over all genes at once and only supports
    LBFGS optimizer.

    Parameters
    ----------
    counts
        Raw counts.
    design_matrix
        Design matrix.
    mu
        Mean estimation for the NB model.
    alpha_hat
        Initial dispersion estimate.
    prior_disp_var
        Prior dispersion variance.
    cr_reg
        Whether to use Cox-Reid regularization. (default: ``True``).
    prior_reg
        Whether to use prior log-residual regularization. (default: ``False``).
    jointly_fit_genes
        Whether to combine all gene-wise problems into one single
        optimization problem.

    Returns
    -------
    Dispersion estimate and whether optimization converged.
    """
    log_alpha_hat = jnp.log(alpha_hat)

    if jointly_fit_genes:
        loss = functools.partial(
            _alpha_mle_loss,
            prior_disp_var=prior_disp_var,
            cr_reg=cr_reg,
            prior_reg=prior_reg,
        )

        def vmap_run(pos):
            pos = pos[None, :]
            out = jax.vmap(loss, in_axes=(1, 0, 1, None, 1))(
                pos, log_alpha_hat, counts, design_matrix, mu
            )
            return out.sum()

        res = _minimize(vmap_run, x0=jnp.log(alpha_hat))
        log_alpha_hat_sol, success = res
        # Broadcast the one optimization result across all genes
        success = jnp.full_like(log_alpha_hat_sol, fill_value=success)

    else:

        def run(pos, log_alpha_hat, counts, design_matrix, mu):
            def loss(pos):
                return _alpha_mle_loss(
                    log_alpha=pos,
                    log_alpha_hat=log_alpha_hat,
                    counts=counts,
                    design_matrix=design_matrix,
                    mu=mu,
                    cr_reg=cr_reg,
                    prior_reg=prior_reg,
                    prior_disp_var=prior_disp_var,
                )

            return _minimize(loss, x0=pos)

        log_alpha_hat_sol, success = jax.vmap(run, in_axes=(0, 0, 1, None, 1))(
            log_alpha_hat[:, None], log_alpha_hat, counts, design_matrix, mu
        )

    return jnp.exp(log_alpha_hat_sol).ravel(), success.ravel()


@jax.jit
def _fit_lin_mu(
    counts: chex.Array,
    size_factors: chex.Array,
    design_matrix: chex.Array,
    min_mu: float = 0.5,
) -> chex.Array:
    """Estimate mean of negative binomial model using a linear regression.

    Used to initialize genewise dispersion models.

    Parameters
    ----------
    counts
        Raw counts.
    size_factors
        Sample-wise scaling factors (obtained from median-of-ratios).
    design_matrix
        Design matrix.
    min_mu
        Lower threshold for fitted means, for numerical stability. (default:
        ``0.5``).

    Returns
    -------
    Estimated mean.
    """
    y = counts / size_factors[:, None]
    coef = _lstsq(design_matrix, y)
    mu_hat = size_factors[:, None] * (design_matrix @ coef)
    # Threshold mu_hat as 1/mu_hat will be used later on.
    return jnp.maximum(mu_hat, min_mu)


@chex.dataclass(frozen=True, kw_only=True)
class _WaldTestGeneArgs:
    """Arguments to _wald_test_single that vary per gene.

    Attributes
    ----------
    disp
        Dispersion estimate.
    lfc
        Log-fold change estimate.
    mu
        Mean estimation for the NB model.
    """

    disp: chex.Array
    lfc: chex.Array
    mu: chex.Array


@chex.dataclass(frozen=True, kw_only=True)
class _WaldTestGlobalArgs:
    """Arguments to _wald_test_single that are global across genes.

    Attributes
    ----------
    design_matrix
        Design matrix.
    ridge_factor
        Regularization factors.
    contrast
        Vector encoding the contrast that is being tested.
    lfc_null
        The log fold change under the null hypothesis.
    """

    design_matrix: chex.Array
    ridge_factor: chex.Array
    contrast: chex.Array
    lfc_null: chex.Array


def _wald_test_single(
    gene_args: _WaldTestGeneArgs,
    global_args: _WaldTestGlobalArgs,
) -> tuple[chex.Array, chex.Array, chex.Array]:
    """Run Wald test for a single gene."""
    # Build covariance matrix estimator
    w = gene_args.mu / (1 + gene_args.mu * gene_args.disp)
    m = (global_args.design_matrix.T * w[None, :]) @ global_args.design_matrix
    h = jnp.linalg.inv(m + global_args.ridge_factor)
    hc = h @ global_args.contrast
    # Evaluate standard error and Wald statistic
    wald_se = jnp.sqrt(hc.T @ m @ hc)
    wald_statistic = (
        global_args.contrast @ (gene_args.lfc - global_args.lfc_null) / wald_se
    )
    wald_p_value = 2 * jax.scipy.stats.norm.sf(jnp.abs(wald_statistic))

    return wald_p_value, wald_statistic, wald_se


@jax.jit
def _wald_test(
    design_matrix: chex.Array,
    disp: chex.Array,
    lfc: chex.Array,
    mu: chex.Array,
    ridge_factor: chex.Array,
    contrast: chex.Array,
    lfc_null: chex.Array,
) -> tuple[chex.Array, chex.Array, chex.Array]:
    """Run Wald test for differential expression.

    Computes Wald statistics, standard error and p-values from
    dispersion and LFC estimates.

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

    Returns
    -------
    wald_p_value
        Estimated p-value.
    wald_statistic
        Wald statistic.
    wald_se
        Standard error of the Wald statistic.
    """
    gene_args = _WaldTestGeneArgs(disp=disp, lfc=lfc, mu=mu.transpose())  # type: ignore[call-arg]
    global_args = _WaldTestGlobalArgs(  # type: ignore[call-arg]
        design_matrix=design_matrix,
        ridge_factor=ridge_factor,
        contrast=contrast,
        lfc_null=lfc_null,
    )
    return jax.vmap(
        _wald_test_single,
        in_axes=(0, None),
    )(gene_args, global_args)


@jax.jit
def _fit_rough_dispersions(
    normed_counts: chex.Array, design_matrix: chex.Array
) -> chex.Array:
    """Rough dispersion estimates from linear model, as per the R code.

    Used as initial estimates in `fit_genewise_dispersions()`

    Parameters
    ----------
    normed_counts
        Array of deseq2-normalized read counts. Rows are samples, columns are
        genes.
    design_matrix
        Array with experiment design information (to split cohorts). Indexed by
        sample barcodes. Unexpanded, *with* intercept.

    Returns
    -------
    Estimated dispersion parameter for each gene.
    """
    num_samples, num_vars = design_matrix.shape
    coef = _lstsq(design_matrix, normed_counts)
    y_hat = design_matrix @ coef
    y_hat = jnp.maximum(y_hat, 1)
    alpha_rde = (
        ((normed_counts - y_hat) ** 2 - y_hat) / ((num_samples - num_vars) * y_hat**2)
    ).sum(0)
    return jnp.maximum(alpha_rde, 0)


@jax.jit
def _fit_moments_dispersions(
    normed_counts: chex.Array, size_factors: chex.Array
) -> chex.Array:
    """Jax-based disperstion estimate based on moments."""
    # mean inverse size factor
    s_mean_inv = (1 / size_factors).mean()
    mu = normed_counts.mean(0)
    sigma = normed_counts.var(0, ddof=1)
    # ddof=1 is to use an unbiased estimator, as in R
    # NaN (variance = 0) are replaced with 0s
    return jnp.nan_to_num((sigma - s_mean_inv * mu) / mu**2)


@chex.dataclass(frozen=True, kw_only=True)
class _NbinomFnGeneArgs:
    counts: chex.Array
    size: chex.Array


@chex.dataclass(frozen=True, kw_only=True)
class _NbinomFnGlobalArgs:
    design_matrix: chex.Array
    offset: chex.Array
    prior_no_shrink_scale: float
    prior_scale: float
    shrink_index: int


@jax.jit
def _nbinom_fn(
    beta: chex.Array,
    gene_args: _NbinomFnGeneArgs,
    global_args: _NbinomFnGlobalArgs,
) -> chex.Array:
    """Return the NB negative likelihood with apeGLM prior for one gene.

    Use for LFC shrinkage.

    Parameters
    ----------
    beta
        2-element array: intercept and LFC coefficients.
    gene_args
        Arguments that vary per gene.
    global_args
        Arguments that are global across genes.

    Returns
    -------
    Sum of the NB negative likelihood and apeGLM prior.
    """
    beta = beta.squeeze()
    num_vars = global_args.design_matrix.shape[-1]

    shrink_mask = jnp.zeros(num_vars)
    shrink_mask = shrink_mask.at[global_args.shrink_index].set(1)
    no_shrink_mask = jnp.ones(num_vars) - shrink_mask

    xbeta = global_args.design_matrix @ beta
    prior = (
        (beta * no_shrink_mask) ** 2 / (2 * global_args.prior_no_shrink_scale**2)
    ).sum() + jnp.log1p((beta[global_args.shrink_index] / global_args.prior_scale) ** 2)

    # Use softplus for potentially better numerical stability
    # logaddexp(a, b) = b + softplus(a - b)
    # Here a = xbeta + offset, b = log(size)
    log_size = jnp.log(gene_args.size)
    log_likelihood_term = log_size + jax.nn.softplus(
        xbeta + global_args.offset - log_size
    )
    nll = (
        gene_args.counts * xbeta
        - (gene_args.counts + gene_args.size) * log_likelihood_term
    ).sum(0)

    return prior - nll


@functools.partial(jax.jit, static_argnames=["shrink_index"])
def _nbinom_glm(
    design_matrix: chex.Array,
    counts: chex.Array,
    size: chex.Array,
    offset: chex.Array,
    prior_no_shrink_scale: float,
    prior_scale: float,
    shrink_index: int = 1,
) -> tuple[chex.Array, chex.Array, chex.Array]:
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
    shrink_index
        Index of the LFC coordinate to shrink. (default: ``1``).

    Returns
    -------
    beta
        2-element array, containing the intercept (first) and the LFC (second).
    inv_hessian
        Inverse of the Hessian of the objective at the estimated MAP LFC.
    converged
        Whether optimization converged.
    """
    num_vars = design_matrix.shape[-1]
    num_genes = counts.shape[1]

    shrink_mask = jnp.zeros(num_vars)
    shrink_mask = shrink_mask.at[shrink_index].set(1)
    no_shrink_mask = jnp.ones(num_vars) - shrink_mask

    # Set optimization scale
    nbinom_gene_args = _NbinomFnGeneArgs(counts=counts, size=size[None, :])  # type: ignore[call-arg]
    nbinom_global_args = _NbinomFnGlobalArgs(  # type: ignore[call-arg]
        design_matrix=design_matrix,
        offset=offset,
        prior_no_shrink_scale=prior_no_shrink_scale,
        prior_scale=prior_scale,
        shrink_index=shrink_index,
    )
    scale_cnst = jax.vmap(
        _nbinom_fn,
        in_axes=(None, 1, None),
    )(
        jnp.zeros(num_vars),
        nbinom_gene_args,
        nbinom_global_args,
    )
    scale_cnst = jnp.maximum(scale_cnst, 1)

    def run(beta, design_matrix, counts, size, offset, scale_cnst):
        gene_args = _NbinomFnGeneArgs(counts=counts, size=size)
        global_args = _NbinomFnGlobalArgs(
            design_matrix=design_matrix,
            offset=offset,
            prior_no_shrink_scale=prior_no_shrink_scale,
            prior_scale=prior_scale,
            shrink_index=shrink_index,
        )

        def loss(beta):
            loss = (
                _nbinom_fn(
                    beta=beta,
                    gene_args=gene_args,
                    global_args=global_args,
                )
                / scale_cnst
            )
            return _maybe_cast_to_float64(loss)

        res = _minimize(loss, x0=beta, g_tol=1e-8)
        return res

    beta_init = (
        jnp.ones((num_vars, num_genes)) * 0.1 * (-1) ** (jnp.arange(num_vars)[:, None])
    )
    out = jax.vmap(run, in_axes=(1, None, 1, 0, None, 0))(
        beta_init, design_matrix, counts, size, offset, scale_cnst
    )

    beta, converged = out[0].T, jnp.asarray(out[1]).ravel()

    def hessian(
        beta: chex.Array, cnst: chex.Array, size: chex.Array, counts: chex.Array
    ) -> chex.Array:
        """Hessian of the function to optimize."""
        # Note: will only work if there is a single shrink index
        beta = beta.squeeze()
        xbeta = design_matrix @ beta
        exp_xbeta_off = jnp.exp(xbeta + offset)
        frac = (counts + size) * size * exp_xbeta_off / (size + exp_xbeta_off) ** 2
        # Build diagonal
        h11 = 1 / prior_no_shrink_scale**2
        h22 = (
            2
            * (prior_scale**2 - beta[shrink_index] ** 2)
            / (prior_scale**2 + beta[shrink_index] ** 2) ** 2
        )

        h = jnp.diag(no_shrink_mask * h11 + shrink_mask * h22)

        return 1 / cnst * ((design_matrix.T * frac) @ design_matrix + jnp.diag(h))

    def inv_hessian_fn(kwargs):
        return jnp.linalg.inv(hessian(**kwargs))

    inv_hessian = jax.vmap(
        inv_hessian_fn,
        in_axes=({"beta": 1, "cnst": None, "size": 0, "counts": 1},),
    )({"beta": beta, "cnst": jnp.array(1.0), "size": size, "counts": counts})

    return beta.T, inv_hessian, jnp.full_like(beta[0], fill_value=converged)


def _await_result(out: Any) -> Any:
    """Convert to numpy and block."""
    # Pandas does not play nicely unless explicit np array creation is done.
    return jax.tree_util.tree_map(
        lambda x: np.array(jax.device_get(x.block_until_ready())), out
    )


def _await_result_decorator(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return _await_result(f(*args, **kwargs))

    return wrapper


class JaxInference(inference.Inference):
    """PyDESeq2 inference with a jax backend.

    This is an accelerated backed that works on GPUs and TPUs. This should be used
    in cases where design matrices are very wide.

    Parameters
    ----------
    jointly_fit_genes
        Whether to use lbfgs for each gene or the sum of all genes.
    lbfgs_after_irls
        Whether to switch to lbfgs for genes where irls does not fully
        converge. If True (default behavior), replicates behavior of the
        default inference class. If False, inference will be much faster on a
        hardware accelerator. The LBFGS step is among the slowest steps in the
        pipeline, so this can be a significant speedup, with the tradeoff of not
        fully reproducing the numpy/scipy code behavior.

    Examples
    --------
    >>> import jax
    >>> import pydeseq2.utils
    >>> from pydeseq2.dds import DeseqDataSet
    >>> from pydeseq2.ds import DeseqStats
    >>> from pydeseq2.jax_inference import JaxInference
    >>> jax.config.update("jax_enable_x64", True)
    >>> counts_df = pydeseq2.utils.load_example_data(
    ...     modality="raw_counts", dataset="synthetic"
    ... )
    >>> metadata = pydeseq2.utils.load_example_data(
    ...     modality="metadata", dataset="synthetic"
    ... )
    >>> dds = DeseqDataSet(
    ...     counts=counts_df,
    ...     metadata=metadata,
    ...     design="~condition",
    ...     inference=JaxInference(),
    ... )
    >>> dds.deseq2()
    >>> stats = DeseqStats(
    ...     dds, contrast=["condition", "B", "A"], inference=JaxInference()
    ... )
    >>> stats.summary()

    Notes
    -----
    There are a few key differences in optimization routines used in this
    jax-based version:

    1. LBFGS is always used in place of LBFGS-B.
    2. No grid search routines are used in case of poor convergence.
    3. For mle-based fitting of dispersions, all gene-wise regressions are summed
       into one optimization problem, so no per-gene convergence can be assessed.
       This summing makes optimization considerably faster, but can be disabled
       with the `jointly_fit_genes` option.
    4. The iterative reweighted least squares method uses a step-halving
       backtracking line search algorithm that improves convergence with the goal
       of avoiding expensive LBFGS steps for poorly converging genes.

    For full reproducibility float64 numerics must be activated in jax using
    `jax.config.update("jax_enable_x64", True)`.

    See Also
    --------
    :class:`~pydeseq2.default_inference.DefaultInference`
    """

    def __init__(
        self, jointly_fit_genes: bool = True, lbfgs_after_irls: bool = True
    ) -> None:
        super().__init__()
        self._jointly_fit_genes = jointly_fit_genes
        self._lbfgs_after_irls = lbfgs_after_irls

    lin_reg_mu = staticmethod(_await_result_decorator(_fit_lin_mu))
    fit_rough_dispersions = staticmethod(_await_result_decorator(_fit_rough_dispersions))

    def fit_moments_dispersions(
        self, normed_counts: np.ndarray, size_factors: np.ndarray | pd.Series
    ) -> np.ndarray:
        """Dispersion estimates based on moments, as per the R code.

        Used as initial estimates in `fit_genewise_dispersions()`

        Parameters
        ----------
        normed_counts
            Array of deseq2-normalized read counts. Rows are samples, columns are
            genes.
        size_factors
            DESeq2 normalization factors.

        Returns
        -------
        Estimated dispersion parameter for each gene.
        """
        if isinstance(size_factors, pd.Series):
            size_factors = size_factors.values
        return _await_result(_fit_moments_dispersions(normed_counts, size_factors))

    def irls(  # type: ignore[override]
        self,
        counts: chex.Array,
        size_factors: chex.Array,
        design_matrix: chex.Array,
        disp: chex.Array,
        min_mu: float = 0.5,
        beta_tol: float = 1e-8,
        max_beta: float = 30.0,
        maxiter: int = 250,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fit a NB GLM wit log-link to predict counts from the design matrix.

        See equations (1-2) in the DESeq2 paper.

        Parameters
        ----------
        counts
            Raw counts matrix.
        size_factors
            Sample-wise scaling factors (obtained from median-of-ratios).
        design_matrix
            Design matrix.
        disp
            Gene-wise dispersion prior.
        min_mu
            Lower bound on estimated means, to ensure numerical stability.
            (default: ``0.5``).
        beta_tol
            Stopping criterion for IRWLS.
        max_beta
            Upper-bound on LFC. (default: ``30``).
        maxiter
            Maximum number of IRLS iterations to perform before switching to
            L-BFGS-B. (default: ``250``).

        Returns
        -------
        beta
            Fitted (basemean, lfc) coefficients of negative binomial GLM.
        mu
            Means estimated from size factors and beta.
        H
            Diagonal of the covariance matrix.
        converged
            Whether IRLS or the optimizer converged. If not and if dimension allows
            it, perform grid search.
        """
        return _await_result(
            _irls_solver(
                counts=counts,
                size_factors=size_factors,
                design_matrix=design_matrix,
                disp=disp,
                min_mu=min_mu,
                beta_tol=beta_tol,
                max_beta=max_beta,
                maxiter=maxiter,
                lbfgs_after_irls=self._lbfgs_after_irls,
            )
        )

    def alpha_mle(
        self,
        counts: np.ndarray,
        design_matrix: np.ndarray,
        mu: np.ndarray,
        alpha_hat: np.ndarray,
        min_disp: float,
        max_disp: float,
        prior_disp_var: float | None = None,
        cr_reg: bool = True,
        prior_reg: bool = False,
        optimizer: Literal["BFGS", "L-BFGS-B"] = "L-BFGS-B",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Dispersion maximum likelihood estimation implementation."""
        del min_disp, max_disp
        del optimizer
        if not prior_disp_var:
            if prior_reg:
                raise ValueError("prior_disp_var is required when prior_reg is True.")
            # This is a no-op and jax just needs something to trace.
            # In `_alpha_mle_loss` we divide by prior_disp_var but only enter
            # that branch via jax.lax.cond if prior_reg is True.
            # Since we cannot divide by None, we arbitrarily set to 1.0.
            prior_disp_var = 1.0
        out = _fit_alpha_mle(
            counts=counts,
            design_matrix=design_matrix,
            mu=mu,
            alpha_hat=alpha_hat,
            prior_disp_var=prior_disp_var,
            cr_reg=cr_reg,
            prior_reg=prior_reg,
            jointly_fit_genes=self._jointly_fit_genes,
        )
        return _await_result(out)

    def wald_test(
        self,
        design_matrix: np.ndarray,
        disp: np.ndarray,
        lfc: np.ndarray,
        mu: np.ndarray,
        ridge_factor: np.ndarray,
        contrast: np.ndarray | pd.Series,
        lfc_null: np.ndarray,
        alt_hypothesis: (
            Literal["greaterAbs", "lessAbs", "greater", "less"] | None
        ) = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Wald test implementation."""
        if alt_hypothesis:
            raise NotImplementedError("Wald test alt hypothesis not implemented.")
        if isinstance(contrast, pd.Series):
            contrast = contrast.values
        out = _wald_test(
            design_matrix=design_matrix,
            disp=disp,
            lfc=lfc,
            mu=mu,
            ridge_factor=ridge_factor,
            contrast=contrast,
            lfc_null=lfc_null,
        )
        return _await_result(out)

    def dispersion_trend_gamma_glm(  # noqa: D102
        self, covariates: pd.Series, targets: pd.Series
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        covariates_w_intercept = covariates.to_frame()
        covariates_w_intercept.insert(0, "intercept", 1)
        covariates_fit = covariates_w_intercept.values
        targets_fit = targets.values

        @jax.jit
        def _fit_gamma_glm(covariates_fit, targets_fit):
            def loss(coeffs):
                mu = covariates_fit @ coeffs
                return jnp.nanmean(targets_fit / mu + jnp.log(mu), axis=0)

            res = _minimize(
                loss,
                x0=jnp.array([1.0, 1.0]),
            )
            return res

        coeffs, success = _await_result(_fit_gamma_glm(covariates_fit, targets_fit))
        predictions = covariates_fit @ coeffs

        return coeffs, predictions, success.ravel()[0]

    def lfc_shrink_nbinom_glm(
        self,
        design_matrix: np.ndarray,
        counts: np.ndarray,
        size: np.ndarray,
        offset: np.ndarray,
        prior_no_shrink_scale: float,
        prior_scale: float,
        optimizer: str,
        shrink_index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit a negative binomial MAP LFC using an apeGLM prior."""
        del optimizer
        out = _nbinom_glm(
            design_matrix=design_matrix,
            counts=counts,
            size=size,
            offset=offset,
            prior_no_shrink_scale=prior_no_shrink_scale,
            prior_scale=prior_scale,
            shrink_index=shrink_index,
        )
        return _await_result(out)
