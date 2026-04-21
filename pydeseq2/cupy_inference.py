"""CuPy-backed inference backend for DESeq2.

All gene-wise ops run on GPU via CuPy. The core kernels are warp-per-gene
RawKernels (one 32-thread warp per gene; P×P Hessians and rhs vectors
register-resident; in-register Cholesky solve) compiled per design-matrix
dimension ``p`` via NVRTC:

* ``irls_iter_pN`` — single-iteration IRLS step: warp-fused W/z accumulation,
  normal-equations assemble, Cholesky solve, in-place μ update, NLL reduction
  — one launch per iteration.
* ``alpha_mle_pN`` — per-gene dispersion MLE via 40-iter golden-section search
  on log α, with Cox-Reid regularization and optional Gaussian prior.

Mu-adjacent arrays (counts, mu, NLL constants) are held in ``(g, n)`` F-order
so that the 32 threads of each warp read one cache line per sample step.

``dispersion_trend_gamma_glm`` uses GPU-computed loss/gradient with a scipy
L-BFGS-B optimizer on host (the problem is 2-parameter so the host round-trip
is negligible). ``lfc_shrink_nbinom_glm`` delegates to the CPU backend (apeGLM
is a per-gene MAP with a non-trivial prior Hessian; not batched here).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from pydeseq2 import inference
from pydeseq2.default_inference import DefaultInference


def _require_cupy():
    try:
        import cupy as cp
    except ImportError as e:
        raise ImportError(
            "CuPyInference requires cupy. Install a CUDA-matched wheel, e.g. "
            "`pip install cupy-cuda12x`."
        ) from e
    return cp


# --------------------------------------------------------------------------- #
# RawKernel: full IRLS iteration, one warp (32 threads) per gene.             #
#                                                                             #
# Grid = (num_genes,). Block = (32,). Each warp's 32 threads collectively     #
# process one gene: the n-axis loop is chunked across the warp so each thread #
# handles ceil(n/32) samples. H (PxP) and rhs (P) accumulators are warp-sum-  #
# reduced via __shfl_xor_sync. All 32 threads then redundantly run the same   #
# in-register Cholesky solve (no shared memory / no barrier needed). NLL is   #
# accumulated per-thread on pass 2 and warp-reduced at the end.               #
#                                                                             #
# Memory layout: Y, mu, C_const are (g, n) C-order (= F-order of (n, g)) so   #
# the 32 threads of each warp read one cache line per sample step (coalesced).#
#                                                                             #
# Compiled per-NUM_VARS via NVRTC -D; the PxP loops fully unroll.             #
# --------------------------------------------------------------------------- #
_IRLS_PN_KERNEL_SRC = r"""
#ifndef NUM_VARS
#define NUM_VARS 3
#endif

#define WARP_SIZE 32
#define WARP_MASK 0xffffffffu

__device__ __forceinline__ double warp_sum(double v) {
    #pragma unroll
    for (int m = WARP_SIZE / 2; m > 0; m >>= 1) {
        v += __shfl_xor_sync(WARP_MASK, v, m);
    }
    return v;
}

extern "C" __global__
void irls_iter_pN(
    const double* __restrict__ X,        // (n, NUM_VARS) row-major
    const double* __restrict__ Y,        // (g, n) row-major  = F-order of (n,g)
    double*       __restrict__ mu,       // (g, n) in-place
    const double* __restrict__ sf,       // (n,)
    const double* __restrict__ log_sf,   // (n,)
    const double* __restrict__ alpha,    // (g,)
    const double* __restrict__ inv_a,    // (g,)
    const double* __restrict__ C_const,  // (g, n) row-major
    double*       __restrict__ beta,     // (NUM_VARS, g) in-place
    double*       __restrict__ nll_out,  // (g,)
    const bool*   __restrict__ active,   // (g,)
    bool*         __restrict__ diverged, // (g,)
    const double  ridge,
    const double  max_beta,
    const double  min_mu,
    const int     n,
    const int     g
) {
    const int gi  = blockIdx.x;
    const int tid = threadIdx.x;
    if (gi >= g) return;
    // Whole warp bails together — safe w.r.t. __shfl_xor_sync convergence.
    if (!active[gi]) return;

    const double a    = alpha[gi];
    const double inva = inv_a[gi];

    double H[NUM_VARS][NUM_VARS];
    double rhs[NUM_VARS];
    #pragma unroll
    for (int p_ = 0; p_ < NUM_VARS; ++p_) {
        rhs[p_] = 0.0;
        #pragma unroll
        for (int q_ = 0; q_ < NUM_VARS; ++q_) H[p_][q_] = 0.0;
    }

    const double* Y_gi  = Y       + (size_t)gi * n;
    const double* C_gi  = C_const + (size_t)gi * n;
    double*       mu_gi = mu      + (size_t)gi * n;

    // Pass 1: each thread handles its slice of n, accumulates H and rhs.
    for (int i = tid; i < n; i += WARP_SIZE) {
        double xrow[NUM_VARS];
        #pragma unroll
        for (int k = 0; k < NUM_VARS; ++k) xrow[k] = X[i * NUM_VARS + k];
        const double mui = mu_gi[i];
        const double yi  = Y_gi[i];
        const double Wi  = mui / (1.0 + mui * a);
        const double zi  = log(mui) - log_sf[i] + yi / mui - 1.0;
        const double Wzi = Wi * zi;
        #pragma unroll
        for (int p_ = 0; p_ < NUM_VARS; ++p_) {
            rhs[p_] = fma(xrow[p_], Wzi, rhs[p_]);
            #pragma unroll
            for (int q_ = 0; q_ < NUM_VARS; ++q_) {
                H[p_][q_] = fma(xrow[p_] * xrow[q_], Wi, H[p_][q_]);
            }
        }
    }

    // Warp-sum reduce H and rhs so every thread ends up with the full sum.
    #pragma unroll
    for (int p_ = 0; p_ < NUM_VARS; ++p_) {
        rhs[p_] = warp_sum(rhs[p_]);
        #pragma unroll
        for (int q_ = 0; q_ < NUM_VARS; ++q_) {
            H[p_][q_] = warp_sum(H[p_][q_]);
        }
    }
    #pragma unroll
    for (int p_ = 0; p_ < NUM_VARS; ++p_) H[p_][p_] += ridge;

    // Cholesky in-place on lower triangle, then forward + back solve.
    // All threads compute redundantly (each has the same reduced H, rhs).
    #pragma unroll
    for (int k = 0; k < NUM_VARS; ++k) {
        double s = H[k][k];
        for (int j = 0; j < k; ++j) s -= H[k][j] * H[k][j];
        const double Lkk = sqrt(s);
        H[k][k] = Lkk;
        const double inv_Lkk = 1.0 / Lkk;
        #pragma unroll
        for (int i = k + 1; i < NUM_VARS; ++i) {
            double t = H[i][k];
            for (int j = 0; j < k; ++j) t -= H[i][j] * H[k][j];
            H[i][k] = t * inv_Lkk;
        }
    }
    #pragma unroll
    for (int i = 0; i < NUM_VARS; ++i) {
        double t = rhs[i];
        for (int j = 0; j < i; ++j) t -= H[i][j] * rhs[j];
        rhs[i] = t / H[i][i];
    }
    double beta_new[NUM_VARS];
    #pragma unroll
    for (int ii = NUM_VARS - 1; ii >= 0; --ii) {
        double t = rhs[ii];
        for (int j = ii + 1; j < NUM_VARS; ++j) t -= H[j][ii] * beta_new[j];
        beta_new[ii] = t / H[ii][ii];
    }

    // Divergence check — all threads see the same beta_new.
    bool div = false;
    #pragma unroll
    for (int p_ = 0; p_ < NUM_VARS; ++p_) {
        if (fabs(beta_new[p_]) > max_beta) div = true;
    }
    if (div) {
        if (tid == 0) diverged[gi] = true;
        return;
    }
    if (tid == 0) {
        #pragma unroll
        for (int p_ = 0; p_ < NUM_VARS; ++p_) beta[p_ * g + gi] = beta_new[p_];
    }

    // Pass 2: update mu in place, accumulate per-thread NLL partials.
    double nll = 0.0;
    for (int i = tid; i < n; i += WARP_SIZE) {
        double xrow[NUM_VARS];
        #pragma unroll
        for (int k = 0; k < NUM_VARS; ++k) xrow[k] = X[i * NUM_VARS + k];
        double xb = 0.0;
        #pragma unroll
        for (int k = 0; k < NUM_VARS; ++k) xb = fma(xrow[k], beta_new[k], xb);
        double mui_new = sf[i] * exp(xb);
        if (mui_new < min_mu) mui_new = min_mu;
        mu_gi[i] = mui_new;

        const double yi       = Y_gi[i];
        const double ci       = C_gi[i];
        const double log_term = log1p(a * mui_new);
        const double ll       = ci - inva * log_term
                                + yi * (log(mui_new) - log_term);
        nll -= ll;
    }
    nll = warp_sum(nll);
    if (tid == 0) nll_out[gi] = nll;
}
"""


# Reasonable cutoff: kernel register footprint is O(p^2); p=8 means 64 doubles
# for H + ~8 for rhs + ~8 for xrow + scratch ≈ 80 doubles (~160 regs). Past
# p=8, register spill hurts more than cuSOLVER.
_MAX_RAWKERNEL_NUM_VARS = 8


# --------------------------------------------------------------------------- #
# RawKernel: per-gene dispersion MLE (alpha_mle) via golden-section search    #
# on [log_min, log_max], with Cox-Reid regularization and optional log-normal #
# prior. One warp (32 threads) per gene; each loss evaluation reduces nb_nll  #
# and (optionally) builds the PxP observed-information matrix via warp-sum,  #
# then takes log|det| via Cholesky. Matches ``utils.fit_alpha_mle``          #
# mathematically; optimizer is golden-section instead of scipy L-BFGS-B.     #
# --------------------------------------------------------------------------- #
_ALPHA_MLE_KERNEL_SRC = r"""
#ifndef NUM_VARS
#define NUM_VARS 3
#endif
#ifndef GSS_ITERS
#define GSS_ITERS 40
#endif

// Golden ratio: phi = (sqrt(5) - 1) / 2
#define GSS_PHI 0.6180339887498949

#define WARP_SIZE 32
#define WARP_MASK 0xffffffffu

__device__ __forceinline__ double warp_sum(double v) {
    #pragma unroll
    for (int m = WARP_SIZE / 2; m > 0; m >>= 1) {
        v += __shfl_xor_sync(WARP_MASK, v, m);
    }
    return v;
}

// Evaluate -log-likelihood + optional Cox-Reid + optional Gaussian prior, at
// log_alpha. All threads in the block collaborate on the reductions and end
// up with the same scalar answer (so control flow stays uniform).
__device__ __forceinline__ double alpha_loss(
    const double log_alpha,
    const double log_alpha_hat,
    const int n, const int gi,
    const double* __restrict__ X,
    const double* __restrict__ Y_gi,
    const double* __restrict__ mu_gi,
    const bool cr_reg, const bool prior_reg,
    const double prior_disp_var
) {
    const int tid = threadIdx.x;
    const double alpha = exp(log_alpha);
    const double inv_a = 1.0 / alpha;

    double nll_local = 0.0;
    double H[NUM_VARS][NUM_VARS];
    #pragma unroll
    for (int p_ = 0; p_ < NUM_VARS; ++p_)
        #pragma unroll
        for (int q_ = 0; q_ < NUM_VARS; ++q_) H[p_][q_] = 0.0;

    for (int i = tid; i < n; i += WARP_SIZE) {
        const double mui = mu_gi[i];
        const double yi  = Y_gi[i];
        const double logbinom = lgamma(yi + inv_a)
                              - lgamma(yi + 1.0)
                              - lgamma(inv_a);
        // NLL rewritten to factor out the n/a*log(a) vs (y+1/a)*log(1/a+mu)
        // cancellation that destroys precision at small alpha:
        //   -logbinom + y * log1p(1/(a*mu)) + (1/a) * log1p(a*mu)
        const double a_mu     = alpha * mui;
        const double one_over = 1.0 / a_mu;
        nll_local += -logbinom
                     + yi    * log1p(one_over)
                     + inv_a * log1p(a_mu);
        if (cr_reg) {
            const double Wi = mui / (1.0 + mui * alpha);
            double xr[NUM_VARS];
            #pragma unroll
            for (int k = 0; k < NUM_VARS; ++k) xr[k] = X[i * NUM_VARS + k];
            #pragma unroll
            for (int p_ = 0; p_ < NUM_VARS; ++p_) {
                #pragma unroll
                for (int q_ = 0; q_ < NUM_VARS; ++q_) {
                    H[p_][q_] = fma(xr[p_] * xr[q_], Wi, H[p_][q_]);
                }
            }
        }
    }

    nll_local = warp_sum(nll_local);
    if (cr_reg) {
        #pragma unroll
        for (int p_ = 0; p_ < NUM_VARS; ++p_)
            #pragma unroll
            for (int q_ = 0; q_ < NUM_VARS; ++q_)
                H[p_][q_] = warp_sum(H[p_][q_]);
    }

    // Full NLL — prefactor n/a*log(a) is already absorbed into the sum
    // via the rewrite above (eliminates catastrophic cancellation).
    double loss = nll_local;

    if (cr_reg) {
        // log|det(H)| via Cholesky, in place on lower triangle.
        #pragma unroll
        for (int k = 0; k < NUM_VARS; ++k) {
            double s = H[k][k];
            for (int j = 0; j < k; ++j) s -= H[k][j] * H[k][j];
            const double Lkk = sqrt(s);
            H[k][k] = Lkk;
            const double inv_Lkk = 1.0 / Lkk;
            #pragma unroll
            for (int i = k + 1; i < NUM_VARS; ++i) {
                double t = H[i][k];
                for (int j = 0; j < k; ++j) t -= H[i][j] * H[k][j];
                H[i][k] = t * inv_Lkk;
            }
        }
        double logdet = 0.0;
        #pragma unroll
        for (int p_ = 0; p_ < NUM_VARS; ++p_) logdet += 2.0 * log(H[p_][p_]);
        loss += 0.5 * logdet;
    }

    if (prior_reg) {
        const double d = log_alpha - log_alpha_hat;
        loss += d * d / (2.0 * prior_disp_var);
    }
    return loss;
}

extern "C" __global__
void alpha_mle_pN(
    const double* __restrict__ X,          // (n, NUM_VARS) row-major
    const double* __restrict__ Y,          // (g, n) F-order (coalesced within warp)
    const double* __restrict__ mu,         // (g, n) F-order
    const double* __restrict__ alpha_hat,  // (g,)
    double*       __restrict__ alpha_out,  // (g,)
    bool*         __restrict__ converged,  // (g,) — always true for GSS path
    const double  log_min,
    const double  log_max,
    const int     cr_reg_int,
    const int     prior_reg_int,
    const double  prior_disp_var,
    const int     n,
    const int     g
) {
    const int gi  = blockIdx.x;
    const int tid = threadIdx.x;
    if (gi >= g) return;

    const double* Y_gi  = Y  + (size_t)gi * n;
    const double* mu_gi = mu + (size_t)gi * n;
    const double log_alpha_hat = log(alpha_hat[gi]);
    const bool cr_reg    = (cr_reg_int    != 0);
    const bool prior_reg = (prior_reg_int != 0);

    // ----- Golden section search on [log_min, log_max] -----
    double a = log_min, b = log_max;
    double c = b - GSS_PHI * (b - a);
    double d = a + GSS_PHI * (b - a);
    double fc = alpha_loss(c, log_alpha_hat, n, gi,
                           X, Y_gi, mu_gi, cr_reg, prior_reg, prior_disp_var);
    double fd = alpha_loss(d, log_alpha_hat, n, gi,
                           X, Y_gi, mu_gi, cr_reg, prior_reg, prior_disp_var);
    for (int k = 0; k < GSS_ITERS; ++k) {
        if (fc < fd) {
            b = d;  d = c;  fd = fc;
            c = b - GSS_PHI * (b - a);
            fc = alpha_loss(c, log_alpha_hat, n, gi,
                            X, Y_gi, mu_gi, cr_reg, prior_reg, prior_disp_var);
        } else {
            a = c;  c = d;  fc = fd;
            d = a + GSS_PHI * (b - a);
            fd = alpha_loss(d, log_alpha_hat, n, gi,
                            X, Y_gi, mu_gi, cr_reg, prior_reg, prior_disp_var);
        }
    }
    const double best_la = (fc < fd) ? c : d;

    if (tid == 0) {
        alpha_out[gi] = exp(best_la);
        converged[gi] = true;
    }
}
"""


class CuPyInference(inference.Inference):
    """Inference routines executed on GPU via CuPy.

    All tensor math runs in ``cupy.float64`` to match the CPU scipy path.
    IRLS, dispersion MLE, Wald tests, rough/moment dispersion, and gamma-GLM
    trend fitting all run on the GPU. ``lfc_shrink_nbinom_glm`` (apeGLM MAP)
    is the only stage still implemented on CPU.

    Parameters
    ----------
    device : str
        ``"cuda"``, ``"gpu"`` (== ``"cuda:0"``), or ``"cuda:N"``.
    sync_every : int
        How often the IRLS loop pulls the convergence flag back to host.
        Lower values react to convergence faster but pay a CPU↔GPU sync
        every iteration. (default: ``4``).
    """

    def __init__(
        self,
        device: str = "cuda:0",
        sync_every: int = 4,
    ):
        self._cp = _require_cupy()
        self._device_id = self._parse_device(device)
        self._cp.cuda.Device(self._device_id).use()
        self._sync_every = sync_every
        self._irls_kernel_cache: dict = {}
        self._alpha_mle_kernel_cache: dict = {}

    def _get_irls_kernel(self, num_vars: int):
        """Return a compiled IRLS RawKernel for this num_vars."""
        if num_vars < 2 or num_vars > _MAX_RAWKERNEL_NUM_VARS:
            raise NotImplementedError(
                f"CuPyInference IRLS supports p in [2, {_MAX_RAWKERNEL_NUM_VARS}]; "
                f"got p={num_vars}."
            )
        if num_vars not in self._irls_kernel_cache:
            self._irls_kernel_cache[num_vars] = self._cp.RawKernel(
                _IRLS_PN_KERNEL_SRC,
                "irls_iter_pN",
                options=(f"-DNUM_VARS={num_vars}",),
            )
        return self._irls_kernel_cache[num_vars]

    def _get_alpha_mle_kernel(self, num_vars: int):
        """Return a compiled alpha_mle RawKernel for this num_vars."""
        if num_vars < 1 or num_vars > _MAX_RAWKERNEL_NUM_VARS:
            raise NotImplementedError(
                f"CuPyInference alpha_mle supports p in [1, {_MAX_RAWKERNEL_NUM_VARS}]; "
                f"got p={num_vars}."
            )
        if num_vars not in self._alpha_mle_kernel_cache:
            self._alpha_mle_kernel_cache[num_vars] = self._cp.RawKernel(
                _ALPHA_MLE_KERNEL_SRC,
                "alpha_mle_pN",
                options=(f"-DNUM_VARS={num_vars}",),
            )
        return self._alpha_mle_kernel_cache[num_vars]

    @staticmethod
    def _parse_device(device: str) -> int:
        if device in ("cuda", "gpu"):
            return 0
        if device.startswith("cuda:"):
            return int(device.split(":", 1)[1])
        raise ValueError(
            f"CuPyInference only supports 'cuda' or 'cuda:N' devices, got {device!r}."
        )

    # ------------------------------------------------------------------ #
    # lin_reg_mu                                                          #
    # ------------------------------------------------------------------ #
    def lin_reg_mu(  # noqa: D102
        self,
        counts: np.ndarray,
        size_factors: np.ndarray,
        design_matrix: np.ndarray,
        min_mu: float,
    ) -> np.ndarray:
        cp = self._cp
        X = cp.ascontiguousarray(cp.asarray(design_matrix, dtype=cp.float64))
        sf = cp.asarray(size_factors, dtype=cp.float64)
        Y = cp.asarray(counts, dtype=cp.float64)

        # Batched normal equations: beta = (X^T X)^-1 X^T y, for every gene.
        Y_normed = Y / sf[:, None]
        beta = cp.linalg.solve(X.T @ X, X.T @ Y_normed)
        mu_hat = cp.maximum(sf[:, None] * (X @ beta), min_mu)
        return cp.asnumpy(mu_hat)

    # ------------------------------------------------------------------ #
    # irls — batched Fisher-scoring for the NB GLM                        #
    # ------------------------------------------------------------------ #
    def irls(  # noqa: D102
        self,
        counts: np.ndarray,
        size_factors: np.ndarray,
        design_matrix: np.ndarray,
        disp: np.ndarray,
        min_mu: float,
        beta_tol: float,
        min_beta: float = -30,
        max_beta: float = 30,
        optimizer: Literal["BFGS", "L-BFGS-B"] = "L-BFGS-B",
        maxiter: int = 250,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cp = self._cp
        from cupyx.scipy.special import gammaln  # type: ignore

        num_genes = counts.shape[1]
        num_vars = design_matrix.shape[1]
        sync_every = max(1, self._sync_every)

        X = cp.ascontiguousarray(cp.asarray(design_matrix, dtype=cp.float64))
        sf = cp.asarray(size_factors, dtype=cp.float64)
        log_sf = cp.log(sf)
        sf_col = sf[:, None]
        Y = cp.asarray(counts, dtype=cp.float64)
        Y_F = cp.ascontiguousarray(Y.T)
        alpha = cp.asarray(disp, dtype=cp.float64)
        alpha_b = alpha[None, :]
        inv_a = 1.0 / alpha_b
        # lgamma constants (depend on Y and disp — disp changes between
        # fit_genewise_dispersions and fit_LFC so we recompute each call).
        C_const_F = cp.ascontiguousarray(
            (gammaln(Y + inv_a) - gammaln(inv_a) - gammaln(Y + 1.0)).T
        )

        # -------- beta init (same QR path as utils.irls_solver) --------
        if cp.linalg.matrix_rank(X) == num_vars:
            Q, R = cp.linalg.qr(X)
            z0 = cp.log(Y / sf_col + 0.1)
            beta = cp.linalg.solve(R, Q.T @ z0)  # (p, g)
        else:
            beta = cp.zeros((num_vars, num_genes), dtype=cp.float64)
            beta[0] = cp.log(Y / sf_col).mean(axis=0)

        ridge = 1e-6 * cp.eye(num_vars, dtype=cp.float64)
        mu = cp.maximum(sf_col * cp.exp(X @ beta), min_mu)
        dev = cp.full(num_genes, 1000.0, dtype=cp.float64)
        active = cp.ones(num_genes, dtype=bool)
        diverged = cp.zeros(num_genes, dtype=bool)

        # Warp-per-gene RawKernel on (g, n) F-order buffers.
        rawkernel = self._get_irls_kernel(num_vars)
        mu_buf = cp.ascontiguousarray(mu.T)
        beta_buf = cp.ascontiguousarray(beta)
        nll_buf = cp.zeros(num_genes, dtype=cp.float64)
        ridge_scalar = cp.float64(1e-6)

        for it in range(maxiter):
            old_dev = dev
            rawkernel(
                (num_genes,),
                (32,),
                (
                    X,
                    Y_F,
                    mu_buf,
                    sf,
                    log_sf,
                    alpha,
                    inv_a[0],
                    C_const_F,
                    beta_buf,
                    nll_buf,
                    active,
                    diverged,
                    ridge_scalar,
                    cp.float64(max_beta),
                    cp.float64(min_mu),
                    np.int32(counts.shape[0]),
                    np.int32(num_genes),
                ),
            )
            active &= ~diverged
            dev = -2.0 * nll_buf
            dev_ratio = cp.abs(dev - old_dev) / (cp.abs(dev) + 0.1)
            active &= dev_ratio > beta_tol

            # Host-device sync only every sync_every iters.
            if (it + 1) % sync_every == 0 and not bool(active.any()):
                break

        beta = beta_buf
        mu = cp.ascontiguousarray(mu_buf.T)  # back to (n, g)
        converged = ~diverged  # diverged genes remain un-refit

        # H diagonal for Cook's distance.
        W_final = mu / (1.0 + mu * alpha_b)
        M = cp.einsum("np,nq,ng->pqg", X, X, W_final, optimize=True) + ridge[:, :, None]
        M_inv = cp.linalg.inv(cp.moveaxis(M, 2, 0))
        H_diag = cp.einsum("np,gpq,nq->ng", X, M_inv, X)
        W_sq = cp.sqrt(W_final)
        H_diag = W_sq * H_diag * W_sq

        # Unthresholded mu for return (matches CPU).
        mu_return = sf_col * cp.exp(X @ beta)

        return (
            cp.asnumpy(beta.T),
            cp.asnumpy(mu_return),
            cp.asnumpy(H_diag),
            cp.asnumpy(converged),
        )

    # ------------------------------------------------------------------ #
    # alpha_mle — dispersion MLE, warp-per-gene golden-section search.   #
    # ------------------------------------------------------------------ #
    def alpha_mle(  # noqa: D102
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
        cp = self._cp
        num_genes = counts.shape[1]
        num_vars = design_matrix.shape[1]
        rawkernel = self._get_alpha_mle_kernel(num_vars)

        X = cp.ascontiguousarray(cp.asarray(design_matrix, dtype=cp.float64))
        Y_F = cp.ascontiguousarray(cp.asarray(counts, dtype=cp.float64).T)
        mu_F = cp.ascontiguousarray(cp.asarray(mu, dtype=cp.float64).T)
        alpha_hat_gpu = cp.asarray(alpha_hat, dtype=cp.float64)
        alpha_out = cp.empty(num_genes, dtype=cp.float64)
        converged = cp.empty(num_genes, dtype=bool)

        rawkernel(
            (num_genes,),
            (32,),
            (
                X,
                Y_F,
                mu_F,
                alpha_hat_gpu,
                alpha_out,
                converged,
                cp.float64(np.log(min_disp)),
                cp.float64(np.log(max_disp)),
                np.int32(1 if cr_reg else 0),
                np.int32(1 if prior_reg else 0),
                cp.float64(prior_disp_var if prior_disp_var is not None else 0.0),
                np.int32(counts.shape[0]),
                np.int32(num_genes),
            ),
        )
        return cp.asnumpy(alpha_out), cp.asnumpy(converged)

    # ------------------------------------------------------------------ #
    # wald_test — fully batched on GPU                                    #
    # ------------------------------------------------------------------ #
    def wald_test(  # noqa: D102
        self,
        design_matrix: np.ndarray,
        disp: np.ndarray,
        lfc: np.ndarray,
        mu: np.ndarray,
        ridge_factor: np.ndarray,
        contrast: np.ndarray,
        lfc_null: np.ndarray,
        alt_hypothesis: (
            Literal["greaterAbs", "lessAbs", "greater", "less"] | None
        ) = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cp = self._cp
        from cupyx.scipy.special import erfc  # type: ignore

        def pnorm_sf(x):
            # 0.5 * erfc(x / sqrt(2)) — survival function of the standard normal.
            return 0.5 * erfc(x * (1.0 / np.sqrt(2.0)))

        X = cp.ascontiguousarray(cp.asarray(design_matrix, dtype=cp.float64))
        alpha = cp.asarray(disp, dtype=cp.float64)
        B = cp.asarray(lfc, dtype=cp.float64)
        Mu = cp.asarray(mu, dtype=cp.float64)
        Rf = cp.asarray(ridge_factor, dtype=cp.float64)
        c = cp.asarray(contrast, dtype=cp.float64)
        lfc_n = cp.asarray(lfc_null, dtype=cp.float64)

        # Single-einsum assemble avoids the (n, p, g) intermediate.
        W = Mu / (1.0 + Mu * alpha[None, :])
        M = cp.einsum("np,nq,ng->pqg", X, X, W, optimize=True)
        M_b = cp.moveaxis(M, 2, 0)
        H_b = cp.linalg.inv(M_b + Rf[None, :, :])

        Hc = H_b @ c
        MHc = cp.einsum("gpq,gq->gp", M_b, Hc)
        wald_se = cp.sqrt(cp.einsum("gp,gp->g", Hc, MHc))

        # lfc_null can come in as: a scalar (broadcast to all genes and
        # coefs), a (p,) array (one value per coef, broadcast across genes),
        # or a (g,) array (one value per gene, applied via contrast).
        if lfc_n.ndim == 0:
            lfc_diff = B - lfc_n
        elif lfc_n.shape == c.shape:
            lfc_diff = B - lfc_n[None, :]
        else:
            lfc_diff = B - lfc_n[:, None]

        if alt_hypothesis is None:
            stat = (lfc_diff @ c) / wald_se
            pval = 2.0 * pnorm_sf(cp.abs(stat))
        elif alt_hypothesis == "greater":
            stat = cp.fmax(lfc_diff @ c / wald_se, 0.0)
            pval = pnorm_sf(stat)
        elif alt_hypothesis == "less":
            stat = cp.fmin(lfc_diff @ c / wald_se, 0.0)
            pval = pnorm_sf(cp.abs(stat))
        elif alt_hypothesis == "greaterAbs":
            Bc = B @ c
            lfc_n_c = lfc_n @ c if lfc_n.shape == c.shape else lfc_n
            stat = cp.sign(Bc) * cp.fmax((cp.abs(Bc) - cp.abs(lfc_n_c)) / wald_se, 0.0)
            pval = 2.0 * pnorm_sf(cp.abs(stat))
        else:  # "lessAbs": min(|above|, |below|) stat, max(pval).
            abs_null = cp.abs(lfc_n @ c if lfc_n.shape == c.shape else lfc_n)
            Bc = B @ c
            stat_above = cp.fmax((Bc + abs_null) / wald_se, 0.0)
            stat_below = cp.fmin((Bc - abs_null) / wald_se, 0.0)
            pick_above = cp.abs(stat_above) <= cp.abs(stat_below)
            stat = cp.where(pick_above, stat_above, stat_below)
            pval = cp.maximum(pnorm_sf(stat_above), pnorm_sf(cp.abs(stat_below)))

        return cp.asnumpy(pval), cp.asnumpy(stat), cp.asnumpy(wald_se)

    # ------------------------------------------------------------------ #
    # fit_rough_dispersions                                               #
    # ------------------------------------------------------------------ #
    def fit_rough_dispersions(  # noqa: D102
        self, normed_counts: np.ndarray, design_matrix: np.ndarray
    ) -> np.ndarray:
        cp = self._cp
        if isinstance(design_matrix, pd.DataFrame):
            design_matrix = design_matrix.values
        num_samples, num_vars = design_matrix.shape
        if num_samples == num_vars:
            raise ValueError(
                "The number of samples and the number of design variables are "
                "equal, i.e., there are no replicates to estimate the dispersion."
            )
        X = cp.ascontiguousarray(cp.asarray(design_matrix, dtype=cp.float64))
        Y = cp.asarray(normed_counts, dtype=cp.float64)
        beta = cp.linalg.solve(X.T @ X, X.T @ Y)
        Yhat = cp.maximum(X @ beta, 1.0)
        alpha = (((Y - Yhat) ** 2 - Yhat) / ((num_samples - num_vars) * Yhat**2)).sum(
            axis=0
        )
        return cp.asnumpy(cp.maximum(alpha, 0.0))

    # ------------------------------------------------------------------ #
    # fit_moments_dispersions                                             #
    # ------------------------------------------------------------------ #
    def fit_moments_dispersions(  # noqa: D102
        self, normed_counts: np.ndarray, size_factors: np.ndarray
    ) -> np.ndarray:
        cp = self._cp
        sf = cp.asarray(size_factors, dtype=cp.float64)
        Y = cp.asarray(normed_counts, dtype=cp.float64)
        # CPU reference drops all-zero genes; upstream code handles re-alignment.
        Y = Y[:, ~(Y == 0).all(axis=0)]
        s_mean_inv = (1.0 / sf).mean()
        mu = Y.mean(axis=0)
        sigma = Y.var(axis=0, ddof=1)
        alpha = cp.nan_to_num((sigma - s_mean_inv * mu) / mu**2)
        return cp.asnumpy(alpha)

    # ------------------------------------------------------------------ #
    # dispersion_trend_gamma_glm — 2-parameter gamma GLM fit (log link), #
    # loss/gradient on GPU via cupy; optimizer is scipy L-BFGS-B on host #
    # since the state is just a 2-vector. Each scipy iter syncs a scalar #
    # loss and 2-element gradient (~50µs sync cost × ~20 iters).         #
    # ------------------------------------------------------------------ #
    def dispersion_trend_gamma_glm(  # noqa: D102
        self, covariates: pd.Series, targets: pd.Series
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        from scipy.optimize import minimize  # type: ignore

        cp = self._cp
        x = cp.asarray(covariates.values, dtype=cp.float64)
        y = cp.asarray(targets.values, dtype=cp.float64)
        X_fit = cp.column_stack([cp.ones_like(x), x])

        def loss(coeffs: np.ndarray) -> float:
            c = cp.asarray(coeffs)
            mu = X_fit @ c
            return float(cp.nanmean(y / mu + cp.log(mu)))

        def grad(coeffs: np.ndarray) -> np.ndarray:
            c = cp.asarray(coeffs)
            mu = X_fit @ c
            factor = (1.0 - y / mu) / mu
            g_vec = cp.nanmean(X_fit * factor[:, None], axis=0)
            return cp.asnumpy(g_vec)

        try:
            res = minimize(
                loss,
                x0=np.array([1.0, 1.0]),
                jac=grad,
                method="L-BFGS-B",
                bounds=[(1e-12, np.inf)],
            )
        except RuntimeWarning:
            return (
                np.array([np.nan, np.nan]),
                np.array([np.nan, np.nan]),
                False,
            )
        coeffs = res.x
        predictions = cp.asnumpy(X_fit @ cp.asarray(coeffs))
        return coeffs, predictions, res.success

    # ------------------------------------------------------------------ #
    # lfc_shrink_nbinom_glm — apeGLM MAP on CPU (no GPU kernel yet).     #
    # ------------------------------------------------------------------ #
    def lfc_shrink_nbinom_glm(  # noqa: D102
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
        return DefaultInference().lfc_shrink_nbinom_glm(
            design_matrix=design_matrix,
            counts=counts,
            size=size,
            offset=offset,
            prior_no_shrink_scale=prior_no_shrink_scale,
            prior_scale=prior_scale,
            optimizer=optimizer,
            shrink_index=shrink_index,
        )
