from typing import Any
from typing import Literal
from typing import cast

import numpy as np
import pandas as pd
from joblib import Parallel  # type: ignore
from joblib import delayed
from joblib import parallel_backend
from scipy.optimize import minimize  # type: ignore

from pydeseq2 import inference
from pydeseq2.dispersions import fit_alpha_mle
from pydeseq2.dispersions import fit_moments_dispersions
from pydeseq2.dispersions import fit_rough_dispersions
from pydeseq2.distributions import nbinomGLM
from pydeseq2.glm import fit_lin_mu
from pydeseq2.glm import irls_solver
from pydeseq2.glm import wald_test
from pydeseq2.misc import get_num_processes


def _column_sliceable_counts(counts: inference.CountMatrix) -> Any:
    """Use CSC storage for efficient repeated sparse column slices."""
    if isinstance(counts, np.ndarray):
        return counts
    return cast(Any, counts).tocsc(copy=False)


def _gene_factors(factors: np.ndarray, gene_idx: int) -> np.ndarray:
    return factors[:, gene_idx] if factors.ndim == 2 else factors


def _gene_counts(counts: Any, gene_idx: int) -> np.ndarray:
    """Return one gene's counts as a dense sample vector."""
    if isinstance(counts, np.ndarray):
        return np.asarray(counts[:, gene_idx]).ravel()
    return counts[:, [gene_idx]].toarray().ravel()


class DefaultInference(inference.Inference):
    """Default DESeq2-related inference methods, using scipy/sklearn/numpy.

    This object contains the interface to the default inference routines and uses joblib internally for parallelization.
    Inherit this class or its parent to write custom inference routines.

    Parameters
    ----------
    joblib_verbosity
        The verbosity level for joblib tasks.
        The higher the value, the more updates are reported. (default: ``0``).
    batch_size
        Number of tasks to allocate to each joblib parallel worker. (default: ``128``).
    n_cpus
        Number of cpus to use.
        If None, all available cpus will be used. (default: ``None``).
    backend
        Joblib backend.
    """

    fit_rough_dispersions = staticmethod(fit_rough_dispersions)  # type: ignore
    fit_moments_dispersions = staticmethod(fit_moments_dispersions)  # type: ignore

    def __init__(
        self,
        joblib_verbosity: int = 0,
        batch_size: int = 128,
        n_cpus: int | None = None,
        backend: str = "loky",
    ):
        self._joblib_verbosity = joblib_verbosity
        self._batch_size = batch_size
        self._n_cpus = get_num_processes(n_cpus)
        self._backend = backend

    @property
    def n_cpus(self) -> int:
        """Number of cpus to use for multiprocessing."""
        return self._n_cpus

    @n_cpus.setter
    def n_cpus(self, n_cpus: int) -> None:
        self._n_cpus = get_num_processes(n_cpus)

    def lin_reg_mu(  # noqa: D102
        self,
        counts: inference.CountMatrix,
        size_factors: np.ndarray,
        design_matrix: np.ndarray,
        min_mu: float,
    ) -> np.ndarray:
        column_counts = _column_sliceable_counts(counts)
        with parallel_backend(self._backend, inner_max_num_threads=1):
            mu_hat_ = np.array(
                Parallel(
                    n_jobs=self.n_cpus,
                    verbose=self._joblib_verbosity,
                    batch_size=self._batch_size,
                )(
                    delayed(fit_lin_mu)(
                        counts=_gene_counts(column_counts, i),
                        size_factors=_gene_factors(size_factors, i),
                        design_matrix=design_matrix,
                        min_mu=min_mu,
                    )
                    for i in range(column_counts.shape[1])
                )
            )
        return mu_hat_.T

    def irls(  # noqa: D102
        self,
        counts: inference.CountMatrix,
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
        column_counts = _column_sliceable_counts(counts)
        with parallel_backend(self._backend, inner_max_num_threads=1):
            res = Parallel(
                n_jobs=self.n_cpus,
                verbose=self._joblib_verbosity,
                batch_size=self._batch_size,
            )(
                delayed(irls_solver)(
                    counts=_gene_counts(column_counts, i),
                    size_factors=_gene_factors(size_factors, i),
                    design_matrix=design_matrix,
                    disp=disp[i],
                    min_mu=min_mu,
                    beta_tol=beta_tol,
                    min_beta=min_beta,
                    max_beta=max_beta,
                    optimizer=optimizer,
                    maxiter=maxiter,
                )
                for i in range(column_counts.shape[1])
            )
        res = zip(*res, strict=False)
        MLE_lfcs_, mu_hat_, hat_diagonals_, converged_ = (np.array(m) for m in res)

        return (
            MLE_lfcs_,
            mu_hat_.T,
            hat_diagonals_.T,
            converged_,
        )

    def alpha_mle(  # noqa: D102
        self,
        counts: inference.CountMatrix,
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
        column_counts = _column_sliceable_counts(counts)
        with parallel_backend(self._backend, inner_max_num_threads=1):
            res = Parallel(
                n_jobs=self.n_cpus,
                verbose=self._joblib_verbosity,
                batch_size=self._batch_size,
            )(
                delayed(fit_alpha_mle)(
                    counts=_gene_counts(column_counts, i),
                    design_matrix=design_matrix,
                    mu=mu[:, i],
                    alpha_hat=alpha_hat[i],
                    min_disp=min_disp,
                    max_disp=max_disp,
                    prior_disp_var=prior_disp_var,
                    cr_reg=cr_reg,
                    prior_reg=prior_reg,
                    optimizer=optimizer,
                )
                for i in range(column_counts.shape[1])
            )
        res = zip(*res, strict=False)
        dispersions_, l_bfgs_b_converged_ = (np.array(m) for m in res)
        return dispersions_, l_bfgs_b_converged_

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
        num_genes = mu.shape[1]
        with parallel_backend(self._backend, inner_max_num_threads=1):
            res = Parallel(
                n_jobs=self.n_cpus,
                verbose=self._joblib_verbosity,
                batch_size=self._batch_size,
            )(
                delayed(wald_test)(
                    design_matrix=design_matrix,
                    disp=disp[i],
                    lfc=lfc[i],
                    mu=mu[:, i],
                    ridge_factor=ridge_factor,
                    contrast=contrast,
                    lfc_null=lfc_null,  # Convert log2 to natural log
                    alt_hypothesis=alt_hypothesis,
                )
                for i in range(num_genes)
            )
        res = zip(*res, strict=False)
        pvals, stats, se = (np.array(m) for m in res)

        return pvals, stats, se

    def dispersion_trend_gamma_glm(  # noqa: D102
        self, covariates: pd.Series, targets: pd.Series
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        covariates_w_intercept = covariates.to_frame()
        covariates_w_intercept.insert(0, "intercept", 1)
        covariates_fit = covariates_w_intercept.values
        targets_fit = targets.values

        def loss(coeffs):
            mu = covariates_fit @ coeffs
            return np.nanmean(targets_fit / mu + np.log(mu), axis=0)

        def grad(coeffs):
            mu = covariates_fit @ coeffs
            return -np.nanmean(
                ((targets_fit / mu - 1)[:, None] * covariates_fit) / mu[:, None], axis=0
            )

        try:
            res = minimize(
                loss,
                x0=np.array([1.0, 1.0]),
                jac=grad,
                method="L-BFGS-B",
                bounds=[(1e-12, np.inf)],
            )
        except RuntimeWarning:  # Could happen if the coefficients fall to zero
            return np.array([np.nan, np.nan]), np.array([np.nan, np.nan]), False

        coeffs = res.x
        return coeffs, covariates_fit @ coeffs, res.success

    def lfc_shrink_nbinom_glm(  # noqa: D102
        self,
        design_matrix: np.ndarray,
        counts: inference.CountMatrix,
        size: np.ndarray,
        offset: np.ndarray,
        prior_no_shrink_scale: float,
        prior_scale: float,
        optimizer: str,
        shrink_index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        column_counts = _column_sliceable_counts(counts)
        with parallel_backend(self._backend, inner_max_num_threads=1):
            res = Parallel(
                n_jobs=self.n_cpus,
                verbose=self._joblib_verbosity,
                batch_size=self._batch_size,
            )(
                delayed(nbinomGLM)(
                    design_matrix=design_matrix,
                    counts=_gene_counts(column_counts, i),
                    size=size[i],
                    offset=_gene_factors(offset, i),
                    prior_no_shrink_scale=prior_no_shrink_scale,
                    prior_scale=prior_scale,
                    optimizer=optimizer,
                    shrink_index=shrink_index,
                )
                for i in range(column_counts.shape[1])
            )
        res = zip(*res, strict=False)
        lfcs, inv_hessians, l_bfgs_b_converged_ = (np.array(m) for m in res)
        return lfcs, inv_hessians, l_bfgs_b_converged_
