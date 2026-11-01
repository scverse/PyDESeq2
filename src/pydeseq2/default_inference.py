from numbers import Integral
from typing import Literal

import numpy as np
import pandas as pd
from joblib import Parallel  # type: ignore
from joblib import delayed
from joblib import parallel_backend
from scipy.optimize import minimize  # type: ignore

from pydeseq2 import inference
from pydeseq2.dispersions import _fit_alpha_mle_lockstep
from pydeseq2.dispersions import fit_alpha_mle
from pydeseq2.dispersions import fit_moments_dispersions
from pydeseq2.dispersions import fit_rough_dispersions
from pydeseq2.distributions import _nbinom_glm_lockstep
from pydeseq2.distributions import nbinomGLM
from pydeseq2.glm import _fit_lin_mu_rows
from pydeseq2.glm import _irls_rows
from pydeseq2.glm import _wald_test_rows
from pydeseq2.glm import fit_lin_mu
from pydeseq2.glm import irls_solver
from pydeseq2.glm import wald_test
from pydeseq2.misc import get_num_processes


def _genes_first(a: np.ndarray) -> np.ndarray:
    """Return a C-contiguous (genes, samples) copy of a (samples, genes) block."""
    return np.ascontiguousarray(a.T)


def _lin_reg_mu_block(counts, size_factors, design_matrix, min_mu):
    res = _fit_lin_mu_rows(
        counts=counts,
        size_factors=size_factors,
        design_matrix=design_matrix,
        min_mu=min_mu,
    )
    if res is not None:
        return res
    return np.array(
        [
            fit_lin_mu(
                counts=row.copy(),
                size_factors=size_factors,
                design_matrix=design_matrix,
                min_mu=min_mu,
            )
            for row in counts
        ]
    )


def _irls_block(counts, disp, **kwargs):
    res = _irls_rows(counts=counts, disp=disp, **kwargs)
    if res is not None:
        return res
    res = [
        irls_solver(counts=row.copy(), disp=d, **kwargs)
        for row, d in zip(counts, disp, strict=True)
    ]
    return tuple(np.array(m) for m in zip(*res, strict=True))


def _alpha_mle_block(counts, mu, alpha_hat, optimizer, **kwargs):
    if optimizer == "L-BFGS-B":
        res = _fit_alpha_mle_lockstep(
            counts=counts, mu=mu, alpha_hat=alpha_hat, **kwargs
        )
        if res is not None:
            return res
    res = [
        fit_alpha_mle(
            counts=row.copy(), mu=m.copy(), alpha_hat=a, optimizer=optimizer, **kwargs
        )
        for row, m, a in zip(counts, mu, alpha_hat, strict=True)
    ]
    return tuple(np.array(m) for m in zip(*res, strict=True))


def _genewise_block(
    counts,
    size_factors,
    design_matrix,
    min_mu,
    alpha_hat,
    min_disp,
    max_disp,
    irls_disp,
    beta_tol,
):
    if irls_disp is None:
        mu = _lin_reg_mu_block(
            counts=counts,
            size_factors=size_factors,
            design_matrix=design_matrix,
            min_mu=min_mu,
        )
    else:
        mu = _irls_block(
            counts=counts,
            size_factors=size_factors,
            design_matrix=design_matrix,
            disp=irls_disp,
            min_mu=min_mu,
            beta_tol=beta_tol,
            min_beta=-30,
            max_beta=30,
            optimizer="L-BFGS-B",
            maxiter=250,
        )[1]
    # alpha_mle reads mu_hat back from a float64 layer.
    mu = np.ascontiguousarray(mu, dtype=np.float64)
    dispersions, converged = _alpha_mle_block(
        counts=counts,
        design_matrix=design_matrix,
        mu=mu,
        alpha_hat=alpha_hat,
        min_disp=min_disp,
        max_disp=max_disp,
        prior_disp_var=None,
        cr_reg=True,
        prior_reg=False,
        optimizer="L-BFGS-B",
    )
    return mu, dispersions, converged


def _map_lfc_block(
    counts,
    size_factors,
    design_matrix,
    mu,
    alpha_hat,
    min_disp,
    max_disp,
    prior_disp_var,
    genewise,
    outlier,
    min_mu,
    beta_tol,
):
    dispersions, converged = _alpha_mle_block(
        counts=counts,
        design_matrix=design_matrix,
        mu=mu,
        alpha_hat=alpha_hat,
        min_disp=min_disp,
        max_disp=max_disp,
        prior_disp_var=prior_disp_var,
        cr_reg=True,
        prior_reg=True,
        optimizer="L-BFGS-B",
    )
    disp = np.clip(dispersions, min_disp, max_disp)
    disp[outlier] = genewise[outlier]
    lfcs, mu_lfc, hat_diagonals, lfc_converged = _irls_block(
        counts=counts,
        size_factors=size_factors,
        design_matrix=design_matrix,
        disp=disp,
        min_mu=min_mu,
        beta_tol=beta_tol,
        min_beta=-30,
        max_beta=30,
        optimizer="L-BFGS-B",
        maxiter=250,
    )
    return dispersions, converged, disp, lfcs, mu_lfc, hat_diagonals, lfc_converged


def _refit_block(
    counts,
    size_factors,
    design_matrix,
    min_mu,
    alpha_hat,
    min_disp,
    max_disp,
    irls_disp,
    beta_tol,
    fitted,
    prior_disp_var,
    outlier_offset,
):
    mu, genewise, _ = _genewise_block(
        counts=counts,
        size_factors=size_factors,
        design_matrix=design_matrix,
        min_mu=min_mu,
        alpha_hat=alpha_hat,
        min_disp=min_disp,
        max_disp=max_disp,
        irls_disp=irls_disp,
        beta_tol=beta_tol,
    )
    dispersions, _ = _alpha_mle_block(
        counts=counts,
        design_matrix=design_matrix,
        mu=mu,
        alpha_hat=fitted,
        min_disp=min_disp,
        max_disp=max_disp,
        prior_disp_var=prior_disp_var,
        cr_reg=True,
        prior_reg=True,
        optimizer="L-BFGS-B",
    )
    # fit_MAP_dispersions' dispersions; the caller checks them against its own.
    clipped = np.clip(genewise, min_disp, max_disp)
    disp = np.clip(dispersions, min_disp, max_disp)
    with np.errstate(all="ignore"):
        outlier = np.log(clipped) > np.log(fitted) + outlier_offset
    disp[outlier] = clipped[outlier]
    lfcs = _irls_block(
        counts=counts,
        size_factors=size_factors,
        design_matrix=design_matrix,
        disp=disp,
        min_mu=min_mu,
        beta_tol=beta_tol,
        min_beta=-30,
        max_beta=30,
        optimizer="L-BFGS-B",
        maxiter=250,
    )[0]
    return genewise, dispersions, disp, lfcs


def _wald_test_block(disp, lfc, mu, **kwargs):
    res = [
        wald_test(disp=d, lfc=b.copy(), mu=m.copy(), **kwargs)
        for d, b, m in zip(disp, lfc, mu, strict=True)
    ]
    return tuple(np.array(m) for m in zip(*res, strict=True))


def _nbinom_glm_block(counts, size, optimizer, **kwargs):
    if optimizer == "L-BFGS-B":
        res = _nbinom_glm_lockstep(counts=counts, size=size, **kwargs)
        if res is not None:
            return res
    res = [
        nbinomGLM(counts=row.copy(), size=s, optimizer=optimizer, **kwargs)
        for row, s in zip(counts, size, strict=True)
    ]
    return tuple(np.array(m) for m in zip(*res, strict=True))


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

    def _gene_blocks(self, n_genes: int) -> tuple[list[slice], int | str]:
        """Split genes into blocks, one joblib task each.

        A block holds ``batch_size`` genes, as many as one joblib batch of per-gene
        tasks, and is sent as one task. With ``batch_size="auto"`` the blocks hold 128
        genes and joblib batches them itself; any other value goes to joblib, which
        rejects it.
        """
        if isinstance(self._batch_size, Integral) and self._batch_size > 0:
            size, joblib_batch_size = int(self._batch_size), 1
        else:
            size, joblib_batch_size = 128, self._batch_size
        blocks = [slice(s, min(s + size, n_genes)) for s in range(0, n_genes, size)]
        return blocks, joblib_batch_size

    def _parallel(self, joblib_batch_size: int | str) -> Parallel:
        # max_nbytes=None: pickle large blocks rather than memmap them to disk.
        return Parallel(
            n_jobs=self.n_cpus,
            verbose=self._joblib_verbosity,
            batch_size=joblib_batch_size,
            max_nbytes=None,
        )

    def lin_reg_mu(  # noqa: D102
        self,
        counts: np.ndarray,
        size_factors: np.ndarray,
        design_matrix: np.ndarray,
        min_mu: float,
    ) -> np.ndarray:
        blocks, joblib_batch_size = self._gene_blocks(counts.shape[1])
        with parallel_backend(self._backend, inner_max_num_threads=1):
            res = self._parallel(joblib_batch_size)(
                delayed(_lin_reg_mu_block)(
                    counts=_genes_first(counts[:, b]),
                    size_factors=size_factors,
                    design_matrix=design_matrix,
                    min_mu=min_mu,
                )
                for b in blocks
            )
        mu_hat_ = np.concatenate(res) if res else np.array([])
        return mu_hat_.T

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
        blocks, joblib_batch_size = self._gene_blocks(counts.shape[1])
        disp = np.asarray(disp)
        with parallel_backend(self._backend, inner_max_num_threads=1):
            res = self._parallel(joblib_batch_size)(
                delayed(_irls_block)(
                    counts=_genes_first(counts[:, b]),
                    size_factors=size_factors,
                    design_matrix=design_matrix,
                    disp=disp[b],
                    min_mu=min_mu,
                    beta_tol=beta_tol,
                    min_beta=min_beta,
                    max_beta=max_beta,
                    optimizer=optimizer,
                    maxiter=maxiter,
                )
                for b in blocks
            )
        res = zip(*res, strict=False)
        MLE_lfcs_, mu_hat_, hat_diagonals_, converged_ = (np.concatenate(m) for m in res)

        return (
            MLE_lfcs_,
            mu_hat_.T,
            hat_diagonals_.T,
            converged_,
        )

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
        blocks, joblib_batch_size = self._gene_blocks(counts.shape[1])
        alpha_hat = np.asarray(alpha_hat)
        with parallel_backend(self._backend, inner_max_num_threads=1):
            res = self._parallel(joblib_batch_size)(
                delayed(_alpha_mle_block)(
                    counts=_genes_first(counts[:, b]),
                    design_matrix=design_matrix,
                    mu=_genes_first(mu[:, b]),
                    alpha_hat=alpha_hat[b],
                    min_disp=min_disp,
                    max_disp=max_disp,
                    prior_disp_var=prior_disp_var,
                    cr_reg=cr_reg,
                    prior_reg=prior_reg,
                    optimizer=optimizer,
                )
                for b in blocks
            )
        res = zip(*res, strict=False)
        dispersions_, l_bfgs_b_converged_ = (np.concatenate(m) for m in res)
        return dispersions_, l_bfgs_b_converged_

    def _genewise_dispersions(
        self,
        counts: np.ndarray,
        size_factors: np.ndarray,
        design_matrix: np.ndarray,
        min_mu: float,
        alpha_hat: np.ndarray,
        min_disp: float,
        max_disp: float,
        irls_disp: np.ndarray | None = None,
        beta_tol: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Fit the gene-wise dispersions in one task per block of genes.

        Each block runs ``lin_reg_mu`` (``irls`` when ``irls_disp`` is given), then
        ``alpha_mle`` on its means. Returns mu_hat as the first call returns it, and
        the dispersions and convergence flags of the second; None when there is no
        gene.
        """
        blocks, joblib_batch_size = self._gene_blocks(counts.shape[1])
        if not blocks:
            return None
        alpha_hat = np.asarray(alpha_hat)
        if irls_disp is not None:
            irls_disp = np.asarray(irls_disp)
        with parallel_backend(self._backend, inner_max_num_threads=1):
            res = self._parallel(joblib_batch_size)(
                delayed(_genewise_block)(
                    counts=_genes_first(counts[:, b]),
                    size_factors=size_factors,
                    design_matrix=design_matrix,
                    min_mu=min_mu,
                    alpha_hat=alpha_hat[b],
                    min_disp=min_disp,
                    max_disp=max_disp,
                    irls_disp=None if irls_disp is None else irls_disp[b],
                    beta_tol=beta_tol,
                )
                for b in blocks
            )
        mu_hat_, dispersions_, converged_ = (
            np.concatenate(m) for m in zip(*res, strict=True)
        )
        return mu_hat_.T, dispersions_, converged_

    def _MAP_dispersions_and_LFC(
        self,
        counts: np.ndarray,
        size_factors: np.ndarray,
        design_matrix: np.ndarray,
        mu: np.ndarray,
        alpha_hat: np.ndarray,
        min_disp: float,
        max_disp: float,
        prior_disp_var: float,
        genewise: np.ndarray,
        outlier: np.ndarray,
        min_mu: float,
        beta_tol: float,
    ) -> tuple[np.ndarray, ...] | None:
        """Fit the MAP dispersions and the LFCs in one task per block.

        Each block runs ``alpha_mle`` with the prior, then ``irls`` on its dispersions
        clipped to [min_disp, max_disp] (``genewise`` where ``outlier``).

        Returns the MAP dispersions and convergence flags as ``alpha_mle`` returns
        them, the dispersions given to ``irls``, then ``irls``'s four outputs; None
        when there is no gene.
        """
        blocks, joblib_batch_size = self._gene_blocks(counts.shape[1])
        if not blocks:
            return None
        alpha_hat = np.asarray(alpha_hat)
        with parallel_backend(self._backend, inner_max_num_threads=1):
            res = self._parallel(joblib_batch_size)(
                delayed(_map_lfc_block)(
                    counts=_genes_first(counts[:, b]),
                    size_factors=size_factors,
                    design_matrix=design_matrix,
                    mu=_genes_first(mu[:, b]),
                    alpha_hat=alpha_hat[b],
                    min_disp=min_disp,
                    max_disp=max_disp,
                    prior_disp_var=prior_disp_var,
                    genewise=genewise[b],
                    outlier=outlier[b],
                    min_mu=min_mu,
                    beta_tol=beta_tol,
                )
                for b in blocks
            )
        disp, converged, irls_disp, lfcs, mu_lfc, hat_diagonals, lfc_converged = (
            np.concatenate(m) for m in zip(*res, strict=True)
        )
        return (
            disp,
            converged,
            irls_disp,
            lfcs,
            mu_lfc.T,
            hat_diagonals.T,
            lfc_converged,
        )

    def _refit_dispersions_and_LFC(
        self,
        counts: np.ndarray,
        size_factors: np.ndarray,
        design_matrix: np.ndarray,
        min_mu: float,
        alpha_hat: np.ndarray,
        min_disp: float,
        max_disp: float,
        irls_disp: np.ndarray | None,
        beta_tol: float,
        fitted: np.ndarray,
        prior_disp_var: float,
        outlier_offset: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """Fit the refit's dispersions and LFCs in one task per block.

        Each block runs the fits of ``_genewise_dispersions``, ``alpha_mle`` with the
        prior from ``fitted``, then ``irls`` on the clipped MAP dispersions (the
        clipped gene-wise ones where ``log genewise > log fitted + outlier_offset``).

        Returns the gene-wise and MAP dispersions as ``alpha_mle`` returns them, the
        dispersions given to ``irls`` and its LFCs; None when there is no gene.
        """
        blocks, joblib_batch_size = self._gene_blocks(counts.shape[1])
        if not blocks:
            return None
        alpha_hat = np.asarray(alpha_hat)
        fitted = np.asarray(fitted)
        if irls_disp is not None:
            irls_disp = np.asarray(irls_disp)
        with parallel_backend(self._backend, inner_max_num_threads=1):
            res = self._parallel(joblib_batch_size)(
                delayed(_refit_block)(
                    counts=_genes_first(counts[:, b]),
                    size_factors=size_factors,
                    design_matrix=design_matrix,
                    min_mu=min_mu,
                    alpha_hat=alpha_hat[b],
                    min_disp=min_disp,
                    max_disp=max_disp,
                    irls_disp=None if irls_disp is None else irls_disp[b],
                    beta_tol=beta_tol,
                    fitted=fitted[b],
                    prior_disp_var=prior_disp_var,
                    outlier_offset=outlier_offset,
                )
                for b in blocks
            )
        genewise, dispersions, disp, lfcs = (
            np.concatenate(m) for m in zip(*res, strict=True)
        )
        return genewise, dispersions, disp, lfcs

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
        disp = np.asarray(disp)
        res = _wald_test_rows(
            design_matrix=design_matrix,
            disp=disp,
            lfc=lfc,
            mu=mu,
            ridge_factor=ridge_factor,
            contrast=contrast,
            lfc_null=lfc_null,  # type: ignore[arg-type]
            alt_hypothesis=alt_hypothesis,
        )
        if res is not None:
            return res
        blocks, joblib_batch_size = self._gene_blocks(mu.shape[1])
        with parallel_backend(self._backend, inner_max_num_threads=1):
            res = self._parallel(joblib_batch_size)(
                delayed(_wald_test_block)(
                    design_matrix=design_matrix,
                    disp=disp[b],
                    lfc=np.ascontiguousarray(lfc[b]),
                    mu=_genes_first(mu[:, b]),
                    ridge_factor=ridge_factor,
                    contrast=contrast,
                    lfc_null=lfc_null,  # Convert log2 to natural log
                    alt_hypothesis=alt_hypothesis,
                )
                for b in blocks
            )
        res = zip(*res, strict=False)  # type: ignore[assignment]
        pvals, stats, se = (np.concatenate(m) for m in res)

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
        counts: np.ndarray,
        size: np.ndarray,
        offset: np.ndarray,
        prior_no_shrink_scale: float,
        prior_scale: float,
        optimizer: str,
        shrink_index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        blocks, joblib_batch_size = self._gene_blocks(counts.shape[1])
        size = np.asarray(size)
        with parallel_backend(self._backend, inner_max_num_threads=1):
            res = self._parallel(joblib_batch_size)(
                delayed(_nbinom_glm_block)(
                    design_matrix=design_matrix,
                    counts=_genes_first(counts[:, b]),
                    size=size[b],
                    offset=offset,
                    prior_no_shrink_scale=prior_no_shrink_scale,
                    prior_scale=prior_scale,
                    optimizer=optimizer,
                    shrink_index=shrink_index,
                )
                for b in blocks
            )
        res = zip(*res, strict=False)
        lfcs, inv_hessians, l_bfgs_b_converged_ = (np.concatenate(m) for m in res)
        return lfcs, inv_hessians, l_bfgs_b_converged_


_GENEWISE_STEPS = {
    name: getattr(DefaultInference, name)
    for name in ("lin_reg_mu", "irls", "alpha_mle", "_genewise_dispersions")
}


def _fits_genewise_in_one_map(obj: inference.Inference) -> bool:
    """Check whether ``obj`` can fit the gene-wise dispersions in one map.

    It can when it is a DefaultInference that overrides none of ``lin_reg_mu``,
    ``irls``, ``alpha_mle`` and ``_genewise_dispersions``, so the one map gives
    what the separate calls give.
    """
    return isinstance(obj, DefaultInference) and all(
        getattr(getattr(obj, name, None), "__func__", None) is method
        for name, method in _GENEWISE_STEPS.items()
    )


_MAP_LFC_STEPS = {
    name: getattr(DefaultInference, name)
    for name in ("alpha_mle", "irls", "_MAP_dispersions_and_LFC")
}


def _fits_MAP_and_LFC_in_one_map(obj: inference.Inference) -> bool:
    """Check whether ``obj`` can fit the MAP dispersions and LFCs in one map.

    It can when it is a DefaultInference that overrides none of ``alpha_mle``,
    ``irls`` and ``_MAP_dispersions_and_LFC``.
    """
    return isinstance(obj, DefaultInference) and all(
        getattr(getattr(obj, name, None), "__func__", None) is method
        for name, method in _MAP_LFC_STEPS.items()
    )


_REFIT_STEPS = {
    **_GENEWISE_STEPS,
    "_refit_dispersions_and_LFC": DefaultInference._refit_dispersions_and_LFC,
}


def _fits_refit_in_one_map(obj: inference.Inference) -> bool:
    """Check whether ``obj`` can run the refit's fits in one map.

    It can when it is a DefaultInference that overrides none of ``lin_reg_mu``,
    ``irls``, ``alpha_mle``, ``_genewise_dispersions`` and
    ``_refit_dispersions_and_LFC``.
    """
    return isinstance(obj, DefaultInference) and all(
        getattr(getattr(obj, name, None), "__func__", None) is method
        for name, method in _REFIT_STEPS.items()
    )
