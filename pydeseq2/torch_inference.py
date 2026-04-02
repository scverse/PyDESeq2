"""GPU-accelerated inference backend for PyDESeq2 using PyTorch.

Implements all methods from the :class:`~pydeseq2.inference.Inference` ABC
with fully vectorized tensor operations across all genes simultaneously.
"""

import warnings
from typing import Literal

import numpy as np
import pandas as pd
import torch

from pydeseq2 import inference
from pydeseq2.gpu_utils import get_device
from pydeseq2.torch_grid_search import torch_grid_fit_alpha
from pydeseq2.torch_grid_search import torch_grid_fit_beta
from pydeseq2.torch_grid_search import torch_grid_fit_shrink_beta


class TorchInference(inference.Inference):
    """GPU-backed DESeq2 inference methods using PyTorch.

    Implements DESeq2 inference routines with fully vectorized PyTorch
    operations for GPU acceleration. All genes are processed
    simultaneously rather than via per-gene parallelization.

    Parameters
    ----------
    device : str or None
        Device string (e.g. ``"cuda"``, ``"cuda:0"``, ``"cpu"``).
        If ``None``, auto-detects CUDA availability.
    """

    def __init__(self, device: str | None = None):
        self.device = get_device(device)

    @torch.no_grad()
    def lin_reg_mu(
        self,
        counts: np.ndarray,
        size_factors: np.ndarray,
        design_matrix: np.ndarray,
        min_mu: float,
    ) -> np.ndarray:
        """Estimate mean via vectorized linear regression on GPU.

        Parameters
        ----------
        counts : np.ndarray
            Raw counts ``(N, G)``.

        size_factors : np.ndarray
            Sample-wise scaling factors ``(N,)``.

        design_matrix : np.ndarray
            Design matrix ``(N, P)``.

        min_mu : float
            Lower threshold for fitted means.

        Returns
        -------
        np.ndarray
            Estimated means ``(N, G)``.
        """
        counts_t = torch.tensor(counts, dtype=torch.float64, device=self.device)
        size_factors_t = torch.tensor(
            size_factors, dtype=torch.float64, device=self.device
        )
        design_matrix_t = torch.tensor(
            design_matrix, dtype=torch.float64, device=self.device
        )

        normed_counts_t = counts_t / size_factors_t[:, None]

        # Solve AX = B for all genes at once: (N, P) @ (P, G) = (N, G)
        coeffs = torch.linalg.lstsq(design_matrix_t, normed_counts_t)[0]

        mu_hat_t = size_factors_t[:, None] * (design_matrix_t @ coeffs)
        mu_hat_t = torch.clamp(mu_hat_t, min=min_mu)

        return mu_hat_t.cpu().numpy()

    @torch.no_grad()
    def irls(
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
        r"""Fit NB GLM with log-link via IRLS, vectorized across genes.

        Parameters
        ----------
        counts : np.ndarray
            Raw counts ``(N, G)``.

        size_factors : np.ndarray
            Sample-wise scaling factors ``(N,)``.

        design_matrix : np.ndarray
            Design matrix ``(N, P)``.

        disp : np.ndarray
            Gene-wise dispersions ``(G,)``.

        min_mu : float
            Lower bound on estimated means.

        beta_tol : float
            Convergence threshold for relative deviance change.

        min_beta : float
            Lower bound on LFC.

        max_beta : float
            Upper bound on LFC.

        optimizer : str
            Ignored (kept for API compatibility).

        maxiter : int
            Maximum IRLS iterations.

        Returns
        -------
        beta : np.ndarray
            Fitted coefficients ``(G, P)``.

        mu : np.ndarray
            Fitted means ``(N, G)``.

        hat_diagonals : np.ndarray
            Hat matrix diagonals ``(N, G)``.

        converged : np.ndarray
            Per-gene convergence flags ``(G,)``.
        """
        counts_t = torch.tensor(counts, dtype=torch.float64, device=self.device)
        size_factors_t = torch.tensor(
            size_factors, dtype=torch.float64, device=self.device
        )
        design_matrix_t = torch.tensor(
            design_matrix, dtype=torch.float64, device=self.device
        )
        disp_t = torch.tensor(disp, dtype=torch.float64, device=self.device)
        eps = torch.finfo(torch.float64).eps

        n_samples, n_genes = counts_t.shape
        n_coeffs = design_matrix_t.shape[1]

        # Initialize beta (P, G) with log base mean as intercept
        beta = torch.zeros(
            (n_coeffs, n_genes),
            dtype=torch.float64,
            device=self.device,
        )
        log_base_mean = torch.log(counts_t / size_factors_t[:, None] + eps).mean(dim=0)
        beta[0, :] = log_base_mean

        dev = torch.full((n_genes,), 1000.0, device=self.device)
        ridge_factor = torch.diag_embed(
            torch.full(
                (n_genes, n_coeffs),
                1e-6,
                device=self.device,
            )
        )

        mu = torch.clamp(
            size_factors_t[:, None] * torch.exp(design_matrix_t @ beta),
            min=min_mu,
        )

        converged = torch.zeros(n_genes, dtype=torch.bool, device=self.device)

        for i in range(maxiter):
            W = mu / (1.0 + mu * disp_t[None, :])
            z = torch.log(mu / size_factors_t[:, None] + eps) + (counts_t - mu) / (
                mu + eps
            )

            # H_g = X^T W_g X + ridge for each gene: (G, P, P)
            sqrt_W = torch.sqrt(W)
            X_weighted = sqrt_W.T.unsqueeze(2) * design_matrix_t.unsqueeze(0)
            H_g = torch.bmm(X_weighted.transpose(1, 2), X_weighted) + ridge_factor

            RHS = design_matrix_t.T @ (W * z)
            beta_hat = torch.linalg.solve(H_g, RHS.T.unsqueeze(2)).squeeze(2)

            old_dev = dev.clone()
            beta = beta_hat.T

            mu = torch.clamp(
                size_factors_t[:, None] * torch.exp(design_matrix_t @ beta),
                min=min_mu,
            )

            # Compute deviance via NB NLL
            alpha_neg1_t = 1.0 / disp_t
            logbinom_t = (
                torch.lgamma(counts_t + alpha_neg1_t)
                - torch.lgamma(counts_t + 1)
                - torch.lgamma(alpha_neg1_t)
            )
            term2 = (counts_t + alpha_neg1_t) * torch.log(
                mu + alpha_neg1_t
            ) - counts_t * torch.log(mu)
            total_nll = (alpha_neg1_t * torch.log(disp_t)) * n_samples + (
                -logbinom_t + term2
            ).sum(dim=0)

            dev = -2 * total_nll
            dev_ratio = torch.abs(dev - old_dev) / (torch.abs(dev) + 0.1)
            converged = dev_ratio < beta_tol

            if torch.all(converged) or i == maxiter - 1:
                break

        # Check for NaNs and fall back to grid search if needed
        irls_converged = ~torch.isnan(beta).any(dim=0)

        if not torch.all(irls_converged):
            if n_coeffs == 2:
                beta_fallback = torch_grid_fit_beta(
                    counts=counts,
                    size_factors=size_factors,
                    design_matrix=design_matrix,
                    disp=disp,
                    min_mu=min_mu,
                    device=self.device,
                )
                beta = torch.tensor(
                    beta_fallback.T,
                    device=self.device,
                    dtype=torch.float64,
                )
            else:
                # For n_coeffs > 2, fall back to CPU grid search
                # per non-converged gene
                from pydeseq2.utils import irls_solver

                nan_mask = torch.isnan(beta).any(dim=0)
                nan_indices = torch.where(nan_mask)[0]
                for idx in nan_indices:
                    i = idx.item()
                    try:
                        result = irls_solver(
                            counts[:, i],
                            size_factors,
                            design_matrix,
                            disp[i],
                            min_mu,
                            beta_tol,
                            min_beta=min_beta,
                            max_beta=max_beta,
                            optimizer=optimizer,
                            maxiter=maxiter,
                        )
                        beta[:, i] = torch.tensor(
                            result[0],
                            dtype=torch.float64,
                            device=self.device,
                        )
                    except (RuntimeError, ValueError):
                        beta[:, i] = 0.0
            converged = torch.zeros(n_genes, dtype=torch.bool, device=self.device)
        else:
            converged = irls_converged

        # Compute hat diagonals using final beta
        W = mu / (1.0 + mu * disp_t[None, :])
        sqrt_W = torch.sqrt(W)
        X_weighted = sqrt_W.T.unsqueeze(2) * design_matrix_t.unsqueeze(0)
        H_g = torch.bmm(X_weighted.transpose(1, 2), X_weighted) + ridge_factor
        H_inv = torch.linalg.inv(H_g)

        hat_diagonals = torch.einsum(
            "np,gpq,nq->gn", design_matrix_t, H_inv, design_matrix_t
        )
        hat_diagonals = sqrt_W * hat_diagonals.T * sqrt_W

        # Return unthresholded mu
        mu = size_factors_t[:, None] * torch.exp(design_matrix_t @ beta)

        return (
            beta.T.cpu().numpy(),
            mu.cpu().numpy(),
            hat_diagonals.cpu().numpy(),
            converged.cpu().numpy(),
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
        """Estimate dispersion via L-BFGS on GPU.

        Parameters
        ----------
        counts : np.ndarray
            Raw counts ``(N, G)``.

        design_matrix : np.ndarray
            Design matrix ``(N, P)``.

        mu : np.ndarray
            Mean estimation ``(N, G)``.

        alpha_hat : np.ndarray
            Initial dispersion estimate ``(G,)``.

        min_disp : float
            Lower threshold for dispersion.

        max_disp : float
            Upper threshold for dispersion.

        prior_disp_var : float, optional
            Prior dispersion variance.

        cr_reg : bool
            Whether to use Cox-Reid regularization.

        prior_reg : bool
            Whether to use prior log-residual regularization.

        optimizer : str
            Ignored (kept for API compatibility).

        Returns
        -------
        alpha : np.ndarray
            Fitted dispersions ``(G,)``.

        converged : np.ndarray
            Per-gene convergence flags ``(G,)``.
        """
        counts_t = torch.tensor(counts, dtype=torch.float64, device=self.device)
        design_matrix_t = torch.tensor(
            design_matrix, dtype=torch.float64, device=self.device
        )
        mu_t = torch.tensor(mu, dtype=torch.float64, device=self.device)
        alpha_hat_t = torch.tensor(alpha_hat, dtype=torch.float64, device=self.device)

        n_samples, n_genes = counts_t.shape

        log_alpha = torch.nn.Parameter(
            torch.log(alpha_hat_t).clone().detach().requires_grad_(True)
        )

        optim = torch.optim.LBFGS(
            [log_alpha], max_iter=20, line_search_fn="strong_wolfe"
        )

        def closure():
            optim.zero_grad()
            alpha_t = torch.exp(log_alpha)
            alpha_t = torch.clamp(alpha_t, min=min_disp, max=max_disp)

            logbinom_t = (
                torch.lgamma(counts_t + 1.0 / alpha_t)
                - torch.lgamma(counts_t + 1)
                - torch.lgamma(1.0 / alpha_t)
            )
            term2 = (counts_t + 1.0 / alpha_t) * torch.log(
                mu_t + 1.0 / alpha_t
            ) - counts_t * torch.log(mu_t)
            nll = (n_samples * 1.0 / alpha_t * torch.log(alpha_t)) + (
                -logbinom_t + term2
            ).sum(dim=0)

            total_loss = nll

            if cr_reg:
                W = mu_t / (1.0 + mu_t * alpha_t[None, :])
                term = torch.bmm(
                    design_matrix_t.transpose(0, 1).unsqueeze(0).expand(n_genes, -1, -1),
                    (W.T.unsqueeze(2) * design_matrix_t.unsqueeze(0)),
                )
                _, logdet = torch.linalg.slogdet(term)
                total_loss = total_loss + 0.5 * logdet

            if prior_reg:
                if prior_disp_var is None:
                    raise ValueError("prior_disp_var required for prior regularization")
                log_alpha_hat_t = torch.log(alpha_hat_t)
                total_loss = total_loss + (log_alpha - log_alpha_hat_t) ** 2 / (
                    2 * prior_disp_var
                )

            loss = total_loss.sum()
            loss.backward()
            return loss

        optim_converged = True
        try:
            optim.step(closure)
            if torch.isnan(log_alpha.data).any():
                optim_converged = False
        except (RuntimeError, ValueError):
            optim_converged = False

        alpha_final = torch.exp(log_alpha.detach())
        alpha_final = torch.clamp(alpha_final, min=min_disp, max=max_disp)

        if not optim_converged:
            warnings.warn(
                "L-BFGS failed for alpha_mle. Falling back to grid search.",
                UserWarning,
                stacklevel=2,
            )
            alpha_final_np = torch_grid_fit_alpha(
                counts=counts,
                design_matrix=design_matrix,
                mu=mu,
                alpha_hat=alpha_hat,
                min_disp=min_disp,
                max_disp=max_disp,
                device=self.device,
                prior_disp_var=prior_disp_var,
                cr_reg=cr_reg,
                prior_reg=prior_reg,
            )
            alpha_final = torch.tensor(
                alpha_final_np,
                device=self.device,
                dtype=torch.float64,
            )
            converged_out = torch.zeros(n_genes, dtype=torch.bool, device=self.device)
        else:
            converged_out = torch.ones(n_genes, dtype=torch.bool, device=self.device)

        return alpha_final.cpu().numpy(), converged_out.cpu().numpy()

    @torch.no_grad()
    def wald_test(
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
        """Run Wald test on GPU.

        Parameters
        ----------
        design_matrix : np.ndarray
            Design matrix ``(N, P)``.

        disp : np.ndarray
            Dispersions ``(G,)``.

        lfc : np.ndarray
            Log-fold changes ``(G, P)``.

        mu : np.ndarray
            Fitted means ``(N, G)``.

        ridge_factor : np.ndarray
            Ridge regularization ``(P, P)``.

        contrast : np.ndarray
            Contrast vector ``(P,)``.

        lfc_null : np.ndarray
            Null hypothesis LFC.

        alt_hypothesis : str or None
            Alternative hypothesis type.

        Returns
        -------
        p_values : np.ndarray
            Wald p-values ``(G,)``.

        statistics : np.ndarray
            Wald statistics ``(G,)``.

        se : np.ndarray
            Standard errors ``(G,)``.
        """
        design_matrix_t = torch.tensor(
            np.asarray(design_matrix, dtype=np.float64),
            dtype=torch.float64,
            device=self.device,
        )
        disp_t = torch.tensor(
            np.asarray(disp, dtype=np.float64),
            dtype=torch.float64,
            device=self.device,
        )
        lfc_t = torch.tensor(
            np.asarray(lfc, dtype=np.float64),
            dtype=torch.float64,
            device=self.device,
        )
        mu_t = torch.tensor(
            np.asarray(mu, dtype=np.float64),
            dtype=torch.float64,
            device=self.device,
        )
        ridge_factor_t = torch.tensor(
            np.asarray(ridge_factor, dtype=np.float64),
            dtype=torch.float64,
            device=self.device,
        )
        contrast_t = torch.tensor(
            np.asarray(contrast, dtype=np.float64),
            dtype=torch.float64,
            device=self.device,
        )
        lfc_null_t = torch.tensor(
            np.asarray(lfc_null, dtype=np.float64),
            dtype=torch.float64,
            device=self.device,
        )

        n_samples, n_coeffs = design_matrix_t.shape
        n_genes = mu_t.shape[1]

        # W = mu / (1 + mu * disp): (N, G)
        W = mu_t / (1.0 + mu_t * disp_t[None, :])

        # M = X^T diag(W_g) X for each gene: (G, P, P)
        M = torch.bmm(
            design_matrix_t.transpose(0, 1).unsqueeze(0).expand(n_genes, -1, -1),
            (W.T.unsqueeze(2) * design_matrix_t.unsqueeze(0)),
        )

        H = torch.linalg.inv(M + ridge_factor_t)
        Hc = H @ contrast_t[None, :, None]

        wald_se = torch.sqrt(torch.bmm(Hc.transpose(1, 2), torch.bmm(M, Hc)).squeeze())

        # Extract per-gene LFC for the contrast
        if lfc_t.ndim > 1:
            lfc_contracted = torch.einsum("gp,p->g", lfc_t, contrast_t)
        else:
            lfc_contracted = lfc_t

        # Compute stat and p-value per alternative hypothesis,
        # matching the CPU implementation in utils.py
        if alt_hypothesis == "greater":
            stat = (
                torch.einsum(
                    "gp,p->g",
                    lfc_t,
                    contrast_t,
                )
                if lfc_t.ndim > 1
                else lfc_t
            )
            stat = (
                torch.fmax(
                    (stat - lfc_null_t) / wald_se,
                    torch.zeros_like(wald_se),
                )
                * contrast_t.sum()
            )
            wald_statistic = stat
            wald_p_value = 1.0 - torch.special.ndtr(stat)
        elif alt_hypothesis == "less":
            stat = (
                torch.einsum(
                    "gp,p->g",
                    lfc_t,
                    contrast_t,
                )
                if lfc_t.ndim > 1
                else lfc_t
            )
            stat = (
                torch.fmin(
                    (stat - lfc_null_t) / wald_se,
                    torch.zeros_like(wald_se),
                )
                * contrast_t.sum()
            )
            wald_statistic = stat
            wald_p_value = 1.0 - torch.special.ndtr(torch.abs(stat))
        elif alt_hypothesis == "greaterAbs":
            lfc_sign = torch.sign(lfc_contracted)
            stat = lfc_sign * torch.fmax(
                (torch.abs(lfc_contracted) - lfc_null_t) / wald_se,
                torch.zeros_like(wald_se),
            )
            wald_statistic = stat
            wald_p_value = 2 * (1.0 - torch.special.ndtr(torch.abs(stat)))
        elif alt_hypothesis == "lessAbs":
            # lessAbs = max(p_above, p_below)
            # where p_above = greater(-|lfc_null|)
            # and p_below = less(|lfc_null|)
            stat_above = torch.fmax(
                (lfc_contracted - (-torch.abs(lfc_null_t))) / wald_se,
                torch.zeros_like(wald_se),
            )
            pval_above = 1.0 - torch.special.ndtr(stat_above)

            stat_below = torch.fmin(
                (lfc_contracted - torch.abs(lfc_null_t)) / wald_se,
                torch.zeros_like(wald_se),
            )
            pval_below = 1.0 - torch.special.ndtr(torch.abs(stat_below))

            # Pick stat with smaller abs, pick larger p-value
            use_above = torch.abs(stat_above) <= torch.abs(stat_below)
            wald_statistic = torch.where(use_above, stat_above, stat_below)
            wald_p_value = torch.fmax(pval_above, pval_below)
        else:
            wald_statistic = (lfc_contracted - lfc_null_t) / wald_se
            wald_p_value = 2 * (1.0 - torch.special.ndtr(torch.abs(wald_statistic)))

        return (
            wald_p_value.cpu().numpy(),
            wald_statistic.cpu().numpy(),
            wald_se.cpu().numpy(),
        )

    @torch.no_grad()
    def fit_rough_dispersions(
        self,
        normed_counts: np.ndarray,
        design_matrix: pd.DataFrame,
    ) -> np.ndarray:
        """Rough dispersion estimates from linear model on GPU.

        Parameters
        ----------
        normed_counts : np.ndarray
            Normalized counts ``(N, G)``.

        design_matrix : pd.DataFrame
            Design matrix ``(N, P)``.

        Returns
        -------
        np.ndarray
            Rough dispersion estimates ``(G,)``.
        """
        normed_counts_t = torch.tensor(
            normed_counts, dtype=torch.float64, device=self.device
        )
        design_matrix_t = torch.tensor(
            design_matrix.values if hasattr(design_matrix, "values") else design_matrix,
            dtype=torch.float64,
            device=self.device,
        )

        n_samples = normed_counts_t.shape[0]
        num_vars = design_matrix_t.shape[1]

        if n_samples == num_vars:
            raise ValueError(
                "The number of samples and the number of design "
                "variables are equal, i.e., there are no replicates "
                "to estimate the dispersion. Please use a design "
                "with fewer variables."
            )

        coeffs = torch.linalg.lstsq(design_matrix_t, normed_counts_t)[0]
        y_hat = design_matrix_t @ coeffs
        y_hat = torch.clamp(y_hat, min=1.0)

        alpha_rde = (
            ((normed_counts_t - y_hat) ** 2 - y_hat)
            / ((n_samples - num_vars) * y_hat**2)
        ).sum(dim=0)

        alpha_rde = torch.clamp(alpha_rde, min=0.0)
        return alpha_rde.cpu().numpy()

    @torch.no_grad()
    def fit_moments_dispersions(
        self, normed_counts: np.ndarray, size_factors: np.ndarray
    ) -> np.ndarray:
        """Dispersion estimates based on moments on GPU.

        Parameters
        ----------
        normed_counts : np.ndarray
            Normalized counts ``(N, G)``.

        size_factors : np.ndarray
            Size factors ``(N,)``.

        Returns
        -------
        np.ndarray
            Moment-based dispersion estimates ``(G,)``.
        """
        normed_counts_t = torch.tensor(
            normed_counts, dtype=torch.float64, device=self.device
        )
        size_factors_t = torch.tensor(
            size_factors, dtype=torch.float64, device=self.device
        )

        all_zeros = (normed_counts_t == 0).all(dim=0)
        normed_counts_filtered = normed_counts_t[:, ~all_zeros]

        s_mean_inv = (1.0 / size_factors_t).mean()
        mu = normed_counts_filtered.mean(dim=0)
        sigma = normed_counts_filtered.var(dim=0, unbiased=True)

        alpha_moments = (sigma - s_mean_inv * mu) / (mu**2)
        alpha_moments = torch.nan_to_num(alpha_moments, nan=0.0)

        final_alpha = torch.zeros(
            normed_counts_t.shape[1],
            dtype=torch.float64,
            device=self.device,
        )
        final_alpha[~all_zeros] = alpha_moments

        return final_alpha.cpu().numpy()

    def dispersion_trend_gamma_glm(
        self, covariates: pd.Series, targets: pd.Series
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """Fit gamma GLM for dispersion trend on GPU.

        Parameters
        ----------
        covariates : pd.Series
            Covariates (mean expression per gene) ``(G,)``.

        targets : pd.Series
            Targets (gene-wise dispersions) ``(G,)``.

        Returns
        -------
        coeffs : np.ndarray
            Regression coefficients ``(2,)``.

        predictions : np.ndarray
            Predicted dispersions ``(G,)``.

        converged : bool
            Whether L-BFGS converged.
        """
        covariates_t = torch.tensor(
            covariates.values,
            dtype=torch.float64,
            device=self.device,
        ).unsqueeze(1)
        targets_t = torch.tensor(
            targets.values,
            dtype=torch.float64,
            device=self.device,
        )

        covariates_w_intercept_t = torch.cat(
            [torch.ones_like(covariates_t), covariates_t], dim=1
        )

        coeffs = torch.nn.Parameter(
            torch.tensor(
                [1.0, 1.0],
                dtype=torch.float64,
                device=self.device,
            ),
            requires_grad=True,
        )

        opt = torch.optim.LBFGS(
            [coeffs],
            max_iter=20,
            line_search_fn="strong_wolfe",
        )

        def closure():
            opt.zero_grad()
            mu_pred = covariates_w_intercept_t @ coeffs
            mu_pred = torch.clamp(mu_pred, min=1e-12)
            loss = (targets_t / mu_pred + torch.log(mu_pred)).nanmean()
            loss.backward()
            return loss

        try:
            opt.step(closure)
            converged = True
        except (RuntimeError, ValueError):
            converged = False

        predictions = (covariates_w_intercept_t @ coeffs.detach()).cpu().numpy()
        coeffs_np = coeffs.detach().cpu().numpy()

        return coeffs_np, predictions, converged

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
        """Fit NB MAP LFC with apeGLM prior on GPU.

        Parameters
        ----------
        design_matrix : np.ndarray
            Design matrix ``(N, P)``.

        counts : np.ndarray
            Raw counts ``(N, G)``.

        size : np.ndarray
            Size parameter (1/dispersion) ``(G,)``.

        offset : np.ndarray
            Log size factors ``(N,)``.

        prior_no_shrink_scale : float
            Prior scale for non-shrunk coefficients.

        prior_scale : float
            Prior scale for the LFC.

        optimizer : str
            Ignored (kept for API compatibility).

        shrink_index : int
            Index of the coefficient to shrink.

        Returns
        -------
        beta : np.ndarray
            Fitted coefficients ``(G, P)``.

        inv_hessian : np.ndarray
            Inverse Hessian ``(G, P, P)``.

        converged : np.ndarray
            Per-gene convergence flags ``(G,)``.
        """
        counts_t = torch.tensor(counts, dtype=torch.float64, device=self.device)
        design_matrix_t = torch.tensor(
            design_matrix, dtype=torch.float64, device=self.device
        )
        size_t = torch.tensor(size, dtype=torch.float64, device=self.device)
        offset_t = torch.tensor(offset, dtype=torch.float64, device=self.device)

        n_samples, n_genes = counts_t.shape
        n_coeffs = design_matrix_t.shape[1]

        # Initialize beta (P, G) with small alternating values
        beta = torch.nn.Parameter(
            torch.ones(
                (n_coeffs, n_genes),
                dtype=torch.float64,
                device=self.device,
            )
            * 0.1
            * (-1) ** (torch.arange(n_coeffs, device=self.device)[:, None]),
            requires_grad=True,
        )

        shrink_mask = torch.zeros(n_coeffs, dtype=torch.float64, device=self.device)
        shrink_mask[shrink_index] = 1
        no_shrink_mask = 1 - shrink_mask

        optim = torch.optim.LBFGS(
            [beta],
            max_iter=100,
            tolerance_grad=1e-10,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optim.zero_grad()

            xbeta = design_matrix_t @ beta

            prior_term = (beta * no_shrink_mask[:, None]) ** 2 / (
                2 * prior_no_shrink_scale**2
            )
            prior = prior_term.sum(dim=0) + torch.log1p(
                (beta[shrink_index, :] / prior_scale) ** 2
            )

            nll_term = (
                counts_t * xbeta
                - (counts_t + size_t)
                * torch.logaddexp(
                    xbeta + offset_t[:, None],
                    torch.log(size_t[None, :]),
                )
            ).sum(dim=0)

            loss = (prior - nll_term).sum()
            loss.backward()
            return loss

        optim_converged = True
        try:
            optim.step(closure)
            if torch.isnan(beta.data).any():
                optim_converged = False
        except (RuntimeError, ValueError):
            optim_converged = False

        beta_final = beta.detach()

        if not optim_converged:
            warnings.warn(
                "L-BFGS failed for lfc_shrink. Falling back to grid search.",
                UserWarning,
                stacklevel=2,
            )
            beta_final_np = torch_grid_fit_shrink_beta(
                counts=counts,
                offset=offset,
                design_matrix=design_matrix,
                size=size,
                prior_no_shrink_scale=prior_no_shrink_scale,
                prior_scale=prior_scale,
                scale_cnst=1.0,
                device=self.device,
                shrink_index=shrink_index,
            )
            beta_final = torch.tensor(
                beta_final_np,
                device=self.device,
                dtype=torch.float64,
            )
            converged_g = torch.zeros(n_genes, dtype=torch.bool, device=self.device)
        else:
            converged_g = torch.ones(n_genes, dtype=torch.bool, device=self.device)

        # Compute inverse Hessian for SE estimation
        xbeta = design_matrix_t @ beta_final
        exp_xbeta_off = torch.exp(xbeta + offset_t[:, None])
        size_expanded = size_t[None, :]
        frac = (
            (counts_t + size_expanded)
            * size_expanded
            * exp_xbeta_off
            / (size_expanded + exp_xbeta_off) ** 2
        )

        h11 = 1 / prior_no_shrink_scale**2
        h22 = (
            2
            * (prior_scale**2 - beta_final[shrink_index, :] ** 2)
            / (prior_scale**2 + beta_final[shrink_index, :] ** 2) ** 2
        )

        # NOTE: Intentionally matching CPU behavior where the prior
        # Hessian diagonal is broadcast-added to every row of the
        # full Hessian matrix (not just the diagonal). This ensures
        # concordance with the CPU DefaultInference implementation.
        diag_val = (
            no_shrink_mask[:, None] * h11 + shrink_mask[:, None] * h22.unsqueeze(0)
        ).T

        X_expanded = design_matrix_t.unsqueeze(0).expand(n_genes, -1, -1)
        frac_expanded = frac.transpose(0, 1).unsqueeze(2)
        X_weighted_per_gene = X_expanded * frac_expanded
        hessian_nll_part = torch.bmm(X_weighted_per_gene.transpose(1, 2), X_expanded)

        full_hessian = hessian_nll_part + diag_val.unsqueeze(1)
        inv_hessian = torch.linalg.inv(full_hessian)

        return (
            beta_final.T.cpu().numpy(),
            inv_hessian.cpu().numpy(),
            converged_g.cpu().numpy(),
        )
