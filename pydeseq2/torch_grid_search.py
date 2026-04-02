"""GPU-accelerated grid search fallbacks for PyDESeq2.

Used when iterative solvers (IRLS, L-BFGS) fail to converge.
"""

import numpy as np
import torch


def torch_vec_nb_nll(
    counts: torch.Tensor, mu: torch.Tensor, alpha: torch.Tensor
) -> torch.Tensor:
    r"""Return the negative log-likelihood of a negative binomial.

    Vectorized PyTorch version for 3D inputs ``(N, G, GP)`` for
    counts/mu and 2D ``(G, GP)`` for alpha, where GP is the number
    of grid points.

    Parameters
    ----------
    counts : torch.Tensor
        Observations ``(N, G, GP)``.

    mu : torch.Tensor
        Mean of the distribution ``(N, G, GP)``.

    alpha : torch.Tensor
        Dispersion of the distribution ``(G, GP)`` or broadcastable.

    Returns
    -------
    torch.Tensor
        Negative log-likelihood ``(G, GP)``.
    """
    n = counts.shape[0]
    alpha_neg1 = 1.0 / alpha

    logbinom = (
        torch.lgamma(counts + alpha_neg1)
        - torch.lgamma(counts + 1)
        - torch.lgamma(alpha_neg1)
    )

    term2 = (counts + alpha_neg1) * torch.log(mu + alpha_neg1) - counts * torch.log(mu)

    total_nll = (n * alpha_neg1 * torch.log(alpha)) + ((-logbinom + term2).sum(dim=0))
    return total_nll


def torch_grid_fit_alpha(
    counts: np.ndarray,
    design_matrix: np.ndarray,
    mu: np.ndarray,
    alpha_hat: np.ndarray,
    min_disp: float,
    max_disp: float,
    device: torch.device,
    prior_disp_var: float | None = None,
    cr_reg: bool = True,
    prior_reg: bool = False,
    grid_length: int = 100,
) -> np.ndarray:
    """Find optimal dispersion via 1D grid search for all genes.

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

    device : torch.device
        Device for tensors.

    prior_disp_var : float, optional
        Prior dispersion variance.

    cr_reg : bool
        Whether to use Cox-Reid regularization.

    prior_reg : bool
        Whether to use prior log-residual regularization.

    grid_length : int
        Number of grid points.

    Returns
    -------
    np.ndarray
        Fitted dispersion for each gene ``(G,)``.
    """
    counts_t = torch.tensor(counts, dtype=torch.float64, device=device)
    design_matrix_t = torch.tensor(design_matrix, dtype=torch.float64, device=device)
    mu_t = torch.tensor(mu, dtype=torch.float64, device=device)
    alpha_hat_t = torch.tensor(alpha_hat, dtype=torch.float64, device=device)

    n_samples, n_genes = counts_t.shape
    n_coeffs = design_matrix_t.shape[1]

    min_log_alpha = torch.log(torch.tensor(min_disp, dtype=torch.float64, device=device))
    max_log_alpha = torch.log(torch.tensor(max_disp, dtype=torch.float64, device=device))

    grid = torch.linspace(min_log_alpha, max_log_alpha, grid_length, device=device)

    def loss_fn(log_alpha_grid: torch.Tensor) -> torch.Tensor:
        alpha = torch.exp(log_alpha_grid)
        alpha_grid_expanded = alpha.unsqueeze(0).unsqueeze(0)
        mu_expanded = mu_t.unsqueeze(2)
        counts_expanded = counts_t.unsqueeze(2)

        nll = torch_vec_nb_nll(counts_expanded, mu_expanded, alpha_grid_expanded)

        reg = torch.zeros_like(nll)

        if cr_reg:
            W = mu_expanded / (1.0 + mu_expanded * alpha_grid_expanded)
            sqrt_W = torch.sqrt(W)
            X_weighted = sqrt_W.permute(1, 2, 0).unsqueeze(
                3
            ) * design_matrix_t.unsqueeze(0).unsqueeze(0)
            term = torch.bmm(
                X_weighted.reshape(-1, n_samples, n_coeffs).transpose(1, 2),
                X_weighted.reshape(-1, n_samples, n_coeffs),
            ).reshape(n_genes, grid_length, n_coeffs, n_coeffs)

            _, logdet = torch.linalg.slogdet(term)
            reg += 0.5 * logdet

        if prior_reg:
            if prior_disp_var is None:
                raise ValueError("prior_disp_var required when prior_reg=True")
            log_alpha_hat_expanded = torch.log(alpha_hat_t).unsqueeze(1)
            log_alpha_grid_expanded = log_alpha_grid.unsqueeze(0)
            reg += (log_alpha_grid_expanded - log_alpha_hat_expanded) ** 2 / (
                2 * prior_disp_var
            )

        return nll + reg

    # Coarse grid search
    ll_grid = loss_fn(grid)
    min_idx = torch.argmin(ll_grid, dim=1)
    delta = grid[1] - grid[0]

    # Fine grid search -- vectorized across genes
    fine_grid_starts = grid[min_idx] - delta
    fine_grid_ends = grid[min_idx] + delta

    # Build per-gene fine grids without Python loop: (G, grid_length)
    t = torch.linspace(0, 1, grid_length, device=device, dtype=torch.float64).unsqueeze(
        0
    )
    fine_grid = fine_grid_starts.unsqueeze(1) + t * (
        fine_grid_ends - fine_grid_starts
    ).unsqueeze(1)

    # Evaluate fine grid -- need to handle per-gene grids
    # Reshape fine_grid to evaluate loss_fn gene-by-gene via broadcasting
    # fine_grid is (G, grid_length). We need alpha (G, grid_length).
    alpha_fine = torch.exp(fine_grid)

    mu_expanded = mu_t.unsqueeze(2).expand(-1, -1, grid_length)
    counts_expanded = counts_t.unsqueeze(2).expand(-1, -1, grid_length)

    nll_fine = torch_vec_nb_nll(counts_expanded, mu_expanded, alpha_fine)
    reg_fine = torch.zeros_like(nll_fine)

    if cr_reg:
        alpha_fine_exp = alpha_fine.unsqueeze(0)
        W = mu_t.unsqueeze(2) / (1.0 + mu_t.unsqueeze(2) * alpha_fine_exp)
        sqrt_W = torch.sqrt(W)
        X_weighted = sqrt_W.permute(1, 2, 0).unsqueeze(3) * design_matrix_t.unsqueeze(
            0
        ).unsqueeze(0)
        term = torch.bmm(
            X_weighted.reshape(-1, n_samples, n_coeffs).transpose(1, 2),
            X_weighted.reshape(-1, n_samples, n_coeffs),
        ).reshape(n_genes, grid_length, n_coeffs, n_coeffs)
        _, logdet = torch.linalg.slogdet(term)
        reg_fine += 0.5 * logdet

    if prior_reg and prior_disp_var is not None:
        log_alpha_hat_expanded = torch.log(alpha_hat_t).unsqueeze(1)
        reg_fine += (fine_grid - log_alpha_hat_expanded) ** 2 / (2 * prior_disp_var)

    ll_fine = nll_fine + reg_fine
    min_idx_fine = torch.argmin(ll_fine, dim=1)
    log_alpha_final = fine_grid[torch.arange(n_genes, device=device), min_idx_fine]

    return torch.exp(log_alpha_final).cpu().numpy()


def torch_grid_fit_beta(
    counts: np.ndarray,
    size_factors: np.ndarray,
    design_matrix: np.ndarray,
    disp: np.ndarray,
    device: torch.device,
    min_mu: float = 0.5,
    grid_length: int = 60,
    min_beta: float = -30,
    max_beta: float = 30,
) -> np.ndarray:
    """Find optimal LFC via 2D grid search for all genes.

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

    device : torch.device
        Device for tensors.

    min_mu : float
        Lower threshold for fitted means.

    grid_length : int
        Number of grid points per dimension.

    min_beta : float
        Lower bound on LFC.

    max_beta : float
        Upper bound on LFC.

    Returns
    -------
    np.ndarray
        Fitted beta ``(G, P)``.
    """
    counts_t = torch.tensor(counts, dtype=torch.float64, device=device)
    size_factors_t = torch.tensor(size_factors, dtype=torch.float64, device=device)
    design_matrix_t = torch.tensor(design_matrix, dtype=torch.float64, device=device)
    disp_t = torch.tensor(disp, dtype=torch.float64, device=device)

    n_samples, n_genes = counts_t.shape
    n_coeffs = design_matrix_t.shape[1]

    assert n_coeffs == 2, (
        "torch_grid_fit_beta currently supports only 2 coefficients. "
        "For multi-factor designs, non-converged genes use CPU fallback."
    )

    x_grid = torch.linspace(
        min_beta, max_beta, grid_length, device=device, dtype=torch.float64
    )
    y_grid = torch.linspace(
        min_beta, max_beta, grid_length, device=device, dtype=torch.float64
    )

    beta_grid_x, beta_grid_y = torch.meshgrid(x_grid, y_grid, indexing="ij")
    beta_grid_flat = torch.stack([beta_grid_x.flatten(), beta_grid_y.flatten()], dim=1)
    num_grid_points = beta_grid_flat.shape[0]

    # xbeta for all samples and grid points: (N, num_grid_points)
    xbeta_all_grid = design_matrix_t @ beta_grid_flat.T
    mu_all_grid = size_factors_t.unsqueeze(1).unsqueeze(2) * torch.exp(
        xbeta_all_grid.unsqueeze(1)
    )
    mu_all_grid = torch.clamp(mu_all_grid, min=min_mu)

    disp_expanded = disp_t.unsqueeze(1).expand(-1, num_grid_points)
    counts_expanded = counts_t.unsqueeze(2).expand(-1, -1, num_grid_points)

    ll_grid = torch_vec_nb_nll(counts_expanded, mu_all_grid, disp_expanded)

    reg_term = 0.5 * (1e-6 * beta_grid_flat**2).sum(dim=1)
    ll_grid += reg_term

    min_idx_flat = torch.argmin(ll_grid, dim=1)
    beta_initial_best = beta_grid_flat[min_idx_flat]

    # Fine grid -- vectorized
    delta_x = x_grid[1] - x_grid[0]
    delta_y = y_grid[1] - y_grid[0]

    t = torch.linspace(0, 1, grid_length, device=device, dtype=torch.float64).unsqueeze(
        0
    )

    fine_x = (beta_initial_best[:, 0] - delta_x).unsqueeze(1) + t * (2 * delta_x)
    fine_y = (beta_initial_best[:, 1] - delta_y).unsqueeze(1) + t * (2 * delta_y)

    # Build per-gene fine grids: (G, grid_length^2, 2)
    fine_gx = fine_x.unsqueeze(2).expand(-1, -1, grid_length)
    fine_gy = fine_y.unsqueeze(1).expand(-1, grid_length, -1)
    fine_beta_grid = torch.stack(
        [fine_gx.reshape(n_genes, -1), fine_gy.reshape(n_genes, -1)],
        dim=2,
    )

    # Evaluate: xbeta = design @ beta^T for each gene
    xbeta_fine = torch.einsum("np,gqp->ngq", design_matrix_t, fine_beta_grid)
    mu_fine = size_factors_t.unsqueeze(1).unsqueeze(2) * torch.exp(xbeta_fine)
    mu_fine = torch.clamp(mu_fine, min=min_mu)

    ll_fine = torch_vec_nb_nll(counts_expanded, mu_fine, disp_expanded)

    # Per-gene regularization for fine grid
    reg_fine = 0.5 * (1e-6 * fine_beta_grid**2).sum(dim=2)
    ll_fine += reg_fine

    min_idx_fine = torch.argmin(ll_fine, dim=1)
    beta_final = fine_beta_grid[torch.arange(n_genes, device=device), min_idx_fine]

    return beta_final.cpu().numpy()


def torch_nbinomFn(
    beta: torch.Tensor,
    design_matrix: torch.Tensor,
    counts: torch.Tensor,
    size: torch.Tensor,
    offset: torch.Tensor,
    prior_no_shrink_scale: float,
    prior_scale: float,
    shrink_index: int = 1,
) -> torch.Tensor:
    """Return the NB negative likelihood with apeGLM prior.

    Parameters
    ----------
    beta : torch.Tensor
        Coefficients ``(P, G)``.

    design_matrix : torch.Tensor
        Design matrix ``(N, P)``.

    counts : torch.Tensor
        Raw counts ``(N, G)``.

    size : torch.Tensor
        Size parameter (1/dispersion) ``(G,)``.

    offset : torch.Tensor
        Log size factors ``(N,)``.

    prior_no_shrink_scale : float
        Prior scale for non-shrunk coefficients.

    prior_scale : float
        Prior scale for the LFC.

    shrink_index : int
        Index of coefficient to shrink.

    Returns
    -------
    torch.Tensor
        Loss per gene ``(G,)``.
    """
    n_coeffs = beta.shape[0]

    shrink_mask = torch.zeros(n_coeffs, dtype=torch.float64, device=beta.device)
    shrink_mask[shrink_index] = 1
    no_shrink_mask = 1 - shrink_mask

    xbeta = design_matrix @ beta

    prior_term = (beta * no_shrink_mask[:, None]) ** 2 / (2 * prior_no_shrink_scale**2)
    prior = prior_term.sum(dim=0) + torch.log1p(
        (beta[shrink_index, :] / prior_scale) ** 2
    )

    nll = (
        counts * xbeta
        - (counts + size)
        * torch.logaddexp(xbeta + offset[:, None], torch.log(size[None, :]))
    ).sum(dim=0)

    return prior - nll


def torch_grid_fit_shrink_beta(
    counts: np.ndarray,
    offset: np.ndarray,
    design_matrix: np.ndarray,
    size: np.ndarray,
    prior_no_shrink_scale: float,
    prior_scale: float,
    scale_cnst: float,
    device: torch.device,
    grid_length: int = 60,
    min_beta: float = -30,
    max_beta: float = 30,
    shrink_index: int = 1,
) -> np.ndarray:
    """Find optimal LFC via 2D grid search with apeGLM prior.

    Parameters
    ----------
    counts : np.ndarray
        Raw counts ``(N, G)``.

    offset : np.ndarray
        Log size factors ``(N,)``.

    design_matrix : np.ndarray
        Design matrix ``(N, P)``.

    size : np.ndarray
        Size parameter (1/dispersion) ``(G,)``.

    prior_no_shrink_scale : float
        Prior scale for non-shrunk coefficients.

    prior_scale : float
        Prior scale for the LFC.

    scale_cnst : float
        Scaling constant for the loss.

    device : torch.device
        Device for tensors.

    grid_length : int
        Number of grid points per dimension.

    min_beta : float
        Lower bound on LFC.

    max_beta : float
        Upper bound on LFC.

    shrink_index : int
        Index of coefficient to shrink.

    Returns
    -------
    np.ndarray
        Fitted beta ``(G, P)``.
    """
    counts_t = torch.tensor(counts, dtype=torch.float64, device=device)
    offset_t = torch.tensor(offset, dtype=torch.float64, device=device)
    design_matrix_t = torch.tensor(design_matrix, dtype=torch.float64, device=device)
    size_t = torch.tensor(size, dtype=torch.float64, device=device)

    n_samples, n_genes = counts_t.shape
    n_coeffs = design_matrix_t.shape[1]

    assert n_coeffs == 2, (
        "torch_grid_fit_shrink_beta currently supports only 2 "
        "coefficients. For multi-factor designs, non-converged genes "
        "use CPU fallback."
    )

    x_grid = torch.linspace(
        min_beta, max_beta, grid_length, device=device, dtype=torch.float64
    )
    y_grid = torch.linspace(
        min_beta, max_beta, grid_length, device=device, dtype=torch.float64
    )

    beta_grid_x, beta_grid_y = torch.meshgrid(x_grid, y_grid, indexing="ij")
    beta_grid_flat = torch.stack([beta_grid_x.flatten(), beta_grid_y.flatten()], dim=1)

    # Vectorized coarse grid: evaluate all grid points at once
    # beta_grid_flat is (num_grid_points, P) -> need (P, G) for each
    # Batch evaluate: beta (P, num_grid_points) broadcast over genes
    # via modified torch_nbinomFn that handles (P, GP) beta for all genes

    # xbeta: (N, P) @ (P, num_grid_points) = (N, num_grid_points)
    xbeta_all = design_matrix_t @ beta_grid_flat.T

    shrink_mask = torch.zeros(n_coeffs, dtype=torch.float64, device=device)
    shrink_mask[shrink_index] = 1
    no_shrink_mask = 1 - shrink_mask

    prior_term = (beta_grid_flat.T * no_shrink_mask[:, None]) ** 2 / (
        2 * prior_no_shrink_scale**2
    )
    prior_all = prior_term.sum(dim=0) + torch.log1p(
        (beta_grid_flat[:, shrink_index] / prior_scale) ** 2
    )

    # NLL for each gene at each grid point
    # xbeta_all: (N, num_grid_points), counts_t: (N, G), size_t: (G,)
    # For each gene g and grid point gp:
    #   nll = sum_n(counts[n,g]*xbeta[n,gp] - (counts[n,g]+size[g])
    #              * logaddexp(xbeta[n,gp]+offset[n], log(size[g])))
    # Vectorize: (N, G, num_grid_points)
    xbeta_exp = xbeta_all.unsqueeze(1)  # (N, 1, GP)
    counts_exp = counts_t.unsqueeze(2)  # (N, G, 1)
    size_exp = size_t.unsqueeze(0).unsqueeze(2)  # (1, G, 1)
    offset_exp = offset_t.unsqueeze(1).unsqueeze(2)  # (N, 1, 1)

    nll_3d = (
        counts_exp * xbeta_exp
        - (counts_exp + size_exp)
        * torch.logaddexp(
            xbeta_exp + offset_exp,
            torch.log(size_exp),
        )
    ).sum(dim=0)  # (G, GP)

    ll_grid = (prior_all.unsqueeze(0) - nll_3d) / scale_cnst

    min_idx_flat = torch.argmin(ll_grid, dim=1)
    beta_initial_best = beta_grid_flat[min_idx_flat]

    # Fine grid -- vectorized
    delta_x = x_grid[1] - x_grid[0]
    delta_y = y_grid[1] - y_grid[0]

    t = torch.linspace(0, 1, grid_length, device=device, dtype=torch.float64).unsqueeze(
        0
    )

    fine_x = (beta_initial_best[:, 0] - delta_x).unsqueeze(1) + t * (2 * delta_x)
    fine_y = (beta_initial_best[:, 1] - delta_y).unsqueeze(1) + t * (2 * delta_y)

    # Per-gene fine grids: (G, GL^2, 2)
    fine_gx = fine_x.unsqueeze(2).expand(-1, -1, grid_length)
    fine_gy = fine_y.unsqueeze(1).expand(-1, grid_length, -1)
    fine_beta_grid = torch.stack(
        [fine_gx.reshape(n_genes, -1), fine_gy.reshape(n_genes, -1)],
        dim=2,
    )

    # Evaluate fine grid: xbeta_fine (N, G, GP)
    xbeta_fine = torch.einsum("np,gqp->ngq", design_matrix_t, fine_beta_grid)

    # Prior for fine grid: (G, GP)
    prior_fine_term = (
        fine_beta_grid * no_shrink_mask.unsqueeze(0).unsqueeze(0)
    ) ** 2 / (2 * prior_no_shrink_scale**2)
    prior_fine = prior_fine_term.sum(dim=2) + torch.log1p(
        (fine_beta_grid[:, :, shrink_index] / prior_scale) ** 2
    )

    # NLL fine: (G, GP)
    counts_exp_f = counts_t.unsqueeze(2)
    size_exp_f = size_t.unsqueeze(0).unsqueeze(2)
    offset_exp_f = offset_t.unsqueeze(1).unsqueeze(2)

    nll_fine = (
        counts_exp_f * xbeta_fine
        - (counts_exp_f + size_exp_f)
        * torch.logaddexp(
            xbeta_fine + offset_exp_f,
            torch.log(size_exp_f),
        )
    ).sum(dim=0)

    ll_fine = (prior_fine - nll_fine) / scale_cnst

    min_idx_fine = torch.argmin(ll_fine, dim=1)
    beta_final = fine_beta_grid[torch.arange(n_genes, device=device), min_idx_fine]

    return beta_final.cpu().numpy()
