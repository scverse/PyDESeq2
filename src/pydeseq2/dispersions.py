"""Estimation of gene-wise and trended dispersions."""

from typing import Literal
from typing import cast

import numpy as np
import pandas as pd
from scipy.optimize import minimize  # type: ignore
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
    nonzero_genes = ~(normed_counts == 0).all(axis=0)
    normed_counts = normed_counts[:, nonzero_genes]
    # Mean inverse size factor. For gene-specific normalization factors, DESeq2
    # first averages factors across genes within each sample, then averages the
    # inverse sample means into one scalar shared by all genes.
    if size_factors.ndim != 1:
        size_factors = size_factors[:, nonzero_genes].mean(axis=1)
    s_mean_inv = (1 / size_factors).mean()
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
