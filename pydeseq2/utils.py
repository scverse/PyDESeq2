import multiprocessing
from math import ceil
from math import floor
from pathlib import Path
from typing import Literal
from typing import cast

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.linalg import solve  # type: ignore
from scipy.optimize import minimize  # type: ignore
from scipy.special import gammaln  # type: ignore
from scipy.special import polygamma  # type: ignore
from scipy.stats import norm  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore

import pydeseq2
from pydeseq2.grid_search import grid_fit_alpha
from pydeseq2.grid_search import grid_fit_beta
from pydeseq2.grid_search import grid_fit_shrink_beta


def load_example_data(
    modality: Literal["raw_counts", "metadata"] = "raw_counts",
    dataset: Literal["synthetic"] = "synthetic",
    debug: bool = False,
    debug_seed: int = 42,
) -> pd.DataFrame:
    """Load synthetic example data.

    May load either metadata or rna-seq data. For now, this function may only return the
    synthetic data provided as part of this repo, but new datasets might be added in the
    future.

    Parameters
    ----------
    modality : str
        Data modality. "raw_counts" or "metadata".

    dataset : str
        The dataset for which to return gene expression data.
        If "synthetic", will return the synthetic data that is used for CI unit tests.
        (default: ``"synthetic"``).

    debug : bool
        If true, subsample 10 samples and 100 genes at random.
        (Note that the "synthetic" dataset is already 10 features 100.)
        (default: ``False``).

    debug_seed : int
        Seed for the debug mode. (default: ``42``).

    Returns
    -------
    pandas.DataFrame
        Requested data modality.
    """
    assert modality in [
        "raw_counts",
        "metadata",
    ], "The modality argument must be one of the following: raw_counts, metadata"

    assert dataset in ["synthetic"], (
        "The dataset argument must be one of the following: synthetic."
    )

    # Load data
    datasets_path = Path(pydeseq2.__file__).parent.parent / "datasets"

    if dataset == "synthetic":
        path_to_data = datasets_path / "synthetic"
        if Path(path_to_data).is_dir():
            # Cast the Paths to strings to have coherent types wrt to the url case (that
            # does not handle Paths), else mypy throws an error.
            path_to_data_counts = str(path_to_data / "test_counts.csv")
            path_to_data_metadata = str(path_to_data / "test_metadata.csv")
        else:
            # if the path does not exist (as is the case in RDT) load it from github
            url_to_data = (
                "https://raw.githubusercontent.com/owkin/"
                "PyDESeq2/main/datasets/synthetic/"
            )
            path_to_data_counts = url_to_data + "/test_counts.csv"
            path_to_data_metadata = url_to_data + "/test_metadata.csv"

        if modality == "raw_counts":
            df = pd.read_csv(
                path_to_data_counts,
                sep=",",
                index_col=0,
            ).T
        elif modality == "metadata":
            df = pd.read_csv(
                path_to_data_metadata,
                sep=",",
                index_col=0,
            )

    if debug:
        # TODO: until we provide a larger dataset, this option is useless
        # subsample 10 samples and 100 genes
        df = df.sample(n=10, axis=0, random_state=debug_seed)
        if modality == "raw_counts":
            df = df.sample(n=100, axis="index", random_state=debug_seed)

    return df


def test_valid_counts(counts: pd.DataFrame | np.ndarray) -> None:
    """Test that the count matrix contains valid inputs.

    More precisely, test that inputs are non-negative integers.

    Parameters
    ----------
    counts : pandas.DataFrame or ndarray
        Raw counts. One column per gene, rows are indexed by sample barcodes.
    """
    if isinstance(counts, pd.DataFrame):
        if counts.isna().any().any():
            raise ValueError("NaNs are not allowed in the count matrix.")
        if not np.issubdtype(counts.to_numpy().dtype, np.number):
            raise ValueError("The count matrix should only contain numbers.")
    else:
        if np.isnan(counts).any().any():
            raise ValueError("NaNs are not allowed in the count matrix.")
        if not np.issubdtype(counts.dtype, np.number):
            raise ValueError("The count matrix should only contain numbers.")
    if (counts % 1 != 0).any().any():
        raise ValueError("The count matrix should only contain integers.")
    if (counts < 0).any().any():
        raise ValueError("The count matrix should only contain non-negative values.")


def dispersion_trend(
    normed_mean: float | np.ndarray,
    coeffs: pd.Series | np.ndarray,
) -> float | np.ndarray:
    r"""Return dispersion trend from normalized counts.

    :math:`a_1/ \mu + a_0`.

    Parameters
    ----------
    normed_mean : float or ndarray
        Mean of normalized counts for a given gene or set of genes.

    coeffs : ndarray or pd.Series
        Fitted dispersion trend coefficients :math:`a_0` and :math:`a_1`.

    Returns
    -------
    float or ndarray
        Dispersion trend :math:`a_1/ \mu + a_0`.
    """
    if isinstance(coeffs, pd.Series):
        return coeffs["a0"] + coeffs["a1"] / normed_mean
    else:
        return coeffs[0] + coeffs[1] / normed_mean


def nb_nll(
    counts: np.ndarray, mu: np.ndarray, alpha: float | np.ndarray
) -> float | np.ndarray:
    r"""Neg log-likelihood of a negative binomial of parameters ``mu`` and ``alpha``.

    Mathematically, if ``counts`` is a vector of counting entries :math:`y_i`
    then the likelihood of each entry :math:`y_i` to be drawn from a negative
    binomial :math:`NB(\mu, \alpha)` is [1]

    .. math::
        p(y_i | \mu, \alpha) = \frac{\Gamma(y_i + \alpha^{-1})}{
            \Gamma(y_i + 1)\Gamma(\alpha^{-1})
        }
        \left(\frac{1}{1 + \alpha \mu} \right)^{1/\alpha}
        \left(\frac{\mu}{\alpha^{-1} + \mu} \right)^{y_i}

    As a consequence, assuming there are :math:`n` entries,
    the total negative log-likelihood for ``counts`` is

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
    counts : ndarray
        Observations.

    mu : ndarray
        Mean of the distribution :math:`\mu`.

    alpha : float or ndarray
        Dispersion of the distribution :math:`\alpha`,
        s.t. the variance is :math:`\mu + \alpha \mu^2`.

    Returns
    -------
    float or ndarray
        Negative log likelihood of the observations counts
        following :math:`NB(\mu, \alpha)`.

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
    counts : ndarray
        Observations.

    mu : float
        Mean of the distribution.

    alpha : float
        Dispersion of the distribution,
        s.t. the variance is :math:`\mu + \alpha\mu^2`.

    Returns
    -------
    float
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
    counts : ndarray
        Raw counts for a given gene.

    size_factors : ndarray
        Sample-wise scaling factors (obtained from median-of-ratios).

    design_matrix : ndarray
        Design matrix.

    disp : float
        Gene-wise dispersion prior.

    min_mu : float
        Lower bound on estimated means, to ensure numerical stability.
        (default: ``0.5``).

    beta_tol : float
        Stopping criterion for IRWLS:
        :math:`\vert dev - dev_{old}\vert / \vert dev + 0.1 \vert < \beta_{tol}`.
        (default: ``1e-8``).

    min_beta : float
        Lower-bound on LFC. (default: ``-30``).

    max_beta : float
        Upper-bound on LFC. (default: ``-30``).

    optimizer : str
        Optimizing method to use in case IRLS starts diverging.
        Accepted values: 'BFGS' or 'L-BFGS-B'.
        NB: only 'L-BFGS-B' ensures that LFCS will
        lay in the [min_beta, max_beta] range. (default: ``'L-BFGS-B'``).

    maxiter : int
        Maximum number of IRLS iterations to perform before switching to L-BFGS-B.
        (default: ``250``).

    Returns
    -------
    beta: ndarray
        Fitted (basemean, lfc) coefficients of negative binomial GLM.

    mu: ndarray
        Means estimated from size factors and beta: :math:`\mu = s_{ij} \exp(\beta^t X)`.

    H: ndarray
        Diagonal of the :math:`W^{1/2} X (X^t W X)^-1 X^t W^{1/2}` covariance matrix.

    converged: bool
        Whether IRLS or the optimizer converged. If not and if dimension allows it,
        perform grid search.
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

    Note: it is possible to pass counts, design_matrix and mu arguments in the form of
    pandas Series, but using numpy arrays makes the code significantly faster.

    Parameters
    ----------
    counts : ndarray
        Raw counts for a given gene.

    design_matrix : ndarray
        Design matrix.

    mu : ndarray
        Mean estimation for the NB model.

    alpha_hat : float
        Initial dispersion estimate.

    min_disp : float
        Lower threshold for dispersion parameters.

    max_disp : float
        Upper threshold for dispersion parameters.

    prior_disp_var : float
        Prior dispersion variance.

    cr_reg : bool
        Whether to use Cox-Reid regularization. (default: ``True``).

    prior_reg : bool
        Whether to use prior log-residual regularization. (default: ``False``).

    optimizer : str
        Optimizing method to use. Accepted values: 'BFGS' or 'L-BFGS-B'.
        (default: ``'L-BFGS-B'``).

    Returns
    -------
    float
        Dispersion estimate.

    bool
        Whether L-BFGS-B converged. If not, dispersion is estimated using grid search.
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


def trimmed_mean(x: np.ndarray, trim: float = 0.1, **kwargs) -> float | np.ndarray:
    """Return trimmed mean.

    Compute the mean after trimming data of its smallest and largest quantiles.

    Parameters
    ----------
    x : ndarray
        Data whose mean to compute.

    trim : float
        Fraction of data to trim at each end. (default: ``0.1``).

    **kwargs
        Keyword arguments, useful to pass axis.

    Returns
    -------
    float or ndarray :
        Trimmed mean.
    """
    assert trim <= 0.5
    if "axis" in kwargs:
        axis = kwargs["axis"]
        s = np.sort(x, axis=axis)
        n = x.shape[axis]
        ntrim = floor(n * trim)
        return np.take(s, np.arange(ntrim, n - ntrim), axis).mean(axis)
    else:
        n = len(x)
        s = np.sort(x)
        ntrim = floor(n * trim)
        return s[ntrim : n - ntrim].mean()


def trimmed_cell_variance(counts: np.ndarray, cells: pd.Series) -> np.ndarray:
    """Return trimmed variance of counts according to condition.

    Compute the variance after trimming data of its smallest and largest elements,
    grouped by cohorts, and return the max across cohorts.
    The trim factor is a function of data size.

    Parameters
    ----------
    counts : ndarray
        Sample-wise gene counts.

    cells : pandas.Series
        Cohort affiliation of each sample.

    Returns
    -------
    ndarray :
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
    features : ndarray
        Data whose trimmed variance to compute.

    trim : float
        Fraction of data to trim at each end. (default: ``0.125``).

    axis : int
        Dimension along which to compute variance. (default: ``0``).

    Returns
    -------
    float or ndarray
        Trimmed variances.
    """
    rm = trimmed_mean(x, trim=trim, axis=axis)
    sqerror = (x - rm) ** 2
    # scale due to trimming of large squares
    return 1.51 * trimmed_mean(sqerror, trim=trim, axis=axis)


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
    counts : ndarray
        Raw counts for a given gene.

    size_factors : ndarray
        Sample-wise scaling factors (obtained from median-of-ratios).

    design_matrix : ndarray
        Design matrix.

    min_mu : float
        Lower threshold for fitted means, for numerical stability. (default: ``0.5``).

    Returns
    -------
    ndarray
        Estimated mean.
    """
    reg = LinearRegression(fit_intercept=False)
    reg.fit(design_matrix, counts / size_factors)
    mu_hat = size_factors * reg.predict(design_matrix)
    # Threshold mu_hat as 1/mu_hat will be used later on.
    return np.maximum(mu_hat, min_mu)


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

    Computes Wald statistics, standard error and p-values from
    dispersion and LFC estimates.

    Parameters
    ----------
    design_matrix : ndarray
        Design matrix.

    disp : float
        Dispersion estimate.

    lfc : ndarray
        Log-fold change estimate (in natural log scale).

    mu : ndarray
        Mean estimation for the NB model.

    ridge_factor : ndarray
        Regularization factors.

    contrast : ndarray
        Vector encoding the contrast that is being tested.

    lfc_null : float
        The (log2) log fold change under the null hypothesis.

    alt_hypothesis : str, optional
        The alternative hypothesis for computing wald p-values.

    Returns
    -------
    wald_p_value : float
        Estimated p-value.

    wald_statistic : float
        Wald statistic.

    wald_se : float
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


def fit_rough_dispersions(
    normed_counts: np.ndarray, design_matrix: pd.DataFrame
) -> np.ndarray:
    """Rough dispersion estimates from linear model, as per the R code.

    Used as initial estimates in :meth:`DeseqDataSet.fit_genewise_dispersions()
    <pydeseq2.dds.DeseqDataSet.fit_genewise_dispersions>`.

    Parameters
    ----------
    normed_counts : ndarray
        Array of deseq2-normalized read counts. Rows: samples, columns: genes.

    design_matrix : pandas.DataFrame
        A DataFrame with experiment design information (to split cohorts).
        Indexed by sample barcodes. Unexpanded, *with* intercept.

    Returns
    -------
    ndarray
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

    Used as initial estimates in :meth:`DeseqDataSet.fit_genewise_dispersions()
    <pydeseq2.dds.DeseqDataSet.fit_genewise_dispersions>`.

    Parameters
    ----------
    normed_counts : ndarray
        Array of deseq2-normalized read counts. Rows: samples, columns: genes.

    size_factors : ndarray
        DESeq2 normalization factors.

    Returns
    -------
    ndarray
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


def n_or_more_replicates(design_matrix: pd.DataFrame, min_replicates: int) -> pd.Series:
    """
    Return a  series indicating whether samples have a minimum number of replicates.

    Checks whether each sample has at least ``min_replicates`` replicates, based on its
    combination of design factors.

    Parameters
    ----------
    design_matrix : pandas.DataFrame
        A DataFrame with experiment design information (to split cohorts).
    min_replicates : int
        The minimum number of replicates to have to pass the threshold.

    Returns
    -------
    pandas.Series
        A boolean series indicating whether each sample has at least ``min_replicates``
        replicates.
    """
    n_or_more = design_matrix.value_counts() >= min_replicates
    replaceable = n_or_more[pd.MultiIndex.from_frame(design_matrix)]
    replaceable.index = design_matrix.index
    return replaceable


def robust_method_of_moments_disp(
    normed_counts: np.ndarray, design_matrix: pd.DataFrame
) -> np.ndarray:
    """Perform dispersion estimation using a method of trimmed moments.

    Used for outlier detection based on Cook's distance.

    Parameters
    ----------
    normed_counts : ndarray
        Array of deseq2-normalized read counts. Rows: samples, columns: genes.

    design_matrix : pandas.DataFrame
        A DataFrame with experiment design information (to split cohorts).
        Indexed by sample barcodes. Unexpanded, *with* intercept.

    Returns
    -------
    ndarray
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


def get_num_processes(n_cpus: int | None) -> int:
    """Return the number of processes to use for multiprocessing.

    Returns the maximum number of available cpus by default.

    Parameters
    ----------
    n_cpus : int, optional
        Desired number of cpus. If ``None``, will return the number of available cpus.
        (default: ``None``).

    Returns
    -------
    int
        Number of processes to spawn.
    """
    if n_cpus is None:
        try:
            n_processes = multiprocessing.cpu_count()
        except NotImplementedError:
            n_processes = 5  # arbitrary default
    else:
        n_processes = n_cpus

    return n_processes


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
    design_matrix : ndarray
        Design matrix.

    counts : ndarray
        Raw counts.

    size : ndarray
        Size parameter of NB family (inverse of dispersion).

    offset : ndarray
        Natural logarithm of size factor.

    prior_no_shrink_scale : float
        Prior variance for the intercept.

    prior_scale : float
        Prior variance for the LFC parameter.

    optimizer : str
        Optimizing method to use in case IRLS starts diverging.
        Accepted values: 'L-BFGS-B', 'BFGS' or 'Newton-CG'. (default: ``'Newton-CG'``).

    shrink_index : int
        Index of the LFC coordinate to shrink. (default: ``1``).

    Returns
    -------
    beta: ndarray
        2-element array, containing the intercept (first) and the LFC (second).

    inv_hessian: ndarray
        Inverse of the Hessian of the objective at the estimated MAP LFC.

    converged: bool
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
    beta : ndarray
        2-element array: intercept and LFC coefficients.

    design_matrix : ndarray
        Design matrix.

    counts : ndarray
        Raw counts.

    size : ndarray
        Size parameter of NB family (inverse of dispersion).

    offset : ndarray
        Natural logarithm of size factor.

    prior_no_shrink_scale : float
        Prior variance for the intercept.

    prior_scale : float
        Prior variance for the intercept.

    shrink_index : int
        Index of the LFC coordinate to shrink. (default: ``1``).

    Returns
    -------
    float
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


def mean_absolute_deviation(x: np.ndarray) -> float:
    """
    Compute a scaled estimator of the mean absolute deviation.

    Used in :meth:`pydeseq2.dds.DeseqDataSet.fit_dispersion_prior()`.

    Parameters
    ----------
    features : ndarray
        1D array whose MAD to compute.

    Returns
    -------
    float
        Mean absolute deviation estimator.
    """
    center = np.median(x)
    return np.median(np.abs(x - center)) / norm.ppf(0.75)


def make_scatter(
    disps: list,
    legend_labels: list,
    x_val: np.ndarray,
    log: bool = True,
    save_path: str | None = None,
    **kwargs,
) -> None:
    """
    Create a scatter plot using matplotlib.

    Used in :meth:`pydeseq2.dds.DeseqDataSet.plot_dispersions()`.

    Parameters
    ----------
    disps : list
        List of ndarrays to plot.

    legend_labels : list
        List of strings that correspond to plotted targets values for legend.

    x_val : ndarray
        1D array to plot (example: ``dds.varm['_normed_means']``).

    log : bool
        Whether or not to log scale features and targets axes (``default=True``).

    save_path : str, optional
        The path where to save the plot. If left None, the plot won't be saved
        (``default=None``).

    **kwargs
        Matplotlib keyword arguments for the scatter plot.
    """
    # Adding more colors if plotting more than 3 traces
    if len(disps) == 3:
        colors = "kbr"
    else:
        colors = "kbrcmyg"

    # Standardizing font; init plot
    plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(dpi=600)

    # log scale axes
    if log is True:
        plt.yscale("log")
        plt.xscale("log")

    # scale axes according to data spread
    ax.set_adjustable("datalim")

    # Set default alpha and s parameters, if not already specified
    kwargs.setdefault("alpha", 0.5)
    kwargs.setdefault("s", 0.6)

    # create scatter plot per trace
    for disp, color in list(zip(disps, colors, strict=False)):
        plt.scatter(x=x_val, y=disp, c=color, **kwargs)

    # label legend + axes
    plt.legend(legend_labels, loc="best")
    plt.xlabel("mean of normalized counts")
    plt.ylabel("dispersion")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def make_MA_plot(
    results_df: pd.DataFrame,
    padj_thresh: float = 0.05,
    log: bool = True,
    save_path: str | None = None,
    lfc_null: float = 0,
    alt_hypothesis: Literal["greaterAbs", "lessAbs", "greater", "less"] | None = None,
    **kwargs,
) -> None:
    """
    Create an log ratio (M)-average (A) plot using matplotlib.

    Useful for looking at log fold-change versus mean expression
    between two groups/samples/etc.
    Uses matplotlib to emulate the ``make_MA()`` function in DESeq2 in R.

    Parameters
    ----------
    results_df : pd.DataFrame
        Resultant dataframe after running DeseqStats() and .summary().

    padj_thresh : float
        P-value threshold to subset scatterplot colors on.

    log : bool
        Whether or not to log scale features and targets axes (``default=True``).

    save_path : str, optional
        The path where to save the plot. If left None, the plot won't be saved
        (``default=None``).

    lfc_null : float
        The (log2) log fold change under the null hypothesis. (default: ``0``).

    alt_hypothesis : str, optional
        The alternative hypothesis for computing wald p-values. (default: ``None``).

    **kwargs
        Matplotlib keyword arguments for the scatter plot.
    """
    colors = results_df["padj"].apply(lambda x: "darkred" if x < padj_thresh else "gray")

    fig, ax = plt.subplots(dpi=600)

    # Set default alpha and s parameters, if not already specified
    kwargs.setdefault("alpha", 0.5)
    kwargs.setdefault("s", 0.2)

    plt.scatter(
        x=results_df["baseMean"],
        y=results_df["log2FoldChange"],
        c=colors,
        **kwargs,
    )

    ax.set_adjustable("datalim")

    if log is True:
        plt.xscale("log")

    plt.xlabel("mean of normalized counts")
    plt.ylabel("log2 fold change")

    plt.axhline(lfc_null, color="red", alpha=0.5, linestyle="--", zorder=3)
    if alt_hypothesis and alt_hypothesis in ["greaterAbs", "lessAbs"]:
        plt.axhline(-lfc_null, color="red", alpha=0.5, linestyle="--", zorder=3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")


# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

# Adapted from https://gist.github.com/agramfort/850437


def lowess(
    features: np.ndarray, targets: np.ndarray, frac: float = 2.0 / 3.0, iter: int = 3
):
    """Run lowess smoothing: Robust locally weighted regression.

    The lowess function fits a nonparametric regression curve to a scatterplot.
    The arrays features and targets contain an equal number of elements; each pair
    (features[i], targets[i]) defines a data point in the scatterplot. The function
    returns the estimated (smooth) values of targets.
    The smoothing span is given by frac. A larger value for frac will result in a
    smoother curve. The number of robustifying iterations is given by iter. The
    function will run faster with a smaller number of iterations.

    Parameters
    ----------
    features : ndarray
        A 1D array of data points.
    targets : ndarray
        A 1D array of target values (with the same shape as features).
    frac : float
        The fraction of the data used when estimating each y-value. (default: ``2/3``).
    iter : int
        The number of robustifying iterations. (default: ``3``).

    Returns
    -------
    ndarray
        Estimated (smooth) values of targets.
    """
    n = len(features)
    r = int(ceil(frac * n))
    h = np.maximum(
        np.array([np.sort(np.abs(features - features[i]))[r] for i in range(n)]), 1e-12
    )
    w = np.clip(
        np.abs(np.nan_to_num((features[:, None] - features[None, :]) / h)), 0.0, 1.0
    )
    w = (1 - w**3) ** 3
    yest = np.zeros(n)
    delta = np.ones(n)
    for _ in range(iter):
        for i in range(n):
            weights = delta * w[:, i]
            b = np.array(
                [np.sum(weights * targets), np.sum(weights * targets * features)]
            )
            A = np.array(
                [
                    [np.sum(weights), np.sum(weights * features)],
                    [np.sum(weights * features), np.sum(weights * features * features)],
                ]
            )
            beta = np.linalg.lstsq(A, b, rcond=None)[0]
            yest[i] = beta[0] + beta[1] * features[i]

        residuals = targets - yest
        s = np.median(np.abs(residuals))
        if s == 0:
            delta = (np.abs(residuals) > 0).astype(float)
        else:
            delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta**2) ** 2

    return yest
