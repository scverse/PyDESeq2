from abc import ABC
from abc import abstractmethod
from typing import Literal

import numpy as np
import pandas as pd
from scipy.sparse import sparray
from scipy.sparse import spmatrix

type CountMatrix = np.ndarray | spmatrix | sparray


class Inference(ABC):
    """Abstract class with DESeq2-related inference methods."""

    @abstractmethod
    def lin_reg_mu(
        self,
        counts: CountMatrix,
        size_factors: np.ndarray,
        design_matrix: np.ndarray,
        min_mu: float,
    ) -> np.ndarray:
        """Estimate mean of negative binomial model using a linear regression.

        Used to initialize genewise dispersion models.

        Parameters
        ----------
        counts
            Raw count matrix.

        size_factors
            Sample-wise scaling factors with shape ``(n_samples,)``, or
            sample-by-gene normalization factors with shape
            ``(n_samples, n_genes)`` (obtained from median-of-ratios and optional
            gene-specific offsets).

        design_matrix
            Design matrix.
        min_mu
            Lower threshold for fitted means, for numerical stability. (default: ``0.5``).

        Returns
        -------
        Estimated mean.
        """

    @abstractmethod
    def irls(
        self,
        counts: CountMatrix,
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
        r"""Fit a NB GLM wit log-link to predict counts from the design matrix.

        See equations (1-2) in the DESeq2 paper.

        Parameters
        ----------
        counts
            Raw count matrix.

        size_factors
            Sample-wise scaling factors with shape ``(n_samples,)``, or
            sample-by-gene normalization factors with shape
            ``(n_samples, n_genes)`` (obtained from median-of-ratios and optional
            gene-specific offsets).

        design_matrix
            Design matrix.
        disp
            Gene-wise dispersion prior.
        min_mu
            Lower bound on estimated means, to ensure numerical stability. (default: ``0.5``).
        beta_tol
            Stopping criterion for IRWLS: :math:`\vert dev - dev_{old}\vert / \vert dev + 0.1 \vert < \beta_{tol}`. (default: ``1e-8``).
        min_beta
            Lower-bound on LFC. (default: ``-30``).
        max_beta
            Upper-bound on LFC. (default: ``-30``).
        optimizer
            Optimizing method to use in case IRLS starts diverging.
            Accepted values: 'BFGS' or 'L-BFGS-B'.
            NB: only 'L-BFGS-B' ensures that LFCS will lay in the [min_beta, max_beta] range. (default: ``'L-BFGS-B'``).
        maxiter
            Maximum number of IRLS iterations to perform before switching to L-BFGS-B. (default: ``250``).

        Returns
        -------
        beta
            Fitted (basemean, lfc) coefficients of negative binomial GLM.
        mu
            Means estimated from size factors and beta: :math:`\mu = s_{ij} \exp(\beta^t X)`.
        H
            Diagonal of the :math:`W^{1/2} X (X^t W X)^-1 X^t W^{1/2}` covariance matrix.
        converged
            Whether IRLS or the optimizer converged.
            If not and if dimension allows it, perform grid search.
        """

    @abstractmethod
    def alpha_mle(
        self,
        counts: CountMatrix,
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
        """Estimate the dispersion parameter of a negative binomial GLM.

        Parameters
        ----------
        counts
            Raw count matrix.

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

    @abstractmethod
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
        """Run Wald test for differential expression.

        Computes Wald statistics, standard error and p-values from dispersion and LFC estimates.

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
        alt_hypothesis
            The alternative hypothesis for computing wald p-values.

        Returns
        -------
        wald_p_value
            Estimated p-value.
        wald_statistic
            Wald statistic.
        wald_se
            Standard error of the Wald statistic.
        """

    @abstractmethod
    def fit_rough_dispersions(
        self, normed_counts: np.ndarray, design_matrix: np.ndarray
    ) -> np.ndarray:
        """'Rough dispersion' estimates from linear model, as per the R code.

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

    @abstractmethod
    def fit_moments_dispersions(
        self, normed_counts: np.ndarray, size_factors: np.ndarray
    ) -> np.ndarray:
        """Dispersion estimates based on moments, as per the R code.

        Used as initial estimates in :meth:`DeseqDataSet.fit_genewise_dispersions() <pydeseq2.dds.DeseqDataSet.fit_genewise_dispersions>`.

        Parameters
        ----------
        normed_counts
            Array of deseq2-normalized read counts. Rows: samples, columns: genes.

        size_factors
            Sample-wise scaling factors with shape ``(n_samples,)``, or
            sample-by-gene normalization factors with shape
            ``(n_samples, n_genes)``.

        Returns
        -------
        Estimated dispersion parameter for each gene.
        """

    @abstractmethod
    def dispersion_trend_gamma_glm(
        self, covariates: pd.Series, targets: pd.Series
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """Fit a gamma glm on gene dispersions.

        The intercept should be concatenated in this method and the first returned coefficient should be the intercept.

        Parameters
        ----------
        covariates
            Covariates for the regression (num_genes,).
        targets
            Targets for the regression (num_genes,).

        Returns
        -------
        coeffs
            Coefficients of the regression.
        predictions
            Predictions of the regression.
        converged
            Whether the optimization converged.
        """

    @abstractmethod
    def lfc_shrink_nbinom_glm(
        self,
        design_matrix: np.ndarray,
        counts: CountMatrix,
        size: np.ndarray,
        offset: np.ndarray,
        prior_no_shrink_scale: float,
        prior_scale: float,
        optimizer: str,
        shrink_index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit a negative binomial MAP LFC using an apeGLM prior.

        Only the LFC is shrinked, and not the intercept.

        Parameters
        ----------
        design_matrix
            Design matrix.

        counts
            Raw count matrix.

        size
            Size parameter of NB family (inverse of dispersion).

        offset
            Natural logarithm of sample-wise size factors with shape
            ``(n_samples,)``, or sample-by-gene normalization factors with shape
            ``(n_samples, n_genes)``.

        prior_no_shrink_scale
            Prior variance for the intercept.
        prior_scale
            Prior variance for the LFC parameter.
        optimizer
            Optimizing method to use in case IRLS starts diverging.
            Accepted values: 'L-BFGS-B', 'BFGS' or 'Newton-CG'.
        shrink_index
            Index of the LFC coordinate to shrink. (default: ``1``).

        Returns
        -------
        beta
            2-element array, containing the intercept (first) and the LFC (second).
        inv_hessian
            Inverse of the Hessian of the objective at the estimated MAP LFC.
        converged
            Whether L-BFGS-B converged for each optimization problem.
        """
