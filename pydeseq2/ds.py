import sys
import time
import warnings
from typing import Literal

# import anndata as ad
import numpy as np
import pandas as pd
from scipy.optimize import root_scalar  # type: ignore
from scipy.stats import false_discovery_control  # type: ignore

from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.inference import Inference
from pydeseq2.utils import lowess
from pydeseq2.utils import make_MA_plot


class DeseqStats:
    """PyDESeq2 statistical tests for differential expression.

    Implements p-value estimation for differential gene expression according
    to the DESeq2 pipeline :cite:p:`DeseqStats-love2014moderated`.

    Also supports apeGLM log-fold change shrinkage :cite:p:`DeseqStats-zhu2019heavy`.

    Parameters
    ----------
    dds : DeseqDataSet
        DeseqDataSet for which dispersion and LFCs were already estimated.

    contrast : list or ndarray
        Either a list of three strings or a numpy array.
        If a list of three strings, it must be in the following format:
        ``['variable_of_interest', 'tested_level', 'ref_level']``.
        Names must correspond to the metadata data passed to the DeseqDataSet.
        E.g., ``['condition', 'B', 'A']`` will measure the LFC of 'condition B' compared
        to 'condition A'.
        If a numpy array, it must be a contrast vector of the same length as the design
        matrix.

    alpha : float
        P-value and adjusted p-value significance threshold (usually 0.05).
        (default: ``0.05``).

    cooks_filter : bool
        Whether to filter p-values based on cooks outliers. (default: ``True``).

    independent_filter : bool
        Whether to perform independent filtering to correct p-value trends.
        (default: ``True``).

    prior_LFC_var : ndarray
        Prior variance for LFCs, used for ridge regularization. (default: ``None``).

    lfc_null : float
        The (log2) log fold change under the null hypothesis. (default: ``0``).

    alt_hypothesis : str, optional
        The alternative hypothesis for computing wald p-values. By default, the normal
        Wald test assesses deviation of the estimated log fold change from the null
        hypothesis, as given by ``lfc_null``.
        One of ``["greaterAbs", "lessAbs", "greater", "less"]`` or ``None``.
        The alternative hypothesis corresponds to what the user wants to find rather
        than the null hypothesis. (default: ``None``).

    inference : Inference
        Implementation of inference routines object instance.
        (default:
        :class:`DefaultInference <pydeseq2.default_inference.DefaultInference>`).

    quiet : bool
        Suppress deseq2 status updates during fit.

    Attributes
    ----------
    base_mean : pandas.Series
        Genewise means of normalized counts.

    lfc_null : float
        The (log2) log fold change under the null hypothesis.

    alt_hypothesis : str, optional
        The alternative hypothesis for computing wald p-values.

    contrast_vector : ndarray
        Vector encoding the contrast (variable being tested).

    contrast_idx : int
        Index of the LFC column corresponding to the variable being tested.

    design_matrix : pandas.DataFrame
        A DataFrame with experiment design information (to split cohorts).
        Indexed by sample barcodes. Depending on the contrast that is provided to the
        DeseqStats object, it may differ from the DeseqDataSet design matrix, as the
        reference level may need to be adapted.

    LFC : pandas.DataFrame
        Estimated log-fold change between conditions and intercept, in natural log scale.

    SE : pandas.Series
        Standard LFC error.

    statistics : pandas.Series
        Wald statistics.

    p_values : pandas.Series
        P-values estimated from Wald statistics.

    padj : pandas.Series
        P-values adjusted for multiple testing.

    results_df : pandas.DataFrame
        Summary of the statistical analysis.

    shrunk_LFCs : bool
        Whether LFCs are shrunk.

    n_processes : int
        Number of threads to use for multiprocessing.

    quiet : bool
        Suppress deseq2 status updates during fit.

    References
    ----------
    .. bibliography::
        :keyprefix: DeseqStats-
    """

    def __init__(
        self,
        dds: DeseqDataSet,
        contrast: list[str] | np.ndarray,
        alpha: float = 0.05,
        cooks_filter: bool = True,
        independent_filter: bool = True,
        prior_LFC_var: np.ndarray | None = None,
        lfc_null: float = 0.0,
        alt_hypothesis: (
            Literal["greaterAbs", "lessAbs", "greater", "less"] | None
        ) = None,
        inference: Inference | None = None,
        quiet: bool = False,
        n_cpus: int | None = None,
    ) -> None:
        assert "LFC" in dds.varm, (
            "Please provide a fitted DeseqDataSet by first running the `deseq2` method."
        )

        self.dds = dds

        self.alpha = alpha
        self.cooks_filter = cooks_filter
        self.independent_filter = independent_filter
        self.base_mean = self.dds.var["_normed_means"].copy()
        self.prior_LFC_var = prior_LFC_var

        if lfc_null < 0 and alt_hypothesis in {"greaterAbs", "lessAbs"}:
            raise ValueError(
                f"The alternative hypothesis being {alt_hypothesis}, please provide a",
                f"positive lfc_null value (got {lfc_null}).",
            )
        self.lfc_null = lfc_null
        self.alt_hypothesis = alt_hypothesis

        # Initialize the design matrix and LFCs. If the chosen reference level are the
        # same as in dds, keep them unchanged. Otherwise, change reference level.
        self.design_matrix = self.dds.obsm["design_matrix"].copy()
        self.LFC = self.dds.varm["LFC"].copy()

        # Check the validity of the contrast (if provided) or build it.
        self.contrast: list[str] | np.ndarray
        if contrast is None:
            raise ValueError(
                """Default contrasts are no longer supported.
                The "contrast" argument must be provided."""
            )
        elif isinstance(contrast, np.ndarray):
            if contrast.shape[0] != self.dds.obsm["design_matrix"].shape[1]:
                raise ValueError(
                    "The contrast vector must have the same length as the design matrix."
                )
            self.contrast = contrast
            self.contrast_vector = contrast
        else:
            self.contrast = contrast
            self._build_contrast_vector()

        # Set a flag to indicate that LFCs are unshrunk
        self.shrunk_LFCs = False
        self.quiet = quiet

        if inference:
            if n_cpus:
                if hasattr(inference, "n_cpus"):
                    inference.n_cpus = n_cpus
                else:
                    warnings.warn(
                        "The provided inference object does not have an n_cpus "
                        "attribute, cannot override `n_cpus`.",
                        UserWarning,
                        stacklevel=2,
                    )

        # Initialize the inference object.
        self.inference = inference or DefaultInference(n_cpus=n_cpus)

        # If the `refit_cooks` attribute of the dds object is True, check that outliers
        # were actually refitted.
        if self.dds.refit_cooks and "replaced" not in self.dds.var:
            raise AttributeError(
                "dds has 'refit_cooks' set to True but Cooks outliers have not been "
                "refitted. Please run 'dds.refit()' first or set 'dds.refit_cooks' "
                "to False."
            )

    @property
    def variables(self):
        """Get the names of the variables used in the model definition."""
        return self.dds.variables

    def summary(
        self,
        **kwargs,
    ) -> None:
        """Run the statistical analysis.

        The results are stored in the ``results_df`` attribute.

        Parameters
        ----------
        **kwargs
            Keyword arguments: providing new values for ``lfc_null`` or
            ``alt_hypothesis`` will override the corresponding ``DeseqStat`` attributes.
        """
        new_lfc_null = kwargs.get("lfc_null", "default")
        new_alt_hypothesis = kwargs.get("alt_hypothesis", "default")

        rerun_summary = False
        if new_lfc_null == "default":
            lfc_null = self.lfc_null
        else:
            lfc_null = new_lfc_null
        if new_alt_hypothesis == "default":
            alt_hypothesis = self.alt_hypothesis
        else:
            alt_hypothesis = new_alt_hypothesis
        if lfc_null < 0 and alt_hypothesis in {"greaterAbs", "lessAbs"}:
            raise ValueError(
                f"The alternative hypothesis being {alt_hypothesis}, please provide a",
                f"positive lfc_null value (got {lfc_null}).",
            )

        if (
            not hasattr(self, "p_values")
            or self.lfc_null != lfc_null
            or self.alt_hypothesis != alt_hypothesis
        ):
            # Estimate p-values with Wald test
            self.lfc_null = lfc_null
            self.alt_hypothesis = alt_hypothesis
            rerun_summary = True
            self.run_wald_test()

        if self.cooks_filter:
            # Filter p-values based on Cooks outliers
            self._cooks_filtering()

        if not hasattr(self, "padj") or rerun_summary:
            if self.independent_filter:
                # Compute adjusted p-values and correct p-value trend
                self._independent_filtering()
            else:
                # Compute adjusted p-values using the Benjamini-Hochberg method, without
                # correcting the p-value trend.
                self._p_value_adjustment()

        # Store the results in a DataFrame, in log2 scale for LFCs.
        self.results_df = pd.DataFrame(index=self.dds.var_names)
        self.results_df["baseMean"] = self.base_mean
        self.results_df["log2FoldChange"] = self.LFC @ self.contrast_vector / np.log(2)
        self.results_df["lfcSE"] = self.SE / np.log(2)
        self.results_df["stat"] = self.statistics
        self.results_df["pvalue"] = self.p_values
        self.results_df["padj"] = self.padj

        if not self.quiet:
            if isinstance(self.contrast, np.ndarray):
                # The contrast vector was directly provided
                print(
                    f"Log2 fold change & Wald test p-value, contrast vector: "
                    f"{self.contrast}"
                )
            else:
                # The factor is categorical
                print(
                    f"Log2 fold change & Wald test p-value: "
                    f"{self.contrast[0]} {self.contrast[1]} vs {self.contrast[2]}"
                )
            print(self.results_df)

    def run_wald_test(self) -> None:
        """Perform a Wald test.

        Get gene-wise p-values for gene over/under-expression.
        """
        num_vars = self.design_matrix.shape[1]

        # Raise a warning if LFCs are shrunk.
        if self.shrunk_LFCs:
            if not self.quiet:
                print(
                    "Note: running Wald test on shrunk LFCs. "
                    "Some sequencing datasets show better performance with the testing "
                    "separated from the use of the LFC prior.",
                    file=sys.stderr,
                )

        mu = (
            np.exp(self.design_matrix @ self.LFC.T)
            .multiply(self.dds.obs["size_factors"], 0)
            .values
        )

        # Set regularization factors.
        if self.prior_LFC_var is not None:
            ridge_factor = np.diag(1 / self.prior_LFC_var**2)
        else:
            ridge_factor = np.diag(np.repeat(1e-6, num_vars))

        design_matrix = self.design_matrix.values
        LFCs = self.LFC.values

        if not self.quiet:
            print("Running Wald tests...", file=sys.stderr)
        start = time.time()
        pvals, stats, se = self.inference.wald_test(
            design_matrix=design_matrix,
            disp=self.dds.var["dispersions"].values,
            lfc=LFCs,
            mu=mu,
            ridge_factor=ridge_factor,
            contrast=self.contrast_vector,
            lfc_null=np.log(2) * self.lfc_null,  # Convert log2 to natural log
            alt_hypothesis=self.alt_hypothesis,
        )
        end = time.time()
        if not self.quiet:
            print(f"... done in {end - start:.2f} seconds.\n", file=sys.stderr)

        self.p_values: pd.Series = pd.Series(pvals, index=self.dds.var_names)
        self.statistics: pd.Series = pd.Series(stats, index=self.dds.var_names)
        self.SE: pd.Series = pd.Series(se, index=self.dds.var_names)

        # Account for possible all_zeroes due to outlier refitting in DESeqDataSet
        if self.dds.refit_cooks and self.dds.var["replaced"].sum() > 0:
            self.SE.loc[self.dds.new_all_zeroes_genes] = 0.0
            self.statistics.loc[self.dds.new_all_zeroes_genes] = 0.0
            self.p_values.loc[self.dds.new_all_zeroes_genes] = 1.0

    # TODO update this to reflect the new contrast format
    def lfc_shrink(self, coeff: str, adapt: bool = True) -> None:
        """LFC shrinkage with an apeGLM prior :cite:p:`DeseqStats-zhu2019heavy`.

        Shrinks LFCs using a heavy-tailed Cauchy prior, leaving p-values unchanged.

        Parameters
        ----------
        coeff : str
            The LFC coefficient to shrink. Must be one of the columns of the LFC matrix.
            (default: ``None``).

        adapt: bool
            Whether to use the MLE estimates of LFC to adapt the prior. If False, the
            prior scale is set to 1. (``default=True``)
        """
        if coeff not in self.LFC.columns:
            raise KeyError(
                f"The coeff argument '{coeff}' should be one the LFC columns. "
                f"The available LFC coeffs are {self.LFC.columns[1:]}."
            )

        coeff_idx = self.LFC.columns.get_loc(coeff)

        design_matrix = self.design_matrix.values
        size = 1.0 / self.dds.var["dispersions"].values
        offset = np.log(self.dds.obs["size_factors"]).values

        # Set priors
        prior_no_shrink_scale = 15
        prior_scale = 1
        if adapt:
            prior_var = self._fit_prior_var(coeff_idx=coeff_idx)
            prior_scale = np.minimum(np.sqrt(prior_var), 1)

        if not self.quiet:
            print("Fitting MAP LFCs...", file=sys.stderr)
        start = time.time()
        lfcs, inv_hessians, l_bfgs_b_converged_ = self.inference.lfc_shrink_nbinom_glm(
            design_matrix=design_matrix,
            counts=self.dds.X[:, self.dds.non_zero_idx],
            size=size[self.dds.non_zero_idx],
            offset=offset,
            prior_no_shrink_scale=prior_no_shrink_scale,
            prior_scale=prior_scale,
            optimizer="L-BFGS-B",
            shrink_index=coeff_idx,
        )
        end = time.time()
        if not self.quiet:
            print(f"... done in {end - start:.2f} seconds.\n", file=sys.stderr)

        new_lfc_values = np.array(lfcs)[:, coeff_idx]
        new_se_values = np.array(
            [
                np.sqrt(np.abs(inv_hess[coeff_idx, coeff_idx]))
                for inv_hess in inv_hessians
            ]
        )
        nan_mask = ~np.isfinite(new_lfc_values) | ~np.isfinite(new_se_values)

        if nan_mask.any():
            warnings.warn(
                f"{nan_mask.sum()} gene(s) had NaN/infinite values during LFC shrinkage,"
                " their LFCs and SEs were not updated.",
                UserWarning,
                stacklevel=2,
            )

        # Only update genes with valid (non-NaN) shrinkage results
        valid_genes = self.dds.non_zero_genes[~nan_mask]
        self.LFC.loc[valid_genes, coeff] = new_lfc_values[~nan_mask]
        self.SE.loc[valid_genes] = new_se_values[~nan_mask]

        self._LFC_shrink_converged = pd.Series(
            pd.array([pd.NA] * len(self.dds.var_names), dtype="boolean"),
            index=self.dds.var_names,
        )
        self._LFC_shrink_converged.loc[self.dds.non_zero_genes] = l_bfgs_b_converged_

        # Set a flag to indicate that LFCs were shrunk
        self.shrunk_LFCs = True

        # Replace in results dataframe, if it exists
        if hasattr(self, "results_df"):
            self.results_df["log2FoldChange"] = self.LFC.iloc[:, coeff_idx] / np.log(2)
            self.results_df["lfcSE"] = self.SE / np.log(2)
            if not self.quiet:
                print(f"Shrunk log2 fold change & Wald test p-value: {coeff}")
                print(self.results_df)

    def plot_MA(self, log: bool = True, save_path: str | None = None, **kwargs):
        """
        Create an log ratio (M)-average (A) plot using matplotlib.

        Useful for looking at log fold-change versus mean expression
        between two groups/samples/etc.
        Uses matplotlib to emulate the ``make_MA()`` function in DESeq2 in R.

        Parameters
        ----------
        log : bool
            Whether or not to log scale x and y axes (``default=True``).

        save_path : str, optional
            The path where to save the plot. If left None, the plot won't be saved
            (``default=None``).

        **kwargs
            Matplotlib keyword arguments for the scatter plot.
        """
        # Raise an error if results_df are missing
        if not hasattr(self, "results_df"):
            raise AttributeError(
                "Trying to make an MA plot but p-values were not computed yet. "
                "Please run the summary() method first."
            )

        make_MA_plot(
            self.results_df,
            padj_thresh=self.alpha,
            log=log,
            save_path=save_path,
            lfc_null=self.lfc_null,
            alt_hypothesis=self.alt_hypothesis,
            **kwargs,
        )

    def _independent_filtering(self) -> None:
        """Compute adjusted p-values using independent filtering.

        Corrects p-value trend (see :cite:p:`DeseqStats-love2014moderated`)
        """
        # Check that p-values are available. If not, compute them.
        if not hasattr(self, "p_values"):
            self.run_wald_test()

        lower_quantile = np.mean(self.base_mean == 0)

        if lower_quantile < 0.95:
            upper_quantile = 0.95
        else:
            upper_quantile = 1

        theta = np.linspace(lower_quantile, upper_quantile, 50)
        cutoffs = np.quantile(self.base_mean, theta)

        result = pd.DataFrame(
            np.nan, index=self.dds.var_names, columns=np.arange(len(theta))
        )

        for i, cutoff in enumerate(cutoffs):
            use = (self.base_mean >= cutoff) & (~self.p_values.isna())
            U2 = self.p_values[use]
            if not U2.empty:
                result.loc[use, i] = false_discovery_control(U2, method="bh")
        num_rej = (result < self.alpha).sum(axis=0).to_numpy().astype(int)
        lowess_res = lowess(theta, num_rej, frac=1 / 5)

        if num_rej.max() <= 10:
            j = 0
        else:
            residual = num_rej[num_rej > 0] - lowess_res[num_rej > 0]
            thresh = lowess_res.max() - np.sqrt(np.mean(residual**2))
            if np.any(num_rej > thresh):
                j = np.where(num_rej > thresh)[0][0]
            else:
                j = 0

        self.padj = result.loc[:, j]

    def _p_value_adjustment(self) -> None:
        """Compute adjusted p-values using the Benjamini-Hochberg method.

        Does not correct the p-value trend.
        This method and the `_independent_filtering` are mutually exclusive.
        """
        if not hasattr(self, "p_values"):
            # Estimate p-values with Wald test
            self.run_wald_test()

        self.padj = pd.Series(np.nan, index=self.dds.var_names)
        self.padj.loc[~self.p_values.isna()] = false_discovery_control(
            self.p_values.dropna(), method="bh"
        )

    def _cooks_filtering(self) -> None:
        """Filter p-values based on Cooks outliers."""
        # Check that p-values are available. If not, compute them.
        if not hasattr(self, "p_values"):
            self.run_wald_test()

        self.p_values[self.dds.cooks_outlier()] = np.nan

    def _fit_prior_var(
        self, coeff_idx: str, min_var: float = 1e-6, max_var: float = 400.0
    ) -> float:
        """Estimate the prior variance of the apeGLM model.

        Returns shrinkage factors.

        Parameters
        ----------
        coeff_idx : str
            Index of the coefficient to shrink.

        min_var : float
            Lower bound for prior variance. (default: ``1e-6``).

        max_var : float
            Upper bound for prior variance. (default: ``400``).

        Returns
        -------
        float
            Estimated prior variance.
        """
        keep = ~self.LFC.iloc[:, coeff_idx].isna()
        S = self.LFC[keep].iloc[:, coeff_idx] ** 2
        D = self.SE[keep] ** 2

        def objective(a: float) -> float:
            # Equation to solve
            coeff = 1 / (2 * (a + D) ** 2)
            return ((S - D) * coeff).sum() / coeff.sum() - a

        # The prior variance is the zero of the above function.
        if objective(min_var) < 0:
            return min_var
        else:
            return root_scalar(objective, bracket=(min_var, max_var)).root

    def _build_contrast_vector(self) -> None:
        """
        Build a vector corresponding to the desired contrast.

        Allows to test any pair of levels without refitting LFCs.
        """
        factor = self.contrast[0]
        alternative = self.contrast[1]
        ref = self.contrast[2]
        self.contrast_vector = self.dds.contrast(
            column=factor, baseline=ref, group_to_compare=alternative
        )
