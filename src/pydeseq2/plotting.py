"""Plots of the fitted model."""

from typing import Literal

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes


def make_scatter(
    disps: list,
    legend_labels: list,
    x_val: np.ndarray,
    log: bool = True,
    save_path: str | None = None,
    ax: Axes | None = None,
    **kwargs,
) -> None:
    """
    Create a scatter plot using matplotlib.

    Used in :meth:`pydeseq2.dds.DeseqDataSet.plot_dispersions()`.

    Parameters
    ----------
    disps
        List of ndarrays to plot.
    legend_labels
        List of strings that correspond to plotted targets values for legend.
    x_val
        1D array to plot (example: ``dds.varm['_normed_means']``).
    log
        Whether or not to log scale features and targets axes (``default=True``).
    save_path
        The path where to save the plot.
        If left None, the plot won't be saved (``default=None``).
    ax
        The axes to plot on.
        If left None, a new figure is created and shown (``default=None``).
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
    show = ax is None
    if ax is None:
        _, ax = plt.subplots(dpi=600, layout="tight")

    # log scale axes
    if log is True:
        ax.set_yscale("log")
        ax.set_xscale("log")

    # scale axes according to data spread
    ax.set_adjustable("datalim")

    # Set default alpha and s parameters, if not already specified
    kwargs.setdefault("alpha", 0.5)
    kwargs.setdefault("s", 0.6)

    # create scatter plot per trace
    for disp, color in list(zip(disps, colors, strict=False)):
        ax.scatter(x=x_val, y=disp, c=color, **kwargs)

    # label legend + axes
    ax.legend(legend_labels, loc="best")
    ax.set_xlabel("mean of normalized counts")
    ax.set_ylabel("dispersion")
    if save_path is not None:
        ax.figure.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()


def make_MA_plot(
    results_df: pd.DataFrame,
    padj_thresh: float = 0.05,
    log: bool = True,
    save_path: str | None = None,
    lfc_null: float = 0,
    alt_hypothesis: Literal["greaterAbs", "lessAbs", "greater", "less"] | None = None,
    ax: Axes | None = None,
    **kwargs,
) -> None:
    """
    Create an log ratio (M)-average (A) plot using matplotlib.

    Useful for looking at log fold-change versus mean expression between two groups/samples/etc.
    Uses matplotlib to emulate the ``make_MA()`` function in DESeq2 in R.

    Parameters
    ----------
    results_df
        Resultant dataframe after running DeseqStats() and .summary().
    padj_thresh
        P-value threshold to subset scatterplot colors on.
    log
        Whether or not to log scale features and targets axes (``default=True``).
    save_path
        The path where to save the plot.
        If left None, the plot won't be saved (``default=None``).
    lfc_null
        The (log2) log fold change under the null hypothesis. (default: ``0``).
    alt_hypothesis
        The alternative hypothesis for computing wald p-values. (default: ``None``).
    ax
        The axes to plot on.
        If left None, a new figure is created (``default=None``).
    **kwargs
        Matplotlib keyword arguments for the scatter plot.
    """
    colors = results_df["padj"].apply(lambda x: "darkred" if x < padj_thresh else "gray")

    if ax is None:
        _, ax = plt.subplots(dpi=600, layout="tight")

    # Set default alpha and s parameters, if not already specified
    kwargs.setdefault("alpha", 0.5)
    kwargs.setdefault("s", 0.2)

    ax.scatter(
        x=results_df["baseMean"],
        y=results_df["log2FoldChange"],
        c=colors,
        **kwargs,
    )

    ax.set_adjustable("datalim")

    if log is True:
        ax.set_xscale("log")

    ax.set_xlabel("mean of normalized counts")
    ax.set_ylabel("log2 fold change")

    ax.axhline(lfc_null, color="red", alpha=0.5, linestyle="--", zorder=3)
    if alt_hypothesis and alt_hypothesis in ["greaterAbs", "lessAbs"]:
        ax.axhline(-lfc_null, color="red", alpha=0.5, linestyle="--", zorder=3)

    if save_path is not None:
        ax.figure.savefig(save_path, bbox_inches="tight")
