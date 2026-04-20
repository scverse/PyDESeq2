"""GPU utility functions for PyDESeq2.

All tensor operations use float64. This requires CUDA or CPU;
MPS (Apple Silicon) does not support float64 and is rejected.
"""

import warnings

import torch


def get_device(device: str | None = None) -> torch.device:
    """Return a ``torch.device``, prioritizing CUDA if available.

    Parameters
    ----------
    device : str or None
        Device string (e.g. ``"cuda"``, ``"cuda:0"``, ``"cpu"``).
        If ``None``, auto-detects CUDA availability.

    Returns
    -------
    torch.device
        Selected device.
    """
    if device is not None and "mps" in str(device):
        raise ValueError(
            "MPS (Apple Silicon) is not supported because "
            "TorchInference requires float64 throughout. "
            "Use device='cpu' or device='cuda'."
        )
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            warnings.warn(
                "CUDA not available. Using CPU for TorchInference.",
                UserWarning,
                stacklevel=2,
            )
            return torch.device("cpu")
    else:
        if device == "cuda" and not torch.cuda.is_available():
            warnings.warn(
                "CUDA requested but not available, falling back to CPU.",
                UserWarning,
                stacklevel=2,
            )
            return torch.device("cpu")
        return torch.device(device)


@torch.no_grad()
def trimmed_mean(x: torch.Tensor, trim: float = 0.1, dim: int = 0) -> torch.Tensor:
    """Return trimmed mean along ``dim``.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    trim : float
        Fraction to trim from each tail. Must be <= 0.5.
    dim : int
        Dimension along which to compute.

    Returns
    -------
    torch.Tensor
        Trimmed mean.
    """
    assert trim <= 0.5
    n = x.shape[dim]
    ntrim = int(n * trim)
    s = torch.sort(x, dim=dim).values
    if dim == 0:
        return s[ntrim : n - ntrim].mean(dim=dim)
    else:
        return s[:, ntrim : n - ntrim].mean(dim=dim)


@torch.no_grad()
def trimmed_variance(x: torch.Tensor, trim: float = 0.125, dim: int = 0) -> torch.Tensor:
    """Return trimmed variance along ``dim``.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    trim : float
        Fraction to trim from each tail.
    dim : int
        Dimension along which to compute.

    Returns
    -------
    torch.Tensor
        Trimmed variance (bias-corrected with factor 1.51).
    """
    rm = trimmed_mean(x, trim=trim, dim=dim)
    sqerror = (x - rm) ** 2
    return 1.51 * trimmed_mean(sqerror, trim=trim, dim=dim)
