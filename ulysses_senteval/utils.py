"""General utility functions."""
import typing as t
import os

import numpy as np
import numpy.typing as npt
import torch


DataType = t.Union[torch.Tensor, npt.NDArray[np.float64]]
MetricType = t.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def expand_path(path: str) -> str:
    """Expand user, variables, and relative file or directory system paths."""
    path = os.path.expandvars(path)
    path = os.path.expanduser(path)
    path = os.path.abspath(path)
    return path


def relative_diff(ref: float, val: float) -> float:
    """Compute relative difference between `ref` and `val`.

    Relative difference = (val - ref) / ref.
    """
    if np.isinf(ref):
        return 0.0 if np.isinf(val) else -np.inf

    return (val - ref) / (1e-12 + ref)
