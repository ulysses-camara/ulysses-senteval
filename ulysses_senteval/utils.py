"""TODO."""
import typing as t
import os

import numpy as np
import numpy.typing as npt
import torch


DataType = t.Union[torch.Tensor, npt.NDArray[np.float64]]
MetricType = t.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def expand_path(path: str) -> str:
    """TODO."""
    path = os.path.expandvars(path)
    path = os.path.expanduser(path)
    path = os.path.abspath(path)
    return path
