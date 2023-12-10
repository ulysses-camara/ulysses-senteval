"""General utility functions."""
import typing as t
import os
import contextlib

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


def is_cuda_vs_multiprocessing_error(err) -> bool:
    """Check if a specific CUDA environment exception has been raised.

    This exception is raised when CUDA is being used in a subprocess, but the main process
    already initialized the CUDA environment.
    """
    return str(err).startswith("Cannot re-initialize CUDA")


@contextlib.contextmanager
def disable_torch_multithreading():
    """Disable PyTorch multithreading to prevent deadlocks in multiprocessing."""
    n_threads = torch.get_num_threads()

    try:
        torch.set_num_threads(1)
        yield

    finally:
        torch.set_num_threads(n_threads)
