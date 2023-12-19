"""General utility functions."""
import typing as t
import os
import contextlib

import numpy as np
import numpy.typing as npt
import torch


MAX_RNG_SEED = 2**32 - 1


EmbeddedDataType = t.Union[torch.Tensor, npt.NDArray[np.float64]]
RawDataType = t.List[str]
PairedRawDataType = t.Tuple[RawDataType, t.Optional[RawDataType]]
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
    """Disable PyTorch multithreading to prevent deadlocks in multiprocessing.

    Read more about this in: https://github.com/pytorch/pytorch/issues/17199.
    """
    n_threads = torch.get_num_threads()

    try:
        torch.set_num_threads(1)
        yield

    finally:
        torch.set_num_threads(n_threads)


def take_inds(iterable: t.Sequence[t.Any], inds: t.Sequence[int], paired: bool = False) -> t.Sequence[t.Any]:
    """Take `inds` from the first axis of `iterable`.

    If paired, will take `inds` from both sequences.
    """
    if paired:
        (it_a, it_b) = iterable
        return (
            take_inds(it_a, inds, paired=False),
            take_inds(it_b, inds, paired=False) if it_b is not None else None,
        )

    if isinstance(iterable, np.ndarray) or torch.is_tensor(iterable):
        return iterable[inds]

    cast_fn = type(iterable)
    return cast_fn([iterable[i] for i in inds])
