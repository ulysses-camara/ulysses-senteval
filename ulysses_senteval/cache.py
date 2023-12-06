"""TODO."""
import typing as t
import os

import torch
import numpy as np

from . import utils


def load_from_cache(cache_dir: str, task: str) -> t.Optional[torch.Tensor]:
    """TODO."""
    cache_output_uri = os.path.join(cache_dir, f"{task}.pt")

    if not os.path.exists(cache_output_uri):
        return None

    return torch.load(cache_output_uri).float()


def save_in_cache(embs: utils.DataType, cache_dir: str, task: str) -> None:
    """TODO."""
    if isinstance(embs, np.ndarray):
        embs = torch.from_numpy(embs)

    embs = embs.float()

    os.makedirs(cache_dir, exist_ok=True)
    cache_output_uri = os.path.join(cache_dir, f"{task}.pt")
    torch.save(embs, cache_output_uri)
