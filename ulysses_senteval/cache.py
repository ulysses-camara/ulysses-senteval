"""Embed cache functions."""
import typing as t
import os

import torch
import numpy as np

from . import utils


def load_from_cache(cache_dir: str, task: str) -> t.Optional[torch.Tensor]:
    """Load cached embedding or return None if not found."""
    cache_output_uri = os.path.join(cache_dir, f"{task}.pt")

    if not os.path.exists(cache_output_uri):
        return None

    return torch.load(cache_output_uri).float()


def save_in_cache(embs: utils.DataType, cache_dir: str, task: str, overwrite: bool = False) -> None:
    """Cache embedding."""
    if isinstance(embs, np.ndarray):
        embs = torch.from_numpy(embs)

    embs = embs.float()

    os.makedirs(cache_dir, exist_ok=True)
    cache_output_uri = os.path.join(cache_dir, f"{task}.pt")

    if not overwrite and os.path.exists(cache_output_uri):
        raise RuntimeError(f"Embedding '{cache_output_uri}' exists but overwrite=False.")

    torch.save(embs, cache_output_uri)
