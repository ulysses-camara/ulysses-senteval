"""Ulysses SentEval asset configuration."""
import typing as t
import os

import pandas as pd
import numpy.typing as npt
import numpy as np
import torch
import torchmetrics
import buscador

from . import utils


__all__ = {
    "TASK_CODE_TO_NAME",
    "TASKS",
    "download_dataset",
    "load_dataset",
}


TASK_CODE_TO_NAME = {
    "F1A": "masked_law_name_in_summaries.csv",
    "F1B": "masked_law_name_in_news.csv",
    "F2": "code_estatutes_cf88.csv",
    "F3": "oab_first_part.csv",
    "F4": "oab_second_part.csv",
    "F5": "trf_examinations.csv",
    "G1": "speech_summaries.csv",
    # "G2A": "TODO",  # TODO: missing, fix this.
    "G2B": "summary_vs_bill.csv",
    "G3": "faqs.csv",
    "G4": "ulysses_sd.csv",
    "T1A": "hatebr_offensive_lang.csv",
    "T1B": "offcombr2.csv",
    "T2A": "factnews_news_bias.csv",
    "T2B": "factnews_news_factuality.csv",
    "T3": "fakebr_size_normalized.csv",
}

TASKS = tuple(TASK_CODE_TO_NAME.keys())


def _build_data_path(task: str, data_dir_path: str) -> str:
    """Build standard dataset URI."""
    data_dir_path = utils.expand_path(data_dir_path)
    dataset_name = TASK_CODE_TO_NAME[task]
    dataset_uri = os.path.join(data_dir_path, dataset_name)
    return dataset_uri


def download_dataset(task: str, data_dir_path: str, force_download: bool = False) -> None:
    """Download `task` dataset if not found locally.

    Parameters
    ----------
    task : str
        Task name.

    data_dir_path : str
        Target directory to search the dataset locally and store it after download.

    force_download : bool, default=False
        If True, download dataset regardless if it already exists locally.

    Returns
    -------
    None
    """
    input_uri = _build_data_path(task=task, data_dir_path=data_dir_path)

    if not force_download and os.path.exists(input_uri):
        return

    # TODO: download data from Fetcher.
    raise NotImplementedError("Dataset download still not implemented; only local files can be used.")


def load_dataset(
    task: str,
    data_dir_path: str,
    local_files_only: bool = False,
) -> t.Tuple[t.Tuple[t.List[str], t.Optional[t.List[str]]], npt.NDArray[t.Union[np.int64, np.float32]], int]:
    """Load `task` dataset.

    If the dataset is not found, will download it if `local_files_only=False`.

    Parameters
    ----------
    task : str
        Task name.

    data_dir_path : str
        Target directory to search the dataset locally and store it after download.

    force_download : bool, default=False
        If True, download dataset regardless if it already exists locally.

    Returns
    -------
    (X_a, X_b) : t.Tuple[t.List[str], t.Optional[t.List[str]]]
        Tuple containing each instance from the dataset.
        For paired datasets, len(X_a) = len(X_b), and X_a have a 1-to-1 correspondence to X_b.
        For datasets with single inputs, X_b is None.

    y : np.NDArray[t.Union[np.int64, np.float32]]
        Target labels.

    n_classes : int
        Number of classes in target `y`.
    """
    input_uri = _build_data_path(task=task, data_dir_path=data_dir_path)

    if not local_files_only:
        download_dataset(task=task, data_dir_path=data_dir_path)

    try:
        df = pd.read_csv(input_uri, index_col=0)

    except (FileNotFoundError, OSError) as err:
        raise FileNotFoundError(f"Could not load data for '{task=}' in URI '{input_uri}' (error message: {err}).") from err

    _, m = df.shape

    if m == 2:
        X_a, X_b = df.iloc[:, 0].values, None
    else:
        X_a, X_b = df.iloc[:, [0, 1]].values.T

    y = df.iloc[:, -1].values
    y = y.astype(float if np.unique(y).size == 2 else int, copy=False)

    n_classes = int(np.unique(y).size)

    return (X_a, X_b), y, n_classes


class NormalizedBinaryMatthewsCorrCoef(torchmetrics.classification.BinaryMatthewsCorrCoef):
    """Matthews Correlation Coefficient normalized to [0, 1] range."""

    def __call__(self, *args: t.Any, **kwargs: t.Any) -> torch.Tensor:
        # pylint: disable='missing-function-docstring'
        return 0.5 * (1.0 + super().__call__(*args, **kwargs))


def get_eval_metric(task: str, n_classes: int):
    """Get evaluation metric for `task`."""
    if task in {"F2"}:
        return torchmetrics.classification.F1Score(num_classes=n_classes, average="macro", task="multiclass")

    return NormalizedBinaryMatthewsCorrCoef()
