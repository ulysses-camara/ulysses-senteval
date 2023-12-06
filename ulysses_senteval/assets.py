"""TODO."""
import typing as t
import os

import pandas as pd
import numpy.typing as npt
import numpy as np
import torch
import torchmetrics
import buscador

from . import utils


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


def download_dataset(task: str) -> None:
    """TODO."""
    print("Downloaded:", task)


def load_data(
    task: str, data_dir_path: str
) -> t.Tuple[t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64], int]:
    """TODO."""
    data_dir_path = utils.expand_path(data_dir_path)

    dataset_name = TASK_CODE_TO_NAME[task]
    input_uri = os.path.join(data_dir_path, dataset_name)

    df = pd.read_csv(input_uri, index_col=0)
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
    def __call__(self, *args: t.Any, **kwargs: t.Any) -> torch.Tensor:
        return 0.5 * (1.0 + super().__call__(*args, **kwargs))


def get_eval_metric(task: str, n_classes: int):
    if task in {"F2"}:
        return torchmetrics.classification.F1Score(num_classes=n_classes, average="macro", task="multiclass")

    return NormalizedBinaryMatthewsCorrCoef()
