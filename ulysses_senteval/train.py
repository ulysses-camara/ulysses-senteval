"""TODO."""
import typing as t
import collections

import numpy as np
import numpy.typing as npt
import torch
import torch.nn
import sklearn.model_selection
import tqdm


DataType = t.Union[torch.Tensor, npt.NDArray[np.float64]]


class LogisticRegression(torch.nn.Module):
    """TODO."""

    # pylint: disable='missing-method-docstring'

    def __init__(self, input_dim: int, output_dim: int):
        self.params = torch.nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.params(X)


def undersample(X: DataType, y: DataType, random_state: int) -> t.Tuple[DataType, DataType]:
    """TODO."""
    rng = np.random.RandomState(random_state)
    classes, freqs = np.unique(y, return_counts=True)

    n_classes = int(classes.size)
    freq_min = int(np.min(freqs))

    sampled_inds = np.empty(n_classes * freq_min, dtype=int)

    for i, cls in enumerate(classes):
        cur_inds = np.flatnonzero(y == cls)
        cur_inds = rng.choice(cur_inds, size=freq_min, replace=False)
        sampled_inds[i * freq_min : (i + 1) * freq_min] = cur_inds

    X = X[sampled_inds, :]
    y = y[sampled_inds]

    return (X, y)


def train(dl_train: torch.utils.data.DataLoader, dl_test: torch.utils.data.DataLoader) -> t.Dict[str, t.Any]:
    """TODO."""
    return {"test_score": 1.0}


def scale_data(X_train: torch.Tensor, X_test: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
    """TODO."""
    avg = torch.mean(X_train, dim=0)
    std = 1e-12 + torch.std(X_train, dim=0)
    X_train = (X_train - avg) / std
    X_test = (X_test - avg) / std
    return (X_train, X_test)


def kfold_train(
    X: DataType,
    y: DataType,
    batch_size: int = 64,
    device: t.Union[torch.device, str] = "cuda:0",
    show_progress_bar: bool = True,
    *,
    n_repeats: int = 5,
    k_fold: int = 5,
    random_state: int = 9847706,
    pbar_desc: t.Optional[str] = None,
) -> t.Dict[str, t.Any]:
    """TODO."""
    reseeder = np.random.RandomState(random_state)
    seeds_undersampling, seeds_kfold = reseeder.randint(0, 2**32 - 1, size=(2, n_repeats))
    seeds_dl = reseeder.randint(0, 2**32 - 1, size=n_repeats * k_fold)

    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)

    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)

    all_results = collections.defaultdict(list)

    pbar = tqdm.tqdm(np.arange(n_repeats * k_fold), desc=pbar_desc)

    for i in np.arange(n_repeats):
        X_cur, y_cur = undersample(X, y, random_state=seeds_undersampling[i])

        splitter = sklearn.model_selection.StratifiedKFold(
            n_splits=k_fold,
            random_state=seeds_kfold[i],
            shuffle=True,
        )

        for j, (inds_train, inds_test) in enumerate(splitter.split(X_cur, y_cur)):
            X_train, X_test = X_cur[inds_train, :], X_cur[inds_test, :]
            y_train, y_test = y_cur[inds_train], y_cur[inds_test]

            assert len(X_train) == len(y_train)
            assert len(X_test) == len(y_test)

            X_train, X_test = scale_data(X_train, X_test)

            torch_rng = torch.Generator().manual_seed(int(seeds_dl[i * k_fold + j]))

            dl_train = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X_train, y_train),
                shuffle=True,
                drop_last=True,
                batch_size=batch_size,
                generator=torch_rng,
            )

            dl_test = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X_test, y_test),
                shuffle=False,
                drop_last=False,
                batch_size=batch_size,
            )

            cur_res = train(dl_train, dl_test)

            for k, v in cur_res.items():
                all_results[k].append(v)

            pbar.update(1)

    return all_results
