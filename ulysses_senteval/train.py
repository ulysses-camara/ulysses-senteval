"""Training pipeline for task classifiers."""
import typing as t
import collections
import itertools
import multiprocessing

import pandas as pd
import numpy as np
import numpy.typing as npt
import torch
import torch.nn
import sklearn.model_selection
import tqdm

from . import utils


class LogisticRegression(torch.nn.Module):
    """Logistic regression module."""

    # pylint: disable='missing-method-docstring'

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.params = torch.nn.Sequential(
            torch.nn.Linear(input_dim, output_dim, bias=True),
            torch.nn.Flatten(start_dim=0) if output_dim == 1 else torch.nn.Identity(),
        )

        torch.nn.init.constant_(self.params[0].bias, 0.0)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.params(X)


def undersample(X: utils.DataType, y: utils.DataType, random_state: int) -> t.Tuple[utils.DataType, utils.DataType]:
    """Undersample majority classes to match the minority class frequency.

    Parameters
    ----------
    X : npt.NDArray[np.float64] of shape (N, M)
        Original embeddings.

    y : npt.NDArray of shape (N,)
        Original targets.

    Returns
    -------
    X_undersampled : npt.NDArray[np.float64] of shape (K, M)
        Undersampled X, where K <= N.

    y_undersampled : npt.NDArray os shape(K,
        Undersampled y, where K <= N.
    """
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


def tune_optim_hyperparameters(
    dl_train: torch.utils.data.DataLoader,
    dl_eval: torch.utils.data.DataLoader,
    train_kwargs: t.Dict[str, t.Any],
) -> t.Dict[str, t.Any]:
    """Tune AdamW optimizer hyper-parameters using grid search."""
    best_metric_val = -np.inf
    best_adamw_optim_kwargs: t.Dict[str, t.Any] = {}

    for lr, b1, b2 in itertools.product((5e-4, 1e-3, 2e-3), (0.80, 0.90), (0.990, 0.999)):
        adamw_optim_kwargs: t.Dict[str, t.Any] = {"lr": lr, "betas": (b1, b2), "weight_decay": 1e-2}

        cur_metric_val = train(
            dl_train=dl_train,
            dl_eval=None,
            dl_test=dl_eval,
            adamw_optim_kwargs=adamw_optim_kwargs,
            n_epochs=8,
            tenacity=-1,
            early_stopping_rel_improv=-1.0,
            **train_kwargs,
        )["metric_test"]

        if best_metric_val < cur_metric_val:
            best_metric_val = cur_metric_val
            best_adamw_optim_kwargs = adamw_optim_kwargs.copy()

    return best_adamw_optim_kwargs


def train(
    dl_train: torch.utils.data.DataLoader,
    dl_eval: t.Optional[torch.utils.data.DataLoader],
    dl_test: t.Optional[torch.utils.data.DataLoader],
    adamw_optim_kwargs: t.Dict[str, t.Any],
    n_classes: int,
    eval_metric: utils.MetricType,
    n_epochs: int,
    tenacity: int,
    early_stopping_rel_improv: float,
    device: str,
    param_init_random_state: int,
) -> t.Dict[str, t.Any]:
    """Classifier train pipeline.

    Parameters
    ----------
    dl_train : torch.utils.data.DataLoader
        Train DataLoader.

    dl_eval : t.Optional[torch.utils.data.DataLoader]
        Evaluation DataLoader.

    dl_test : t.Optional[torch.utils.data.DataLoader]
        Test DataLoader.

    adamw_optim_kwargs : t.Dict[str, t.Any]
        Arguments for AdamW optimizer.

    n_classes : int
        Number of classes.

    eval_metric : utils.MetricType
        Evaluation metric used after each epoch in the validation split and in test split after the
        training is finished.
        Only used if `dl_eval` and `dl_test` are not None simultaneously.

    n_epochs : int
        Number of training epochs.

    tenacity : int
        Maximum number of subsequent epochs without sufficient validation loss decrease for early
        stopping.
        Only used if `dl_eval` is not None.

    early_stopping_rel_improv : float
        Minimum relative difference between best validation loss and current validation loss to
        consider it an actual improvement.
        Only used if `dl_eval` is not None.

    device : str
        Device to run training and validation.

    param_init_random_state : int
        Random seed to initialize classifer parameters.

    Returns
    -------
    output : t.Dict[str, t.Any]
        Train, validation, and test split statistics.
    """
    if hasattr(eval_metric, "to"):
        eval_metric = eval_metric.to(device)

    is_binary_classification = n_classes == 2
    _, input_dim = next(iter(dl_test))[0].shape

    with torch.random.fork_rng(devices=["cpu"]):
        torch.random.manual_seed(param_init_random_state)

        classifier = LogisticRegression(
            input_dim=input_dim,
            output_dim=1 if is_binary_classification else n_classes,
        )

    classifier = classifier.to(device)

    optim = torch.optim.AdamW(classifier.parameters(), **adamw_optim_kwargs)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=0.05, total_iters=len(dl_train) // 4)
    loss_fn = torch.nn.BCEWithLogitsLoss() if is_binary_classification else torch.nn.CrossEntropyLoss()

    early_stopping_count = 0
    best_loss_eval = np.inf

    output: t.Dict[str, t.Any] = {
        "loss_per_epoch_train": [],
        "loss_per_epoch_eval": [],
        "metric_per_epoch_eval": [],
        "loss_test": np.nan,
        "metric_test": np.nan,
    }

    def no_grad_epoch(dl: torch.utils.data.DataLoader) -> t.Tuple[float, float]:
        classifier.eval()
        all_preds: t.List[torch.Tensor] = []
        all_true: t.List[torch.Tensor] = []

        with torch.no_grad():
            for X_batch, y_batch in dl:
                X_batch = X_batch.to(device)
                y_preds = classifier(X_batch)
                all_preds.append(y_preds.cpu())
                all_true.append(y_batch.cpu())

            concat_all_preds = torch.cat(all_preds).to(device)
            concat_all_true = torch.cat(all_true).to(device)

            loss = float(loss_fn(concat_all_preds, concat_all_true).cpu().item())
            metric_val = float(eval_metric(concat_all_preds, concat_all_true).cpu().item())

        return (loss, metric_val)

    for epoch in np.arange(n_epochs):
        classifier.train()
        all_loss_train: t.List[float] = []

        for X_batch, y_batch in dl_train:
            (X_batch, y_batch) = (X_batch.to(device), y_batch.to(device))
            optim.zero_grad()
            y_preds = classifier(X_batch)
            loss_train = loss_fn(y_preds, y_batch)
            loss_train.backward()
            optim.step()
            warmup_scheduler.step()

            all_loss_train.append(float(loss_train.detach().cpu().item()))

        output["loss_per_epoch_train"].append(float(np.mean(all_loss_train)))

        if dl_eval is not None:
            loss_eval, metric_eval = no_grad_epoch(dl_eval)

            output["loss_per_epoch_eval"].append(float(loss_eval))
            output["metric_per_epoch_eval"].append(float(metric_eval))

            rel_diff = -utils.relative_diff(ref=best_loss_eval, val=loss_eval)

            if rel_diff >= early_stopping_rel_improv:
                best_loss_eval = loss_eval
                early_stopping_count = 0
            else:
                early_stopping_count += 1

            if early_stopping_count > tenacity:
                break

    if dl_test is not None:
        loss_test, metric_test = no_grad_epoch(dl_test)
        output["loss_test"] = loss_test
        output["metric_test"] = metric_test

    return output


def scale_data(
    X_train: torch.Tensor, X_eval: torch.Tensor, X_test: torch.Tensor
) -> t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply data standardization in each data split."""
    avg = torch.mean(X_train, dim=0)
    std = 1e-12 + torch.std(X_train, dim=0)

    X_train = (X_train - avg) / std
    X_test = (X_test - avg) / std
    X_eval = (X_eval - avg) / std

    return (X_train, X_eval, X_test)


def _summarize_metrics(all_results: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
    """Summarize metrics collected from training."""
    output: t.Dict[str, t.Any] = {}

    for k, v in all_results.items():
        val_type = type(v[0])

        if hasattr(val_type, "__len__"):
            lens = list(map(len, v))
            max_len = int(max(lens))

            avg_per_epoch, std_per_epoch = (
                pd.DataFrame(
                    {
                        "epoch": itertools.chain(*[np.arange(1, 1 + li) for li in lens]),
                        k: itertools.chain(*v),
                    }
                )
                .groupby("epoch")
                .agg(("mean", "std"))
                .values.T
            )

            output[f"avg_{k}"] = avg_per_epoch
            output[f"std_{k}"] = np.nan_to_num(std_per_epoch, nan=0.0, copy=False)

        else:
            output[f"avg_{k}"] = float(np.mean(v))
            output[f"std_{k}"] = float(np.std(v, ddof=1))

    return output


def kfold_train(
    X: utils.DataType,
    y: utils.DataType,
    n_classes: int,
    eval_metric: utils.MetricType,
    *,
    batch_size: int = 128,
    eval_frac: float = 0.20,
    device: t.Union[torch.device, str] = "cuda:0",
    show_progress_bar: bool = True,
    n_repeats: int = 5,
    k_fold: int = 5,
    n_epochs: int = 100,
    tenacity: int = 5,
    early_stopping_rel_improv: float = 0.0025,
    random_state: int = 9847706,
    pbar_desc: t.Optional[str] = None,
) -> t.Dict[str, t.Any]:
    """Apply repeated 5-fold cross validation in the provided data.

    Classes will be re-balanced using undersampling every cross validation repetition.

    Parameters
    ----------
    X : utils.DataType of shape (N, M)
        Data embeddings.

    y : utils.DataType of shape (N,)
        Target labels.

    n_classes : int
        Number of classes.

    eval_metric : utils.MetricType
        Evaluation metric used after each epoch in the validation split and in test split after the
        training is finished.

    Keyword-only parameters
    ----------
    batch_size : int, default=128
        Training and evaluation batch size.

    eval_frac : float, default=0.20
        Evaluation split fraction with respect to the train split size.
        Evaluation instances are randomly sampled after class balancing.

    device : t.Union[torch.device, str], default="cuda:0"
        Device to run training and validation.

    show_progress_bar : bool, default=True
        If True, show progress bar.

    n_repeats : int, default=5
        Number of cross validation repetitions.
        For each repetition, all random number generators are reseeded, and classes are rebalanced.

    k_fold : int, default=5
        Number of folds for each cross validation.

    n_epochs : int, default=100
        Maximum number of training epochs.

    tenacity : int, default=5
        Maximum number of subsequent epochs without sufficient validation loss decrease for early
        stopping.

    early_stopping_rel_improv : float, default=0.0025
        Minimum relative difference between best validation loss and current validation loss to
        consider it an actual improvement.

    random_state : int, default=9847706
        Random state to keep all results reproducible.

    pbar_desc : t.Optional[str], default=None
        Progress bar description. Only used in `show_progress_bar=True`.

    Returns
    -------
    output : t.Dict[str, t.Any]
        Summarized train, validation, and test split statistics.
    """
    reseeder = np.random.RandomState(random_state)
    seeds_undersampling, seeds_kfold = reseeder.randint(0, 2**32 - 1, size=(2, n_repeats))
    seeds_param_init, seeds_dl = reseeder.randint(0, 2**32 - 1, size=(2, n_repeats * k_fold))

    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)

    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)

    all_results = collections.defaultdict(list)

    pbar = tqdm.tqdm(
        total=n_repeats * k_fold,
        desc=pbar_desc,
        disable=not show_progress_bar,
        unit="partition",
        leave=False,
    )

    for i in np.arange(n_repeats):
        X_cur, y_cur = undersample(X, y, random_state=seeds_undersampling[i])

        splitter = sklearn.model_selection.StratifiedKFold(
            n_splits=k_fold,
            random_state=seeds_kfold[i],
            shuffle=True,
        )

        for j, (inds_train_eval, inds_test) in enumerate(splitter.split(X_cur, y_cur)):
            eval_size = int(np.ceil(eval_frac * inds_train_eval.size))
            reseeder.shuffle(inds_train_eval)
            (inds_train, inds_eval) = (inds_train_eval[eval_size:], inds_train_eval[:eval_size])

            (X_train, X_eval, X_test) = (X_cur[inds_train, :], X_cur[inds_eval, :], X_cur[inds_test, :])
            (y_train, y_eval, y_test) = (y_cur[inds_train], y_cur[inds_eval], y_cur[inds_test])

            assert len(X_train) == len(y_train)
            assert len(X_eval) == len(y_eval)
            assert len(X_test) == len(y_test)
            assert len(X_train) >= max(len(X_eval), len(X_test))

            (X_train, X_eval, X_test) = scale_data(X_train, X_eval, X_test)

            torch_rng = torch.Generator().manual_seed(int(seeds_dl[i * k_fold + j]))

            dl_train = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X_train, y_train),
                shuffle=True,
                drop_last=True,
                batch_size=batch_size,
                generator=torch_rng,
            )

            dl_eval = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X_eval, y_eval),
                shuffle=False,
                drop_last=False,
                batch_size=batch_size,
            )

            dl_test = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X_test, y_test),
                shuffle=False,
                drop_last=False,
                batch_size=batch_size,
            )

            train_kwargs: t.Dict[str, t.Any] = {
                "n_classes": n_classes,
                "eval_metric": eval_metric,
                "device": device,
                "param_init_random_state": seeds_param_init[i * k_fold + j],
            }

            adamw_optim_kwargs = tune_optim_hyperparameters(
                dl_train=dl_train,
                dl_eval=dl_eval,
                train_kwargs=train_kwargs,
            )

            cur_res = train(
                dl_train=dl_train,
                dl_eval=dl_eval,
                dl_test=dl_test,
                adamw_optim_kwargs=adamw_optim_kwargs,
                n_epochs=n_epochs,
                tenacity=tenacity,
                early_stopping_rel_improv=early_stopping_rel_improv,
                **train_kwargs,
            )

            for k, v in cur_res.items():
                all_results[k].append(v)

            pbar.update(1)

    output = _summarize_metrics(all_results)

    return output
