"""Training pipeline for task classifiers."""
import typing as t
import collections
import itertools
import functools
import warnings
import copy

import pandas as pd
import numpy as np
import numpy.typing as npt
import torch
import torch.nn
import sklearn.model_selection
import tqdm

from . import utils
from . import stats


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
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
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


def _single_kfold(
    repetition_id: int,
    random_state: int,
    *,
    X: t.Union[torch.Tensor, utils.PairedRawDataType],
    y: torch.Tensor,
    n_classes: int,
    eval_metric: utils.MetricType,
    batch_size: int,
    eval_frac: float,
    device: t.Union[torch.device, str],
    show_progress_bar: bool,
    k_fold: int,
    n_epochs: int,
    tenacity: int,
    early_stopping_rel_improv: float,
    pbar_desc: t.Optional[str],
    pbar: t.Optional[tqdm.tqdm] = None,
    lazy_embedder: t.Optional[t.Any] = None,
    kwargs_embed: t.Optional[t.Dict[str, t.Any]] = None,
):
    """Perform single k-fold cross validation; isolated to support multiprocessing."""
    reseeder = np.random.RandomState(random_state)
    (seed_undersampling, seed_kfold) = reseeder.randint(0, utils.MAX_RNG_SEED, size=2)
    (seeds_param_init, seeds_dl) = reseeder.randint(0, utils.MAX_RNG_SEED, size=(2, k_fold))
    pbar_random_delay = 1.0 + reseeder.random()

    do_lazy_embedding = not torch.is_tensor(X)

    (X_cur, y_cur) = stats.undersample(X, y, random_state=seed_undersampling)

    splitter = sklearn.model_selection.StratifiedKFold(
        n_splits=k_fold,
        random_state=seed_kfold,
        shuffle=True,
    )

    if pbar is None:
        pbar = tqdm.tqdm(
            total=k_fold,
            desc=f"{pbar_desc} - rep: {repetition_id + 1:<2}",
            disable=not show_progress_bar,
            unit="partition",
            leave=True,
            position=repetition_id,
            delay=pbar_random_delay,
        )

    all_results = collections.defaultdict(list)

    for j, (inds_train_eval, inds_test) in enumerate(splitter.split(X_cur if not do_lazy_embedding else X_cur[0], y_cur)):
        eval_size = int(np.ceil(eval_frac * inds_train_eval.size))
        reseeder.shuffle(inds_train_eval)
        (inds_train, inds_eval) = (inds_train_eval[eval_size:], inds_train_eval[:eval_size])

        # NOTE: can't index X_* directly here, since it can be either a tensor or a tuple of lists.
        (X_train, X_eval, X_test) = (
            utils.take_inds(X_cur, inds_train, paired=do_lazy_embedding),
            utils.take_inds(X_cur, inds_eval, paired=do_lazy_embedding),
            utils.take_inds(X_cur, inds_test, paired=do_lazy_embedding),
        )

        (y_train, y_eval, y_test) = (y_cur[inds_train], y_cur[inds_eval], y_cur[inds_test])

        if do_lazy_embedding:
            assert len(X_train) == len(X_eval) == len(X_test) == 2

            # NOTE: copying 'lazy_embedder' to prevent any kind of embedding contamination
            #       across cross validation runs.
            lazy_embedder_copy = copy.deepcopy(lazy_embedder)

            X_train = lazy_embedder_copy.embed(*X_train, **kwargs_embed, data_split="train")
            X_eval = lazy_embedder_copy.embed(*X_eval, **kwargs_embed, data_split="eval")
            X_test = lazy_embedder_copy.embed(*X_test, **kwargs_embed, data_split="test")

        assert len(X_train) == len(y_train)
        assert len(X_eval) == len(y_eval)
        assert len(X_test) == len(y_test)
        assert (len(X_train) + len(X_eval)) >= len(X_test)

        (X_train, X_eval, X_test) = stats.scale_data(X_train, X_eval, X_test)

        torch_rng = torch.Generator().manual_seed(int(seeds_dl[j]))

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
            "param_init_random_state": seeds_param_init[j],
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

    return all_results


def kfold_train(
    X: t.Union[utils.EmbeddedDataType, utils.PairedRawDataType],
    y: utils.EmbeddedDataType,
    n_classes: int,
    eval_metric: utils.MetricType,
    *,
    n_processes: int = 5,
    batch_size: int = 128,
    eval_frac: float = 0.20,
    device: t.Union[torch.device, str] = "cuda:0",
    show_progress_bar: bool = True,
    n_repeats: int = 10,
    k_fold: int = 5,
    n_epochs: int = 100,
    tenacity: int = 5,
    early_stopping_rel_improv: float = 0.0025,
    random_state: int = 9847706,
    pbar_desc: t.Optional[str] = None,
    lazy_embedder: t.Optional[t.Any] = None,
    kwargs_embed: t.Optional[t.Dict[str, t.Any]] = None,
) -> t.Tuple[t.Dict[str, t.Any], pd.DataFrame]:
    """Apply repeated 5-fold cross validation in the provided data.

    Classes will be re-balanced using undersampling every cross validation repetition.

    Parameters
    ----------
    X : utils.EmbeddedDataType of shape (N, M) or utils.PairedRawDataType
        Data embeddings.

        If (X_a, X_b) tuple, will embed data every `n_repeats * k_fold` repetition. This is adequate for
        lazy embedders, that must take different actions based on train-eval-test splits. In this case,
        you must provide `lazy_embedder` too.

    y : utils.EmbeddedDataType of shape (N,)
        Target labels.

    n_classes : int
        Number of classes.

    eval_metric : utils.MetricType
        Evaluation metric used after each epoch in the validation split and in test split after the
        training is finished.

    Keyword-only parameters
    ----------
    n_processes : int, default=5
        Number of processes to compute cross validation repetitions.
        If n_processes <= 1, will disable multiprocessing.
        Selecting values higher than `n_repeats` will not speed up computations.

    batch_size : int, default=128
        Training and evaluation batch size.

    eval_frac : float, default=0.20
        Evaluation split fraction with respect to the train split size.
        Evaluation instances are randomly sampled after class balancing.

    device : t.Union[torch.device, str], default="cuda:0"
        Device to run training and validation.

    show_progress_bar : bool, default=True
        If True, show progress bar.

    n_repeats : int, default=10
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

    lazy_embedder : t.Optional[t.Any], default=None
        Embedder instance (e.g., UlyssesSentEval instance).
        Used only if X has not been embeded yet (is in the form (X_a, X_b) tuple).

    kwargs_embed : t.Optional[t.Dict[str, t.Any]], default=None
        Additional arguments for embedding. These are passed directly to `embed` method.
        Used only if X has not been embedded yet, i.e., in the form of a tuple of raw data (X_a, X_b).

    Returns
    -------
    output : t.Dict[str, t.Any]
        Summarized train, validation, and test split statistics.
    """
    reseeder = np.random.RandomState(random_state)

    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)

    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)

    if torch.is_tensor(X):
        X.share_memory_()

    if torch.is_tensor(y):
        y.share_memory_()

    all_results = collections.defaultdict(list)

    fn_kfold = functools.partial(
        _single_kfold,
        X=X,
        y=y,
        n_classes=n_classes,
        eval_metric=eval_metric,
        batch_size=batch_size,
        eval_frac=eval_frac,
        device=device,
        show_progress_bar=show_progress_bar,
        k_fold=k_fold,
        n_epochs=n_epochs,
        tenacity=tenacity,
        early_stopping_rel_improv=early_stopping_rel_improv,
        pbar_desc=pbar_desc,
        lazy_embedder=lazy_embedder,
        kwargs_embed=kwargs_embed,
    )

    repetition_ids = np.arange(n_repeats)
    repetition_seeds = reseeder.randint(0, utils.MAX_RNG_SEED, size=n_repeats)
    args = list(zip(repetition_ids, repetition_seeds))
    n_processes = min(n_repeats, n_processes)

    if n_processes > 1:
        try:
            with utils.disable_torch_multithreading(), torch.multiprocessing.Pool(processes=n_processes) as ppool:
                for cur_res in ppool.starmap(fn_kfold, args):
                    for k, v in cur_res.items():
                        all_results[k].extend(v)

        except RuntimeError as err:
            if not utils.is_cuda_vs_multiprocessing_error(err):
                raise err from None

            warnings.warn(
                "You attempted to use multiprocessing, but the CUDA environment had been previously used in the main "
                "process. Unfortunately, this prevents training classifiers in a multiprocessing setup. As a result, "
                "the training process will fallback to using a single process.",
                RuntimeWarning,
            )
            n_processes = 1

    if n_processes <= 1:
        pbar = tqdm.tqdm(
            total=n_repeats * k_fold,
            desc=pbar_desc,
            disable=not show_progress_bar,
            unit="partition",
            leave=True,
        )

        for rep_id, rep_random_state in args:
            cur_res = fn_kfold(repetition_id=rep_id, random_state=rep_random_state, pbar=pbar)
            for k, v in cur_res.items():
                all_results[k].extend(v)

    boostrap_seed = reseeder.randint(0, utils.MAX_RNG_SEED)
    (aggregated_results, all_results) = stats.summarize_metrics(all_results, k_fold=k_fold, random_state=boostrap_seed)

    return (aggregated_results, all_results)
