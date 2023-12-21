"""Auxiliary functions for statistical operations."""
import typing as t
import itertools
import functools

import pandas as pd
import numpy as np
import numpy.typing as npt
import scipy.stats
import torch

from . import utils


def undersample(
    X: t.Union[utils.EmbeddedDataType, utils.PairedRawDataType], y: utils.EmbeddedDataType, random_state: int
) -> t.Tuple[utils.EmbeddedDataType, utils.EmbeddedDataType]:
    """Undersample majority classes to match the minority class frequency.

    Parameters
    ----------
    X : npt.NDArray[np.float64] of shape (N, M) or utils.PairedRawDataType
        Original embeddings.

    y : npt.NDArray of shape (N,)
        Original targets.

    Returns
    -------
    X_undersampled : npt.NDArray[np.float64] of shape (K, M)
        Undersampled X, where K <= N.

    y_undersampled : npt.NDArray of shape(K,)
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

    if torch.is_tensor(X):
        X = X[sampled_inds, :]

    else:
        (X_a, X_b) = X
        X_a = [X_a[i] for i in sampled_inds]
        X_b = [X_b[i] for i in sampled_inds] if X_b is not None else None
        X = (X_a, X_b)

    y = y[sampled_inds]

    return (X, y)


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


def avg_ci_with_bootstrap(vals: npt.NDArray[np.float64], random_state: int) -> t.Tuple[float, float]:
    """Estimate confidence intervals for the average of `vals` using empirical bootstrap."""
    if vals.size == 1:
        return (float(vals.values), float(vals.values))

    global_mean = float(np.mean(vals))

    (diff_low, diff_high) = scipy.stats.bootstrap(
        data=np.atleast_2d(vals),
        statistic=lambda x: global_mean - float(np.mean(x)),
        n_resamples=int(10**4),
        confidence_level=0.99,
        random_state=random_state,
    ).confidence_interval

    return (global_mean + diff_low, global_mean + diff_high)


def summarize_metrics(
    all_results: t.Dict[str, t.Any], k_fold: int, random_state: int
) -> t.Tuple[t.Dict[str, t.Any], pd.DataFrame]:
    """Summarize metrics collected from training."""
    seeder = np.random.RandomState(random_state)
    rng_seeds_per_key = {k: seeder.randint(0, utils.MAX_RNG_SEED) for k in sorted(all_results.keys())}

    output: t.Dict[str, t.Any] = {}
    all_dfs: t.List[pd.DataFrame] = []

    for k, v in all_results.items():
        val_type = type(v[0])

        if hasattr(val_type, "__len__"):
            lens = list(map(len, v))
            max_len = int(max(lens))

            df_stat_per_epoch = pd.DataFrame(
                {
                    "kfold_repetition": itertools.chain(*[[1 + i // k_fold] * li for i, li in enumerate(lens)]),
                    "kfold_partition": itertools.chain(*[[1 + i % k_fold] * li for i, li in enumerate(lens)]),
                    "train_epoch": itertools.chain(*[1 + np.arange(li) for li in lens]),
                    k: itertools.chain(*v),
                }
            )

        else:
            df_stat_per_epoch = pd.DataFrame(
                {
                    "kfold_repetition": 1 + (np.arange(len(v)) // k_fold),
                    "kfold_partition": 1 + (np.arange(len(v)) % k_fold),
                    "train_epoch": -1,
                    k: v,
                }
            )

        metric_grouped_by_epoch = df_stat_per_epoch.groupby("train_epoch")[k]

        (avg_per_epoch, std_per_epoch) = metric_grouped_by_epoch.agg(("mean", "std")).values.T
        std_per_epoch = np.nan_to_num(std_per_epoch, nan=0.0, copy=False)

        avg_ci_with_bootstrap_ = functools.partial(avg_ci_with_bootstrap, random_state=rng_seeds_per_key[k])
        ci_per_epoch = metric_grouped_by_epoch.agg(avg_ci_with_bootstrap_).values
        ci_per_epoch = np.vstack(ci_per_epoch)
        (ci_low, ci_high) = ci_per_epoch.T

        if avg_per_epoch.size == 1:
            avg_per_epoch = float(avg_per_epoch)
            std_per_epoch = float(std_per_epoch)
            ci_low = float(ci_low.squeeze())
            ci_high = float(ci_high.squeeze())

        output[f"avg_{k}"] = avg_per_epoch
        output[f"std_{k}"] = std_per_epoch
        output[f"ci99_low_{k}"] = ci_low
        output[f"ci99_high_{k}"] = ci_high

        df_stat_per_epoch = df_stat_per_epoch.melt(
            id_vars=["kfold_repetition", "kfold_partition", "train_epoch"],
            var_name="metric",
            value_name="value",
            ignore_index=True,
        )

        all_dfs.append(df_stat_per_epoch)

    df_all_results = pd.concat(all_dfs, ignore_index=True)

    return (output, df_all_results)
