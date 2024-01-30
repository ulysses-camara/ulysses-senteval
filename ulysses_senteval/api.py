"""User-facing resources."""
import typing as t
import os
import warnings
import itertools

import pandas as pd
import numpy as np
import torch
import sklearn.base

from . import train
from . import assets
from . import utils
from . import cache


__all__ = [
    "UlyssesSentEval",
    "UlyssesSentEvalLazy",
]


class UlyssesSentEval:
    """Sentence embedding evaluator in Brazilian legal tasks.

    Parameters
    ----------
    sentence_model : t.Any
        Sentence model used to produce embeddings for evaluation.

        This class supports ``sentence_transformers.SentenceTransformer`` by default.

        If you are using other architectures, you likely will need to override this class `embed` method.
        See `embed` method documentation for more details.

        NOTE: The `sentence_model` is not copied, rather this class stores a reference to it. Thus, any
        changes in the model after this class instantiation will take effect.

    Keyword-only parameters
    -----------------------
    tasks : t.Sequence[str], str or "all", default="all"
        Sequence of strings or a single string specifying which tasks the `sentence_model` should be
        evaluated on.

        To evaluate on all tasks, set `tasks="all"` (this is the default behaviour).

        Available tasks are:

            - "F1A": "masked_law_name_in_summaries.csv"
            - "F1B": "masked_law_name_in_news.csv"
            - "F2": "code_estatutes_cf88.csv"
            - "F3": "oab_first_part.csv"
            - "F4": "oab_second_part.csv"
            - "F5": "trf_examinations.csv"
            - "F6": "stj_summary.csv"
            - "G1": "bill_summary_to_topics.csv"
            - "G2A": "sts_state_news.csv"
            - "G2B": "summary_vs_bill.csv"
            - "G3": "faqs.csv"
            - "G4": "ulysses_sd.csv"
            - "T1A": "hatebr_offensive_lang.csv"
            - "T1B": "offcombr2.csv"
            - "T2A": "factnews_news_bias.csv"
            - "T2B": "factnews_news_factuality.csv"
            - "T3": "fakebr_size_normalized.csv"
            - "T4": "tampered_leg.csv"

    data_dir_path : str, default="./ulysses_senteval_datasets"
        Ulysses SentEval dataset directory path. If not found, will download data in the specified
        location.

    cache_embed_key : t.Optional[str], default=None
        Key to cache embeddings or fetch cached embeddings. Can be any string that uniquely identifies
        `sentence_model`; typically the model name.

        If not provided, embeddings will not be cached, and previously cached embedding will not be used.

    cache_dir : str, default="./cache"
        Directory to look for cached embeddings. Used only if `cache_embed_key` is not None.

    disable_multiprocessing : bool, default=False
        If True, evaluate embedding model using just the main process.
        Note that this parameter prevents only this package creating multiprocesses; external multiprocessing
        sources (e.g., custom embedding scheme) must be handled by the user.

    lazy_embedding : bool, default=False
        If True, data is embedded only after k-fold train-eval-test split. This means that, given a

        Enabling this option automatically disables embed caching.

        N x k-fold repeated cross-validation involves embedding the data N * k times, which can be computationally
        expensive. However, this setup is mandatory to prevent train-evaluation contamination for dynamically
        constructed models, such as TF-IDF models. If this is your case, we recommend using the UlyssesSentEvalLazy
        class instead, which comes with an adapted embed method compatible with the scikit-learn API.
    """

    def __init__(
        self,
        sentence_model: t.Any,
        *,
        tasks: t.Union[str, t.Sequence[str]] = "all",
        data_dir_path: str = "./ulysses_senteval_datasets",
        cache_embed_key: t.Optional[str] = None,
        cache_dir: str = "./cache",
        disable_multiprocessing: bool = False,
        lazy_embedding: bool = False,
    ):
        tasks = assets.TASKS if tasks == "all" else tasks

        if isinstance(tasks, str):
            tasks = (tasks,)

        unknown_tasks = set()

        for task in tasks:
            if task not in frozenset(assets.TASKS):
                unknown_tasks.add(task)

        if unknown_tasks:
            unknown_task_string = ", ".join(sorted(unknown_tasks))
            raise ValueError(
                f"Some tasks are not valid task names: {unknown_task_string}.\nPlease select tasks from: {assets.TASKS}."
            )

        self.tasks = tasks
        self.sentence_model = sentence_model
        self.data_dir_path = utils.expand_path(data_dir_path)
        self.disable_multiprocessing = disable_multiprocessing
        self.lazy_embedding = lazy_embedding

        self._lazy_sentence_model = sentence_model if self.lazy_embedding else None

        if cache_embed_key is not None and self.lazy_embedding:
            warnings.warn("Ignoring 'cache_embed_key' since lazy_embedding=True.", UserWarning)
            cache_embed_key = None

        self.cache_dir = os.path.join(utils.expand_path(cache_dir), cache_embed_key) if cache_embed_key is not None else None

    def embed(
        self,
        X_a: t.List[str],
        X_b: t.Optional[t.List[str]],
        task: str,
        data_split: str,
        **kwargs: t.Any,
    ) -> utils.EmbeddedDataType:
        """Embed task sentences using `self.sentence_model` (provided during initialization).

        This method embed (X_a, X_b) as (E_a, E_b, |E_a - E_b|, E_a * E_b) if X_b is not None, where
        E_i = self.sentence_model(X_i, **kwargs), and returns E_a if X_b is None.

        To use another embedding strategy, simply override this method (see ``Examples``).

        Parameters
        ----------
        X_a : t.List[str] of length (N,)
            Task first input.

        X_b : t.List[str] of length (N,) or None
            Task second input. May be None.
            If not None, then this list has 1-to-1 correspondence to `X_a`.

        task : str
            Task name. Can be used to embed data conditionally to the task.
            Ignored by default; to use this, you must override this method (see ``Examples``).

        data_split : {'all', 'train', 'eval', 'test'}
            Data split being embedded.
            Useful to properly embed data with a lazy algorithm (e.g. TF-IDF) without train-test contamination.
            Ignored by default; to use this, you must override this method (see ``Examples``).

        **kwargs : t.Any
            Any additional arguments passed by the user using `.evaluate(kwargs_embed=...)`.

        Returns
        -------
        embed : torch.Tensor of shape (N, D) or npt.NDArray[np.float64] of shape (N, D)
            Embeded (X_a, X_b) as a single vector per instance.

        Examples
        --------
        Below we provide a simple example for overriding the `embed` method. This is necessary if you are
        using architectures other than ``sentence_transformers.SentenceTransformer``, or if you want to
        implement alternative embedding strategies. The code displayed below is the default implementation
        for this method; you can freely modify it as long as you return either a torch Tensor (recommended)
        or a numpy Array.

        >>> class CustomSentEval(UlyssesSentEval):
        >>>     def embed(self,
        >>>               X_a: t.List[str],
        >>>               X_b: t.Optional[t.List[str]],
        >>>               task: str,
        >>>               data_split: str,
        >>>               **kwargs: t.Any) -> utils.EmbeddedDataType:
        >>>         embs_a = self.sentence_model.encode(X_a, convert_to_tensor=True, **kwargs).cpu()
        >>>         if X_b is None:
        >>>             return embs_a
        >>>         embs_b = self.sentence_model.encode(X_b, convert_to_tensor=True, **kwargs).cpu()
        >>>         return torch.hstack((embs_a, embs_b, torch.abs(embs_a - embs_b), embs_a * embs_b))
        """
        # pylint: disable='unused-argument'
        assert data_split in {"all", "train", "eval", "test"}

        if self.lazy_embedding:
            warnings.warn(
                "The default 'embed' method is inappropriate when lazy_embedding=True, since it does not "
                "take into account the data split being embedded (train, eval, or test). Please provide "
                f"your own 'embed' method by inheriting from '{self.__class__.__name__}'.",
                UserWarning,
            )

        embs_a = self.sentence_model.encode(X_a, convert_to_tensor=True, **kwargs).cpu()

        if X_b is None:
            return embs_a

        if task in {"F3", "F5"}:
            X_b = list(itertools.chain(*[x_b.split(" [SEP] ") for x_b in X_b]))
            embs_b = self.sentence_model.encode(X_b, convert_to_tensor=True, **kwargs).cpu()
            embs_b = embs_b.reshape(len(embs_a), -1)
            return torch.hstack((embs_a, embs_b))

        embs_b = self.sentence_model.encode(X_b, convert_to_tensor=True, **kwargs).cpu()

        # NOTE: standard output as per 'SentEval: An Evaluation Toolkit for Universal Sentence Representations'.
        return torch.hstack((embs_a, embs_b, torch.abs(embs_a - embs_b), embs_a * embs_b))

    def evaluate_in_task(
        self,
        task: str,
        *,
        kwargs_embed: t.Optional[t.Dict[str, t.Any]] = None,
        kwargs_train: t.Optional[t.Dict[str, t.Any]] = None,
        return_all_results: bool = False,
        ignore_cached: bool = False,
    ) -> t.Tuple[t.Dict[str, t.Any], pd.DataFrame]:
        """Evaluate `self.sentence_model` in a single `task`.

        Parameters
        ----------
        task : str
            Name of the task to evaluate `self.sentence_model`.

        Keyword-only parameters
        ------------------
        kwargs_embed : t.Optional[t.Dict[str, t.Any]], default=None
            Additional arguments for embedding. These are passed directly to `embed` method.

        kwargs_train : t.Optional[t.Dict[str, t.Any]], default=None
            Additional arguments for task classifier training.
            See documentation of `evaluate` method for more information.

        return_all_results : bool, default=False
            If True, return a `pandas.DataFrame` containing statistics for every epoch, k-fold
            repetition, and data split;
            If False, return just statistics aggregated per epoch.

        ignore_cached : bool, default=False
            If True, previously cached embeddings are ignored, and newly created embeddings will
            overwrite existent files.
            This argument only has effect if `self.cache_dir` has been provided during initialization.

        Returns
        -------
        agg_results : t.Dict[str, t.Any]
            Results aggregated per epoch. Contains the following keys:

            - `avg_loss_train_per_epoch` (npt.NDArray): avg. (across k-fold & CV repetitions) train loss per epoch;
            - `avg_loss_eval_per_epoch` (npt.NDArray): avg. (across k-folds & CV repetitions) validation loss per epoch;
            - `avg_metric_eval_per_epoch` (npt.NDArray): avg. (across k-folds & CV repetitions) validation metric per epoch;
            - `avg_loss_test` (float): avg. (across CV repetitions) test loss;
            - `avg_metric_test` (float): avg. (across CV repetitions) test metric;
            - `std_loss_train_per_epoch` (npt.NDArray): std. (across k-fold & CV repetitions) train loss per epoch;
            - `std_loss_eval_per_epoch` (npt.NDArray): std. (across k-folds & CV repetitions) validation loss per epoch;
            - `std_metric_eval_per_epoch` (npt.NDArray): std. (across k-folds & CV repetitions) validation metric per epoch;
            - `std_loss_test` (float): std. (across CV repetitions) test loss;
            - `std_metric_test` (float): std. (across CV repetitions) test metric;

        all_results : pd.DataFrame (optional, only if `return_all_results=True`)
            Pandas DataFrame containing all unaggregated statistics collected from the training procedure.
            Useful if you want to analyze the raw data, or compute your own aggregated statistics.
            Contains the following columns: ("kfold_repetition", "train_epoch", "metric", "value").
        """
        kwargs_embed = kwargs_embed or {}
        kwargs_train = kwargs_train or {}

        if self.disable_multiprocessing and "n_processes" in kwargs_train:
            raise ValueError(
                f"You can not specify '{kwargs_train['n_processes']=}' when {self.disable_multiprocessing=}, "
                "since 'n_processes=1' will be forced. Please remove 'n_processes' from 'kwargs_train'."
            )

        (X_a, X_b), y, n_classes = assets.load_dataset(
            task,
            data_dir_path=self.data_dir_path,
            local_files_only=False,
        )

        assert len(X_a) == len(y)
        assert X_b is None or len(X_b) == len(y)

        embs: t.Optional[utils.EmbeddedDataType] = None

        if not self.lazy_embedding and not ignore_cached and self.cache_dir is not None:
            embs = cache.load_from_cache(cache_dir=self.cache_dir, task=task)

        if not self.lazy_embedding and embs is None:
            force_main_process = False
            if not self.disable_multiprocessing:
                # NOTE: creating a child process to embed data, since using CUDA in the main process
                #       will invoke a RuntimeError during classifier training if multiprocessing is enabled:
                #           "Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing,
                #            you must use the 'spawn' start method"
                try:
                    with utils.disable_torch_multithreading(), torch.multiprocessing.Pool(1) as ppool:
                        embs = ppool.apply(self.embed, args=(X_a, X_b, task, "all"), kwds=kwargs_embed)

                except RuntimeError as err:
                    if utils.is_cuda_vs_multiprocessing_error(err):
                        warnings.warn(
                            "Was not possible to embed data using a subprocess. This means that the CUDA environment "
                            "had been previously initialized, thus training task classifiers in parallel might not be "
                            "possible. If this happens to be the case, the training procedure will fallback to a single "
                            "process.",
                            RuntimeWarning,
                        )
                        force_main_process = True

                    else:
                        raise err

            if force_main_process or self.disable_multiprocessing:
                embs = self.embed(X_a, X_b, task=task, data_split="all", **kwargs_embed)

            embed_not_cached = True

        else:
            embed_not_cached = False

        assert self.lazy_embedding or embs is not None
        assert self.lazy_embedding or len(embs) == len(y)

        if not self.lazy_embedding and self.cache_dir is not None and (embed_not_cached or ignore_cached):
            cache.save_in_cache(embs, cache_dir=self.cache_dir, task=task, overwrite=ignore_cached)

        eval_metric = assets.get_eval_metric(task=task, n_classes=n_classes)

        assert hasattr(eval_metric, "__call__")

        extra_args: t.Dict[str, t.Any] = {}

        if self.disable_multiprocessing:
            extra_args["n_processes"] = 1

        if self.lazy_embedding:
            kwargs_embed = kwargs_embed or {}
            kwargs_embed = kwargs_embed.copy()
            kwargs_embed["task"] = task

        if task in {"F3", "F5"}:
            kwargs_train.setdefault("hidden_dims", [256])

        (aggregated_results, all_results) = train.kfold_train(
            X=embs if not self.lazy_embedding else (X_a, X_b),
            y=y,
            n_classes=n_classes,
            eval_metric=eval_metric,
            pbar_desc=f"Task: {task:<4}",
            lazy_embedder=self if self.lazy_embedding else None,
            kwargs_embed=kwargs_embed if self.lazy_embedding else None,
            **kwargs_train,
            **extra_args,
        )

        aggregated_results = dict(aggregated_results)

        if return_all_results:
            return (aggregated_results, all_results)

        return aggregated_results

    def evaluate(
        self,
        *,
        kwargs_embed: t.Optional[t.Dict[str, t.Any]] = None,
        kwargs_train: t.Optional[t.Dict[str, t.Any]] = None,
        return_all_results: bool = False,
        ignore_cached: bool = False,
    ) -> t.Tuple[t.Dict[str, t.Dict[str, t.Any]], pd.DataFrame]:
        """Evaluate `self.sentence_model` in each tasks specified during initialization.

        Keyword-only parameters
        ------------------
        kwargs_embed : t.Optional[t.Dict[str, t.Any]], default=None
            Additional arguments for embedding. These are passed directly to `embed` method.

            In the default `embed` method, these arguments are passed to `self.sentence_model.encode`
            method (which assumes it is a proper `sentence_transformer.SentenceTransformers`). If this
            behavior is undesired, you will need to provide your own `embed` method.

        kwargs_train : t.Optional[t.Dict[str, t.Any]], default=None
            Additional arguments for task classifier training.

            Computational configuration:

            - `n_processes`: (int, default=5)
                Number of processes to compute cross validation repetitions.
                If n_processes <= 1, will disable multiprocessing.
                Selecting values higher than `n_repeats` will not speed up computations.
            - `device`: (t.Union[torch.device, str], default="cuda:0")
                Device to run training and validation.
            - `show_progress_bar`: (bool, default=True)
                If True, show progress bar.

            The following training hyper-parameters are available:

            WARNING: to compute results comparable to publish results, hyper-parameters should be kept
            as their default value.

            - `batch_size`: (int, default=128)
                Training and evaluation batch size.
            - `hidden_dims` : (t.Optional[t.List[int]], default=None)
                Hidden dimensions of the feed-forward classifier.
                If None (or empty), the resulting classifier will be a Logistic Regression.
                Each hidden layer has batch normalization and ReLU activation.
            - `eval_frac`: (float, default=0.20)
                Evaluation split fraction with respect to the train split size.
                Evaluation instances are randomly sampled after class balancing.
            - `n_repeats`: (int, default=10)
                Number of cross validation repetitions.
                For each repetition, all random number generators are reseeded, and classes are rebalanced.
            - `k_fold`: (int, default=5)
                Number of folds for each cross validation.
            - `n_epochs`: (int, default=100)
                Maximum number of training epochs.
            - `tenacity`: (int, default=5)
                Maximum number of subsequent epochs without sufficient validation loss decrease for early
                stopping.
            - `early_stopping_rel_improv`: (float, default=0.0025)
                Minimum relative difference between best validation loss and current validation loss to
                consider it an actual improvement.

        return_all_results : bool, default=False
            If True, return a `pandas.DataFrame` containing statistics for every epoch, k-fold
            repetition, and data split;
            If False, return just statistics aggregated per task and per epoch.

        ignore_cached : bool, default=False
            If True, previously cached embeddings are ignored, and newly created embeddings will
            overwrite existent files.
            This argument only has effect if `self.cache_dir` has been provided during initialization.

        Returns
        -------
        agg_results : t.Dict[str, t.Dict[str, t.Any]]
            Results aggregated per task and per epoch. Keys are task names. Values are dictionaries, each containing
            the following keys:

            - `avg_loss_train_per_epoch` (npt.NDArray): avg. (across k-fold & CV repetitions) train loss per epoch;
            - `avg_loss_eval_per_epoch` (npt.NDArray): avg. (across k-folds & CV repetitions) validation loss per epoch;
            - `avg_metric_eval_per_epoch` (npt.NDArray): avg. (across k-folds & CV repetitions) validation metric per epoch;
            - `avg_loss_test` (float): avg. (across CV repetitions) test loss;
            - `avg_metric_test` (float): avg. (across CV repetitions) test metric;
            - `std_loss_train_per_epoch` (npt.NDArray): std. (across k-fold & CV repetitions) train loss per epoch;
            - `std_loss_eval_per_epoch` (npt.NDArray): std. (across k-folds & CV repetitions) validation loss per epoch;
            - `std_metric_eval_per_epoch` (npt.NDArray): std. (across k-folds & CV repetitions) validation metric per epoch;
            - `std_loss_test` (float): std. (across CV repetitions) test loss;
            - `std_metric_test` (float): std. (across CV repetitions) test metric;

        all_results : pd.DataFrame (optional, only if `return_all_results=True`)
            Pandas DataFrame containing all unaggregated statistics collected from the training procedure.
            Useful if you want to analyze the raw data, or compute your own aggregated statistics.
            Contains the following columns: ("task", "kfold_repetition", "train_epoch", "metric", "value").
        """
        results_per_task: t.Dict[str, t.Any] = {}

        for task in self.tasks:
            assets.download_dataset(task, data_dir_path=self.data_dir_path)

        all_dfs: t.List[pd.DataFrame] = []

        for task in self.tasks:
            cur_res, cur_all_results = self.evaluate_in_task(
                task=task,
                kwargs_embed=kwargs_embed,
                kwargs_train=kwargs_train,
                return_all_results=True,
                ignore_cached=ignore_cached,
            )

            results_per_task[task] = cur_res

            cur_all_results.insert(loc=0, column="task", value=task, allow_duplicates=False)
            all_dfs.append(cur_all_results)

        all_results = pd.concat(all_dfs, ignore_index=True)

        if return_all_results:
            return (results_per_task, all_results)

        return results_per_task


class UlyssesSentEvalLazy(UlyssesSentEval):
    """Ulysses SentEval adapted for lazy embedding models.

    If you are using a lazy embedder, like TF-IDF models, you should use this class instead of the
    regular UlyssesSentEval.

    This class API follows the UlyssesSentEval API, except it enforces lazy_embedding=True, and also
    the embedding method is adapted to the scikit-learn API.
    """

    def __init__(self, sentence_model: t.Any, **kwargs: t.Any):
        super().__init__(sentence_model=sentence_model, **kwargs, lazy_embedding=True)

    def embed(
        self,
        X_a: t.List[str],
        X_b: t.Optional[t.List[str]],
        task: str,
        data_split: str,
        **kwargs: t.Any,
    ) -> utils.EmbeddedDataType:
        """Embed data taking into consideration different `data_split`."""
        if data_split == "all":
            raise ValueError(
                "You can not embed all data at once using a lazy embedder. Please call 'embed' passing "
                "'data_split=train' and 'data_split=test' separately (and, optionally, 'data_split=eval')."
            )

        if data_split == "train":
            # NOTE: cloning self.sentence_model is not strictly necessary, since we're fitting the new
            #       model on top of any previously fitted values. Also, this model is cloned for each
            #       partitioning of k-fold cross validation to avoid any contamination across cross
            #       validation runs.
            self._lazy_sentence_model = sklearn.base.clone(self.sentence_model)
            self._lazy_sentence_model.fit(X_a if X_b is None else [*X_a, *X_b])

        embs_a = self._lazy_sentence_model.transform(X_a)
        embs_a = embs_a.astype(np.float32, copy=False)
        embs_a = embs_a if isinstance(embs_a, np.ndarray) else embs_a.toarray()

        if X_b is None:
            return torch.from_numpy(embs_a).float()

        embs_b = self._lazy_sentence_model.transform(X_b)
        embs_b = embs_b.astype(np.float32, copy=False)
        embs_b = embs_b if isinstance(embs_b, np.ndarray) else embs_b.toarray()

        out = np.hstack((embs_a, embs_b, np.abs(embs_a - embs_b), embs_a * embs_b))
        out = torch.from_numpy(out)
        return out.float()
