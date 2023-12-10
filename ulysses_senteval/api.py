"""User-facing resources."""
import typing as t
import os
import warnings

import numpy as np
import torch

from . import train
from . import assets
from . import utils
from . import cache


__all__ = [
    "UlyssesSentEval",
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
            - "G1": "speech_summaries.csv"
            - "G2A": "TODO"
            - "G2B": "summary_vs_bill.csv"
            - "G3": "faqs.csv"
            - "G4": "ulysses_sd.csv"
            - "T1A": "hatebr_offensive_lang.csv"
            - "T1B": "offcombr2.csv"
            - "T2A": "factnews_news_bias.csv"
            - "T2B": "factnews_news_factuality.csv"
            - "T3": "fakebr_size_normalized.csv"

    data_dir_path : str, default="./ulysses_senteval_datasets"
        Ulysses SentEval dataset directory path. If not found, will download data in the specified
        location.

    cache_embed_key : t.Optional[str], default=None
        Key to cache embeddings or fetch cached embeddings. Can be any string that uniquely identifies
        `sentence_model`; typically the model name.

        If not provided, embeddings will not be cached, and previously cached embedding will not be used.

    cache_dir : str, default="./cache"
        Directory to look for cached embeddings. Used only if `cache_embed_key` is not None.
    """

    def __init__(
        self,
        sentence_model: t.Any,
        tasks: t.Union[str, t.Sequence[str]] = "all",
        data_dir_path: str = "./ulysses_senteval_datasets",
        cache_embed_key: t.Optional[str] = None,
        cache_dir: str = "./cache",
        disable_multiprocessing: bool = False,
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

        self.cache_dir = os.path.join(utils.expand_path(cache_dir), cache_embed_key) if cache_embed_key is not None else None

    def embed(self, X_a: t.List[str], X_b: t.Optional[t.List[str]], task: str, **kwargs: t.Any) -> utils.DataType:
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

        **kwargs : t.Any
            Any additional arguments passed by the user using `.evaluate(kwargs_embed=...)`.

        Returns
        -------
        embed : torch.Tensor of shape (N, D) or npt.NDArray[np.float64] of shape (N, D)
            Embeded (X_a, X_b).

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
        >>>               **kwargs: t.Any) -> utils.DataType:
        >>>         embs_a = self.sentence_model.encode(X_a, convert_to_tensor=True, **kwargs).cpu()
        >>>         if X_b is None:
        >>>             return embs_a
        >>>         embs_b = self.sentence_model.encode(X_b, convert_to_tensor=True, **kwargs).cpu()
        >>>         return torch.hstack((embs_a, embs_b, torch.abs(embs_a - embs_b), embs_a * embs_b))
        """
        # pylint: disable='unused-argument'
        embs_a = self.sentence_model.encode(X_a, convert_to_tensor=True, **kwargs).cpu()
        if X_b is None:
            return embs_a
        embs_b = self.sentence_model.encode(X_b, convert_to_tensor=True, **kwargs).cpu()
        # NOTE: standard output as per 'SentEval: An Evaluation Toolkit for Universal Sentence Representations'.
        return torch.hstack((embs_a, embs_b, torch.abs(embs_a - embs_b), embs_a * embs_b))

    def evaluate_in_task(
        self,
        task: str,
        *,
        kwargs_embed: t.Optional[t.Dict[str, t.Any]] = None,
        kwargs_train: t.Optional[t.Dict[str, t.Any]] = None,
        ignore_cached: bool = False,
    ) -> t.Dict[str, t.Any]:
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
            TODO: add which parameters can be modified here.

        ignore_cached : bool, default=False
            If True, previously cached embeddings are ignored, and newly created embeddings will
            overwrite existent files.
            This argument only has effect if `self.cache_dir` has been provided during initialization.

        Returns
        -------
        results : TODO
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

        embs: t.Optional[utils.DataType] = None

        if not ignore_cached and self.cache_dir is not None:
            embs = cache.load_from_cache(cache_dir=self.cache_dir, task=task)

        if embs is None:
            force_main_process = False
            if not self.disable_multiprocessing:
                # NOTE: creating a child process to embed data, since using CUDA in the main process
                #       will invoke a RuntimeError during classifier training if multiprocessing is enabled:
                #           "Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing,
                #            you must use the 'spawn' start method"
                try:
                    with utils.disable_torch_multithreading(), torch.multiprocessing.Pool(1) as ppool:
                        embs = ppool.apply(self.embed, args=(X_a, X_b, task), kwds=kwargs_embed)

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

            if force_main_process or self.disable_multiprocessing:
                embs = self.embed(X_a, X_b, task=task, **kwargs_embed)

            embed_not_cached = True

        else:
            embed_not_cached = False

        assert embs is not None
        assert len(embs) == len(y)

        if self.cache_dir is not None and (embed_not_cached or ignore_cached):
            cache.save_in_cache(embs, cache_dir=self.cache_dir, task=task, overwrite=ignore_cached)

        eval_metric = assets.get_eval_metric(task=task, n_classes=n_classes)

        assert hasattr(eval_metric, "__call__")

        extra_args: t.Dict[str, t.Any] = {}

        if self.disable_multiprocessing:
            extra_args["n_processes"] = 1

        all_results = train.kfold_train(
            X=embs,
            y=y,
            n_classes=n_classes,
            eval_metric=eval_metric,
            pbar_desc=f"Task: {task:<4}",
            **kwargs_train,
            **extra_args,
        )

        return dict(all_results)

    def evaluate(
        self,
        *,
        kwargs_embed: t.Optional[t.Dict[str, t.Any]] = None,
        kwargs_train: t.Optional[t.Dict[str, t.Any]] = None,
        ignore_cached: bool = False,
    ) -> t.Dict[str, t.Any]:
        """Evaluate `self.sentence_model` in each tasks specified during initialization.

        Keyword-only parameters
        ------------------
        kwargs_embed : t.Optional[t.Dict[str, t.Any]], default=None
            Additional arguments for embedding. These are passed directly to `embed` method.

        kwargs_train : t.Optional[t.Dict[str, t.Any]], default=None
            Additional arguments for task classifier training.
            TODO: add which parameters can be modified here.

        ignore_cached : bool, default=False
            If True, previously cached embeddings are ignored, and newly created embeddings will
            overwrite existent files.
            This argument only has effect if `self.cache_dir` has been provided during initialization.

        Returns
        -------
        results : TODO
        """
        results_per_task: t.Dict[str, t.Any] = {}

        for task in self.tasks:
            assets.download_dataset(task, data_dir_path=self.data_dir_path)

        for task in self.tasks:
            cur_res = self.evaluate_in_task(
                task=task,
                kwargs_embed=kwargs_embed,
                kwargs_train=kwargs_train,
                ignore_cached=ignore_cached,
            )

            results_per_task[task] = cur_res

        return results_per_task
