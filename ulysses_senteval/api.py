"""TODO"""
import typing as t
import os

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
    """TODO"""

    def __init__(
        self,
        sentence_model: t.Any,
        tasks: t.Union[str, t.Sequence[str]] = "all",
        data_dir_path: str = "./ulysses_senteval",
        cache_embed_key: t.Optional[str] = None,
        cache_dir: str = "./cache",
        *,
        config: t.Optional[t.Dict[str, t.Any]] = None,
    ):
        """TODO"""
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
        self.data_dir_path = data_dir_path

        self.cache_dir = os.path.join(utils.expand_path(cache_dir), cache_embed_key) if cache_embed_key is not None else None

    def embed(self, X_a: t.List[str], X_b: t.Optional[t.List[str]], task: str, **kwargs: t.Any) -> utils.DataType:
        """TODO"""
        # pylint: disable='unused-argument'
        embs_a = self.sentence_model.encode(X_a, convert_to_tensor=True, **kwargs)
        embs_b = self.sentence_model.encode(X_b, convert_to_tensor=True, **kwargs) if X_b is not None else None

        if embs_b is not None:
            # NOTE: standard output as per 'SentEval: An Evaluation Toolkit for Universal Sentence Representations'.
            out = torch.hstack((embs_a, embs_b, torch.abs(embs_a - embs_b), embs_a * embs_b))
        else:
            out = embs_a

        return out.cpu()

    def evaluate_in_task(
        self,
        task: str,
        *,
        kwargs_embed: t.Optional[t.Dict[str, t.Any]] = None,
        kwargs_train: t.Optional[t.Dict[str, t.Any]] = None,
        ignore_cached: bool = False,
    ) -> t.Dict[str, t.Any]:
        """TODO"""
        kwargs_embed = kwargs_embed or {}
        kwargs_train = kwargs_train or {}

        (X_a, X_b), y, n_classes = assets.load_data(task, data_dir_path=self.data_dir_path)

        if not ignore_cached and self.cache_dir is not None:
            embs = cache.load_from_cache(cache_dir=self.cache_dir, task=task)

        if embs is None:
            embs = self.embed(X_a, X_b, task=task, **kwargs_embed)

        if not ignore_cached and self.cache_dir is not None:
            cache.save_in_cache(embs, cache_dir=self.cache_dir, task=task)

        eval_metric = assets.get_eval_metric(task=task, n_classes=n_classes)

        all_results = train.kfold_train(
            X=embs,
            y=y,
            n_classes=n_classes,
            eval_metric=eval_metric,
            pbar_desc=f"Task: {task:<4}",
            **kwargs_train,
        )

        return dict(all_results)

    def evaluate(
        self,
        *,
        kwargs_embed: t.Optional[t.Dict[str, t.Any]] = None,
        kwargs_train: t.Optional[t.Dict[str, t.Any]] = None,
    ) -> t.Dict[str, t.Any]:
        """TODO"""
        results_per_task: t.Dict[str, t.Any] = {}

        for task in self.tasks:
            assets.download_dataset(task)

        for task in self.tasks:
            cur_res = self.evaluate_in_task(
                task=task,
                kwargs_embed=kwargs_embed,
                kwargs_train=kwargs_train,
            )

            results_per_task[task] = cur_res

        return results_per_task
