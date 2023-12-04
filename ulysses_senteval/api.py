"""TODO"""
import typing as t

import numpy as np

from . import train
from . import assets


__all__ = [
    "UlyssesSentEval",
]


class UlyssesSentEval:
    """TODO"""

    def __init__(
        self,
        sentence_model: t.Any,
        tasks: t.Union[str, t.Sequence[str]] = "all",
        batch_size_embed: int = 32,
        data_dir_path: str = "./ulysses_senteval",
        *,
        config: t.Optional[t.Dict[str, t.Any]] = None,
    ):
        """TODO"""
        self.tasks = assets.TASKS if tasks == "all" else tasks
        self.sentence_model = sentence_model

        self.data_dir_path = data_dir_path

        batch_size_embed = int(batch_size_embed)

        if batch_size_embed <= 0:
            raise ValueError(f"'batch_size_embed' must be >= 1 (got {batch_size_embed}).")

        self.batch_size_embed = batch_size_embed

        unknown_tasks = set()

        for task in self.tasks:
            if task not in frozenset(assets.TASKS):
                unknown_tasks.add(task)

        if unknown_tasks:
            unknown_task_string = ", ".join(sorted(unknown_tasks))
            raise ValueError(
                f"Some tasks are not valid task names: {unknown_task_string}.\n"
                f"Please select tasks from: {assets.TASKS}."
            )

    def embed(self, X_a: t.List[str], X_b: t.Optional[t.List[str]], task: str, **kwargs) -> train.DataType:
        """TODO"""
        embs_a = self.sentence_model.encode(X_a, show_progress_bar=True, batch_size=self.batch_size_embed)

        if X_b is None:
            return embs_a

        embs_b = self.sentence_model.encode(X_b, show_progress_bar=True, batch_size=self.batch_size_embed)

        # NOTE: standard output as per 'SentEval: An Evaluation Toolkit for Universal Sentence Representations'.
        out = np.hstack((embs_a, embs_b, np.abs(embs_a - embs_b), embs_a * embs_b))

        return out

    def evaluate_in_task(
        self,
        task: str,
        *,
        kwargs_embed: t.Optional[t.Dict[str, t.Any]] = None,
        kwargs_train: t.Optional[t.Dict[str, t.Any]] = None,
    ) -> t.Dict[str, t.Any]:
        """TODO"""
        kwargs_embed = kwargs_embed or {}
        kwargs_train = kwargs_train or {}

        (X_a, X_b), y = assets.load_data(task, data_dir_path=self.data_dir_path)
        embs = self.embed(X_a, X_b, task=task, **kwargs_embed)
        all_results = train.kfold_train(X=embs, y=y, pbar_desc=f"{task:<4}", **kwargs_train)

        return all_results

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
