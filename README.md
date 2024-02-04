# Ulysses SentEval: evaluation of sentence embeddings in Brazilian legal domain

Evaluation of textual embeddings for the Brazilian legal domain, similar to the [SentEval](https://github.com/facebookresearch/SentEval) package for the English language and general domain.

---

![Diagram of all Ulysses SentEval tasks.](./assets/ulysses_senteval_tasks_v0.drawio.png)

1. [Installation](#installation)
2. [Task list](#task-list)
3. [Evaluation specifications](#evaluation-and-usage-specifications)
4. [Examples](#examples)
5. [Additional options](#additional-options)
    1. [Embed cache](#embed-cache)
    2. [Evaluation using multiprocessing](#evaluation-using-multiprocessing)
    3. [Changing embedding method and parameters](#changing-embedding-method-and-parameters)
    4. [Using lazy embedders](#using-lazy-embedders)
6. [Paired data for semantic search and strong baseline pretrained sentence model](#paired-data-and-baseline-sentence-model)
7. [License](#license)
8. [Citation](#citation)
9. [References](#references)

---

## Instalation

```bash
python -m pip install -U "git+https://github.com/ulysses-camara/ulysses-senteval"
```

---

## Task list

Task ID | Category | 
:--     | :--      |
TO      | DO.      |

---

## Evaluation and usage specifications

- For each task, a classifier is trained on top of the embedding model under evaluation;
- The embedding model is not updated during the training, just the classifier attached on top of it;
- The classifier architecture is a Logistic Regression except for tasks `F3` and `F5`, in which case it is a feedforward network with 256 hidden units instead;
- The optimizer used is AdamW with weight decay `0.01`;
- Although the train hyper-parameters can be changed using the appropriate API parameters, the standard values must be used to compare models;
- Every pseudo-random number generation is controlled during the execution of this package, hence multiple runs will hold the exact same results;
- Each task is validated using $10x5$-fold repeated cross validation;
- Each training procedure is strictly balanced (all classes have the exact same number of instances) using undersampling;
- Before each training procedure, a quick hyper-parameter search is issues using grid search for 8 epochs in the following search domain:
    - learning rate $\eta$: ${5e-4, 1e-3, 2e-3}$;
    - Adam's $\beta_1$ ${0.80, 0.90}$; and
    - Adam's $\beta_2$: ${0.990, 0.999}$.
- Task datasets are downloaded automatically.

---


## Examples

Below we provide a minimal usage example to evaluate a Sentence Transformer:

```python
import ulysses_senteval
import sentence_transformers

sbert = sentence_transformers.SentenceTransformer("path/to/sbert", device="cuda:0")
evaluator = ulysses_senteval.UlyssesSentEval(sbert)
res = evaluator.evaluate()
print(res)
```

## Additional options

### Recover all unaggregated results

This package aggregates results by epoch/k-fold partitions/train repetitions automatically.
More specifically, the arithmetic average, standard deviation, and lower and upper bounds to the 99% confidence intervals (estimated using boostrapping) are returned for evaluation metric and loss function values for train, validation, and test splits.

However, we provide an option to recover all unaggregated results aswell, so you can visualize all results and/or aggregate the data using your own methods.

```python
import typing as t
import pandas as pd
import sentence_transformers

sbert = sentence_transformers.SentenceTransformer("path/to/sbert", device="cuda:0")
evaluator = ulysses_senteval.UlyssesSentEval(sbert)
res_agg, res_nonagg = evaluator.evaluate(return_all_results=True)

# Data types:
res_agg: t.Dict[str, t.Dict[str, t.Any]]
res_nonagg: pd.DataFrame

# >>> res_nonagg:
#       task  kfold_repetition  kfold_partition  train_epoch                metric     value
# 0       F3                 1                1            1  loss_per_epoch_train  1.799476
# 1       F3                 1                1            2  loss_per_epoch_train  1.141445
# 2       F3                 1                1            3  loss_per_epoch_train  0.874659
# 3       F3                 1                1            4  loss_per_epoch_train  0.604623
# 4       F3                 1                1            5  loss_per_epoch_train  0.402588
# ...    ...               ...              ...          ...                   ...       ...
# 48442   T4                10                1           -1           metric_test  0.158026
# 48443   T4                10                2           -1           metric_test  0.123642
# 48444   T4                10                3           -1           metric_test  0.076494
# 48445   T4                10                4           -1           metric_test  0.186534
# 48446   T4                10                5           -1           metric_test  0.144988
```

### Embed cache

Since embedding models are not optimized during evaluation, the task embeddings can be cached for future usage.
Caching embedding is useful only when changing the train hyper-parameters, but note that different hyper-parameter setup may invalidate standard
comparison of distinct models, so use this resource at your own risk.
Caching is disabled for lazy embedders (e.g., TF-IDF), since they need to be rebuilt for each k-fold partition.

To enable embed caching, you just need to specify the `cache_embed_key` parameter to some value that uniquely identifies your embedding model.
You can use your model's name, for example.

```python
evaluator = ulysses_senteval.UlyssesSentEval(
    sbert,
    cache_embed_key="unique_identifier_for_your_model",
    cache_dir="./cache",  # This is the default value.
)
```

### Evaluation using multiprocessing

You can enable multiprocessing during your evaluation, such that classifiers are trained in parallel up to the number of specified number of processes.

```python
evaluator = ulysses_senteval.UlyssesSentEval(sbert)
res = evaluator.evaluate(kwargs_train={"n_processes": 5})
print(res)
```

Run `help(ulysses_senteval.UlyssesSentEval.evaluate)` for more information. 

> __Known issues__:
  mixing CUDA and multiprocessing is *awkward*, and can lead to some unexpected behaviors.
  For instance, you can not initialize CUDA environment before spawning child processes, or else using CUDA in children processes will be disabled.
  This prevents you from training and evaluating your model using CUDA and multiprocessing in the same process.
  Also, mixing multithreading and multiprocessing can lead to deadlocks.
  We try to circumvent these known issues by disabling multithreading and/or multiprocessing whenever appropriate automatically and/or initializing CUDA only in children processes.
  In the worst case scenario, your execution should fallback to a single process.

### Changing embedding method and parameters

The standard embedding method is inspired by the original SentEval

```python
```

### Using lazy embedders

```python
```

---

## Paired data and baseline sentence model

---

## License
[MIT.](./LICENSE)

---

## Citation

```bibtex
@paper{
}
```

---

## References

[SentEval: An Evaluation Toolkit for Universal Sentence Representations](https://aclanthology.org/L18-1269) (Conneau & Kiela, LREC 2018)
