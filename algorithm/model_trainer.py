from typing import Optional, Dict

import numpy as np

from model.base_model import BaseModel


def train_model(
        model: BaseModel,
        dataset: Dict[str, np.ndarray],
        test_split: Optional[float] = 0.1,
        **kwargs
):
    n = next(iter(dataset.values())).shape[0]

    # shuffle dataset
    shuffle_perm = np.random.permutation(n)
    dataset = {k: v[shuffle_perm] for k, v in dataset.items()}

    # train test split
    if test_split is not None:
        n_test = int(test_split * n)

        dataset_test = {k: v[:n_test] for k, v in dataset.items()}
        dataset_train = {k: v[n_test:] for k, v in dataset.items()}
    else:
        dataset_train = dataset_test = dataset

    # train model
    model.fit(dataset_train, **kwargs)

    # test model
    model.test(dataset_test)
