"""
Utility functions
"""

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import yaml


def parse_configuration(filename="config.yaml"):
    with open(filename, "r") as f:
        data = yaml.safe_load(f)
    return data


def read_dataset(config, dataset="mnist", normalize=False, flatten=False):
    if dataset == "mnist":
        x, y = fetch_openml(
            "mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto"
        )
        if not flatten:
            x = x.reshape(-1, 28, 28)
    elif dataset == "fashion-mnist":
        x, y = fetch_openml(
            "Fashion-MNIST", version=1, return_X_y=True, as_frame=False, parser="auto"
        )
        if not flatten:
            x = x.reshape(-1, 28, 28)
    else:
        raise ValueError("Invalid dataset name")

    if normalize:
        x = x / 255.0

    # Correct configuration if needed
    config["class"] = len(np.unique(y))
    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=0.2, random_state=0, stratify=y
    )  # random_state acts as a seed
    train_y = train_y.astype(np.int64)
    test_y = test_y.astype(np.int64)
    print("Data Loaded! {} classes, {} data".format(config["class"], len(x)))
    return train_x, train_y, test_x, test_y, config
