from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
import torch
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, TensorDataset


@dataclass
class Split:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor


def _standardize(x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_val_s = scaler.transform(x_val)
    x_test_s = scaler.transform(x_test)
    return x_train_s, x_val_s, x_test_s


def make_classification_split(name: Literal["iris", "wine", "breast_cancer"], test_size: float = 0.2, val_size: float = 0.2, seed: int = 42) -> Split:
    if name == "iris":
        ds = datasets.load_iris()
    elif name == "wine":
        ds = datasets.load_wine()
    elif name == "breast_cancer":
        ds = datasets.load_breast_cancer()

    x_train, x_test, y_train, y_test = train_test_split(ds.data, ds.target, test_size=test_size, random_state=seed, stratify=ds.target)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size, random_state=seed, stratify=y_train)

    x_train, x_val, x_test = _standardize(x_train, x_val, x_test)

    return Split(
        x_train=torch.tensor(x_train, dtype=torch.float32),
        y_train=torch.tensor(y_train, dtype=torch.long),
        x_val=torch.tensor(x_val, dtype=torch.float32),
        y_val=torch.tensor(y_val, dtype=torch.long),
        x_test=torch.tensor(x_test, dtype=torch.float32),
        y_test=torch.tensor(y_test, dtype=torch.long),
    )


def make_regression_split(name: Literal["diabetes", "linnerud", "california"], test_size: float = 0.2, val_size: float = 0.2, seed: int = 42) -> Split:
    if name == "diabetes":
        ds = datasets.load_diabetes()
        y = ds.target.astype(np.float32)
    elif name == "linnerud":
        ds = datasets.load_linnerud()
        y = ds.target[:, 0].astype(np.float32)
    elif name == "california":
        ds = datasets.fetch_california_housing()
        y = ds.target.astype(np.float32)

    x = ds.data.astype(np.float32)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size, random_state=seed)

    x_train, x_val, x_test = _standardize(x_train, x_val, x_test)

    return Split(
        x_train=torch.tensor(x_train, dtype=torch.float32),
        y_train=torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1),
        x_val=torch.tensor(x_val, dtype=torch.float32),
        y_val=torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1),
        x_test=torch.tensor(x_test, dtype=torch.float32),
        y_test=torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1),
    )


def to_datasets(split: Split) -> Tuple[Dataset, Dataset, Dataset]:
    train = TensorDataset(split.x_train, split.y_train)
    val = TensorDataset(split.x_val, split.y_val)
    test = TensorDataset(split.x_test, split.y_test)
    return train, val, test


