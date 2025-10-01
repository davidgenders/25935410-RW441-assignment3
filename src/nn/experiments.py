from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from .data import Split, make_classification_split, make_regression_split, to_datasets
from .evaluation import evaluate_classification, evaluate_regression
from .models import OneHiddenMLP
from .strategies import UncertaintySamplingConfig, sensitivity_sampling, uncertainty_sampling
from .training import TrainConfig, train_passive


@dataclass
class ActiveConfig:
    initial_labeled: int = 20
    query_batch: int = 10
    max_labels: int = 200


def _build_loaders(train: TensorDataset, val: TensorDataset, test: TensorDataset, batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Create data loaders with appropriate shuffling
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(val, batch_size=batch_size, shuffle=False),
        DataLoader(test, batch_size=batch_size, shuffle=False),
    )


def _make_model(input_dim: int, hidden_units: int, output_dim: int) -> nn.Module:
    return OneHiddenMLP(input_dim=input_dim, hidden_units=hidden_units, output_dim=output_dim)


def run_passive_classification(dataset_name: Literal["iris", "wine", "breast_cancer"], hidden_units: int = 64, config: TrainConfig = TrainConfig()) -> Dict[str, float]:
    split = make_classification_split(dataset_name)
    train, val, test = to_datasets(split)
    input_dim = split.x_train.shape[1]
    num_classes = int(split.y_train.max().item() + 1)
    model = _make_model(input_dim, hidden_units, num_classes)
    loss_fn = nn.CrossEntropyLoss()
    loaders = _build_loaders(train, val, test, config.batch_size)
    train_passive(model, loaders[0], loaders[1], loss_fn, config)
    return evaluate_classification(model, loaders[2])


def run_passive_regression(dataset_name: Literal["diabetes", "wine_quality", "california"], hidden_units: int = 64, config: TrainConfig = TrainConfig()) -> Dict[str, float]:
    split = make_regression_split(dataset_name)
    train, val, test = to_datasets(split)
    input_dim = split.x_train.shape[1]
    model = _make_model(input_dim, hidden_units, 1)
    loss_fn = nn.MSELoss()
    loaders = _build_loaders(train, val, test, config.batch_size)
    train_passive(model, loaders[0], loaders[1], loss_fn, config)
    return evaluate_regression(model, loaders[2])


def run_active_classification(
    dataset_name: Literal["iris", "wine", "breast_cancer"],
    strategy: Literal["uncertainty", "sensitivity"],
    uncertainty_method: Literal["entropy", "margin", "least_confidence"] = "entropy",
    hidden_units: int = 64,
    train_config: TrainConfig = TrainConfig(),
    active_config: ActiveConfig = ActiveConfig(),
) -> Dict[str, float]:
    split = make_classification_split(dataset_name)
    input_dim = split.x_train.shape[1]
    num_classes = int(split.y_train.max().item() + 1)
    model = _make_model(input_dim, hidden_units, num_classes)
    loss_fn = nn.CrossEntropyLoss()

    # Set up initial labeled and unlabeled pools
    num_train = split.x_train.shape[0]
    labeled = torch.randperm(num_train)[: active_config.initial_labeled]
    unlabeled = torch.tensor([i for i in range(num_train) if i not in set(labeled.tolist())], dtype=torch.long)

    x_pool = split.x_train.clone()
    y_pool = split.y_train.clone()

    # Active learning loop
    while labeled.numel() < min(active_config.max_labels, num_train):
        train_ds = TensorDataset(x_pool[labeled], y_pool[labeled])
        val_ds = TensorDataset(split.x_val, split.y_val)
        test_ds = TensorDataset(split.x_test, split.y_test)
        loaders = _build_loaders(train_ds, val_ds, test_ds, train_config.batch_size)
        # Reset model weights for fresh training
        model.apply(lambda m: isinstance(m, nn.Linear) and m.reset_parameters())
        train_passive(model, loaders[0], loaders[1], loss_fn, train_config)

        if unlabeled.numel() == 0:
            break

        # Select new samples to label
        if strategy == "uncertainty":
            sel = uncertainty_sampling(
                model,
                x_pool[unlabeled].to("cpu"),
                active_config.query_batch,
                UncertaintySamplingConfig(mode="classification", method=uncertainty_method),
            )
        else:
            sel = sensitivity_sampling(model, x_pool[unlabeled].to("cpu"), active_config.query_batch)

        # Update labeled and unlabeled pools
        newly_selected = unlabeled[sel]
        labeled = torch.unique(torch.cat([labeled, newly_selected]))
        mask = torch.ones_like(unlabeled, dtype=torch.bool)
        mask[sel] = False
        unlabeled = unlabeled[mask]

        if labeled.numel() >= active_config.max_labels:
            break

    # Final evaluation on test set
    final_train = TensorDataset(x_pool[labeled], y_pool[labeled])
    val_ds = TensorDataset(split.x_val, split.y_val)
    test_ds = TensorDataset(split.x_test, split.y_test)
    loaders = _build_loaders(final_train, val_ds, test_ds, train_config.batch_size)
    train_passive(model, loaders[0], loaders[1], loss_fn, train_config)
    return evaluate_classification(model, loaders[2])


def run_active_regression(
    dataset_name: Literal["diabetes", "wine_quality", "california"],
    strategy: Literal["uncertainty", "sensitivity"],
    uncertainty_method: Literal["entropy", "margin", "least_confidence"] = "entropy",
    hidden_units: int = 64,
    train_config: TrainConfig = TrainConfig(),
    active_config: ActiveConfig = ActiveConfig(),
) -> Dict[str, float]:
    split = make_regression_split(dataset_name)
    input_dim = split.x_train.shape[1]
    model = _make_model(input_dim, hidden_units, 1)
    loss_fn = nn.MSELoss()

    # Set up initial labeled and unlabeled pools
    num_train = split.x_train.shape[0]
    labeled = torch.randperm(num_train)[: active_config.initial_labeled]
    unlabeled = torch.tensor([i for i in range(num_train) if i not in set(labeled.tolist())], dtype=torch.long)

    x_pool = split.x_train.clone()
    y_pool = split.y_train.clone()

    # Active learning loop
    while labeled.numel() < min(active_config.max_labels, num_train):
        train_ds = TensorDataset(x_pool[labeled], y_pool[labeled])
        val_ds = TensorDataset(split.x_val, split.y_val)
        test_ds = TensorDataset(split.x_test, split.y_test)
        loaders = _build_loaders(train_ds, val_ds, test_ds, train_config.batch_size)
        # Reset model weights for fresh training
        model.apply(lambda m: isinstance(m, nn.Linear) and m.reset_parameters())
        train_passive(model, loaders[0], loaders[1], loss_fn, train_config)

        if unlabeled.numel() == 0:
            break

        # Select new samples to label
        if strategy == "uncertainty":
            sel = uncertainty_sampling(
                model,
                x_pool[unlabeled].to("cpu"),
                active_config.query_batch,
                UncertaintySamplingConfig(mode="regression", method=uncertainty_method),
            )
        else:
            sel = sensitivity_sampling(model, x_pool[unlabeled].to("cpu"), active_config.query_batch)

        # Update labeled and unlabeled pools
        newly_selected = unlabeled[sel]
        labeled = torch.unique(torch.cat([labeled, newly_selected]))
        mask = torch.ones_like(unlabeled, dtype=torch.bool)
        mask[sel] = False
        unlabeled = unlabeled[mask]

        if labeled.numel() >= active_config.max_labels:
            break

    # Final evaluation on test set
    final_train = TensorDataset(x_pool[labeled], y_pool[labeled])
    val_ds = TensorDataset(split.x_val, split.y_val)
    test_ds = TensorDataset(split.x_test, split.y_test)
    loaders = _build_loaders(final_train, val_ds, test_ds, train_config.batch_size)
    train_passive(model, loaders[0], loaders[1], loss_fn, train_config)
    return evaluate_regression(model, loaders[2])


