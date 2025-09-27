from typing import Dict

import torch
from sklearn.metrics import accuracy_score, f1_score, log_loss, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from .training import predict


def evaluate_classification(model: nn.Module, loader: DataLoader, device: str = "cpu") -> Dict[str, float]:
    y_true, y_pred = predict(model, loader, device=device)
    y_true_np = y_true.numpy()
    y_pred_labels = torch.argmax(y_pred, dim=1).numpy()
    # Softmax probabilities for probabilistic metrics
    if y_pred.shape[1] > 1:
        y_prob = torch.softmax(y_pred, dim=1).numpy()
    else:
        # Binary with single logit: map to two-class probs
        p1 = torch.sigmoid(y_pred.squeeze(-1)).numpy()
        y_prob = torch.stack([torch.from_numpy(1.0 - p1), torch.from_numpy(p1)], dim=1).numpy()
    return {
        "accuracy": float(accuracy_score(y_true_np, y_pred_labels)),
        "f1_macro": float(f1_score(y_true_np, y_pred_labels, average="macro")),
        "log_loss": float(log_loss(y_true_np, y_prob)),
        # AUROC: handle binary vs multiclass
        "auroc_ovr": float(roc_auc_score(y_true_np, y_prob, multi_class="ovr" if y_prob.shape[1] > 2 else "raise" if y_prob.shape[1] == 2 else "ovr")) if y_prob.shape[1] != 2 else float(roc_auc_score(y_true_np, y_prob[:, 1])),
    }


def evaluate_regression(model: nn.Module, loader: DataLoader, device: str = "cpu") -> Dict[str, float]:
    y_true, y_pred = predict(model, loader, device=device)
    y_true_np = y_true.numpy().squeeze(-1)
    y_pred_np = y_pred.numpy().squeeze(-1)
    return {
        # Compute RMSE without relying on the 'squared' argument for older sklearn versions
        "rmse": float(mean_squared_error(y_true_np, y_pred_np) ** 0.5),
        "mae": float(mean_absolute_error(y_true_np, y_pred_np)),
        "r2": float(r2_score(y_true_np, y_pred_np)),
    }


