from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    learning_rate: float = 1e-2
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 200
    patience: int = 20
    device: str = "cpu"


def _move_batch(batch: Tuple[torch.Tensor, torch.Tensor], device: str):
    x, y = batch
    return x.to(device), y.to(device)


def train_passive(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    loss_fn: nn.Module,
    config: TrainConfig,
) -> Dict[str, float]:
    device = torch.device(config.device)
    model.to(device)
    # Use Adam for more stable optimization across settings
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    best_val = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    epochs_no_improve = 0

    for epoch in range(config.max_epochs):
        model.train()
        total_loss = 0.0
        num_examples = 0
        for batch in train_loader:
            xb, yb = _move_batch(batch, config.device)
            optimizer.zero_grad(set_to_none=True)
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            # Gradient clipping to prevent exploding updates
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            batch_size = xb.shape[0]
            total_loss += loss.item() * batch_size
            num_examples += batch_size

        train_epoch_loss = total_loss / max(1, num_examples)

        val_epoch_loss = float("nan")
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                total_val = 0.0
                num_val = 0
                for batch in val_loader:
                    xb, yb = _move_batch(batch, config.device)
                    preds = model(xb)
                    loss = loss_fn(preds, yb)
                    total_val += loss.item() * xb.shape[0]
                    num_val += xb.shape[0]
                val_epoch_loss = total_val / max(1, num_val)

            if val_epoch_loss + 1e-12 < best_val:
                best_val = val_epoch_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "train_loss": float(train_epoch_loss),
        "val_loss": float(val_epoch_loss),
        "best_val_loss": float(best_val) if best_state is not None else float(val_epoch_loss),
        "epochs": epoch + 1,
    }


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval().to(device)
    y_true = []
    y_pred = []
    for xb, yb in loader:
        xb = xb.to(device)
        preds = model(xb)
        y_true.append(yb.cpu())
        y_pred.append(preds.cpu())
    return torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0)


