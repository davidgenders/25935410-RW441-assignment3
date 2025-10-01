from dataclasses import dataclass
from typing import Literal, Tuple

import torch
from torch import nn


def top_k_indices(scores: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, scores.numel())
    values, indices = torch.topk(scores.view(-1), k)
    return indices


@dataclass
class UncertaintySamplingConfig:
    mode: Literal["classification", "regression"] = "classification"
    method: Literal["entropy", "margin", "least_confidence"] = "entropy"


@torch.no_grad()
def uncertainty_sampling(
    model: nn.Module,
    pool_x: torch.Tensor,
    k: int,
    config: UncertaintySamplingConfig,
) -> torch.Tensor:
    model.eval()
    logits = model(pool_x)
    if config.mode == "classification":
        probs = torch.softmax(logits, dim=-1)
        if config.method == "entropy":
            scores = -(probs * (probs.clamp_min(1e-12).log())).sum(dim=-1)
        elif config.method == "margin":
            top2 = torch.topk(probs, k=2, dim=-1).values
            scores = (top2[:, 0] - top2[:, 1]).neg()
        else:  # least_confidence
            max_conf, _ = torch.max(probs, dim=-1)
            scores = (1.0 - max_conf)
        return top_k_indices(scores, k)
    else:
        scores = -torch.abs(logits.squeeze(-1))
        return top_k_indices(scores, k)


def sensitivity_scores(
    model: nn.Module,
    inputs: torch.Tensor,
    target_dim: int | None = None,
) -> torch.Tensor:
    model.eval()
    x = inputs.clone().detach().requires_grad_(True)
    outputs = model(x)
    if outputs.ndim == 2 and outputs.shape[1] > 1:
        if target_dim is not None:
            values = outputs[:, target_dim]
            grad = torch.autograd.grad(values.sum(), x, create_graph=False)[0]
            scores = grad.pow(2).sum(dim=1).sqrt()
            return scores.detach()
        else:
            # Full Jacobian norm per sample
            batch_size, num_classes = outputs.shape
            scores_list = []
            for c in range(num_classes):
                grad_c = torch.autograd.grad(outputs[:, c].sum(), x, retain_graph=True, create_graph=False)[0]
                scores_list.append(grad_c.pow(2))
            # Sum over classes, then L2 over input dims
            sum_sq = torch.stack(scores_list, dim=0).sum(dim=0)
            scores = sum_sq.sum(dim=1).sqrt()
            return scores.detach()
    else:
        values = outputs.squeeze(-1)
        grad = torch.autograd.grad(values.sum(), x, create_graph=False)[0]
        scores = grad.pow(2).sum(dim=1).sqrt()
        return scores.detach()


def sensitivity_sampling(model: nn.Module, pool_x: torch.Tensor, k: int) -> torch.Tensor:
    scores = sensitivity_scores(model, pool_x)
    return top_k_indices(scores, k)


