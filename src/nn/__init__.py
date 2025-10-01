"""
Active Learning for Neural Networks (ALNN).

This package provides:
- One-hidden-layer MLP models implemented in PyTorch
- Passive learning trainer (SGD) with weight decay and early stopping
- Active learning strategies: Uncertainty Sampling and Sensitivity Analysis
- Dataset utilities for common classification and regression tasks
- Experiment runner CLI to compare strategies across tasks
"""

__all__ = [
    "models",
    "training",
    "strategies",
    "data",
    "evaluation",
]


