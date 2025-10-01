from typing import Callable, Optional

import torch
from torch import nn


class OneHiddenMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_units: int,
        output_dim: int,
        hidden_activation: Optional[Callable[[], nn.Module]] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if hidden_activation is None:
            hidden_activation = nn.ReLU

        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.output_dim = output_dim

        # Simple two-layer MLP
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_units, bias=bias),
            hidden_activation(),
            nn.Linear(hidden_units, output_dim, bias=bias),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Initialize weights using Kaiming uniform for ReLU activations
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=0.0, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


