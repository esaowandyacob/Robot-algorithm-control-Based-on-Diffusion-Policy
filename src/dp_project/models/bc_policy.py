from __future__ import annotations

import torch
from torch import nn


class BehaviorCloningPolicy(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        horizon: int,
        hidden_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = observation_dim
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                ]
            )
            input_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, horizon * action_dim))
        self.network = nn.Sequential(*layers)
        self.horizon = horizon
        self.action_dim = action_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        flattened_observations = observations.reshape(observations.shape[0], -1)
        return self.network(flattened_observations).reshape(-1, self.horizon, self.action_dim)
