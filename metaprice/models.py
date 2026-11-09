"""Demand MLP, attention composition module, and parameter helpers."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class DemandMLP(nn.Module):
    """Compact MLP for demand prediction.

    Input: scalar price (1-dim)
    Output: scalar predicted demand (non-negative via softplus)
    """

    def __init__(self, hidden_dim: int = 64, num_hidden: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = 1
        for _ in range(num_hidden):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Softplus())
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def params_to_vector(model: nn.Module) -> torch.Tensor:
    """Flatten all model parameters into a single vector."""
    return torch.cat([p.reshape(-1) for p in model.parameters()])


def vector_to_params(vec: torch.Tensor, model: nn.Module) -> None:
    """Load a flat parameter vector back into a model."""
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(vec[offset : offset + numel].reshape(p.shape))
        offset += numel


def clone_params(model: nn.Module) -> List[torch.Tensor]:
    """Return a list of cloned parameter tensors."""
    return [p.clone() for p in model.parameters()]


def load_params(model: nn.Module, params: List[torch.Tensor]) -> None:
    """Load parameters from a list into the model."""
    for p, new_p in zip(model.parameters(), params, strict=True):
        p.data.copy_(new_p)


def compose_parameters(models: List[nn.Module], weights: torch.Tensor) -> List[torch.Tensor]:
    """Return the attention-weighted initialization from aligned learners.

    The hierarchy learners are fixed during stage-two training. Detaching their
    parameters keeps gradients focused on the attention network while preserving
    the path from attention weights through adaptation to query loss.
    """
    if len(models) != len(weights):
        raise ValueError("Each hierarchy learner requires one attention weight.")

    parameter_sets = [list(model.parameters()) for model in models]
    parameter_count = len(parameter_sets[0])
    if any(len(params) != parameter_count for params in parameter_sets):
        raise ValueError("Hierarchy learners must share the same parameter layout.")

    return [
        sum(
            weight * params[index].detach()
            for weight, params in zip(weights, parameter_sets, strict=True)
        )
        for index in range(parameter_count)
    ]


class AttentionComposer(nn.Module):
    """Attention-based composition of hierarchy-specific parameter vectors.

    Takes a concatenated representation of 3 hierarchy embeddings and produces
    softmax weights over them.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_hierarchies: int = 3) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_hierarchies)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """v: (input_dim,) -> weights: (num_hierarchies,) summing to 1."""
        a = F.relu(self.fc1(v))
        w = F.softmax(self.fc2(a), dim=-1)
        return w


def compute_hierarchy_representation(model: nn.Module, device: str = "cpu") -> torch.Tensor:
    """Compute a representation vector from a meta-learner.

    Uses the penultimate layer weights flattened as the representation.
    """
    params = list(model.parameters())
    if len(params) >= 4:
        rep = torch.cat([params[-4].reshape(-1), params[-3].reshape(-1)])
    else:
        rep = torch.cat([p.reshape(-1) for p in params])
    return rep.detach().to(device)
