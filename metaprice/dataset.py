"""Dataset and task containers for meta-learning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch


@dataclass
class Task:
    """A single product task with support and query sets plus hierarchy info."""

    stock_code: str
    description: str
    support_prices: np.ndarray
    support_demands: np.ndarray
    query_prices: np.ndarray
    query_demands: np.ndarray
    category: str = "General"
    price_tier: str = "Mid-range"
    elasticity: str = "Moderately Elastic"
    norm_stats: Optional[Dict[str, float]] = None

    def support_tensors(self, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.support_prices, dtype=torch.float32, device=device).unsqueeze(-1)
        y = torch.tensor(self.support_demands, dtype=torch.float32, device=device).unsqueeze(-1)
        return x, y

    def query_tensors(self, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.query_prices, dtype=torch.float32, device=device).unsqueeze(-1)
        y = torch.tensor(self.query_demands, dtype=torch.float32, device=device).unsqueeze(-1)
        return x, y


@dataclass
class TaskDataset:
    """Collection of tasks for a split (train/val/test)."""

    tasks: List[Task] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, idx: int) -> Task:
        return self.tasks[idx]

    def tasks_by_hierarchy(self, hierarchy: str) -> Dict[str, List[Task]]:
        """Group tasks by a given hierarchy attribute (category, price_tier, elasticity)."""
        groups: Dict[str, List[Task]] = {}
        for t in self.tasks:
            key = getattr(t, hierarchy)
            groups.setdefault(key, []).append(t)
        return groups
