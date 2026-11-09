"""Baseline methods: Transfer Learning, Multi-Task Learning, Standard MAML."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from metaprice.dataset import Task, TaskDataset
from metaprice.meta_learning import meta_train_step
from metaprice.models import DemandMLP


def train_transfer_learning(
    train_ds: TaskDataset,
    cfg: Dict[str, Any],
    device: str = "cpu",
) -> DemandMLP:
    """Pre-train a demand model on all training product data (support + query)."""
    model_cfg = cfg.get("model", {})

    model = DemandMLP(
        hidden_dim=model_cfg.get("hidden_dim", 64),
        num_hidden=model_cfg.get("num_hidden", 2),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Collect all data
    all_x: List[torch.Tensor] = []
    all_y: List[torch.Tensor] = []
    for task in train_ds.tasks:
        sx, sy = task.support_tensors(device)
        qx, qy = task.query_tensors(device)
        all_x.extend([sx, qx])
        all_y.extend([sy, qy])

    X = torch.cat(all_x, dim=0)
    Y = torch.cat(all_y, dim=0)

    for _ in range(100):
        optimizer.zero_grad()
        pred = model(X)
        loss = F.mse_loss(pred, Y)
        loss.backward()
        optimizer.step()

    return model


def finetune_transfer(
    model: DemandMLP,
    task: Task,
    steps: int = 10,
    lr: float = 1e-3,
    device: str = "cpu",
) -> DemandMLP:
    """Fine-tune a pre-trained model on a task's support set."""
    import copy

    finetuned = copy.deepcopy(model)
    optimizer = torch.optim.SGD(finetuned.parameters(), lr=lr)
    sx, sy = task.support_tensors(device)

    for _ in range(steps):
        optimizer.zero_grad()
        pred = finetuned(sx)
        loss = F.mse_loss(pred, sy)
        loss.backward()
        optimizer.step()

    return finetuned


def train_multitask(
    train_ds: TaskDataset,
    cfg: Dict[str, Any],
    device: str = "cpu",
) -> DemandMLP:
    """Train a multi-task model on all support sets from training products."""
    model_cfg = cfg.get("model", {})
    bl_cfg = cfg.get("baselines", {}).get("multitask", {})

    model = DemandMLP(
        hidden_dim=model_cfg.get("hidden_dim", 64),
        num_hidden=model_cfg.get("num_hidden", 2),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=bl_cfg.get("lr", 1e-3),
    )

    # Use only support sets
    all_x: List[torch.Tensor] = []
    all_y: List[torch.Tensor] = []
    for task in train_ds.tasks:
        sx, sy = task.support_tensors(device)
        all_x.append(sx)
        all_y.append(sy)

    X = torch.cat(all_x, dim=0)
    Y = torch.cat(all_y, dim=0)

    epochs = bl_cfg.get("epochs", 100)
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(X)
        loss = F.mse_loss(pred, Y)
        loss.backward()
        optimizer.step()

    return model


def finetune_multitask(
    model: DemandMLP,
    task: Task,
    steps: int = 5,
    lr: float = 0.01,
    device: str = "cpu",
) -> DemandMLP:
    """Fine-tune the multi-task model on a task's support set."""
    import copy

    finetuned = copy.deepcopy(model)
    optimizer = torch.optim.SGD(finetuned.parameters(), lr=lr)
    sx, sy = task.support_tensors(device)

    for _ in range(steps):
        optimizer.zero_grad()
        pred = finetuned(sx)
        loss = F.mse_loss(pred, sy)
        loss.backward()
        optimizer.step()

    return finetuned


def train_standard_maml(
    train_ds: TaskDataset,
    val_ds: TaskDataset,
    cfg: Dict[str, Any],
    device: str = "cpu",
) -> DemandMLP:
    """Train a flat MAML meta-learner over all training tasks (no hierarchy)."""
    model_cfg = cfg.get("model", {})
    bl_cfg = cfg.get("baselines", {}).get("maml", {})

    model = DemandMLP(
        hidden_dim=model_cfg.get("hidden_dim", 64),
        num_hidden=model_cfg.get("num_hidden", 2),
    ).to(device)

    inner_steps = bl_cfg.get("inner_steps", 5)
    inner_lr = bl_cfg.get("inner_lr", 0.01)
    outer_lr = bl_cfg.get("outer_lr", 1e-3)
    meta_batch = bl_cfg.get("meta_batch", 10)
    epochs = bl_cfg.get("epochs", 50)

    optimizer = torch.optim.Adam(model.parameters(), lr=outer_lr)
    rng = np.random.RandomState(cfg.get("seed", 42) + 2000)

    for epoch in range(epochs):
        indices = rng.choice(len(train_ds), size=min(meta_batch, len(train_ds)), replace=False)
        batch = [train_ds[int(i)] for i in indices]

        optimizer.zero_grad()
        loss = meta_train_step(model, batch, inner_steps, inner_lr, device)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"  [Standard MAML] Epoch {epoch + 1}/{epochs}, loss={float(loss):.4f}")

    return model
