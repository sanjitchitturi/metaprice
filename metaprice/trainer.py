"""Training loops for Stage 1 (hierarchical meta-learners) and Stage 2 (attention)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

from metaprice.dataset import Task, TaskDataset
from metaprice.meta_learning import compute_query_loss, inner_adapt, meta_train_step
from metaprice.models import (
    AttentionComposer,
    DemandMLP,
    clone_params,
    compose_parameters,
    compute_hierarchy_representation,
    load_params,
)


def _sample_batch(tasks: List[Task], batch_size: int, rng: np.random.RandomState) -> List[Task]:
    """Sample a mini-batch of tasks."""
    if len(tasks) == 0:
        return []
    indices = rng.choice(len(tasks), size=min(batch_size, len(tasks)), replace=False)
    return [tasks[i] for i in indices]


def train_hierarchy_meta_learner(
    tasks: List[Task],
    val_tasks: List[Task],
    cfg: Dict[str, Any],
    device: str = "cpu",
    hierarchy_name: str = "",
) -> DemandMLP:
    """Stage 1: Train a single MAML meta-learner on tasks from one hierarchy group."""
    model_cfg = cfg.get("model", {})
    meta_cfg = cfg.get("meta", {})

    if not tasks:
        raise ValueError("Cannot train a hierarchy on an empty task set.")

    model = DemandMLP(
        hidden_dim=model_cfg.get("hidden_dim", 64),
        num_hidden=model_cfg.get("num_hidden", 2),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=meta_cfg.get("outer_lr", 1e-3),
        weight_decay=meta_cfg.get("weight_decay", 0.0),
    )

    inner_steps = meta_cfg.get("inner_steps", 5)
    inner_lr = meta_cfg.get("inner_lr", 0.01)
    meta_batch = meta_cfg.get("meta_batch", 10)
    epochs = meta_cfg.get("stage1_epochs", 50)
    grad_clip = meta_cfg.get("grad_clip", 5.0)
    seed = cfg.get("seed", 42)
    rng = np.random.RandomState(seed)

    best_val_loss = float("inf")
    best_params = clone_params(model)
    patience = 10
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        batch = _sample_batch(tasks, meta_batch, rng)
        if not batch:
            break

        optimizer.zero_grad()
        loss = meta_train_step(model, batch, inner_steps, inner_lr, device)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # Validation
        if not val_tasks:
            best_params = [p.detach().clone() for p in model.parameters()]
        if val_tasks and ((epoch + 1) % 5 == 0 or epoch + 1 == epochs):
            model.eval()
            val_loss = 0.0
            orig = clone_params(model)
            for vt in val_tasks:
                load_params(model, orig)
                sx, sy = vt.support_tensors(device)
                qx, qy = vt.query_tensors(device)
                adapted = inner_adapt(model, sx, sy, inner_steps, inner_lr)
                with torch.no_grad():
                    load_params(model, adapted)
                    pred = model(qx)
                    val_loss += float(torch.nn.functional.mse_loss(pred, qy))
            load_params(model, orig)
            val_loss /= max(len(val_tasks), 1)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = clone_params(model)
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                print(f"  [{hierarchy_name}] Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 10 == 0:
            print(f"  [{hierarchy_name}] Epoch {epoch + 1}/{epochs}, loss={float(loss):.4f}")

    load_params(model, best_params)
    return model


def train_all_hierarchy_learners(
    train_ds: TaskDataset,
    val_ds: TaskDataset,
    cfg: Dict[str, Any],
    device: str = "cpu",
) -> Dict[str, Dict[str, DemandMLP]]:
    """Train hierarchy-specific meta-learners for all 3 hierarchies.

    Returns nested dict: hierarchy_type -> group_name -> trained DemandMLP.
    """
    hierarchy_types = ["category", "price_tier", "elasticity"]
    all_learners: Dict[str, Dict[str, DemandMLP]] = {}

    for h_type in hierarchy_types:
        print(f"\nTraining hierarchy: {h_type}")
        train_groups = train_ds.tasks_by_hierarchy(h_type)
        val_groups = val_ds.tasks_by_hierarchy(h_type)

        learners: Dict[str, DemandMLP] = {}
        for group_name, group_tasks in train_groups.items():
            if len(group_tasks) < 2:
                print(f"  Skipping group '{group_name}' (only {len(group_tasks)} tasks)")
                continue
            val_group = val_groups.get(group_name, [])
            print(f"  Group '{group_name}': {len(group_tasks)} train, {len(val_group)} val tasks")
            learner = train_hierarchy_meta_learner(
                group_tasks, val_group, cfg, device, f"{h_type}/{group_name}"
            )
            learners[group_name] = learner

        # Ensure at least one learner exists; create a default if needed
        if not learners:
            print(f"  No groups trained for {h_type}. Training on all tasks as fallback.")
            learners["_default"] = train_hierarchy_meta_learner(
                train_ds.tasks, val_ds.tasks, cfg, device, f"{h_type}/_default"
            )

        all_learners[h_type] = learners

    return all_learners


def _get_learner_for_task(
    task: Task,
    learners: Dict[str, Dict[str, DemandMLP]],
    h_type: str,
) -> DemandMLP:
    """Get the learner for a task's hierarchy group, falling back to a default."""
    group = getattr(task, h_type)
    group_learners = learners.get(h_type, {})
    if group in group_learners:
        return group_learners[group]
    # Fallback: use the first available learner in this hierarchy
    return next(iter(group_learners.values()))


def train_attention_network(
    learners: Dict[str, Dict[str, DemandMLP]],
    val_ds: TaskDataset,
    cfg: Dict[str, Any],
    device: str = "cpu",
) -> AttentionComposer:
    """Train the paper's parameter-level attention composition network."""
    meta_cfg = cfg.get("meta", {})
    model_cfg = cfg.get("model", {})
    epochs = meta_cfg.get("stage2_epochs", 20)
    inner_steps = meta_cfg.get("inner_steps", 5)
    inner_lr = meta_cfg.get("inner_lr", 0.01)
    grad_clip = meta_cfg.get("grad_clip", 5.0)
    hidden_dim = model_cfg.get("hidden_dim", 64)

    # Compute representation dimension
    sample_model = next(iter(next(iter(learners.values())).values()))
    rep_dim = compute_hierarchy_representation(sample_model, device).shape[0]
    input_dim = rep_dim * 3

    composer = AttentionComposer(input_dim, hidden_dim, num_hierarchies=3).to(device)
    optimizer = torch.optim.Adam(composer.parameters(), lr=meta_cfg.get("outer_lr", 1e-3))

    rng = np.random.RandomState(cfg.get("seed", 42) + 1000)

    hierarchy_reps: dict[int, torch.Tensor] = {}
    hierarchy_models: dict[int, list[DemandMLP]] = {}

    for idx in range(len(val_ds)):
        task = val_ds[idx]
        cat_model = _get_learner_for_task(task, learners, "category")
        price_model = _get_learner_for_task(task, learners, "price_tier")
        elast_model = _get_learner_for_task(task, learners, "elasticity")
        hierarchy_models[idx] = [cat_model, price_model, elast_model]

        h_cat = compute_hierarchy_representation(cat_model, device)
        h_price = compute_hierarchy_representation(price_model, device)
        h_elast = compute_hierarchy_representation(elast_model, device)
        hierarchy_reps[idx] = torch.cat([h_cat, h_price, h_elast])

    for epoch in range(epochs):
        total_loss = 0.0
        perm = rng.permutation(len(val_ds))

        for idx in perm:
            task = val_ds[int(idx)]
            sx, sy = task.support_tensors(device)
            qx, qy = task.query_tensors(device)
            models = hierarchy_models[int(idx)]
            weights = composer(hierarchy_reps[int(idx)])
            composed_params = compose_parameters(models, weights)
            adapted = inner_adapt(models[0], sx, sy, inner_steps, inner_lr, composed_params)
            loss = compute_query_loss(models[0], adapted, qx, qy)

            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(composer.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.detach().item()

        avg_loss = total_loss / max(len(val_ds), 1)
        if (epoch + 1) % 5 == 0:
            print(f"  [Attention] Epoch {epoch + 1}/{epochs}, loss={avg_loss:.4f}")

    return composer


def save_models(
    learners: Dict[str, Dict[str, DemandMLP]],
    composer: AttentionComposer,
    output_dir: str,
) -> None:
    """Save all trained models to disk."""
    os.makedirs(output_dir, exist_ok=True)
    manifest = {"format_version": 1, "learners": {}}
    for h_type, group_learners in learners.items():
        manifest["learners"][h_type] = {}
        for index, (group_name, model) in enumerate(group_learners.items()):
            filename = f"learner_{h_type}_{index}.pt"
            path = os.path.join(output_dir, filename)
            torch.save(model.state_dict(), path)
            manifest["learners"][h_type][group_name] = filename
    torch.save(composer.state_dict(), os.path.join(output_dir, "attention_composer.pt"))
    (Path(output_dir) / "checkpoint-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Models saved to {output_dir}")


def load_models(
    output_dir: str, cfg: Dict[str, Any], device: str = "cpu"
) -> tuple[Dict[str, Dict[str, DemandMLP]], AttentionComposer]:
    """Load hierarchy learners and composer from a trusted training directory."""
    root = Path(output_dir)
    manifest = json.loads((root / "checkpoint-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint manifest version")
    model_cfg = cfg.get("model", {})
    learners: Dict[str, Dict[str, DemandMLP]] = {}
    for hierarchy in ("category", "price_tier", "elasticity"):
        groups = manifest.get("learners", {}).get(hierarchy, {})
        if not groups:
            raise ValueError(f"No saved learners for {hierarchy}")
        learners[hierarchy] = {}
        for group, filename in groups.items():
            if Path(filename).name != filename or not filename.endswith(".pt"):
                raise ValueError("Invalid checkpoint filename")
            model = DemandMLP(
                hidden_dim=model_cfg.get("hidden_dim", 64),
                num_hidden=model_cfg.get("num_hidden", 2),
            ).to(device)
            model.load_state_dict(
                torch.load(root / filename, map_location=device, weights_only=True)
            )
            model.eval()
            learners[hierarchy][group] = model
    sample = next(iter(learners["category"].values()))
    input_dim = compute_hierarchy_representation(sample, device).numel() * 3
    composer = AttentionComposer(input_dim, model_cfg.get("hidden_dim", 64)).to(device)
    composer.load_state_dict(
        torch.load(root / "attention_composer.pt", map_location=device, weights_only=True)
    )
    composer.eval()
    return learners, composer
