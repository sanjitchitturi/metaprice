"""Evaluation: RMSE, revenue ratio, adaptation latency, summary tables."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import numpy as np
import torch

from metaprice.dataset import TaskDataset
from metaprice.meta_learning import inner_adapt, predict_with_params
from metaprice.models import (
    AttentionComposer,
    DemandMLP,
    clone_params,
    compose_parameters,
    compute_hierarchy_representation,
    load_params,
)
from metaprice.trainer import _get_learner_for_task
from metaprice.utils import Timer, revenue_ratio, rmse


def evaluate_metaprice(
    test_ds: TaskDataset,
    learners: Dict[str, Dict[str, DemandMLP]],
    composer: AttentionComposer,
    cfg: Dict[str, Any],
    device: str = "cpu",
) -> Dict[str, Any]:
    """Evaluate MetaPrice using paper-defined parameter-level composition."""
    meta_cfg = cfg.get("meta", {})
    inner_steps = meta_cfg.get("inner_steps", 5)
    inner_lr = meta_cfg.get("inner_lr", 0.01)

    rmses: List[float] = []
    rev_ratios: List[float] = []
    latencies: List[float] = []

    for task in test_ds.tasks:
        timer = Timer()
        with timer:
            cat_model = _get_learner_for_task(task, learners, "category")
            price_model = _get_learner_for_task(task, learners, "price_tier")
            elast_model = _get_learner_for_task(task, learners, "elasticity")

            sx, sy = task.support_tensors(device)
            qx, qy = task.query_tensors(device)

            h_cat = compute_hierarchy_representation(cat_model, device)
            h_price = compute_hierarchy_representation(price_model, device)
            h_elast = compute_hierarchy_representation(elast_model, device)
            v = torch.cat([h_cat, h_price, h_elast])

            with torch.no_grad():
                weights = composer(v)
            composed_params = compose_parameters([cat_model, price_model, elast_model], weights)
            initial_params = [param.detach().requires_grad_(True) for param in composed_params]
            adapted = inner_adapt(cat_model, sx, sy, inner_steps, inner_lr, initial_params)
            with torch.no_grad():
                preds = predict_with_params(cat_model, adapted, qx).cpu().numpy().flatten()

        latencies.append(timer.elapsed_ms)

        # Denormalize for proper RMSE and revenue computation
        targets = task.query_demands
        query_prices = task.query_prices
        stats = task.norm_stats
        if stats:
            preds = preds * stats["d_range"] + stats["d_min"]
            targets = targets * stats["d_range"] + stats["d_min"]
            query_prices = query_prices * stats["p_range"] + stats["p_min"]

        rmses.append(rmse(preds, targets))
        rev_ratios.append(revenue_ratio(preds, targets, query_prices))

    return {
        "model": "MetaPrice",
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "revenue_ratio_mean": float(np.mean(rev_ratios)),
        "revenue_ratio_std": float(np.std(rev_ratios)),
        "latency_mean_ms": float(np.mean(latencies)),
        "latency_std_ms": float(np.std(latencies)),
    }


def evaluate_baseline_maml(
    test_ds: TaskDataset,
    model: DemandMLP,
    inner_steps: int = 5,
    inner_lr: float = 0.01,
    device: str = "cpu",
    name: str = "Standard MAML",
) -> Dict[str, Any]:
    """Evaluate a MAML-style baseline on test tasks."""
    original_params = clone_params(model)
    rmses: List[float] = []
    rev_ratios: List[float] = []
    latencies: List[float] = []

    for task in test_ds.tasks:
        load_params(model, original_params)
        timer = Timer()
        with timer:
            sx, sy = task.support_tensors(device)
            adapted = inner_adapt(model, sx, sy, inner_steps, inner_lr)
            load_params(model, [p.detach() for p in adapted])

            qx, qy = task.query_tensors(device)
            with torch.no_grad():
                preds = model(qx).cpu().numpy().flatten()

        latencies.append(timer.elapsed_ms)
        targets = task.query_demands
        query_prices = task.query_prices
        stats = task.norm_stats
        if stats:
            preds = preds * stats["d_range"] + stats["d_min"]
            targets = targets * stats["d_range"] + stats["d_min"]
            query_prices = query_prices * stats["p_range"] + stats["p_min"]

        rmses.append(rmse(preds, targets))
        rev_ratios.append(revenue_ratio(preds, targets, query_prices))

    load_params(model, original_params)
    return {
        "model": name,
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "revenue_ratio_mean": float(np.mean(rev_ratios)),
        "revenue_ratio_std": float(np.std(rev_ratios)),
        "latency_mean_ms": float(np.mean(latencies)),
        "latency_std_ms": float(np.std(latencies)),
    }


def evaluate_finetuned_baseline(
    test_ds: TaskDataset,
    base_model: DemandMLP,
    finetune_fn: Callable,
    device: str = "cpu",
    name: str = "Transfer Learning",
) -> Dict[str, Any]:
    """Evaluate a baseline that uses fine-tuning (Transfer or Multi-Task)."""
    rmses: List[float] = []
    rev_ratios: List[float] = []
    latencies: List[float] = []

    for task in test_ds.tasks:
        timer = Timer()
        with timer:
            finetuned = finetune_fn(base_model, task, device=device)
            qx, qy = task.query_tensors(device)
            with torch.no_grad():
                preds = finetuned(qx).cpu().numpy().flatten()

        latencies.append(timer.elapsed_ms)
        targets = task.query_demands
        query_prices = task.query_prices
        stats = task.norm_stats
        if stats:
            preds = preds * stats["d_range"] + stats["d_min"]
            targets = targets * stats["d_range"] + stats["d_min"]
            query_prices = query_prices * stats["p_range"] + stats["p_min"]

        rmses.append(rmse(preds, targets))
        rev_ratios.append(revenue_ratio(preds, targets, query_prices))

    return {
        "model": name,
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "revenue_ratio_mean": float(np.mean(rev_ratios)),
        "revenue_ratio_std": float(np.std(rev_ratios)),
        "latency_mean_ms": float(np.mean(latencies)),
        "latency_std_ms": float(np.std(latencies)),
    }


def print_results_table(results: List[Dict[str, Any]]) -> None:
    """Print a formatted comparison table of all models."""
    print("\n" + "=" * 80)
    print(f"{'Model':<25} {'RMSE':>15} {'RevRatio':>15} {'Latency(ms)':>15}")
    print("-" * 80)
    for r in results:
        rmse_str = f"{r['rmse_mean']:.4f} +/- {r['rmse_std']:.4f}"
        rr_str = f"{r['revenue_ratio_mean']:.4f} +/- {r['revenue_ratio_std']:.4f}"
        lat_str = f"{r['latency_mean_ms']:.1f} +/- {r['latency_std_ms']:.1f}"
        print(f"{r['model']:<25} {rmse_str:>15} {rr_str:>15} {lat_str:>15}")
    print("=" * 80)
