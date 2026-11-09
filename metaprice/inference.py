"""Price recommendation for a single new product."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch

from metaprice.dataset import Task
from metaprice.meta_learning import inner_adapt, predict_with_params
from metaprice.models import (
    AttentionComposer,
    DemandMLP,
    compose_parameters,
    compute_hierarchy_representation,
)
from metaprice.trainer import _get_learner_for_task


def recommend_price(
    task: Task,
    learners: Dict[str, Dict[str, DemandMLP]],
    composer: AttentionComposer,
    cfg: Dict[str, Any],
    num_candidates: int = 100,
    device: str = "cpu",
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Recommend optimal price for a new product task.

    Returns (best_price, predicted_revenue, candidate_prices, predicted_demands).
    All returned values are in original (denormalized) scale.
    """
    meta_cfg = cfg.get("meta", {})
    inner_steps = meta_cfg.get("inner_steps", 5)
    inner_lr = meta_cfg.get("inner_lr", 0.01)

    cat_model = _get_learner_for_task(task, learners, "category")
    price_model = _get_learner_for_task(task, learners, "price_tier")
    elast_model = _get_learner_for_task(task, learners, "elasticity")

    sx, sy = task.support_tensors(device)

    candidates_norm = np.linspace(0.0, 1.0, num_candidates)
    cand_tensor = torch.tensor(candidates_norm, dtype=torch.float32, device=device).unsqueeze(-1)

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
        pred_demands_norm = (
            predict_with_params(cat_model, adapted, cand_tensor).cpu().numpy().flatten()
        )

    stats = task.norm_stats
    if stats:
        candidates = candidates_norm * stats["p_range"] + stats["p_min"]
        pred_demands = pred_demands_norm * stats["d_range"] + stats["d_min"]
    else:
        candidates = candidates_norm
        pred_demands = pred_demands_norm

    pred_demands = np.maximum(pred_demands, 0.0)
    revenues = candidates * pred_demands
    best_idx = int(np.argmax(revenues))
    best_price = float(candidates[best_idx])
    best_revenue = float(revenues[best_idx])

    return best_price, best_revenue, candidates, pred_demands
