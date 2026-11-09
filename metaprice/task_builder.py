"""Build few-shot tasks from preprocessed product data."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from metaprice.dataset import Task, TaskDataset
from metaprice.hierarchy import assign_category, assign_elasticity, assign_price_tier


def build_tasks(
    product_data: Dict[str, pd.DataFrame],
    descriptions: Dict[str, str],
    support_size: int = 5,
    cfg: Dict[str, Any] | None = None,
    norm_stats: Dict[str, Dict[str, float]] | None = None,
) -> List[Task]:
    """Convert product data dict into a list of Task objects."""
    if cfg is None:
        cfg = {}

    hierarchy_cfg = cfg.get("hierarchy", {})
    price_cfg = hierarchy_cfg.get("price_tiers", {})
    elast_cfg = hierarchy_cfg.get("elasticity", {})

    budget_max = price_cfg.get("budget_max", 2.0)
    midrange_max = price_cfg.get("midrange_max", 5.0)
    high_thresh = elast_cfg.get("high_threshold", -0.5)
    low_thresh = elast_cfg.get("low_threshold", -0.2)

    if norm_stats is None:
        norm_stats = {}

    if not isinstance(support_size, int) or isinstance(support_size, bool) or support_size < 1:
        raise ValueError("support_size must be a positive integer.")
    tasks: List[Task] = []
    for code, pdf in product_data.items():
        prices = pdf["price"].values
        demands = pdf["demand"].values
        if len(prices) <= support_size:
            raise ValueError(f"Product {code}: support_size must leave at least one query point.")
        if not np.isfinite(prices).all() or not np.isfinite(demands).all():
            raise ValueError(f"Product {code}: price and demand values must be finite.")

        support_p = prices[:support_size]
        support_d = demands[:support_size]
        query_p = prices[support_size:]
        query_d = demands[support_size:]

        desc = descriptions.get(code, "")
        cat = assign_category(desc)

        # Use only prices available in the support slice for tier assignment.
        stats = norm_stats.get(code, {})
        raw_avg_price = stats.get("p_min", 0) + stats.get("p_range", 1) * float(np.mean(support_p))
        tier = assign_price_tier(raw_avg_price, budget_max, midrange_max)
        elast = assign_elasticity(support_p, support_d, high_thresh, low_thresh)

        tasks.append(
            Task(
                stock_code=code,
                description=desc,
                support_prices=support_p,
                support_demands=support_d,
                query_prices=query_p,
                query_demands=query_d,
                category=cat,
                price_tier=tier,
                elasticity=elast,
                norm_stats=stats if stats else None,
            )
        )

    return tasks


def split_tasks(
    tasks: List[Task],
    n_train: int = 100,
    n_val: int = 25,
    n_test: int = 25,
    seed: int = 42,
) -> Tuple[TaskDataset, TaskDataset, TaskDataset]:
    """Randomly split tasks into train / val / test sets."""
    if len(tasks) < 3:
        raise ValueError(
            "At least three product tasks are required for nonempty train/val/test splits."
        )
    for count in (n_train, n_val, n_test):
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("Split sizes must be positive integers.")
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(tasks))

    total_needed = n_train + n_val + n_test
    if len(tasks) < total_needed:
        # Scale proportionally
        ratio_tr = n_train / total_needed
        ratio_val = n_val / total_needed
        n_train = min(len(tasks) - 2, max(1, int(len(tasks) * ratio_tr)))
        n_val = min(len(tasks) - n_train - 1, max(1, int(len(tasks) * ratio_val)))
        n_test = len(tasks) - n_train - n_val
        print(
            f"Not enough tasks ({len(tasks)}). Using train={n_train}, val={n_val}, test={n_test}."
        )

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val : n_train + n_val + n_test]

    train_ds = TaskDataset([tasks[i] for i in train_idx])
    val_ds = TaskDataset([tasks[i] for i in val_idx])
    test_ds = TaskDataset([tasks[i] for i in test_idx])

    print(f"Split: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    return train_ds, val_ds, test_ds


def get_descriptions(df: pd.DataFrame) -> Dict[str, str]:
    """Extract StockCode -> Description mapping from the cleaned dataframe."""
    if "Description" not in df.columns:
        return {}
    sub = df[["StockCode", "Description"]].drop_duplicates("StockCode")
    return dict(zip(sub["StockCode"].astype(str), sub["Description"].fillna(""), strict=True))
