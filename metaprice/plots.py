"""Save comparison plots for evaluation results."""

from __future__ import annotations

import os
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_comparison_plots(
    results: List[Dict[str, Any]],
    plot_dir: str = "plots",
) -> None:
    """Save bar charts comparing all models on RMSE, Revenue Ratio, and Latency."""
    os.makedirs(plot_dir, exist_ok=True)

    models = [r["model"] for r in results]
    x = np.arange(len(models))

    # RMSE comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    means = [r["rmse_mean"] for r in results]
    stds = [r["rmse_std"] for r in results]
    ax.bar(x, means, yerr=stds, capsize=5, color=["#4c72b0", "#55a868", "#c44e52", "#8172b2"])
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("RMSE")
    ax.set_title("Demand Forecasting Accuracy (RMSE)")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "rmse_comparison.png"), dpi=150)
    plt.close()

    # Revenue Ratio comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    means = [r["revenue_ratio_mean"] for r in results]
    stds = [r["revenue_ratio_std"] for r in results]
    ax.bar(x, means, yerr=stds, capsize=5, color=["#4c72b0", "#55a868", "#c44e52", "#8172b2"])
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Revenue Ratio")
    ax.set_title("Revenue Performance (Revenue Ratio)")
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "revenue_ratio_comparison.png"), dpi=150)
    plt.close()

    # Latency comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    means = [r["latency_mean_ms"] for r in results]
    stds = [r["latency_std_ms"] for r in results]
    ax.bar(x, means, yerr=stds, capsize=5, color=["#4c72b0", "#55a868", "#c44e52", "#8172b2"])
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Adaptation Latency (ms)")
    ax.set_title("Computational Efficiency")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "latency_comparison.png"), dpi=150)
    plt.close()

    print(f"Plots saved to {plot_dir}/")
