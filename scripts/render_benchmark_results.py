"""Render a transparent aggregate view from benchmark-metrics.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", default="outputs/synthetic-benchmark/benchmark-metrics.json")
    parser.add_argument("--output", default="outputs/synthetic-benchmark/rmse-summary.png")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    aggregate = report["aggregate"]
    names = list(aggregate)
    means = [aggregate[name]["mean_rmse"] for name in names]
    errors = [aggregate[name]["sample_std_rmse"] or 0.0 for name in names]

    figure, axis = plt.subplots(figsize=(9, 5))
    bars = axis.bar(names, means, yerr=errors, capsize=5, color="#2b6f89", edgecolor="#173f4d")
    axis.set_ylabel("Mean held-out RMSE across seeds")
    axis.set_title("Synthetic product-holdout benchmark")
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.tick_params(axis="x", rotation=15)
    for bar, value in zip(bars, means, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom"
        )
    figure.text(
        0.5,
        0.01,
        report["protocol"],
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)


if __name__ == "__main__":
    main()
