"""Render the checked-in synthetic RMSE comparison from measured metrics."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "results"


def main() -> None:
    rows = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(9.8, 3.8))
    names = [row["model"] for row in rows]
    values = [row["rmse_mean"] for row in rows]
    bars = ax.barh(names, values, color="#356e73")
    ax.invert_yaxis()
    ax.set_xlabel("Mean query RMSE · lower is better")
    ax.set_xlim(0, max(values) * 1.2)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            value + max(values) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
        )
    fig.tight_layout()
    fig.savefig(RESULTS / "comparison.png", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()
