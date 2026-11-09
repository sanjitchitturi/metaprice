"""Run the synthetic protocol across explicit seeds and aggregate raw run metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError(
            "Provide one or more unique comma-separated integer seeds."
        )
    return seeds


def summarise(runs: list[dict[str, Any]]) -> dict[str, Any]:
    methods = {result["model"] for run in runs for result in run["metrics"]}
    summary: dict[str, Any] = {}
    for method in sorted(methods):
        values = [
            result["rmse_mean"]
            for run in runs
            for result in run["metrics"]
            if result["model"] == method
        ]
        summary[method] = {
            "runs": len(values),
            "mean_rmse": float(np.mean(values)),
            "sample_std_rmse": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="examples/synthetic/config.yaml")
    parser.add_argument("--data-path", default="examples/synthetic/transactions.csv")
    parser.add_argument("--seeds", type=parse_seeds, default=[17, 29, 43])
    parser.add_argument("--output-dir", default="outputs/synthetic-benchmark")
    args = parser.parse_args()

    config = (ROOT / args.config).resolve()
    data_path = (ROOT / args.data_path).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []

    for seed in args.seeds:
        run_dir = output_dir / f"seed-{seed}"
        subprocess.run(
            [
                sys.executable,
                "main.py",
                "--config",
                str(config),
                "--data_path",
                str(data_path),
                "--mode",
                "train",
                "--seed",
                str(seed),
                "--output-dir",
                str(run_dir),
                "--plot-dir",
                str(run_dir / "plots"),
            ],
            cwd=ROOT,
            check=True,
        )
        runs.append(
            {
                "seed": seed,
                "output_dir": run_dir.name,
                "metrics": json.loads((run_dir / "metrics.json").read_text(encoding="utf-8")),
            }
        )

    report = {
        "protocol": "synthetic product-holdout; not a real-world pricing evaluation",
        "config_sha256": sha256_file(config),
        "data_sha256": sha256_file(data_path),
        "seeds": args.seeds,
        "runs": runs,
        "aggregate": summarise(runs),
    }
    (output_dir / "benchmark-metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
