from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import torch

from metaprice.baselines import (
    finetune_multitask,
    finetune_transfer,
    train_multitask,
    train_standard_maml,
    train_transfer_learning,
)
from metaprice.config import load_config
from metaprice.evaluate import (
    evaluate_baseline_maml,
    evaluate_finetuned_baseline,
    evaluate_metaprice,
    print_results_table,
)
from metaprice.inference import recommend_price
from metaprice.models import DemandMLP
from metaprice.plots import save_comparison_plots
from metaprice.preprocessing import build_product_data, load_and_clean
from metaprice.seed import set_seed
from metaprice.task_builder import build_tasks, get_descriptions, split_tasks
from metaprice.trainer import (
    load_models,
    save_models,
    train_all_hierarchy_learners,
    train_attention_network,
)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes() -> Dict[str, str]:
    root = Path(__file__).resolve().parent
    files = [root / "main.py", *sorted((root / "metaprice").glob("*.py"))]
    return {str(path.relative_to(root)): sha256_file(str(path)) for path in files}


def runtime_metadata() -> Dict[str, str | None]:
    """Capture the runtime needed to interpret a recorded experiment."""
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None

    return {
        "git_revision": revision,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()).lower(),
    }


def checkpoint_filenames(checkpoint_manifest: Dict[str, Any]) -> list[str]:
    return [
        filename
        for groups in checkpoint_manifest["learners"].values()
        for filename in groups.values()
    ] + ["attention_composer.pt", "transfer_learning.pt", "multitask.pt", "standard_maml.pt"]


def save_run_manifest(
    output_dir: str,
    data_path: str,
    cfg: Dict[str, Any],
    train_ds,
    val_ds,
    test_ds,
    checkpoint_files: list[str] | None = None,
) -> str:
    """Record the exact input and disjoint product assignment for this run."""
    splits = {
        name: [task.stock_code for task in dataset.tasks]
        for name, dataset in (("train", train_ds), ("validation", val_ds), ("test", test_ds))
    }
    codes = [code for split in splits.values() for code in split]
    if len(codes) != len(set(codes)):
        raise ValueError("A product appears in more than one split.")
    os.makedirs(output_dir, exist_ok=True)
    path = Path(output_dir) / "run-manifest.json"
    manifest = {
        "dataset_sha256": sha256_file(data_path),
        "dataset_filename": Path(data_path).name,
        "evaluation_protocol": (
            "product-disjoint, full-curve interpolation and normalization "
            "before support/query split"
        ),
        "source_sha256": source_hashes(),
        "runtime": runtime_metadata(),
        "configuration": cfg,
        "splits": splits,
        "checkpoint_sha256": {
            filename: sha256_file(str(Path(output_dir) / filename))
            for filename in (checkpoint_files or [])
        },
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path)


def evaluate_all_models(test_ds, learners, composer, tl_model, mt_model, maml_model, cfg, device):
    """Apply the same held-out evaluation to trained or reloaded models."""
    bl_cfg = cfg.get("baselines", {})
    tl_cfg = bl_cfg.get("transfer", {})
    tl_results = evaluate_finetuned_baseline(
        test_ds,
        tl_model,
        lambda m, t, device=device: finetune_transfer(
            m, t, tl_cfg.get("finetune_steps", 10), tl_cfg.get("finetune_lr", 1e-3), device
        ),
        device,
        "Transfer Learning",
    )
    mt_cfg = bl_cfg.get("multitask", {})
    mt_results = evaluate_finetuned_baseline(
        test_ds,
        mt_model,
        lambda m, t, device=device: finetune_multitask(
            m, t, mt_cfg.get("finetune_steps", 5), mt_cfg.get("finetune_lr", 0.01), device
        ),
        device,
        "Multi-Task Learning",
    )
    maml_cfg = bl_cfg.get("maml", {})
    maml_results = evaluate_baseline_maml(
        test_ds,
        maml_model,
        maml_cfg.get("inner_steps", 5),
        maml_cfg.get("inner_lr", 0.01),
        device,
        "Standard MAML",
    )
    mp_results = evaluate_metaprice(test_ds, learners, composer, cfg, device)
    return [tl_results, mt_results, maml_results, mp_results]


def load_baseline(output_dir: str, filename: str, cfg: Dict[str, Any], device: str) -> DemandMLP:
    model_cfg = cfg.get("model", {})
    model = DemandMLP(model_cfg.get("hidden_dim", 64), model_cfg.get("num_hidden", 2)).to(device)
    model.load_state_dict(
        torch.load(Path(output_dir) / filename, map_location=device, weights_only=True)
    )
    model.eval()
    return model


def run_train(cfg: Dict[str, Any], data_path: str) -> None:
    device = cfg.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

    print("Loading and cleaning data...")
    df = load_and_clean(data_path)

    print("Building product data...")
    data_cfg = cfg.get("data", {})
    product_data, norm_stats = build_product_data(
        df,
        min_pairs=data_cfg.get("min_pairs", 3),
        num_points=data_cfg.get("num_points", 50),
    )

    if len(product_data) == 0:
        print("ERROR: No products with enough data. Cannot proceed.")
        sys.exit(1)

    descriptions = get_descriptions(df)

    print("Building tasks...")
    tasks = build_tasks(
        product_data, descriptions, data_cfg.get("support_size", 5), cfg, norm_stats
    )

    splits_cfg = cfg.get("splits", {})
    train_ds, val_ds, test_ds = split_tasks(
        tasks,
        n_train=splits_cfg.get("train", 100),
        n_val=splits_cfg.get("val", 25),
        n_test=splits_cfg.get("test", 25),
        seed=cfg.get("seed", 42),
    )

    output_dir = cfg.get("output_dir", "outputs")

    # Stage 1: Train hierarchy-specific meta-learners
    print("\n=== Stage 1: Training hierarchy-specific meta-learners ===")
    learners = train_all_hierarchy_learners(train_ds, val_ds, cfg, device)

    # Stage 2: Train attention composition network
    print("\n=== Stage 2: Training attention network ===")
    composer = train_attention_network(learners, val_ds, cfg, device)

    # Train baselines
    print("\n=== Training baselines ===")
    print("Training Transfer Learning baseline...")
    tl_model = train_transfer_learning(train_ds, cfg, device)

    print("Training Multi-Task Learning baseline...")
    mt_model = train_multitask(train_ds, cfg, device)

    print("Training Standard MAML baseline...")
    maml_model = train_standard_maml(train_ds, val_ds, cfg, device)

    # Save models
    save_models(learners, composer, output_dir)
    torch.save(tl_model.state_dict(), os.path.join(output_dir, "transfer_learning.pt"))
    torch.save(mt_model.state_dict(), os.path.join(output_dir, "multitask.pt"))
    torch.save(maml_model.state_dict(), os.path.join(output_dir, "standard_maml.pt"))
    checkpoint_manifest = json.loads(
        (Path(output_dir) / "checkpoint-manifest.json").read_text(encoding="utf-8")
    )
    checkpoint_files = checkpoint_filenames(checkpoint_manifest)
    manifest_path = save_run_manifest(
        output_dir, data_path, cfg, train_ds, val_ds, test_ds, checkpoint_files
    )
    print(f"Run manifest saved to {manifest_path}")

    # Evaluate
    print("\n=== Evaluating on test set ===")
    all_results = evaluate_all_models(
        test_ds, learners, composer, tl_model, mt_model, maml_model, cfg, device
    )
    print_results_table(all_results)

    # Save metrics
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # Save plots
    plot_dir = cfg.get("plot_dir", "plots")
    save_comparison_plots(all_results, plot_dir)


def load_recorded_test_set(cfg: Dict[str, Any], data_path: str):
    """Verify a saved run and reconstruct its recorded test tasks."""
    output_dir = cfg.get("output_dir", "outputs")
    manifest = json.loads((Path(output_dir) / "run-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset_sha256") != sha256_file(data_path):
        raise ValueError("Dataset checksum differs from the training run.")
    if manifest.get("configuration") != cfg:
        raise ValueError("Configuration differs from the training run.")
    if manifest.get("source_sha256") != source_hashes():
        raise ValueError(
            "Source code differs from the training run; retrain before evaluation or inference."
        )

    df = load_and_clean(data_path)
    data_cfg = cfg.get("data", {})
    product_data, norm_stats = build_product_data(
        df, data_cfg.get("min_pairs", 3), data_cfg.get("num_points", 50)
    )
    tasks = build_tasks(
        product_data, get_descriptions(df), data_cfg.get("support_size", 5), cfg, norm_stats
    )
    split_cfg = cfg.get("splits", {})
    train_ds, val_ds, test_ds = split_tasks(
        tasks,
        split_cfg.get("train", 100),
        split_cfg.get("val", 25),
        split_cfg.get("test", 25),
        cfg.get("seed", 42),
    )
    actual = {
        name: [task.stock_code for task in dataset.tasks]
        for name, dataset in (("train", train_ds), ("validation", val_ds), ("test", test_ds))
    }
    if manifest.get("splits") != actual:
        raise ValueError("Product split differs from the training run.")
    hashes = manifest.get("checkpoint_sha256")
    if not hashes:
        raise ValueError("Run manifest has no checkpoint hashes.")
    checkpoint_manifest = json.loads(
        (Path(output_dir) / "checkpoint-manifest.json").read_text(encoding="utf-8")
    )
    if set(hashes) != set(checkpoint_filenames(checkpoint_manifest)):
        raise ValueError("Checkpoint manifest does not match the recorded run.")
    for filename, expected in hashes.items():
        if Path(filename).name != filename or not filename.endswith(".pt"):
            raise ValueError("Invalid checkpoint filename in run manifest.")
        if sha256_file(str(Path(output_dir) / filename)) != expected:
            raise ValueError(f"Checkpoint checksum differs: {filename}")

    return test_ds


def run_eval(cfg: Dict[str, Any], data_path: str) -> None:
    """Evaluate saved weights on the exact recorded dataset and split."""
    device = cfg.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but unavailable; use the training device or update the config."
        )
    test_ds = load_recorded_test_set(cfg, data_path)
    output_dir = cfg.get("output_dir", "outputs")
    learners, composer = load_models(output_dir, cfg, device)
    tl_model = load_baseline(output_dir, "transfer_learning.pt", cfg, device)
    mt_model = load_baseline(output_dir, "multitask.pt", cfg, device)
    maml_model = load_baseline(output_dir, "standard_maml.pt", cfg, device)
    results = evaluate_all_models(
        test_ds, learners, composer, tl_model, mt_model, maml_model, cfg, device
    )
    print_results_table(results)
    path = Path(output_dir) / "metrics-eval.json"
    path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Checkpoint evaluation saved to {path}")


def run_infer(cfg: Dict[str, Any], data_path: str) -> None:
    """Recommend prices for the recorded held-out tasks using saved weights."""
    device = cfg.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but unavailable; use the training device or update the config."
        )
    test_ds = load_recorded_test_set(cfg, data_path)
    learners, composer = load_models(cfg.get("output_dir", "outputs"), cfg, device)

    print("\n=== Price Recommendations ===")
    for task in test_ds.tasks[:5]:
        best_price, best_rev, _, _ = recommend_price(task, learners, composer, cfg, device=device)
        print(
            f"Product {task.stock_code} ({task.category}, {task.price_tier}): "
            f"Recommended price = {best_price:.2f}, Predicted revenue = {best_rev:.2f}"
        )


def apply_run_overrides(
    cfg: Dict[str, Any], *, seed: int | None, output_dir: str | None, plot_dir: str | None
) -> Dict[str, Any]:
    """Apply CLI-only run locations and seed values without modifying the source config."""
    resolved = dict(cfg)
    if seed is not None:
        resolved["seed"] = seed
    if output_dir is not None:
        resolved["output_dir"] = output_dir
    if plot_dir is not None:
        resolved["plot_dir"] = plot_dir
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MetaPrice: Hierarchical Meta-Learning for Dynamic Pricing"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Path to YAML config file"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to raw dataset (CSV or XLSX)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval", "infer"],
        help="Run mode: train, eval saved checkpoints, or infer from saved checkpoints",
    )
    parser.add_argument("--seed", type=int, help="Override the configuration seed for this run")
    parser.add_argument(
        "--output-dir", help="Override the configuration output directory for this run"
    )
    parser.add_argument("--plot-dir", help="Override the configuration plot directory for this run")
    args = parser.parse_args()

    cfg = apply_run_overrides(
        load_config(args.config),
        seed=args.seed,
        output_dir=args.output_dir,
        plot_dir=args.plot_dir,
    )
    set_seed(cfg.get("seed", 42))

    if args.mode == "train":
        run_train(cfg, args.data_path)
    elif args.mode == "eval":
        run_eval(cfg, args.data_path)
    elif args.mode == "infer":
        run_infer(cfg, args.data_path)


if __name__ == "__main__":
    main()
