import hashlib
import json
from types import SimpleNamespace

import pytest

from main import apply_run_overrides, run_eval, run_infer, save_run_manifest


def _dataset(*codes):
    return SimpleNamespace(tasks=[SimpleNamespace(stock_code=code) for code in codes])


def test_run_manifest_records_input_and_product_splits(tmp_path):
    source = tmp_path / "transactions.csv"
    source.write_bytes(b"stock,price\nA,2\n")
    output = tmp_path / "out"
    path = save_run_manifest(
        str(output),
        str(source),
        {"seed": 42},
        _dataset("A"),
        _dataset("B"),
        _dataset("C"),
    )
    data = json.loads((output / "run-manifest.json").read_text())
    assert path == str(output / "run-manifest.json")
    assert data["dataset_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert data["splits"] == {"train": ["A"], "validation": ["B"], "test": ["C"]}
    assert "main.py" in data["source_sha256"]
    assert "metaprice/task_builder.py" in data["source_sha256"]
    assert {"python", "platform", "torch", "cuda_available", "git_revision"} <= set(data["runtime"])


def test_run_manifest_rejects_product_overlap(tmp_path):
    source = tmp_path / "data.csv"
    source.write_bytes(b"data")
    with pytest.raises(ValueError, match="more than one split"):
        save_run_manifest(
            str(tmp_path), str(source), {}, _dataset("A"), _dataset("A"), _dataset("B")
        )
    assert not (tmp_path / "run-manifest.json").exists()


def test_checkpoint_eval_rejects_changed_input_before_loading_weights(tmp_path):
    source = tmp_path / "data.csv"
    source.write_bytes(b"original")
    cfg = {"output_dir": str(tmp_path / "out"), "device": "cpu"}
    save_run_manifest(
        cfg["output_dir"], str(source), cfg, _dataset("A"), _dataset("B"), _dataset("C")
    )
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        run_eval(cfg, str(source))


def test_checkpoint_inference_rejects_changed_input_before_loading_weights(tmp_path):
    source = tmp_path / "data.csv"
    source.write_bytes(b"original")
    cfg = {"output_dir": str(tmp_path / "out"), "device": "cpu"}
    save_run_manifest(
        cfg["output_dir"], str(source), cfg, _dataset("A"), _dataset("B"), _dataset("C")
    )
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        run_infer(cfg, str(source))


def test_checkpoint_eval_rejects_changed_source(tmp_path):
    source = tmp_path / "data.csv"
    source.write_bytes(b"original")
    cfg = {"output_dir": str(tmp_path / "out"), "device": "cpu"}
    save_run_manifest(
        cfg["output_dir"], str(source), cfg, _dataset("A"), _dataset("B"), _dataset("C")
    )
    manifest = json.loads((tmp_path / "out" / "run-manifest.json").read_text())
    manifest["source_sha256"]["main.py"] = "changed"
    (tmp_path / "out" / "run-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Source code differs"):
        run_eval(cfg, str(source))


def test_run_overrides_do_not_mutate_loaded_configuration():
    original = {"seed": 42, "output_dir": "outputs", "plot_dir": "plots"}
    resolved = apply_run_overrides(
        original, seed=9, output_dir="outputs/seed-9", plot_dir="plots/seed-9"
    )
    assert original == {"seed": 42, "output_dir": "outputs", "plot_dir": "plots"}
    assert resolved == {"seed": 9, "output_dir": "outputs/seed-9", "plot_dir": "plots/seed-9"}
