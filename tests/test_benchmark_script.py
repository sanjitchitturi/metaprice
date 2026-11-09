import importlib.util
from pathlib import Path


def _benchmark_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_synthetic_benchmark.py"
    spec = importlib.util.spec_from_file_location("synthetic_benchmark", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summarise_aggregates_the_evaluation_schema():
    benchmark = _benchmark_module()
    summary = benchmark.summarise(
        [
            {"metrics": [{"model": "MetaPrice", "rmse_mean": 2.0}]},
            {"metrics": [{"model": "MetaPrice", "rmse_mean": 4.0}]},
        ]
    )
    assert summary == {"MetaPrice": {"runs": 2, "mean_rmse": 3.0, "sample_std_rmse": 2**0.5}}


def test_parse_seeds_rejects_duplicate_values():
    benchmark = _benchmark_module()
    try:
        benchmark.parse_seeds("7,7")
    except Exception as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("Duplicate seeds must be rejected.")
