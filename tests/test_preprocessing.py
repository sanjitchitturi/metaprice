import pandas as pd
import pytest

from metaprice.config import load_config
from metaprice.preprocessing import interpolate_to_n, load_and_clean, normalize_product


def test_cleaning_aliases_cancellations_and_invalid_rows(tmp_path):
    path = tmp_path / "transactions.csv"
    pd.DataFrame(
        {
            "InvoiceNo": ["1", "C2", "3", "4"],
            "StockCode": ["A"] * 4,
            "Quantity": [2, 3, 0, 1],
            "UnitPrice": [5, 6, 7, -1],
        }
    ).to_csv(path, index=False)
    result = load_and_clean(str(path))
    assert len(result) == 1
    assert result.iloc[0]["Price"] == 5


def test_missing_columns_are_rejected(tmp_path):
    path = tmp_path / "invalid.csv"
    pd.DataFrame({"Quantity": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing required columns"):
        load_and_clean(str(path))


def test_xlsx_input(tmp_path):
    path = tmp_path / "transactions.xlsx"
    pd.DataFrame({"StockCode": ["A"], "Quantity": [2], "Price": [3]}).to_excel(path, index=False)
    assert len(load_and_clean(str(path))) == 1


def test_interpolation_preserves_endpoints_and_normalization_is_finite():
    source = pd.DataFrame({"price": [1.0, 2.0, 3.0], "demand": [10.0, 8.0, 6.0]})
    curve = interpolate_to_n(source, 5)
    assert len(curve) == 5
    assert curve.iloc[0].tolist() == [1.0, 10.0]
    assert curve.iloc[-1].tolist() == [3.0, 6.0]
    normalized, stats = normalize_product(curve)
    assert normalized.min().tolist() == [0.0, 0.0]
    assert normalized.max().tolist() == [1.0, 1.0]
    assert stats["p_range"] == 2.0


def test_default_config_loads():
    assert load_config("configs/default.yaml")["device"] == "cpu"
