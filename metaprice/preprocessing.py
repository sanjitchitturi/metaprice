"""Raw data cleaning, aggregation, and interpolation."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

# Common column name variants
_COL_MAP = {
    "invoice": "Invoice",
    "invoiceno": "Invoice",
    "invoice_no": "Invoice",
    "stockcode": "StockCode",
    "stock_code": "StockCode",
    "description": "Description",
    "quantity": "Quantity",
    "invoicedate": "InvoiceDate",
    "invoice_date": "InvoiceDate",
    "price": "Price",
    "unitprice": "Price",
    "unit_price": "Price",
    "customerid": "CustomerID",
    "customer_id": "CustomerID",
    "customer id": "CustomerID",
    "country": "Country",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to a canonical set."""
    rename = {}
    for col in df.columns:
        key = col.strip().lower().replace(" ", "")
        if key in _COL_MAP:
            rename[col] = _COL_MAP[key]
    df = df.rename(columns=rename)
    return df


def load_and_clean(data_path: str) -> pd.DataFrame:
    """Load raw data from CSV or XLSX and perform cleaning."""
    if data_path.endswith(".xlsx") or data_path.endswith(".xls"):
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path, encoding="latin-1")

    df = _normalize_columns(df)

    required = {"StockCode", "Quantity", "Price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}")

    for column in ("Quantity", "Price"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    initial_len = len(df)
    df = df[np.isfinite(df["Quantity"]) & np.isfinite(df["Price"])]

    # Remove cancelled invoices (start with C)
    if "Invoice" in df.columns:
        df["Invoice"] = df["Invoice"].astype(str)
        df = df[~df["Invoice"].str.startswith("C")]

    # Remove rows with non-positive quantity or price
    df = df[df["Quantity"] > 0]
    df = df[df["Price"] > 0]

    # Drop rows missing StockCode
    df = df.dropna(subset=["StockCode"])

    # Drop duplicates
    df = df.drop_duplicates()

    print(f"Preprocessing: {initial_len} -> {len(df)} rows after cleaning.")
    return df


def aggregate_price_demand(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Aggregate transactions into price-demand pairs per product.

    Returns dict mapping StockCode -> DataFrame with columns [price, demand].
    """
    products: Dict[str, pd.DataFrame] = {}
    grouped = df.groupby("StockCode")

    for code, group in grouped:
        agg = group.groupby("Price")["Quantity"].mean().reset_index()
        agg.columns = ["price", "demand"]
        agg = agg.sort_values("price").reset_index(drop=True)
        products[str(code)] = agg

    return products


def interpolate_to_n(pdf: pd.DataFrame, n: int = 50) -> pd.DataFrame:
    """If product has fewer than n price-demand pairs, interpolate to n.
    If n or more, take first n after sorting."""
    if not isinstance(n, int) or n < 2 or pdf.empty:
        raise ValueError("Interpolation requires data and at least two output points.")
    if len(pdf) >= n:
        return pdf.iloc[:n].reset_index(drop=True)

    prices = pdf["price"].values
    demands = pdf["demand"].values

    new_prices = np.linspace(prices.min(), prices.max(), n)
    new_demands = np.interp(new_prices, prices, demands)
    # Ensure non-negative demand
    new_demands = np.maximum(new_demands, 0.0)

    return pd.DataFrame({"price": new_prices, "demand": new_demands})


def normalize_product(pdf: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Min-max normalize price to [0,1] and demand to [0,1] per product.

    Returns normalized DataFrame and stats dict for inverse transform.
    """
    p_min, p_max = float(pdf["price"].min()), float(pdf["price"].max())
    d_min, d_max = float(pdf["demand"].min()), float(pdf["demand"].max())

    p_range = p_max - p_min if p_max > p_min else 1.0
    d_range = d_max - d_min if d_max > d_min else 1.0

    normed = pd.DataFrame(
        {
            "price": (pdf["price"] - p_min) / p_range,
            "demand": (pdf["demand"] - d_min) / d_range,
        }
    )

    stats = {
        "p_min": p_min,
        "p_max": p_max,
        "p_range": p_range,
        "d_min": d_min,
        "d_max": d_max,
        "d_range": d_range,
    }
    return normed, stats


def build_product_data(
    df: pd.DataFrame, min_pairs: int = 3, num_points: int = 50
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, float]]]:
    """Full pipeline: aggregate, filter/interpolate, normalize.

    Returns:
        products: dict of StockCode -> normalized DataFrame[price, demand]
        norm_stats: dict of StockCode -> normalization stats for inverse transform
    """
    if not isinstance(min_pairs, int) or min_pairs < 3:
        raise ValueError("min_pairs must be an integer of at least three.")
    products = aggregate_price_demand(df)

    filtered: Dict[str, pd.DataFrame] = {}
    norm_stats: Dict[str, Dict[str, float]] = {}
    for code, pdf in products.items():
        if len(pdf) < min_pairs:
            continue
        interpolated = interpolate_to_n(pdf, num_points)
        normed, stats = normalize_product(interpolated)
        filtered[code] = normed
        norm_stats[code] = stats

    print(
        f"Products with >= {min_pairs} price points (interpolated to {num_points}): {len(filtered)}"
    )

    if len(filtered) < 10:
        print(
            f"WARNING: Only {len(filtered)} products available. "
            "Results may not be meaningful. Consider a larger dataset."
        )

    return filtered, norm_stats
