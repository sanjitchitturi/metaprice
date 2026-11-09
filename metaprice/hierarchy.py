"""Hierarchy assignment: category, price tier, elasticity."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

# Keyword rules for category inference from product descriptions
_CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "Electronics": [
        "led",
        "lamp",
        "light",
        "bulb",
        "torch",
        "battery",
        "usb",
        "phone",
        "clock",
        "radio",
        "speaker",
        "charger",
        "cable",
        "computer",
        "laptop",
        "camera",
        "electronic",
    ],
    "Storage": [
        "box",
        "tin",
        "jar",
        "basket",
        "bag",
        "case",
        "drawer",
        "container",
        "storage",
        "chest",
        "trunk",
        "rack",
        "shelf",
        "holder",
        "organiser",
        "organizer",
    ],
    "Stationery": [
        "pen",
        "pencil",
        "notebook",
        "paper",
        "card",
        "letter",
        "envelope",
        "sticker",
        "stamp",
        "tape",
        "ribbon",
        "wrap",
        "tag",
        "label",
        "diary",
        "journal",
        "pad",
        "chalk",
    ],
    "Decorations": [
        "candle",
        "flower",
        "vase",
        "garland",
        "bunting",
        "star",
        "heart",
        "angel",
        "fairy",
        "christmas",
        "xmas",
        "tree",
        "wreath",
        "ornament",
        "decoration",
        "hanging",
        "frame",
        "mirror",
        "cushion",
        "plaque",
    ],
}


def assign_category(description: str) -> str:
    """Assign a product category based on keyword matching in description."""
    desc_lower = description.lower() if isinstance(description, str) else ""
    best_cat = "General"
    best_count = 0
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in desc_lower)
        if count > best_count:
            best_count = count
            best_cat = cat
    return best_cat


def assign_price_tier(avg_price: float, budget_max: float = 2.0, midrange_max: float = 5.0) -> str:
    """Assign price tier based on average product price."""
    if avg_price < budget_max:
        return "Budget"
    elif avg_price <= midrange_max:
        return "Mid-range"
    else:
        return "Premium"


def assign_elasticity(
    support_prices: np.ndarray,
    support_demands: np.ndarray,
    high_threshold: float = -0.5,
    low_threshold: float = -0.2,
) -> str:
    """Assign elasticity cluster from support set only (no query leakage).

    Uses Pearson correlation between price and demand as a proxy for elasticity.
    """
    if len(support_prices) < 3:
        return "Moderately Elastic"

    std_p = np.std(support_prices)
    std_d = np.std(support_demands)
    if std_p < 1e-9 or std_d < 1e-9:
        return "Inelastic"

    corr = float(np.corrcoef(support_prices, support_demands)[0, 1])

    if corr <= high_threshold:
        return "Highly Elastic"
    elif corr <= low_threshold:
        return "Moderately Elastic"
    else:
        return "Inelastic"
