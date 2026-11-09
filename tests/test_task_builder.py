import pandas as pd

from metaprice.task_builder import build_tasks


def test_price_tier_uses_support_prices_not_query_prices():
    raw_prices = [1.0, 1.25, 1.5, 1.75, 1.9, 9.0, 10.0]
    curve = pd.DataFrame(
        {
            "price": [(price - 1.0) / 9.0 for price in raw_prices],
            "demand": [1.0, 0.9, 0.8, 0.7, 0.6, 0.2, 0.1],
        }
    )
    stats = {"p_min": 1.0, "p_range": 9.0}

    task = build_tasks({"P1": curve}, {"P1": "Paper"}, 5, norm_stats={"P1": stats})[0]

    assert task.price_tier == "Budget"
