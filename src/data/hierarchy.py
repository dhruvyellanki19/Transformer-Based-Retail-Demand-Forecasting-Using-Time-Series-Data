"""
src/data/hierarchy.py
---------------------
M5 Dataset Hierarchical Structure — Phase 1, Step 1.3.

Documents all 12 M5 aggregation levels and provides helper functions
to build aggregated DataFrames at each level for EDA and WRMSSE computation.
"""

from dataclasses import dataclass
from typing import List

import pandas as pd

# ── Level definitions ─────────────────────────────────────────────────────────

LEVELS = {
    1: {"name": "Total", "group_cols": [], "n_series": 1},
    2: {"name": "State", "group_cols": ["state_id"], "n_series": 3},
    3: {"name": "Store", "group_cols": ["store_id"], "n_series": 10},
    4: {"name": "Category", "group_cols": ["cat_id"], "n_series": 3},
    5: {"name": "Department", "group_cols": ["dept_id"], "n_series": 7},
    6: {"name": "State x Category", "group_cols": ["state_id", "cat_id"], "n_series": 9},
    7: {"name": "State x Department", "group_cols": ["state_id", "dept_id"], "n_series": 21},
    8: {"name": "Store x Category", "group_cols": ["store_id", "cat_id"], "n_series": 30},
    9: {"name": "Store x Department", "group_cols": ["store_id", "dept_id"], "n_series": 70},
    10: {"name": "Product (item)", "group_cols": ["item_id"], "n_series": 3049},
    11: {"name": "State x Product", "group_cols": ["state_id", "item_id"], "n_series": 9147},
    12: {"name": "Store x Product", "group_cols": ["store_id", "item_id"], "n_series": 30490},
}

TOTAL_SERIES = 30490

# States, stores, categories, departments as per M5 spec
STATES = ["CA", "TX", "WI"]
STORES = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"]
CATEGORIES = ["FOODS", "HOBBIES", "HOUSEHOLD"]
DEPARTMENTS = ["FOODS_1", "FOODS_2", "FOODS_3", "HOBBIES_1", "HOBBIES_2", "HOUSEHOLD_1", "HOUSEHOLD_2"]

# Evaluation uses sales_train_evaluation (extends to d_1941)
# Validation window: d_1886–d_1913 (28 days)
# Test window:       d_1914–d_1941 (28 days)
TRAIN_END_DATE = "2016-03-27"
VAL_START_DATE = "2016-03-28"
VAL_END_DATE = "2016-04-24"
TEST_START_DATE = "2016-04-25"
TEST_END_DATE = "2016-05-22"


def print_hierarchy() -> None:
    """Print all 12 aggregation levels to stdout."""
    print("\nM5 Dataset Hierarchy")
    print("=" * 60)
    for level, info in LEVELS.items():
        print(f"  Level {level:2d}: {info['name']:<25} ({info['n_series']:>6,} series)")
    print("=" * 60)
    print(f"  TOTAL bottom-level series: {TOTAL_SERIES:,}")
    print()


def aggregate_level(long_df: pd.DataFrame, level: int) -> pd.DataFrame:
    """
    Aggregate a long-format sales DataFrame to the specified M5 level.

    Parameters
    ----------
    long_df : pd.DataFrame
        Long-format DataFrame with columns: date, sales, and all ID columns.
    level : int
        M5 aggregation level (1–12).

    Returns
    -------
    pd.DataFrame
        Aggregated daily sales summed to the requested level.
        Columns: [group_cols..., date, sales].
    """
    if level not in LEVELS:
        raise ValueError(f"Level must be 1–12, got {level}")

    group_cols = LEVELS[level]["group_cols"]

    if not group_cols:
        # Level 1: total
        agg = long_df.groupby("date", as_index=False)["sales"].sum()
        agg.insert(0, "level", "Total")
        return agg

    return long_df.groupby(group_cols + ["date"], as_index=False)["sales"].sum()


def get_level_info(level: int) -> dict:
    """Return metadata dict for a given level."""
    if level not in LEVELS:
        raise ValueError(f"Level must be 1–12, got {level}")
    return LEVELS[level]


if __name__ == "__main__":
    print_hierarchy()
