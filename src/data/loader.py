"""
src/data/loader.py
------------------
Data loading utilities for the M5 Retail Forecasting project.
Loads raw CSV files from data/raw/ and returns validated DataFrames.
"""

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")


def load_sales_train_validation() -> pd.DataFrame:
    """Load sales_train_validation.csv. Expected: 30,490 rows x 1,919 cols."""
    path = RAW_DIR / "sales_train_validation.csv"
    logger.info(f"Loading {path}")
    df = pd.read_csv(path)
    assert df.shape == (
        30490,
        1919,
    ), f"Unexpected shape {df.shape}; expected (30490, 1919)"
    return df


def load_sales_train_evaluation() -> pd.DataFrame:
    """Load sales_train_evaluation.csv. Expected: 30,490 rows x 1,947 cols."""
    path = RAW_DIR / "sales_train_evaluation.csv"
    logger.info(f"Loading {path}")
    df = pd.read_csv(path)
    assert df.shape == (
        30490,
        1947,
    ), f"Unexpected shape {df.shape}; expected (30490, 1947)"
    return df


def load_calendar() -> pd.DataFrame:
    """Load calendar.csv. Expected: 1,969 rows x 14 cols."""
    path = RAW_DIR / "calendar.csv"
    logger.info(f"Loading {path}")
    df = pd.read_csv(path)
    assert df.shape == (1969, 14), f"Unexpected shape {df.shape}; expected (1969, 14)"
    return df


def load_sell_prices() -> pd.DataFrame:
    """Load sell_prices.csv. Expected: ~6.8M rows x 4 cols."""
    path = RAW_DIR / "sell_prices.csv"
    logger.info(f"Loading {path}")
    df = pd.read_csv(path)
    assert df.shape[1] == 4, f"Unexpected column count {df.shape[1]}; expected 4"
    return df


def load_all_raw() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience loader returning (sales_val, sales_eval, calendar, prices)."""
    return (
        load_sales_train_validation(),
        load_sales_train_evaluation(),
        load_calendar(),
        load_sell_prices(),
    )
