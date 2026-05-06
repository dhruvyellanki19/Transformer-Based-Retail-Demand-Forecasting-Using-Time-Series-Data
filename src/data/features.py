"""
src/data/features.py
---------------------
Feature engineering module — memory-efficient implementation.

Reads data/processed/sales_clean.parquet, computes features in batches
of series groups to stay within ~6 GB RAM, and writes:
    data/features/features_all.parquet

All feature functions are pure and independently testable.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

FORECAST_HORIZON = 28
FEATURES_DIR = Path("data/features")
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_MAPPING_PATH = Path("configs/feature_category_mappings.json")

LAG_DAYS = [7, 14, 21, 28, 35, 42, 56, 91, 182, 364]
STATIC_CATEGORICALS = ["cat_id", "dept_id", "store_id", "state_id"]
ROLLING_COLS = [
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_mean_56",
    "rolling_mean_91",
    "rolling_std_7",
    "rolling_std_28",
    "rolling_max_7",
    "rolling_min_7",
    "rolling_skew_28",
]


# ---------------------------------------------------------------------------
# Step 4.1 — Calendar Features (Cyclical Encoding)
# ---------------------------------------------------------------------------


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical calendar encodings and binary flags. Works on any-size DF."""
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek.astype("int8")
    df["day_of_month"] = df["date"].dt.day.astype("int8")
    df["day_of_year"] = df["date"].dt.dayofyear.astype("int16")
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype("int8")
    df["month"] = df["date"].dt.month.astype("int8")
    df["year"] = df["date"].dt.year.astype("int16")

    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7).astype("float32")
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7).astype("float32")
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12).astype("float32")
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12).astype("float32")
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365).astype("float32")
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365).astype("float32")

    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")
    df["is_month_start"] = (df["day_of_month"] == 1).astype("int8")
    df["is_month_end"] = (df["date"] == df["date"] + pd.offsets.MonthEnd(0)).astype("int8")

    for col in ["dow_sin", "dow_cos", "month_sin", "month_cos", "doy_sin", "doy_cos"]:
        assert df[col].between(-1, 1).all(), f"{col} has values outside [-1, 1]"

    return df


# ---------------------------------------------------------------------------
# Steps 4.2 + 4.3 — Lag + Rolling Features (per-series, memory safe)
# ---------------------------------------------------------------------------


def _compute_lag_rolling_for_series(grp: pd.DataFrame) -> pd.DataFrame:
    """Compute all lag + rolling features for a single series group."""
    sales = grp["sales"].values.astype("float32")

    for lag in LAG_DAYS:
        col = f"lag_{lag}"
        shifted = np.empty_like(sales)
        shifted[:lag] = np.nan
        shifted[lag:] = sales[:-lag]
        grp[col] = shifted.astype("float32")

    assert grp["lag_28"].iloc[:FORECAST_HORIZON].isna().all(), "lag_28 must be NaN for the first 28 rows of each series"

    # Shift by FORECAST_HORIZON before rolling (leakage prevention)
    shifted_sales = np.empty_like(sales)
    shifted_sales[:FORECAST_HORIZON] = np.nan
    shifted_sales[FORECAST_HORIZON:] = sales[:-FORECAST_HORIZON]
    padded = pd.Series(shifted_sales)

    rolling_configs = [
        ("rolling_mean_7", 7, "mean"),
        ("rolling_mean_14", 14, "mean"),
        ("rolling_mean_28", 28, "mean"),
        ("rolling_mean_56", 56, "mean"),
        ("rolling_mean_91", 91, "mean"),
        ("rolling_std_7", 7, "std"),
        ("rolling_std_28", 28, "std"),
        ("rolling_max_7", 7, "max"),
        ("rolling_min_7", 7, "min"),
        ("rolling_skew_28", 28, "skew"),
    ]
    for col, window, agg in rolling_configs:
        r = padded.rolling(window, min_periods=1)
        grp[col] = getattr(r, agg)().values.astype("float32")

    return grp


# ---------------------------------------------------------------------------
# Step 4.4 — Price Features
# ---------------------------------------------------------------------------


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add price change and relative price features."""
    df = df.copy()
    df = df.sort_values(["id", "date"])

    df["price_change_1w"] = df.groupby("id")["sell_price"].pct_change(1).astype("float32")
    df["price_change_4w"] = df.groupby("id")["sell_price"].pct_change(4).astype("float32")

    weekly_cat_mean = df.groupby(["cat_id", "wm_yr_wk"])["sell_price"].transform("mean")
    df["price_vs_category_mean"] = (df["sell_price"] / weekly_cat_mean.replace(0, np.nan)).astype("float32")

    weekly_store_mean = df.groupby(["store_id", "wm_yr_wk"])["sell_price"].transform("mean")
    df["price_vs_store_mean"] = (df["sell_price"] / weekly_store_mean.replace(0, np.nan)).astype("float32")

    df["price_momentum"] = (
        df.groupby("id")["price_change_1w"].transform(lambda x: x.rolling(4, min_periods=1).mean()).astype("float32")
    )
    df["is_price_reduced"] = (df["price_change_1w"] < -0.01).astype("int8")
    return df


# ---------------------------------------------------------------------------
# Step 4.5 — Static Metadata Encoding
# ---------------------------------------------------------------------------


def add_static_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """Integer-encode static categorical columns for TFT embedding layers."""
    df = df.copy()
    for col in STATIC_CATEGORICALS:
        df[col] = pd.Categorical(df[col]).codes.astype("int8")
    return df


def encode_static_metadata(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Encode static categoricals and return label mappings for reproducibility."""
    encoded = df.copy()
    mappings: dict[str, dict[str, int]] = {}
    for col in STATIC_CATEGORICALS:
        category = pd.Categorical(encoded[col])
        encoded[col] = category.codes.astype("int16")
        mappings[col] = {str(label): int(code) for code, label in enumerate(category.categories)}
    return encoded, mappings


def write_category_mappings(mappings: dict[str, dict[str, int]], output_path: str | Path = DEFAULT_MAPPING_PATH) -> Path:
    """Persist static categorical label mappings for downstream reproducibility."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mappings, indent=2, sort_keys=True))
    return path


# ---------------------------------------------------------------------------
# Step 4.6 — time_idx Column
# ---------------------------------------------------------------------------


def add_time_idx(df: pd.DataFrame) -> pd.DataFrame:
    """Add integer time index (days since first date = 2011-01-29)."""
    df = df.copy()
    start_date = pd.Timestamp("2011-01-29")
    df["time_idx"] = ((df["date"] - start_date).dt.days).astype("int16")
    assert df["time_idx"].between(0, 1940).all(), "time_idx out of range [0, 1940]"
    monotonic = df.sort_values(["id", "date"]).groupby("id")["time_idx"].apply(lambda s: s.is_monotonic_increasing)
    assert monotonic.all(), "time_idx must be monotonically increasing within each series"
    return df


def finalize_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Drop join-only columns, fill remaining feature nulls, and assert final feature hygiene."""
    final_df = df.copy()
    drop_cols = [col for col in ["d", "wm_yr_wk"] if col in final_df.columns]
    final_df = final_df.drop(columns=drop_cols)

    event_fill_unknown = ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]
    existing_event_cols = [col for col in event_fill_unknown if col in final_df.columns]
    for col in existing_event_cols:
        final_df[col] = final_df[col].fillna("None")

    lag_cols = [f"lag_{d}" for d in LAG_DAYS]
    feature_fill_zero = lag_cols + ROLLING_COLS + [
        "price_change_1w",
        "price_change_4w",
        "price_vs_category_mean",
        "price_vs_store_mean",
        "price_momentum",
    ]
    existing_fill_cols = [col for col in feature_fill_zero if col in final_df.columns]
    final_df[existing_fill_cols] = final_df[existing_fill_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    feature_cols = [col for col in final_df.columns if col not in {"date"}]
    null_counts = final_df[feature_cols].isna().sum()
    assert int(null_counts.sum()) == 0, f"Final feature dataset has nulls: {null_counts[null_counts > 0].to_dict()}"
    return final_df


# ---------------------------------------------------------------------------
# Step 4.7 — Final Feature Dataset Assembly (memory-efficient batched)
# ---------------------------------------------------------------------------


def build_feature_dataset(
    df: pd.DataFrame,
    output_path: str = "data/features/features_all.parquet",
    batch_size: int = 500,
    mapping_output_path: str | Path = DEFAULT_MAPPING_PATH,
) -> Path:
    """
    Build the full feature dataset from the clean DataFrame.
    Processes lag/rolling features in batches of `batch_size` series
    to keep peak RAM usage low.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned, long-format DataFrame from preprocessor.
    output_path : str
        Destination Parquet path.
    batch_size : int
        Number of series IDs to process per batch (default 500 ≈ 1-2 GB peak).
    """
    # Step 1 — calendar + price + encoding + time_idx (vectorised — fast)
    logger.info("Adding calendar features…")
    df = add_calendar_features(df)
    logger.info("Adding price features…")
    df = add_price_features(df)
    logger.info("Adding static encodings and time_idx…")
    df, mappings = encode_static_metadata(df)
    write_category_mappings(mappings, mapping_output_path)
    df = add_time_idx(df)

    # Downcast sales to float32 to save memory
    df["sales"] = df["sales"].astype("float32")
    df["sell_price"] = df["sell_price"].astype("float32")

    # Step 2 — lag + rolling in batches streaming to disk
    logger.info("Computing lag + rolling features in batches (streaming to disk)...")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    all_ids = df["id"].unique()
    n_batches = (len(all_ids) + batch_size - 1) // batch_size
    total_rows = 0
    writer = None

    for i in range(n_batches):
        batch_ids = all_ids[i * batch_size : (i + 1) * batch_size]
        batch = df[df["id"].isin(batch_ids)].copy()
        batch = batch.sort_values(["id", "date"])
        batch = batch.groupby("id", group_keys=False, as_index=False).apply(_compute_lag_rolling_for_series)
        batch = batch[batch["lag_28"].notna()]

        # Fill any remaining NaN (rolling at series start) with 0
        lag_cols = [f"lag_{d}" for d in LAG_DAYS]
        feature_cols = lag_cols + ROLLING_COLS
        batch[feature_cols] = batch[feature_cols].fillna(0).astype("float32")
        batch = finalize_feature_frame(batch)

        # Write to parquet
        table = pa.Table.from_pandas(batch, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out, table.schema)
        writer.write_table(table)

        total_rows += len(batch)
        logger.info(f"  Batch {i+1}/{n_batches}: {len(batch_ids)} series processed, {len(batch)} rows written")

    if writer:
        writer.close()

    logger.info(f"Feature dataset saved to {out} | shape: ({total_rows}, {len(batch.columns)})")
    logger.info("Final columns: %s", batch.columns.tolist())
    return out
