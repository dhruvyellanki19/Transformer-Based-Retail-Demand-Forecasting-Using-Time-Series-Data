"""
src/data/preprocessor.py
------------------------
Data cleaning and preprocessing module.
Handles missing prices, negative sales, event flags, and discontinued items.
Outputs cleaned data to data/processed/sales_clean.parquet.

Phase 1 update: vectorised SNAP lookup (was .apply() — too slow on 58M rows);
                added split_train_val_test() to produce final EDA-ready splits.
"""

import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

# Canonical date-based splits (from execution plan)
TRAIN_END = "2016-03-27"
VAL_START = "2016-03-28"
VAL_END = "2016-04-24"
TEST_START = "2016-04-25"
TEST_END = "2016-05-22"

TRAIN_END_TS = pd.Timestamp(TRAIN_END)
VAL_START_TS = pd.Timestamp(VAL_START)
VAL_END_TS = pd.Timestamp(VAL_END)
TEST_START_TS = pd.Timestamp(TEST_START)
TEST_END_TS = pd.Timestamp(TEST_END)


def melt_sales(sales_df: pd.DataFrame) -> pd.DataFrame:
    """Convert wide-format sales DataFrame to long format."""
    day_cols = [c for c in sales_df.columns if c not in ID_COLS]
    long_df = sales_df.melt(id_vars=ID_COLS, value_vars=day_cols, var_name="d", value_name="sales")
    logger.info(f"Melted sales to {len(long_df):,} rows")
    return long_df


def merge_calendar(long_df: pd.DataFrame, calendar_df: pd.DataFrame) -> pd.DataFrame:
    """Join long sales with calendar on 'd' column."""
    merged = long_df.merge(calendar_df, on="d", how="left")
    merged["date"] = pd.to_datetime(merged["date"])
    return merged


def merge_prices(df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    """Join merged DataFrame with sell_prices on [store_id, item_id, wm_yr_wk]."""
    return df.merge(prices_df, on=["store_id", "item_id", "wm_yr_wk"], how="left")


def handle_missing_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Forward/backfill sell_price per (store_id, item_id)."""
    logger.info("Handling missing sell prices...")
    df = df.sort_values(["id", "date"]).copy()
    df["sell_price"] = df.groupby(["store_id", "item_id"])["sell_price"].transform(lambda x: x.ffill().bfill())
    bad = df[(df["sales"] > 0) & df["sell_price"].isna()]
    assert len(bad) == 0, f"{len(bad)} rows have sales>0 but null sell_price"
    return df


def handle_negative_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Clip negative sales to 0."""
    n_neg = (df["sales"] < 0).sum()
    logger.info(f"Clipping {n_neg} negative sales to 0")
    df["sales"] = df["sales"].clip(lower=0)
    assert df["sales"].min() >= 0, "Sales still has negative values after clipping"
    return df


def add_event_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary event/SNAP flags — vectorised per-state SNAP lookup.
    (Replaces slow row-wise .apply() for 58M-row DataFrame.)
    """
    df = df.copy()
    df["is_event"] = df["event_name_1"].notna().astype("int8")

    # Vectorised SNAP: create a single 'snap' column by selecting the right state column
    snap_ca = df["snap_CA"].where(df["state_id"] == "CA", 0)
    snap_tx = df["snap_TX"].where(df["state_id"] == "TX", 0)
    snap_wi = df["snap_WI"].where(df["state_id"] == "WI", 0)
    df["is_snap"] = (snap_ca | snap_tx | snap_wi).astype("int8")

    event_type_map = {
        "Cultural": "is_cultural",
        "National": "is_national",
        "Religious": "is_religious",
        "Sporting": "is_sporting",
    }
    for event_type, col in event_type_map.items():
        df[col] = (df["event_type_1"] == event_type).astype("int8")

    return df


def add_is_active_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Mark series as inactive after their last nonzero sale date."""
    last_sale = df[df["sales"] > 0].groupby("id")["date"].max().rename("last_sale_date")
    df = df.merge(last_sale, on="id", how="left")
    df["is_active"] = (df["date"] <= df["last_sale_date"]).astype("int8")
    df.drop(columns=["last_sale_date"], inplace=True)
    return df


def _finalize_split_assertions(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    strict_validation: bool,
) -> None:
    assert train["date"].max() < VAL_START_TS, "Leakage: train bleeds into val"
    assert val["date"].max() < TEST_START_TS, "Leakage: val bleeds into test"
    if strict_validation:
        assert train["id"].nunique() == 30490, f"Train has {train['id'].nunique()} series, expected 30490"
        assert val["id"].nunique() == 30490, f"Val has {val['id'].nunique()} series, expected 30490"
        assert test["id"].nunique() == 30490, f"Test has {test['id'].nunique()} series, expected 30490"


def _split_from_parquet_streaming(
    source: Path,
    out: Path,
    strict_validation: bool,
    return_dataframes: bool,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    split_paths = {
        "train": out / "train.parquet",
        "val": out / "val.parquet",
        "test": out / "test.parquet",
    }
    writers: dict[str, pq.ParquetWriter] = {}
    id_sets = {name: set() for name in split_paths}
    counts = {name: 0 for name in split_paths}
    date_bounds = {name: [None, None] for name in split_paths}
    collected = {name: [] for name in split_paths} if return_dataframes else None

    parquet_file = pq.ParquetFile(source)
    for batch in parquet_file.iter_batches(batch_size=500_000):
        batch_df = batch.to_pandas()
        split_frames = {
            "train": batch_df[batch_df["date"] <= TRAIN_END_TS].copy(),
            "val": batch_df[(batch_df["date"] >= VAL_START_TS) & (batch_df["date"] <= VAL_END_TS)].copy(),
            "test": batch_df[(batch_df["date"] >= TEST_START_TS) & (batch_df["date"] <= TEST_END_TS)].copy(),
        }
        for name, frame in split_frames.items():
            if frame.empty:
                continue
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if name not in writers:
                writers[name] = pq.ParquetWriter(split_paths[name], table.schema)
            writers[name].write_table(table)

            counts[name] += len(frame)
            id_sets[name].update(frame["id"].unique().tolist())
            min_date = frame["date"].min()
            max_date = frame["date"].max()
            current_min, current_max = date_bounds[name]
            date_bounds[name][0] = min_date if current_min is None else min(current_min, min_date)
            date_bounds[name][1] = max_date if current_max is None else max(current_max, max_date)
            if collected is not None:
                collected[name].append(frame)

    for writer in writers.values():
        writer.close()

    if collected is not None:
        train = pd.concat(collected["train"], ignore_index=True) if collected["train"] else pd.DataFrame()
        val = pd.concat(collected["val"], ignore_index=True) if collected["val"] else pd.DataFrame()
        test = pd.concat(collected["test"], ignore_index=True) if collected["test"] else pd.DataFrame()
    else:
        train = val = test = None

    assert date_bounds["train"][1] < VAL_START_TS, "Leakage: train bleeds into val"
    assert date_bounds["val"][1] < TEST_START_TS, "Leakage: val bleeds into test"
    if strict_validation:
        assert len(id_sets["train"]) == 30490, f"Train has {len(id_sets['train'])} series, expected 30490"
        assert len(id_sets["val"]) == 30490, f"Val has {len(id_sets['val'])} series, expected 30490"
        assert len(id_sets["test"]) == 30490, f"Test has {len(id_sets['test'])} series, expected 30490"

    logger.info(
        "Train: %s rows  |  Val: %s rows  |  Test: %s rows",
        f"{counts['train']:,}",
        f"{counts['val']:,}",
        f"{counts['test']:,}",
    )
    return train, val, test


def split_train_val_test(
    features_df: pd.DataFrame | str | Path,
    features_dir: str = "data/features",
    strict_validation: bool = True,
    return_dataframes: bool = True,
) -> tuple:
    """
    Split the feature DataFrame into train / val / test by date.

    Splits:
      Train : 2011-01-29 → 2016-03-27  (d_1  – d_1885)
      Val   : 2016-03-28 → 2016-04-24  (d_1886 – d_1913, 28 days)
      Test  : 2016-04-25 → 2016-05-22  (d_1914 – d_1941, 28 days)
    """
    out = Path(features_dir)
    out.mkdir(parents=True, exist_ok=True)

    if isinstance(features_df, (str, Path)):
        source = Path(features_df)
        train, val, test = _split_from_parquet_streaming(source, out, strict_validation, return_dataframes)
    else:
        train = features_df[features_df["date"] <= TRAIN_END].copy()
        val = features_df[(features_df["date"] >= VAL_START) & (features_df["date"] <= VAL_END)].copy()
        test = features_df[(features_df["date"] >= TEST_START) & (features_df["date"] <= TEST_END)].copy()
        _finalize_split_assertions(train, val, test, strict_validation)
        train.to_parquet(out / "train.parquet", index=False)
        val.to_parquet(out / "val.parquet", index=False)
        test.to_parquet(out / "test.parquet", index=False)
        logger.info(f"Train: {len(train):,} rows  |  Val: {len(val):,} rows  |  Test: {len(test):,} rows")
    return train, val, test


def run_preprocessing(
    sales_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    output_path: str = "data/processed/sales_clean.parquet",
) -> pd.DataFrame:
    """Full preprocessing pipeline. Saves output parquet and returns DataFrame."""
    df = melt_sales(sales_df)
    df = merge_calendar(df, calendar_df)
    df = merge_prices(df, prices_df)
    df = handle_missing_prices(df)
    df = handle_negative_sales(df)
    df = add_event_flags(df)
    df = add_is_active_flag(df)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    logger.info(f"Saved clean data to {out} | shape: {df.shape}")
    return df
