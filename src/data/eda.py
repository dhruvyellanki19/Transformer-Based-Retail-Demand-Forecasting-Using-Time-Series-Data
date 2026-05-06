"""
Utilities for the Phase 2 exploratory data analysis notebook.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGURES_DIR = Path("outputs/figures")
VALIDATION_END_DATE = pd.Timestamp("2016-04-24")


def ensure_figures_dir() -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR


def build_shape_summary(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, frame in frames.items():
        memory_mb = frame.memory_usage(deep=True).sum() / (1024**2)
        rows.append(
            {
                "dataset": name,
                "rows": int(frame.shape[0]),
                "cols": int(frame.shape[1]),
                "memory_mb": round(memory_mb, 2),
            }
        )
    return pd.DataFrame(rows)


def save_table_figure(df: pd.DataFrame, output_path: str | Path, title: str, index: bool = False) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    display_df = df.reset_index() if index else df.copy()
    nrows = max(len(display_df), 1)
    height = max(1.8, 0.45 * nrows + 1.2)

    fig, ax = plt.subplots(figsize=(max(8, len(display_df.columns) * 2), height))
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=12)
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def compute_zero_sales_rate(df: pd.DataFrame, group_col: str | None = None) -> float | pd.DataFrame:
    if group_col is None:
        return float((df["sales"] == 0).mean())

    result = (
        df.assign(is_zero=df["sales"].eq(0))
        .groupby(group_col, observed=True)["is_zero"]
        .mean()
        .rename("zero_rate")
        .reset_index()
    )
    return result


def compute_event_day_lift(df: pd.DataFrame) -> dict[str, float]:
    grouped = df.groupby("is_event", observed=True)["sales"].mean()
    non_event_mean = float(grouped.get(0, np.nan))
    event_mean = float(grouped.get(1, np.nan))
    lift_pct = float(((event_mean / non_event_mean) - 1) * 100) if non_event_mean else np.nan
    return {
        "non_event_mean": non_event_mean,
        "event_mean": event_mean,
        "lift_pct": lift_pct,
    }


def compute_snap_lift_by_state(df: pd.DataFrame) -> pd.DataFrame:
    stats = (
        df.groupby(["state_id", "is_snap"], observed=True)["sales"]
        .mean()
        .unstack(fill_value=np.nan)
        .rename(columns={0: "non_snap_mean", 1: "snap_mean"})
        .reset_index()
    )
    stats["lift_pct"] = ((stats["snap_mean"] / stats["non_snap_mean"]) - 1) * 100
    return stats[["state_id", "non_snap_mean", "snap_mean", "lift_pct"]]


def compute_category_distribution_stats(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("cat_id", observed=True)["sales"]
    stats = grouped.agg(["mean", "median", "std", "max"]).reset_index()
    p99 = grouped.quantile(0.99).rename("p99").reset_index(drop=True)
    skew = grouped.skew().rename("skewness").reset_index(drop=True)
    kurt = grouped.apply(pd.Series.kurt).rename("kurtosis").reset_index(drop=True)
    stats["p99"] = p99
    stats["skewness"] = skew
    stats["kurtosis"] = kurt
    return stats


def compute_series_sparsity(sales_df: pd.DataFrame, day_cols: list[str]) -> pd.DataFrame:
    zero_rate = sales_df[day_cols].eq(0).mean(axis=1)
    return sales_df[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]].assign(sparsity_ratio=zero_rate)


def first_last_sale_dates(sales_df: pd.DataFrame, day_cols: list[str], calendar_map: pd.Series) -> pd.DataFrame:
    values = sales_df[day_cols].to_numpy()
    non_zero_mask = values > 0

    first_idx = np.argmax(non_zero_mask, axis=1)
    last_idx = values.shape[1] - 1 - np.argmax(non_zero_mask[:, ::-1], axis=1)
    has_sales = non_zero_mask.any(axis=1)

    first_days = np.where(has_sales, np.array(day_cols, dtype=object)[first_idx], None)
    last_days = np.where(has_sales, np.array(day_cols, dtype=object)[last_idx], None)

    first_dates = pd.to_datetime(pd.Series(first_days).map(calendar_map))
    last_dates = pd.to_datetime(pd.Series(last_days).map(calendar_map))

    return sales_df[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]].assign(
        first_sale_date=first_dates,
        last_sale_date=last_dates,
    )


def load_validation_eda_frame(clean_path: str | Path = "data/processed/sales_clean.parquet") -> pd.DataFrame:
    clean_df = pd.read_parquet(clean_path)
    eda_df = clean_df.loc[clean_df["date"] <= VALIDATION_END_DATE].copy()
    return eda_df.sort_values(["id", "date"]).reset_index(drop=True)
