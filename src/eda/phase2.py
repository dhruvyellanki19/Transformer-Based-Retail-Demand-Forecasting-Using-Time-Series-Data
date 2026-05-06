from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.tsa.seasonal import STL

from src.data.eda import load_validation_eda_frame
from src.data.loader import (
    load_calendar,
    load_sales_train_evaluation,
    load_sales_train_validation,
    load_sell_prices,
)
from src.data.preprocessor import melt_sales, merge_calendar, merge_prices, run_preprocessing

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
EDA_CACHE_PATH = Path("data/processed/eda_long_phase2.parquet")
CLEAN_EDA_PATH = Path("data/processed/sales_clean.parquet")
VALIDATION_END_DATE = pd.Timestamp("2016-04-24")

sns.set_theme(style="whitegrid")


def ensure_eda_dirs() -> None:
    EDA_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _validate_long_frame(long_df: pd.DataFrame, expected_rows: int) -> bool:
    required_columns = {"id", "date", "sales"}
    if not required_columns.issubset(long_df.columns):
        return False
    if len(long_df) != expected_rows:
        return False
    if not pd.api.types.is_datetime64_any_dtype(long_df["date"]):
        return False
    return True


def _read_long_frame(path: Path, expected_rows: int) -> pd.DataFrame | None:
    try:
        long_df = pd.read_parquet(path)
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        return None

    if not _validate_long_frame(long_df, expected_rows):
        return None

    long_df = long_df.copy()
    long_df["date"] = pd.to_datetime(long_df["date"])
    return long_df.sort_values(["id", "date"]).reset_index(drop=True)


def _rebuild_clean_frame(
    sales_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    prices_df: pd.DataFrame,
) -> pd.DataFrame:
    clean_df = run_preprocessing(sales_df, calendar_df, prices_df, output_path=str(CLEAN_EDA_PATH))
    clean_df = clean_df.copy()
    clean_df["date"] = pd.to_datetime(clean_df["date"])
    return clean_df.sort_values(["id", "date"]).reset_index(drop=True)


def _load_clean_frame(
    sales_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    prices_df: pd.DataFrame,
) -> pd.DataFrame:
    try:
        clean_df = load_validation_eda_frame(CLEAN_EDA_PATH)
        if not _validate_long_frame(clean_df, sales_df.shape[0] * (sales_df.shape[1] - len(ID_COLS))):
            raise ValueError("Invalid cleaned EDA frame")
    except Exception:
        clean_df = _rebuild_clean_frame(sales_df, calendar_df, prices_df)
    clean_df = clean_df.copy()
    clean_df["date"] = pd.to_datetime(clean_df["date"])
    return clean_df.loc[clean_df["date"] <= VALIDATION_END_DATE].sort_values(["id", "date"]).reset_index(drop=True)


def _event_rows(long_df: pd.DataFrame) -> pd.DataFrame:
    primary = (
        long_df.loc[long_df["event_name_1"].notna(), ["date", "event_name_1", "event_type_1", "sales"]]
        .rename(columns={"event_name_1": "event_name", "event_type_1": "event_type"})
        .copy()
    )
    secondary = (
        long_df.loc[long_df["event_name_2"].notna(), ["date", "event_name_2", "event_type_2", "sales"]]
        .rename(columns={"event_name_2": "event_name", "event_type_2": "event_type"})
        .copy()
    )
    events = pd.concat([primary, secondary], ignore_index=True)
    if events.empty:
        return events
    events = events.drop_duplicates(subset=["date", "event_name", "event_type"])
    daily_sales = long_df.groupby("date")["sales"].sum().rename("daily_sales")
    events["daily_sales"] = events["date"].map(daily_sales)
    return events


def build_shape_summary(
    sales_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    evaluation_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    summary_rows = [
        {
            "dataset": "sales_train_validation",
            "rows": sales_df.shape[0],
            "columns": sales_df.shape[1],
            "id_columns": len(ID_COLS),
            "day_columns": len([col for col in sales_df.columns if col not in ID_COLS]),
            "memory_mb": round(sales_df.memory_usage(deep=True).sum() / 1024**2, 2),
        }
    ]
    if evaluation_df is not None:
        summary_rows.append(
            {
                "dataset": "sales_train_evaluation",
                "rows": evaluation_df.shape[0],
                "columns": evaluation_df.shape[1],
                "id_columns": len(ID_COLS),
                "day_columns": len([col for col in evaluation_df.columns if col not in ID_COLS]),
                "memory_mb": round(evaluation_df.memory_usage(deep=True).sum() / 1024**2, 2),
            }
        )
    summary_rows.extend(
        [
            {
                "dataset": "calendar",
                "rows": calendar_df.shape[0],
                "columns": calendar_df.shape[1],
                "id_columns": np.nan,
                "day_columns": np.nan,
                "memory_mb": round(calendar_df.memory_usage(deep=True).sum() / 1024**2, 2),
            },
            {
                "dataset": "sell_prices",
                "rows": prices_df.shape[0],
                "columns": prices_df.shape[1],
                "id_columns": np.nan,
                "day_columns": np.nan,
                "memory_mb": round(prices_df.memory_usage(deep=True).sum() / 1024**2, 2),
            },
        ]
    )
    summary = pd.DataFrame(summary_rows)
    return summary


def build_long_sales_frame(
    sales_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    ensure_eda_dirs()
    expected_rows = sales_df.shape[0] * (sales_df.shape[1] - len(ID_COLS))

    if cache_path is not None and cache_path.exists():
        cached = _read_long_frame(cache_path, expected_rows)
        if cached is not None:
            return cached

    long_df = melt_sales(sales_df)
    assert len(long_df) == expected_rows, f"Expected {expected_rows:,} rows after melt, got {len(long_df):,}"
    long_df = merge_calendar(long_df, calendar_df)
    long_df = merge_prices(long_df, prices_df)
    long_df["date"] = pd.to_datetime(long_df["date"])
    long_df = long_df.sort_values(["id", "date"]).reset_index(drop=True)

    if cache_path is not None:
        try:
            long_df.to_parquet(cache_path, index=False)
        except Exception:
            pass

    return long_df


def load_phase2_data(
    cache_path: Path = EDA_CACHE_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sales_df = load_sales_train_validation()
    sales_eval_df = load_sales_train_evaluation()
    calendar_df = load_calendar()
    prices_df = load_sell_prices()
    raw_long_df = build_long_sales_frame(sales_df, calendar_df, prices_df, cache_path=cache_path)
    clean_long_df = _load_clean_frame(sales_df, calendar_df, prices_df)
    return sales_df, sales_eval_df, calendar_df, prices_df, raw_long_df, clean_long_df


def compute_series_sparsity(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby("id")
        .agg(
            zero_days=("sales", lambda s: int((s == 0).sum())),
            total_days=("sales", "size"),
            dept_id=("dept_id", "first"),
            store_id=("store_id", "first"),
        )
        .reset_index()
    )
    grouped["sparsity_ratio"] = grouped["zero_days"] / grouped["total_days"]

    non_zero = df[df["sales"] > 0]
    first_non_zero = non_zero.groupby("id")["date"].min().rename("first_non_zero_date")
    last_non_zero = non_zero.groupby("id")["date"].max().rename("last_non_zero_date")
    grouped = grouped.merge(first_non_zero, on="id", how="left")
    grouped = grouped.merge(last_non_zero, on="id", how="left")
    return grouped


def compute_event_lift(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    events = _event_rows(df)
    daily = (
        df.assign(is_event=df[["event_name_1", "event_name_2"]].notna().any(axis=1))
        .groupby("date")
        .agg(sales=("sales", "sum"), is_event=("is_event", "max"))
        .reset_index()
    )

    event_avg = daily.loc[daily["is_event"], "sales"].mean()
    non_event_avg = daily.loc[~daily["is_event"], "sales"].mean()
    total = pd.Series(
        {
            "event_day_avg_sales": float(event_avg),
            "non_event_day_avg_sales": float(non_event_avg),
            "lift_pct": float(((event_avg - non_event_avg) / non_event_avg) * 100) if non_event_avg else np.nan,
        }
    )

    if events.empty:
        by_type = pd.DataFrame(columns=["event_type", "event_day_avg_sales"])
    else:
        by_type = events.groupby("event_type", dropna=False)["daily_sales"].mean().rename("event_day_avg_sales").reset_index()
    by_type["non_event_day_avg_sales"] = non_event_avg
    by_type["lift_pct"] = ((by_type["event_day_avg_sales"] - non_event_avg) / non_event_avg) * 100 if non_event_avg else np.nan
    return total, by_type.sort_values("lift_pct", ascending=False).reset_index(drop=True)


def display_figure(fig: plt.Figure) -> None:
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def display_table_figure(df: pd.DataFrame, title: str) -> None:
    fig_height = max(2.5, 0.45 * (len(df) + 1))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    ax.set_title(title, fontsize=12, pad=10)
    display_figure(fig)


def display_shape_summary(summary_df: pd.DataFrame) -> None:
    display_table_figure(summary_df, "Phase 2 Shape Summary")


def add_common_analysis_columns(long_df: pd.DataFrame) -> pd.DataFrame:
    df = long_df.copy()
    df["is_event"] = df[["event_name_1", "event_name_2"]].notna().any(axis=1)
    df["event_name"] = df["event_name_1"].combine_first(df["event_name_2"])
    df["event_type"] = df["event_type_1"].combine_first(df["event_type_2"])
    df["is_snap"] = (
        ((df["state_id"] == "CA") & (df["snap_CA"] == 1))
        | ((df["state_id"] == "TX") & (df["snap_TX"] == 1))
        | ((df["state_id"] == "WI") & (df["snap_WI"] == 1))
    )
    df["day_of_week"] = df["date"].dt.day_name()
    df["month"] = df["date"].dt.month
    return df


def sample_rows(df: pd.DataFrame, n: int, seed: int = 42, non_zero_only: bool = False) -> pd.DataFrame:
    base = df[df["sales"] > 0] if non_zero_only else df
    if len(base) <= n:
        return base.copy()
    return base.sample(n=n, random_state=seed)


def category_sales_stats(long_df: pd.DataFrame) -> pd.DataFrame:
    stats_df = (
        long_df.groupby("cat_id")["sales"]
        .agg(
            mean="mean",
            median="median",
            std="std",
            max="max",
            p99=lambda s: s.quantile(0.99),
            skew=lambda s: s.skew(),
            kurtosis=lambda s: s.kurt(),
        )
        .reset_index()
    )
    return stats_df.round(4)


def top_bottom_items(long_df: pd.DataFrame, top_n: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    totals = (
        long_df.groupby(["item_id", "cat_id"])["sales"].sum().sort_values(ascending=False).reset_index(name="total_sales")
    )
    top = totals.head(top_n).copy()
    bottom = totals.tail(top_n).sort_values("total_sales", ascending=True).copy()
    return top, bottom


def daily_total_sales(long_df: pd.DataFrame) -> pd.Series:
    return long_df.groupby("date")["sales"].sum().sort_index()


def rolling_sales(daily_sales: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sales": daily_sales,
            "rolling_7": daily_sales.rolling(7, min_periods=1).mean(),
            "rolling_28": daily_sales.rolling(28, min_periods=1).mean(),
        }
    )


def stl_components(series: pd.Series, period: int) -> pd.DataFrame:
    result = STL(series, period=period, robust=True).fit()
    return pd.DataFrame(
        {
            "observed": series,
            "trend": result.trend,
            "seasonal": result.seasonal,
            "resid": result.resid,
        }
    )


def structural_breaks(trend: pd.Series, top_n: int = 5) -> pd.DataFrame:
    diff = trend.diff().abs().dropna()
    breaks = diff.nlargest(top_n).rename("absolute_change").reset_index()
    breaks.columns = ["date", "absolute_change"]
    return breaks


def price_change_features(long_df: pd.DataFrame) -> pd.DataFrame:
    df = long_df.sort_values(["store_id", "item_id", "date"]).copy()
    df["price_change_pct"] = df.groupby(["store_id", "item_id"])["sell_price"].pct_change()
    return df


def snap_lift_by_state(long_df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for state in ["CA", "TX", "WI"]:
        snap_col = f"snap_{state}"
        state_daily = (
            long_df[long_df["state_id"] == state]
            .groupby("date")
            .agg(sales=("sales", "sum"), snap=(snap_col, "max"))
            .reset_index()
        )
        snap_avg = state_daily.loc[state_daily["snap"] == 1, "sales"].mean()
        non_snap_avg = state_daily.loc[state_daily["snap"] == 0, "sales"].mean()
        lift_pct = ((snap_avg - non_snap_avg) / non_snap_avg) * 100 if non_snap_avg else np.nan
        results.append(
            {
                "state_id": state,
                "snap_avg_sales": snap_avg,
                "non_snap_avg_sales": non_snap_avg,
                "lift_pct": lift_pct,
            }
        )
    return pd.DataFrame(results).sort_values("lift_pct", ascending=False)


def event_rankings(long_df: pd.DataFrame) -> pd.DataFrame:
    events = _event_rows(long_df)
    daily = long_df.groupby("date")["sales"].sum().rename("daily_sales").reset_index()
    baseline = daily["daily_sales"].mean()
    rankings = events.groupby(["event_name", "event_type"])["daily_sales"].mean().reset_index(name="event_avg_sales")
    rankings = rankings.sort_values("event_avg_sales", ascending=False)
    rankings["lift_pct_vs_overall_mean"] = ((rankings["event_avg_sales"] - baseline) / baseline) * 100
    return rankings


def list_named_events(long_df: pd.DataFrame) -> pd.DataFrame:
    events = _event_rows(long_df)
    if events.empty:
        return pd.DataFrame(columns=["event_name", "event_type", "num_days"])
    return (
        events.groupby(["event_name", "event_type"], dropna=False)["date"]
        .nunique()
        .rename("num_days")
        .reset_index()
        .sort_values(["event_type", "event_name"], na_position="last")
        .reset_index(drop=True)
    )


def price_correlations(long_df: pd.DataFrame) -> pd.DataFrame:
    valid = long_df.dropna(subset=["sell_price"])
    rows = []
    for category, subset in valid.groupby("cat_id"):
        corr = subset["sell_price"].corr(subset["sales"])
        rows.append({"cat_id": category, "price_sales_corr": corr})
    return pd.DataFrame(rows).sort_values("cat_id").reset_index(drop=True)


def volatile_items(long_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    volatility = (
        long_df.dropna(subset=["sell_price"])
        .groupby(["item_id", "store_id"])["sell_price"]
        .std()
        .rename("price_std")
        .reset_index()
        .sort_values("price_std", ascending=False)
    )
    return volatility.head(top_n)


def representative_series(sales_df: pd.DataFrame) -> list[str]:
    day_cols = [col for col in sales_df.columns if col not in ID_COLS]
    series_meta = sales_df[ID_COLS].copy()
    series_meta["sparsity_ratio"] = (sales_df[day_cols] == 0).mean(axis=1)
    picks: list[str] = []

    for category in ["FOODS", "HOBBIES", "HOUSEHOLD"]:
        subset = series_meta[series_meta["cat_id"] == category]
        picks.append(subset.sort_values("sparsity_ratio").iloc[0]["id"])

    picks.append(series_meta.sort_values("sparsity_ratio").iloc[0]["id"])
    picks.append(series_meta.sort_values("sparsity_ratio", ascending=False).iloc[0]["id"])
    return list(dict.fromkeys(picks))


def extract_series(sales_df: pd.DataFrame, series_id: str) -> pd.Series:
    row = sales_df.loc[sales_df["id"] == series_id]
    if row.empty:
        raise KeyError(f"Unknown series_id: {series_id}")
    day_cols = [col for col in sales_df.columns if col not in ID_COLS]
    values = row.iloc[0][day_cols].astype(float).values
    dates = pd.date_range("2011-01-29", periods=len(day_cols), freq="D")
    return pd.Series(values, index=dates, name=series_id)


def stationarity_tests(series: pd.Series) -> dict[str, float]:
    from statsmodels.tsa.stattools import adfuller, kpss

    try:
        adf_pvalue = adfuller(series, autolag="AIC")[1]
    except Exception:
        adf_pvalue = np.nan
    try:
        kpss_pvalue = kpss(series, regression="c", nlags="auto")[1]
    except Exception:
        kpss_pvalue = np.nan
    return {"adf_pvalue": adf_pvalue, "kpss_pvalue": kpss_pvalue}


def feature_correlation_frame(long_df: pd.DataFrame) -> pd.DataFrame:
    df = add_common_analysis_columns(long_df)
    df["day_of_week_num"] = df["date"].dt.dayofweek
    df["month_num"] = df["date"].dt.month
    return df[["sales", "sell_price", "is_event", "is_snap", "day_of_week_num", "month_num"]].dropna()


def correlation_significance(feature_df: pd.DataFrame) -> pd.Series:
    corr = feature_df["sell_price"].corr(feature_df["sales"])
    t_stat, p_value = stats.ttest_ind(
        feature_df.loc[feature_df["sell_price"] >= feature_df["sell_price"].median(), "sales"],
        feature_df.loc[feature_df["sell_price"] < feature_df["sell_price"].median(), "sales"],
        equal_var=False,
    )
    return pd.Series({"price_sales_corr": corr, "t_stat": t_stat, "p_value": p_value})


def summary_lines(
    sparsity_ratio: float,
    category_stats_df: pd.DataFrame,
    event_lift: pd.Series,
    snap_lift_df: pd.DataFrame,
    price_corr_df: pd.DataFrame,
    break_df: pd.DataFrame,
) -> list[str]:
    most_sparse_state = snap_lift_df.sort_values("lift_pct", ascending=False).iloc[0]["state_id"]
    strongest_price_category = price_corr_df.sort_values("price_sales_corr").iloc[0]["cat_id"]
    break_dates = ", ".join(pd.to_datetime(break_df["date"]).dt.strftime("%Y-%m-%d").head(3).tolist())
    return [
        f"Overall daily-sales sparsity is {sparsity_ratio:.2%}, confirming heavy zero inflation in the bottom-level series.",
        "FOODS is the densest category with the strongest weekly seasonality, while HOBBIES and HOUSEHOLD are more intermittent.",
        f"Event days lift aggregate demand by {event_lift['lift_pct']:.2f}% versus non-event days, supporting explicit event features in downstream models.",
        f"SNAP effects are strongest in {most_sparse_state}, reinforcing state-specific benefit flags in forecasting features.",
        f"Price-demand correlation is most negative in {strongest_price_category}, supporting price elasticity features and robust losses over plain MSE.",
        f"Major structural breaks in aggregate demand appear around {break_dates}, so rolling features and calendar effects should remain prominent in baselines and transformers.",
    ]


def write_readme_summary(lines: Iterable[str], readme_path: Path = Path("README.md")) -> Path:
    content = "\n".join(
        [
            "# Transformer-Based Retail Demand Forecasting Using Time Series Data",
            "",
            "## Phase 2 EDA Summary",
            "",
            *[f"- {line}" for line in lines],
            "",
            "## Artefacts",
            "",
            "- `notebooks/01_eda.ipynb` contains the full Phase 2 walkthrough.",
            "- Figures are rendered inline in `notebooks/01_eda.ipynb`.",
            "- `data/processed/eda_long_phase2.parquet` caches the merged Phase 2 long-format dataset for reruns.",
            "",
        ]
    )
    readme_path.write_text(content)
    return readme_path


def calendar_heatmap_frame(daily_sales: pd.Series) -> pd.DataFrame:
    heatmap_df = daily_sales.rename("sales").to_frame()
    heatmap_df["year"] = heatmap_df.index.year
    heatmap_df["week"] = heatmap_df.index.isocalendar().week.astype(int)
    heatmap_df["weekday"] = heatmap_df.index.dayofweek
    pivot = heatmap_df.pivot_table(index=["year", "week"], columns="weekday", values="sales", aggfunc="mean")
    pivot.columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return pivot


def super_bowl_week_frame(long_df: pd.DataFrame) -> pd.DataFrame:
    events = _event_rows(long_df)
    events = events[events["event_name"].fillna("").str.contains("SuperBowl", case=False)]
    daily = long_df.groupby("date")["sales"].sum().sort_index()
    rows = []
    for event_date in sorted(events["date"].drop_duplicates()):
        event_week_start = event_date - pd.Timedelta(days=3)
        before = daily.loc[event_week_start - pd.Timedelta(days=28) : event_week_start - pd.Timedelta(days=1)].mean()
        during = daily.loc[event_week_start : event_week_start + pd.Timedelta(days=6)].mean()
        after = daily.loc[event_week_start + pd.Timedelta(days=7) : event_week_start + pd.Timedelta(days=34)].mean()
        rows.append({"super_bowl_date": event_date, "before_avg": before, "during_avg": during, "after_avg": after})
    return pd.DataFrame(rows)


def thanksgiving_week_frame(long_df: pd.DataFrame) -> pd.DataFrame:
    events = _event_rows(long_df)
    events = events[events["event_name"].fillna("").str.contains("Thanksgiving", case=False)]
    daily = long_df.groupby("date")["sales"].sum().sort_index()
    rows = []
    for event_date in sorted(events["date"].drop_duplicates()):
        week_start = event_date - pd.Timedelta(days=3)
        before = daily.loc[week_start - pd.Timedelta(days=28) : week_start - pd.Timedelta(days=1)].mean()
        during = daily.loc[week_start : week_start + pd.Timedelta(days=6)].mean()
        after = daily.loc[week_start + pd.Timedelta(days=7) : week_start + pd.Timedelta(days=34)].mean()
        rows.append({"thanksgiving_date": event_date, "before_avg": before, "during_avg": during, "after_avg": after})
    return pd.DataFrame(rows)


def christmas_comparison(long_df: pd.DataFrame) -> pd.Series:
    daily = long_df.groupby("date")["sales"].sum().sort_index()
    christmas_days = daily[(daily.index.month == 12) & (daily.index.day.isin([24, 25]))]
    december_baseline = daily[daily.index.month == 12]
    lift_pct = ((christmas_days.mean() - december_baseline.mean()) / december_baseline.mean()) * 100
    return pd.Series({"christmas_avg": christmas_days.mean(), "december_avg": december_baseline.mean(), "lift_pct": lift_pct})


def weekday_totals(long_df: pd.DataFrame) -> pd.DataFrame:
    ordered_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    totals = (
        long_df.assign(day_name=long_df["date"].dt.day_name())
        .groupby(["cat_id", "day_name"])["sales"]
        .sum()
        .reset_index()
    )
    totals["day_name"] = pd.Categorical(totals["day_name"], categories=ordered_days, ordered=True)
    return totals.sort_values(["cat_id", "day_name"])


def month_totals(long_df: pd.DataFrame) -> pd.DataFrame:
    totals = long_df.assign(month=long_df["date"].dt.month).groupby("month")["sales"].sum().reset_index()
    return totals


def highest_lowest_days(long_df: pd.DataFrame) -> pd.DataFrame:
    daily = long_df.groupby("date")["sales"].sum().sort_values()
    lowest = daily[daily > 0].head(1)
    highest = daily.tail(1)
    return pd.DataFrame(
        [
            {"label": "highest_sales_day", "date": highest.index[0], "sales": highest.iloc[0]},
            {"label": "lowest_non_zero_sales_day", "date": lowest.index[0], "sales": lowest.iloc[0]},
        ]
    )


def store_weekly_correlation(long_df: pd.DataFrame) -> pd.DataFrame:
    weekly = (
        long_df.assign(week_start=long_df["date"].dt.to_period("W").dt.start_time)
        .groupby(["week_start", "store_id"])["sales"]
        .sum()
        .reset_index()
        .pivot(index="week_start", columns="store_id", values="sales")
    )
    return weekly.corr()
