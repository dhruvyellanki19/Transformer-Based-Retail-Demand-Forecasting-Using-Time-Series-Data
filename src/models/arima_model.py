"""
src/models/arima_model.py
--------------------------
ARIMA / SARIMA baseline helpers for the Phase 6 notebook.

The notebook is the primary user-facing deliverable. This module keeps the
baseline logic reusable and testable so notebook cells stay readable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)

PREDICTIONS_DIR = Path("outputs/predictions")
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
CONFIGS_DIR = Path("configs")
CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

ARIMA_SAMPLE_PATH = CONFIGS_DIR / "arima_sample_ids.txt"
ARIMA_VAL_PREDS_PATH = PREDICTIONS_DIR / "arima_val_preds.csv"
ARIMA_TEST_PREDS_PATH = PREDICTIONS_DIR / "arima_test_preds.csv"
ARIMA_ORDER_LOG_PATH = PREDICTIONS_DIR / "arima_orders.csv"
ARIMA_FAILURES_PATH = PREDICTIONS_DIR / "arima_failures.csv"
ARIMA_EXOG_COLUMNS = ["is_weekend", "is_event", "is_snap", "sell_price"]


def build_series_metadata(train_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize one row per series for Phase 6 stratified sampling."""
    required = {"id", "state_id", "cat_id", "dept_id", "sales"}
    missing = required - set(train_df.columns)
    assert not missing, f"train_df missing columns: {sorted(missing)}"

    meta = (
        train_df.groupby("id")
        .agg(
            state_id=("state_id", "first"),
            cat_id=("cat_id", "first"),
            dept_id=("dept_id", "first"),
            mean_sales=("sales", "mean"),
            zero_ratio=("sales", lambda s: float((s == 0).mean())),
        )
        .reset_index()
    )

    ranks = meta["mean_sales"].rank(method="first")
    meta["velocity_bucket"] = pd.qcut(ranks, q=3, labels=["low", "mid", "high"]).astype(str)
    return meta


def filter_non_sparse_series(metadata_df: pd.DataFrame, max_sparsity: float = 0.70) -> pd.DataFrame:
    """Keep only series with sparsity below the configured threshold."""
    eligible = metadata_df.loc[metadata_df["zero_ratio"] < max_sparsity].copy()
    assert not eligible.empty, "No eligible series remain after sparsity filtering"
    ranks = eligible["mean_sales"].rank(method="first")
    eligible["velocity_bucket"] = pd.qcut(ranks, q=3, labels=["low", "mid", "high"]).astype(str)
    return eligible


def select_stratified_sample_ids(
    metadata_df: pd.DataFrame,
    sample_size: int = 200,
    random_state: int = 42,
) -> list[str]:
    """
    Select a deterministic stratified sample with coverage guarantees.

    Coverage goals:
    - all states
    - all categories
    - all departments
    - low / mid / high velocity buckets
    """
    assert sample_size > 0, "sample_size must be positive"
    eligible = metadata_df.copy()
    assert len(eligible) >= sample_size, "Not enough eligible series to sample from"

    rng = np.random.default_rng(random_state)
    selected_ids: list[str] = []
    selected_set: set[str] = set()

    def pick_one(mask: pd.Series) -> None:
        pool = eligible.loc[mask & ~eligible["id"].isin(selected_set)]
        assert not pool.empty, "Coverage requirement cannot be satisfied with current eligible pool"
        chosen = pool.sample(n=1, random_state=int(rng.integers(0, 1_000_000)))
        series_id = str(chosen.iloc[0]["id"])
        selected_ids.append(series_id)
        selected_set.add(series_id)

    for state in sorted(eligible["state_id"].unique()):
        pick_one(eligible["state_id"] == state)
    for cat in sorted(eligible["cat_id"].unique()):
        pick_one(eligible["cat_id"] == cat)
    for dept in sorted(eligible["dept_id"].unique()):
        pick_one(eligible["dept_id"] == dept)
    for bucket in sorted(eligible["velocity_bucket"].unique().tolist()):
        pick_one(eligible["velocity_bucket"] == bucket)

    remaining = sample_size - len(selected_ids)
    if remaining <= 0:
        return selected_ids[:sample_size]

    residual = eligible.loc[~eligible["id"].isin(selected_set)].copy()
    residual["state_cat"] = residual["state_id"].astype(str) + "_" + residual["cat_id"].astype(str)
    counts = residual["state_cat"].value_counts().sort_index()
    proportional = counts / counts.sum() * remaining
    base = np.floor(proportional).astype(int)
    remainder = remaining - int(base.sum())
    if remainder > 0:
        fractional = (proportional - base).sort_values(ascending=False)
        for stratum in fractional.index[:remainder]:
            base.loc[stratum] += 1

    for stratum, n_take in base.items():
        if n_take <= 0:
            continue
        pool = residual.loc[residual["state_cat"] == stratum]
        if pool.empty:
            continue
        take = min(n_take, len(pool))
        chosen = pool.sample(n=take, random_state=int(rng.integers(0, 1_000_000)))
        for series_id in chosen["id"].tolist():
            if series_id not in selected_set:
                selected_ids.append(str(series_id))
                selected_set.add(str(series_id))

    if len(selected_ids) < sample_size:
        leftover = residual.loc[~residual["id"].isin(selected_set)]
        chosen = leftover.sample(n=sample_size - len(selected_ids), random_state=int(rng.integers(0, 1_000_000)))
        selected_ids.extend([str(x) for x in chosen["id"].tolist()])

    sample_meta = eligible.loc[eligible["id"].isin(selected_ids)]
    assert sample_meta["state_id"].nunique() == eligible["state_id"].nunique()
    assert sample_meta["cat_id"].nunique() == eligible["cat_id"].nunique()
    assert sample_meta["dept_id"].nunique() == eligible["dept_id"].nunique()
    assert set(sample_meta["velocity_bucket"].unique()) == set(eligible["velocity_bucket"].unique())
    return selected_ids[:sample_size]


def save_sample_ids(sample_ids: list[str], output_path: str | Path = ARIMA_SAMPLE_PATH) -> Path:
    """Persist sampled IDs in the plan-required txt format."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sample_ids) + "\n")
    return path


def fit_auto_arima(series: pd.Series, exogenous: pd.DataFrame | None = None, m: int = 7):
    """Fit pmdarima auto_arima to a single series with optional exogenous features."""
    import pmdarima as pm

    model = pm.auto_arima(
        series,
        X=exogenous,
        seasonal=True,
        m=m,
        max_p=3,
        max_q=3,
        max_P=2,
        max_Q=2,
        d=None,
        information_criterion="aic",
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
    )
    return model


def extract_model_summary(model: Any, series_id: str) -> dict[str, Any]:
    """Extract order metadata from a fitted pmdarima model."""
    return {
        "id": series_id,
        "order": tuple(model.order),
        "seasonal_order": tuple(model.seasonal_order),
        "aic": float(model.aic()),
    }


def fit_sarima_with_orders(
    series: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    exogenous: pd.DataFrame | None = None,
):
    """Refit a SARIMAX model with fixed orders and optional exogenous regressors."""
    import pmdarima as pm

    model = pm.ARIMA(order=order, seasonal_order=seasonal_order, suppress_warnings=True)
    return model.fit(series, X=exogenous)


def forecast_nonnegative(model: Any, steps: int = 28, exogenous: pd.DataFrame | None = None) -> np.ndarray:
    """Generate a clipped non-negative forecast."""
    forecast = np.asarray(model.predict(n_periods=steps, X=exogenous), dtype=float)
    return np.maximum(forecast, 0.0)


def build_prediction_frame(
    series_id: str,
    dates: pd.Series,
    predicted: np.ndarray,
    actual: pd.Series,
) -> pd.DataFrame:
    """Build a standard prediction dataframe."""
    return pd.DataFrame(
        {
            "id": series_id,
            "date": pd.to_datetime(dates).values,
            "predicted": np.asarray(predicted, dtype=float),
            "actual": np.asarray(actual, dtype=float),
        }
    )


def summarize_order_modes(order_log_df: pd.DataFrame) -> dict[str, Any]:
    """Return the modal p,d,q and seasonal orders across successfully fit series."""
    order_parts = pd.DataFrame(order_log_df["order"].tolist(), columns=["p", "d", "q"])
    seasonal_parts = pd.DataFrame(order_log_df["seasonal_order"].tolist(), columns=["P", "D", "Q", "m"])
    mode_summary = {
        "p_mode": int(order_parts["p"].mode().iloc[0]),
        "d_mode": int(order_parts["d"].mode().iloc[0]),
        "q_mode": int(order_parts["q"].mode().iloc[0]),
        "P_mode": int(seasonal_parts["P"].mode().iloc[0]),
        "D_mode": int(seasonal_parts["D"].mode().iloc[0]),
        "Q_mode": int(seasonal_parts["Q"].mode().iloc[0]),
        "m_mode": int(seasonal_parts["m"].mode().iloc[0]),
    }
    mode_summary["order_mode"] = (mode_summary["p_mode"], mode_summary["d_mode"], mode_summary["q_mode"])
    mode_summary["seasonal_order_mode"] = (
        mode_summary["P_mode"],
        mode_summary["D_mode"],
        mode_summary["Q_mode"],
        mode_summary["m_mode"],
    )
    return mode_summary


def compute_group_metrics(
    predictions_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    """Compute MAE/RMSE/MAPE by a metadata grouping column."""
    merged = predictions_df.merge(metadata_df[["id", group_col]], on="id", how="left")
    rows: list[dict[str, Any]] = []
    for group_value, grp in merged.groupby(group_col):
        metrics = compute_all_metrics(grp["actual"].values, grp["predicted"].values)
        rows.append({group_col: group_value, **metrics, "rows": int(len(grp))})
    return pd.DataFrame(rows).sort_values(group_col).reset_index(drop=True)


def run_arima_baseline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sample_ids: list[str],
    exog_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fit the Phase 6 ARIMA/SARIMA baseline over the sampled series.

    Returns
    -------
    order_log_df, failures_df, val_preds_df, test_preds_df
    """
    order_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    val_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []
    exog_cols = exog_cols or ARIMA_EXOG_COLUMNS

    for series_id in sample_ids:
        series_train = train_df.loc[train_df["id"] == series_id].sort_values("date")["sales"].astype(float)
        train_series_df = train_df.loc[train_df["id"] == series_id].sort_values("date")
        series_val = val_df.loc[val_df["id"] == series_id].sort_values("date")
        series_test = test_df.loc[test_df["id"] == series_id].sort_values("date")
        train_exog = train_series_df[exog_cols].astype(float)
        val_exog = series_val[exog_cols].astype(float)
        test_exog = series_test[exog_cols].astype(float)

        try:
            auto_model = fit_auto_arima(series_train, exogenous=train_exog)
            summary = extract_model_summary(auto_model, series_id)
            summary["exog_cols"] = ",".join(exog_cols)
            order_rows.append(summary)

            val_model = fit_sarima_with_orders(
                series_train,
                summary["order"],
                summary["seasonal_order"],
                exogenous=train_exog,
            )
            val_pred = forecast_nonnegative(val_model, steps=len(series_val), exogenous=val_exog)
            val_frames.append(build_prediction_frame(series_id, series_val["date"], val_pred, series_val["sales"]))

            train_plus_val = pd.concat([series_train, series_val["sales"].astype(float)], ignore_index=True)
            train_plus_val_exog = pd.concat([train_exog, val_exog], ignore_index=True)
            test_model = fit_sarima_with_orders(
                train_plus_val,
                summary["order"],
                summary["seasonal_order"],
                exogenous=train_plus_val_exog,
            )
            test_pred = forecast_nonnegative(test_model, steps=len(series_test), exogenous=test_exog)
            test_frames.append(build_prediction_frame(series_id, series_test["date"], test_pred, series_test["sales"]))
        except Exception as exc:  # pragma: no cover - runtime-heavy failures handled in notebook
            logger.warning("ARIMA failed for %s: %s", series_id, exc)
            failures.append({"id": series_id, "error": str(exc)})

    order_log_df = pd.DataFrame(order_rows)
    failures_df = pd.DataFrame(failures)
    val_preds_df = pd.concat(val_frames, ignore_index=True) if val_frames else pd.DataFrame(columns=["id", "date", "predicted", "actual"])
    test_preds_df = pd.concat(test_frames, ignore_index=True) if test_frames else pd.DataFrame(columns=["id", "date", "predicted", "actual"])

    order_log_df.to_csv(ARIMA_ORDER_LOG_PATH, index=False)
    val_preds_df.to_csv(ARIMA_VAL_PREDS_PATH, index=False)
    test_preds_df.to_csv(ARIMA_TEST_PREDS_PATH, index=False)
    failures_df.to_csv(ARIMA_FAILURES_PATH, index=False)
    return order_log_df, failures_df, val_preds_df, test_preds_df
