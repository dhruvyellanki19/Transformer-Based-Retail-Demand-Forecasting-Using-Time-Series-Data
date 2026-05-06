"""
src/evaluation/metrics.py
--------------------------
Evaluation metrics: MAE, RMSE, MAPE, sMAPE, WRMSSE, and TFT quantile / probabilistic helpers.
All functions are pure — no side effects or I/O.
"""

import numpy as np
import pandas as pd

# TFT / pytorch-forecasting quantile output names → nominal probability masses (τ in pinball loss).
TFT_QUANTILE_TAUS: dict[str, float] = {
    "p02": 0.02,
    "p10": 0.10,
    "p25": 0.25,
    "p50": 0.50,
    "p75": 0.75,
    "p90": 0.90,
    "p98": 0.98,
}
def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error."""
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual: np.ndarray, predicted: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error. Avoids division by zero with eps."""
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs((actual - predicted) / (np.abs(actual) + eps))) * 100)


def wrmsse(
    predictions_df: pd.DataFrame,
    train_df: pd.DataFrame,
    weights: np.ndarray,
) -> float:
    """
    Weighted Root Mean Squared Scaled Error (WRMSSE).

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Columns: id, actual, predicted.
    train_df : pd.DataFrame
        Training sales used to compute scaling denominator per series.
    weights : np.ndarray
        Per-series weights summing to 1 across all 12 aggregation levels.

    Returns
    -------
    float
        WRMSSE score.
    """
    # Compute scaling denominator: mean squared naive 1-step error per series
    denominators = {}
    for series_id, grp in train_df.groupby("id"):
        sales = grp.sort_values("date")["sales"].values
        naive_errors = np.diff(sales) ** 2
        denominators[series_id] = float(np.mean(naive_errors)) if len(naive_errors) > 0 else 1.0

    # Compute RMSSE per series
    scores = []
    series_ids = predictions_df["id"].unique()
    for i, series_id in enumerate(series_ids):
        grp = predictions_df[predictions_df["id"] == series_id]
        actual = grp["actual"].values
        pred = grp["predicted"].values
        mse = np.mean((actual - pred) ** 2)
        denom = denominators.get(series_id, 1.0)
        rmsse_val = np.sqrt(mse / max(denom, 1e-8))
        scores.append(rmsse_val * weights[i % len(weights)])

    return float(np.sum(scores))


def smape(actual: np.ndarray, predicted: np.ndarray, eps: float = 1e-8) -> float:
    """Symmetric Mean Absolute Percentage Error.

    Better handles zero-sales days than MAPE because the denominator uses
    the average of |actual| and |predicted| rather than |actual| alone.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    numerator = np.abs(actual - predicted)
    denominator = (np.abs(actual) + np.abs(predicted)) / 2.0 + eps
    return float(np.mean(numerator / denominator) * 100)


def pinball_loss(actual: np.ndarray, q_pred: np.ndarray, tau: float) -> float:
    """
    Mean pinball (quantile) loss for a single quantile level τ ∈ (0, 1).

    L_τ(y, q) = (τ - 1)(y - q) if y < q else τ(y - q).
    """
    y = np.asarray(actual, dtype=float)
    q = np.asarray(q_pred, dtype=float)
    diff = y - q
    loss = np.where(diff >= 0.0, tau * diff, (tau - 1.0) * diff)
    return float(np.mean(loss))


def empirical_interval_coverage(
    actual: np.ndarray,
    q_low: np.ndarray,
    q_high: np.ndarray,
) -> float:
    """Fraction of actual values in [q_low, q_high] (inclusive)."""
    y = np.asarray(actual, dtype=float)
    lo = np.asarray(q_low, dtype=float)
    hi = np.asarray(q_high, dtype=float)
    inside = (y >= lo) & (y <= hi)
    return float(np.mean(inside))


def mean_interval_width(q_low: np.ndarray, q_high: np.ndarray) -> float:
    """Mean width q_high - q_low."""
    lo = np.asarray(q_low, dtype=float)
    hi = np.asarray(q_high, dtype=float)
    return float(np.mean(hi - lo))


def compute_tft_probabilistic_metrics(
    df: pd.DataFrame,
    low_col: str = "p10",
    high_col: str = "p90",
    nominal_interval: float = 0.80,
) -> dict:
    """
    Aggregate probabilistic metrics for a TFT (or any) quantile CSV.

    Expects columns: ``actual``, ``low_col``, ``high_col``, and optionally
    any keys from ``TFT_QUANTILE_TAUS`` for mean pinball loss.

    Parameters
    ----------
    nominal_interval
        Target coverage for [p10, p90] (informative only; not used in computation).
    """
    required = {"actual", low_col, high_col}
    if df is None or len(df) == 0 or not required.issubset(df.columns):
        return {
            "n_rows": 0,
            "empirical_coverage_p10_p90": float("nan"),
            "mean_width_p10_p90": float("nan"),
            "nominal_interval_target": nominal_interval,
            "mean_pinloss_over_quantiles": float("nan"),
        }

    cols = ["actual", low_col, high_col]
    for c in TFT_QUANTILE_TAUS:
        if c in df.columns and c not in cols:
            cols.append(c)
    work = df[cols].copy()
    work = work.dropna()
    for c in work.columns:
        if c != "actual":
            work[c] = work[c].clip(lower=0.0)
    work["actual"] = work["actual"].astype(float)

    y = work["actual"].to_numpy()
    lo = work[low_col].to_numpy()
    hi = work[high_col].to_numpy()

    cover = empirical_interval_coverage(y, lo, hi)
    width = mean_interval_width(lo, hi)

    pinballs: list[float] = []
    for col, tau in TFT_QUANTILE_TAUS.items():
        if col in work.columns:
            pinballs.append(pinball_loss(y, work[col].to_numpy(), tau))

    mean_pin = float(np.mean(pinballs)) if pinballs else float("nan")

    return {
        "n_rows": int(len(work)),
        "empirical_coverage_p10_p90": round(cover, 4),
        "mean_width_p10_p90": round(width, 4),
        "nominal_interval_target": nominal_interval,
        "mean_pinloss_over_quantiles": round(mean_pin, 4),
    }


def compute_all_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict:
    """Return dict of MAE, RMSE, MAPE, sMAPE for convenience."""
    return {
        "MAE": mae(actual, predicted),
        "RMSE": rmse(actual, predicted),
        "MAPE": mape(actual, predicted),
        "sMAPE": smape(actual, predicted),
    }
