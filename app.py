"""
app.py
------
Streamlit dashboard for Transformer-Based Retail Demand Forecasting.

Loads pre-computed prediction CSVs from outputs/predictions/ and lets users:
  - Pick any series ID
  - Choose Validation or Test split
  - Toggle which models to show
  - See actual vs predicted charts, per-series metrics, overall comparison,
    residual analysis, and TFT probabilistic metrics.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import compute_all_metrics, compute_tft_probabilistic_metrics

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PREDS_DIR = ROOT / "outputs" / "predictions"

PRED_FILES = {
    ("ARIMA", "Validation"): PREDS_DIR / "arima_val_preds.csv",
    ("ARIMA", "Test"):       PREDS_DIR / "arima_test_preds.csv",
    ("LSTM", "Validation"):  PREDS_DIR / "lstm_val_preds.csv",
    ("LSTM", "Test"):        PREDS_DIR / "lstm_test_preds.csv",
    ("Informer", "Validation"): PREDS_DIR / "informer_val_preds.csv",
    ("Informer", "Test"):       PREDS_DIR / "informer_test_preds.csv",
    ("TFT", "Validation"):   PREDS_DIR / "tft_val_preds.csv",
    ("TFT", "Test"):         PREDS_DIR / "tft_test_preds.csv",
}

MODEL_COLORS = {
    "ARIMA":    "#EF553B",
    "LSTM":     "#4C78A8",
    "Informer": "#F58518",
    "TFT":      "#54A24B",
}

VAL_START = pd.Timestamp("2016-03-28")
TEST_START = pd.Timestamp("2016-04-25")

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading prediction files…")
def load_all_predictions() -> dict:
    """Load all prediction CSVs and normalise to id/date/predicted/actual columns."""
    data = {}
    for (model, split), path in PRED_FILES.items():
        if not path.exists():
            data[(model, split)] = None
            continue

        df = pd.read_csv(path)

        # TFT has 'step' instead of 'date', and quantile columns
        if "p50" in df.columns and "predicted" not in df.columns:
            df["predicted"] = df["p50"]
        if "date" not in df.columns and "step" in df.columns:
            base = VAL_START if split == "Validation" else TEST_START
            df["date"] = base + pd.to_timedelta(df["step"], unit="D")

        df["date"] = pd.to_datetime(df["date"])
        df["predicted"] = df["predicted"].clip(lower=0)
        df["actual"] = df["actual"].clip(lower=0)
        df["model"] = model
        df["split"] = split
        data[(model, split)] = df

    return data


@st.cache_data(show_spinner=False)
def build_id_index(all_ids: tuple) -> pd.DataFrame:
    """Build a lookup DataFrame with parsed M5 metadata columns from series IDs."""
    rows = []
    for sid in all_ids:
        p = sid.split("_")
        if len(p) < 5:
            continue
        rows.append({
            "id":       sid,
            "category": p[0],
            "dept":     f"{p[0]}_{p[1]}",
            "state":    p[3],
            "store":    f"{p[3]}_{p[4]}",
            "item":     p[2],
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def get_available_ids(preds: dict) -> list[str]:
    """Union of series IDs across deep models (LSTM/Informer/TFT)."""
    ids: set[str] = set()
    for (model, _), df in preds.items():
        if df is not None and model in ("LSTM", "Informer", "TFT"):
            ids.update(df["id"].unique().tolist())
    return sorted(ids)


@st.cache_data(show_spinner=False)
def load_model_comparison() -> pd.DataFrame | None:
    path = PREDS_DIR / "model_comparison.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data(show_spinner=False)
def load_tft_probabilistic() -> pd.DataFrame | None:
    path = PREDS_DIR / "tft_probabilistic_metrics.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def parse_series_info(series_id: str) -> dict:
    """Parse metadata from M5 series ID like FOODS_1_017_CA_1_evaluation."""
    parts = series_id.split("_")
    info = {"id": series_id}
    if len(parts) >= 1:
        info["category"] = parts[0]
    if len(parts) >= 2:
        info["dept"] = f"{parts[0]}_{parts[1]}"
    if len(parts) >= 5:
        info["store"] = f"{parts[3]}_{parts[4]}"
        info["state"] = parts[3]
    return info


def compute_per_series_metrics(preds: dict, series_id: str, split: str, models: list[str]) -> pd.DataFrame:
    rows = []
    for model in models:
        df = preds.get((model, split))
        if df is None:
            continue
        sub = df[df["id"] == series_id]
        if sub.empty:
            rows.append({"Model": model, "MAE": None, "RMSE": None, "sMAPE": None, "N": 0})
            continue
        m = compute_all_metrics(sub["actual"].to_numpy(), sub["predicted"].to_numpy())
        rows.append({
            "Model": model,
            "MAE": round(m["MAE"], 4),
            "RMSE": round(m["RMSE"], 4),
            "sMAPE (%)": round(m["sMAPE"], 2),
            "N points": len(sub),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot builders
# ---------------------------------------------------------------------------

def build_forecast_chart(preds: dict, series_id: str, split: str, models: list[str], show_tft_band: bool) -> go.Figure:
    fig = go.Figure()

    # Draw actual once (from LSTM or first available model)
    actual_drawn = False
    for model in ("LSTM", "Informer", "ARIMA", "TFT"):
        df = preds.get((model, split))
        if df is None:
            continue
        sub = df[df["id"] == series_id]
        if sub.empty:
            continue
        sub = sub.sort_values("date")
        fig.add_trace(go.Scatter(
            x=sub["date"], y=sub["actual"],
            mode="lines",
            name="Actual",
            line=dict(color="#111111", width=2),
            showlegend=not actual_drawn,
        ))
        actual_drawn = True
        break

    # TFT confidence band
    if "TFT" in models and show_tft_band:
        tft_df = preds.get(("TFT", split))
        if tft_df is not None:
            src = tft_df
            if "p10" in src.columns and "p90" in src.columns:
                sub = src[src["id"] == series_id].sort_values("date")
                if not sub.empty:
                    fig.add_trace(go.Scatter(
                        x=pd.concat([sub["date"], sub["date"].iloc[::-1]]),
                        y=pd.concat([sub["p90"], sub["p10"].iloc[::-1]]),
                        fill="toself",
                        fillcolor="rgba(84,162,75,0.15)",
                        line=dict(color="rgba(255,255,255,0)"),
                        name="TFT p10–p90",
                        showlegend=True,
                    ))

    # Predicted lines per model
    for model in models:
        df = preds.get((model, split))
        if df is None:
            continue
        sub = df[df["id"] == series_id]
        if sub.empty:
            continue
        sub = sub.sort_values("date")
        fig.add_trace(go.Scatter(
            x=sub["date"], y=sub["predicted"],
            mode="lines",
            name=f"{model} predicted",
            line=dict(color=MODEL_COLORS.get(model, "#888888"), width=1.8, dash="dash"),
        ))

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Sales (units)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=40, b=40),
        height=420,
    )
    return fig


def build_residual_chart(preds: dict, series_id: str, split: str, models: list[str]) -> go.Figure:
    fig = go.Figure()
    for model in models:
        df = preds.get((model, split))
        if df is None:
            continue
        sub = df[df["id"] == series_id]
        if sub.empty:
            continue
        sub = sub.sort_values("date")
        residuals = sub["actual"].to_numpy() - sub["predicted"].to_numpy()
        fig.add_trace(go.Bar(
            x=sub["date"], y=residuals,
            name=model,
            marker_color=MODEL_COLORS.get(model, "#888888"),
            opacity=0.7,
        ))

    fig.add_hline(y=0, line_dash="dot", line_color="black", line_width=1)
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Residual (Actual − Predicted)",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=40, b=40),
        height=350,
    )
    return fig


def build_comparison_bar(comparison_df: pd.DataFrame, split: str, metric: str) -> go.Figure:
    subset = comparison_df[comparison_df["Split"] == split].copy()
    subset = subset.sort_values(metric)

    fig = go.Figure(go.Bar(
        x=subset["Model"],
        y=subset[metric],
        marker_color=[MODEL_COLORS.get(m, "#888") for m in subset["Model"]],
        text=[f"{v:.4f}" for v in subset[metric]],
        textposition="outside",
    ))
    fig.update_layout(
        yaxis_title=metric,
        xaxis_title="Model",
        margin=dict(l=30, r=20, t=30, b=30),
        height=320,
        yaxis=dict(rangemode="tozero"),
    )
    return fig


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

def render_series_info(series_id: str):
    info = parse_series_info(series_id)
    cols = st.columns(4)
    fields = [
        ("Category",  info.get("category", "—")),
        ("Department", info.get("dept",    "—")),
        ("Store",      info.get("store",   "—")),
        ("State",      info.get("state",   "—")),
    ]
    for col, (label, value) in zip(cols, fields):
        col.metric(label, value)


def render_overall_comparison(comparison_df: pd.DataFrame, tft_prob_df: pd.DataFrame | None, split: str):
    st.subheader("Overall Model Comparison (all series)")

    # Filter to selected split
    df_split = comparison_df[comparison_df["Split"] == split].copy()

    # Metric tabs
    metric_tab, bar_tab = st.tabs(["Metrics Table", "Bar Charts"])

    with metric_tab:
        styled = df_split.drop(columns=["Split"], errors="ignore").set_index("Model")
        st.dataframe(
            styled.style.highlight_min(axis=0, color="#d4edda").format("{:.4f}"),
            use_container_width=True,
        )

    with bar_tab:
        for metric in ["MAE", "RMSE", "sMAPE"]:
            if metric in df_split.columns:
                st.plotly_chart(
                    build_comparison_bar(comparison_df, split, metric),
                    use_container_width=True,
                    key=f"bar_{metric}_{split}",
                )

    # TFT probabilistic panel
    if tft_prob_df is not None:
        st.subheader("TFT Probabilistic Metrics")
        row = tft_prob_df[tft_prob_df["split"] == split]
        if not row.empty:
            r = row.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Interval Coverage (p10–p90)", f"{r.get('empirical_coverage_p10_p90', 'N/A'):.2%}",
                      help="Fraction of actuals inside the 80% prediction interval")
            c2.metric("Nominal Target", f"{r.get('nominal_interval_target', 0.8):.0%}")
            c3.metric("Mean Interval Width", f"{r.get('mean_width_p10_p90', 'N/A'):.4f}")
            c4.metric("Mean Pinball Loss", f"{r.get('mean_pinloss_over_quantiles', 'N/A'):.4f}")
        else:
            st.info(f"No TFT probabilistic metrics found for split '{split}'.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Retail Demand Forecasting",
        page_icon="📈",
        layout="wide",
    )

    st.title("Retail Demand Forecasting Dashboard")
    st.caption("M5 Competition · ARIMA · LSTM · Informer · TFT")

    # Load data
    preds = load_all_predictions()
    all_ids = get_available_ids(preds)
    comparison_df = load_model_comparison()
    tft_prob_df = load_tft_probabilistic()

    if not all_ids:
        st.error("No prediction files found in `outputs/predictions/`. Please run the model notebooks first.")
        return

    # -----------------------------------------------------------------------
    # Sidebar
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.header("Controls")

        split = st.radio("Data Split", ["Validation", "Test"], index=1)

        st.divider()
        st.markdown("**Select Series**")

        idx = build_id_index(tuple(all_ids))

        category = st.selectbox("Category", sorted(idx["category"].unique()))

        dept_options = sorted(idx[idx["category"] == category]["dept"].unique())
        dept = st.selectbox("Department", dept_options)

        state_options = sorted(idx[idx["dept"] == dept]["state"].unique())
        state = st.selectbox("State", state_options)

        store_options = sorted(
            idx[(idx["dept"] == dept) & (idx["state"] == state)]["store"].unique()
        )
        store = st.selectbox("Store", store_options)

        filtered_ids = idx[(idx["dept"] == dept) & (idx["store"] == store)]
        item_options = sorted(filtered_ids["item"].unique())
        item = st.selectbox("Item #", item_options)

        series_id = filtered_ids[filtered_ids["item"] == item]["id"].iloc[0]

        st.divider()

        available_models = []
        for model in ["ARIMA", "LSTM", "Informer", "TFT"]:
            df = preds.get((model, split))
            if df is not None and not df[df["id"] == series_id].empty:
                available_models.append(model)

        selected_models = st.multiselect(
            "Models to display",
            options=["ARIMA", "LSTM", "Informer", "TFT"],
            default=available_models,
        )

        show_tft_band = st.checkbox("Show TFT confidence band (p10–p90)", value=True)

    # -----------------------------------------------------------------------
    # Main content
    # -----------------------------------------------------------------------
    st.subheader(f"Series: `{series_id}`")
    render_series_info(series_id)

    # Check if any model has data for this series
    has_data = any(
        preds.get((m, split)) is not None and not preds[(m, split)][preds[(m, split)]["id"] == series_id].empty
        for m in selected_models
    )

    if not has_data:
        st.warning(f"No predictions available for **{series_id}** in the **{split}** split with the selected models.")
    else:
        # --- Actual vs Predicted ---
        st.subheader("Actual vs Predicted")
        forecast_fig = build_forecast_chart(preds, series_id, split, selected_models, show_tft_band)
        st.plotly_chart(forecast_fig, use_container_width=True)

        # --- Per-Series Metrics ---
        st.subheader("Per-Series Metrics")
        metrics_df = compute_per_series_metrics(preds, series_id, split, selected_models)
        if not metrics_df.empty:
            st.dataframe(
                metrics_df.set_index("Model")
                    .style.highlight_min(axis=0, color="#d4edda")
                    .format({
                        "MAE": "{:.4f}",
                        "RMSE": "{:.4f}",
                        "sMAPE (%)": "{:.2f}",
                    }),
                use_container_width=True,
            )

            # Mini bar chart for quick scan
            metric_choice = st.radio(
                "Metric to visualise", ["MAE", "RMSE", "sMAPE (%)"],
                horizontal=True, key="series_metric_radio"
            )
            fig_mini = go.Figure(go.Bar(
                x=metrics_df["Model"],
                y=metrics_df[metric_choice],
                marker_color=[MODEL_COLORS.get(m, "#888") for m in metrics_df["Model"]],
                text=[f"{v:.4f}" if v is not None else "N/A" for v in metrics_df[metric_choice]],
                textposition="outside",
            ))
            fig_mini.update_layout(
                yaxis_title=metric_choice,
                margin=dict(l=30, r=20, t=20, b=30),
                height=280,
                yaxis=dict(rangemode="tozero"),
            )
            st.plotly_chart(fig_mini, use_container_width=True)

        # --- TFT Quantile Bands for this series ---
        tft_df = preds.get(("TFT", split))
        if "TFT" in selected_models and tft_df is not None:
            tft_sub = tft_df[tft_df["id"] == series_id]
            if not tft_sub.empty and "p10" in tft_sub.columns:
                with st.expander("TFT Quantile Detail for this series"):
                    tft_sub = tft_sub.sort_values("date")
                    fig_q = go.Figure()
                    quantile_pairs = [("p02", "p98"), ("p10", "p90"), ("p25", "p75")]
                    alphas = [0.08, 0.13, 0.20]
                    for (lo, hi), alpha in zip(quantile_pairs, alphas):
                        if lo in tft_sub.columns and hi in tft_sub.columns:
                            fig_q.add_trace(go.Scatter(
                                x=pd.concat([tft_sub["date"], tft_sub["date"].iloc[::-1]]),
                                y=pd.concat([tft_sub[hi], tft_sub[lo].iloc[::-1]]),
                                fill="toself",
                                fillcolor=f"rgba(84,162,75,{alpha})",
                                line=dict(color="rgba(255,255,255,0)"),
                                name=f"{lo}–{hi}",
                            ))
                    fig_q.add_trace(go.Scatter(x=tft_sub["date"], y=tft_sub["actual"], mode="lines", name="Actual", line=dict(color="#111111", width=2)))
                    fig_q.add_trace(go.Scatter(x=tft_sub["date"], y=tft_sub["p50"], mode="lines", name="TFT p50", line=dict(color=MODEL_COLORS["TFT"], width=1.8, dash="dash")))
                    fig_q.update_layout(
                        xaxis_title="Date", yaxis_title="Sales",
                        hovermode="x unified",
                        height=380,
                        margin=dict(l=40, r=20, t=30, b=40),
                        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
                    )
                    st.plotly_chart(fig_q, use_container_width=True)

        # --- Residual Analysis ---
        st.subheader("Residual Analysis")
        residual_fig = build_residual_chart(preds, series_id, split, selected_models)
        st.plotly_chart(residual_fig, use_container_width=True)

    # --- Overall comparison (always shown) ---
    st.divider()
    if comparison_df is not None:
        render_overall_comparison(comparison_df, tft_prob_df, split)
    else:
        st.info("`outputs/predictions/model_comparison.csv` not found. Run the evaluation notebook to generate it.")


if __name__ == "__main__":
    main()
