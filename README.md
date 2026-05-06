# Transformer-Based Retail Demand Forecasting

End-to-end demand forecasting pipeline on the [M5 Walmart competition dataset](https://www.kaggle.com/competitions/m5-forecasting-accuracy) — 30,000 store-item time series, five years of daily sales history, 28-day forecast horizon.

Four models are implemented and compared: a classical **ARIMA/SARIMA** baseline, an **LSTM**, an **Informer** Transformer, and a **Temporal Fusion Transformer (TFT)**. A Streamlit dashboard lets you interactively explore all model predictions and metrics.

---

## Project Summary

| | |
|---|---|
| Dataset | M5 Forecasting — Accuracy (Kaggle) |
| Series | ~30,000 store × item combinations |
| Forecast horizon | 28 days |
| Primary model | Temporal Fusion Transformer (TFT) |
| Baselines | ARIMA/SARIMA · LSTM · Informer |
| Evaluation metrics | MAE · RMSE · sMAPE · WRMSSE |
| Dashboard | Streamlit (`app.py`) |

---

## Key EDA Findings

- Overall daily-sales sparsity is **68.2%** — heavy zero inflation in bottom-level series.
- **FOODS** is the densest category with the strongest weekly seasonality; HOBBIES and HOUSEHOLD are more intermittent.
- Event days lift aggregate demand by **−5.3%** versus non-event days, supporting explicit event features.
- SNAP effects are strongest in WI, reinforcing state-specific benefit flags.
- Price-demand correlation is most negative in HOUSEHOLD, supporting price elasticity features.
- Major structural breaks appear around 2014-11-30 and 2015-07 — rolling features and calendar effects capture this.

---

## Repository Layout

```
├── app.py                      # Streamlit dashboard
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   ├── 02_arima.ipynb          # ARIMA/SARIMA baseline
│   ├── 03_lstm.ipynb           # LSTM training and evaluation
│   ├── 04_informer.ipynb       # Informer Transformer training
│   ├── 05_tft.ipynb            # TFT training, Optuna tuning, interpretability
│   └── 06_evaluation.ipynb     # Cross-model comparison and plots
├── src/
│   ├── data/                   # loader, preprocessor, feature engineering, EDA helpers
│   ├── eda/                    # EDA module
│   ├── models/                 # arima_model, lstm_model, informer_model, tft_model
│   └── evaluation/             # metrics: MAE, RMSE, MAPE, sMAPE, WRMSSE, pinball
├── configs/
│   ├── tft_config.yaml         # TFT hyperparameters (Optuna-tuned)
│   ├── arima_sample_ids.txt    # 251 stratified series used for ARIMA baseline
│   └── feature_category_mappings.json
├── tests/                      # Pytest unit and integration tests
├── data/
│   ├── raw/                    # M5 CSVs (not tracked — download from Kaggle)
│   ├── processed/              # sales_clean.parquet (not tracked)
│   └── features/               # train · val · test parquet splits (not tracked)
├── outputs/
│   ├── models/                 # Saved checkpoints (not tracked)
│   ├── predictions/            # Forecast CSVs per model (not tracked)
│   └── figures/                # EDA and evaluation plots
├── requirements.txt
└── pyproject.toml
```

---

## Setup

### 1. Clone and create environment

```bash
git clone https://github.com/dhruvyellanki19/Transformer-Based-Retail-Demand-Forecasting-Using-Time-Series-Data.git
cd Transformer-Based-Retail-Demand-Forecasting-Using-Time-Series-Data

conda create -n retail-forecast python=3.10
conda activate retail-forecast
pip install -r requirements.txt
```

### 2. Launch the dashboard

The prediction CSVs and model weights are pre-generated. Just run:

```bash
streamlit run app.py
```

Open `http://localhost:8501`. Use the sidebar to drill down by **Category → Department → State → Store → Item** and compare all four models on any series.

### 3. Explore the notebooks (optional)

```bash
jupyter lab
```

| Notebook | What it does |
|---|---|
| `01_eda.ipynb` | Full EDA — sparsity, seasonality, event lift, price analysis |
| `02_arima.ipynb` | ARIMA/SARIMA baseline on 251 stratified series |
| `03_lstm.ipynb` | 2-layer LSTM training, predictions, evaluation |
| `04_informer.ipynb` | Informer Transformer training and evaluation |
| `05_tft.ipynb` | TFT training, Optuna hyperparameter tuning, interpretability |
| `06_evaluation.ipynb` | Side-by-side model comparison with plots |

---

## Model Architecture

### ARIMA/SARIMA (`notebooks/02_arima.ipynb`)
- `pmdarima` auto-ARIMA with stepwise order selection
- Exogenous features: `is_weekend`, `is_event`, `is_snap`, `sell_price`
- Stratified sample of 251 series covering all states, categories, and velocity buckets

### LSTM (`notebooks/03_lstm.ipynb`)
- 2-layer stacked LSTM, `hidden_size=128`, Huber loss
- 90-day encoder window → FC layer → 28-day forecast
- Per-series mean normalisation + StandardScaler on features

### Informer (`notebooks/04_informer.ipynb`)
- Multi-head self-attention encoder (2 layers) with **convolutional distilling**
- Cross-attention decoder (1 layer)
- `seq_len=180`, `label_len=48`, `pred_len=28`, `d_model=256`, `n_heads=8`

### TFT — Primary Model (`notebooks/05_tft.ipynb`)
- `pytorch-forecasting` `TemporalFusionTransformer`
- Variable Selection Networks, LSTM local encoder, multi-head attention
- **7-quantile output** (p02 → p98) via `QuantileLoss`
- Optuna tuning: 50 trials over `hidden_size`, `attention_head_size`, `dropout`, `hidden_continuous_size`, `learning_rate`
- Best val loss: **0.559** at `hidden_size=32`, `dropout=0.139`, `lr=5.6e-4`

---

## Results (held-out test set)

| Model | MAE | RMSE | sMAPE |
|---|---|---|---|
| ARIMA | — | — | — |
| LSTM | **1.2055** | 2.8674 | 147.88% |
| Informer | 1.2056 | **2.7442** | **141.87%** |
| TFT (p50) | 1.3775 | 3.1287 | 157.75% |

> LSTM and Informer are evaluated as point forecasters. TFT is a probabilistic model — its p50 median is shown here for reference. TFT's primary value is its **7-quantile uncertainty output** (p02–p98) and variable importance interpretability, not point accuracy.

**TFT probabilistic metrics (test set):**
- Empirical coverage of p10–p90 interval: **46.4%**
- Mean interval width: **4.18**
- Mean pinball loss: **0.366**

---

## Dashboard Features

The Streamlit app (`app.py`) loads pre-computed prediction CSVs and provides:

- **Hierarchical series selector** — drill down by Category → Department → State → Store → Item
- **Actual vs Predicted chart** — all models overlaid, with optional TFT p10–p90 confidence band
- **Per-series metrics table** — MAE, RMSE, sMAPE computed live for the selected series
- **Residual analysis** — bar chart of (actual − predicted) per model per day
- **TFT quantile detail** — nested p02/p25/p75/p98 bands for the selected series
- **Overall comparison** — metrics table and bar charts across all 5,000 series

---

## Reproducibility

All training notebooks set seeds at the top:

```python
torch.manual_seed(42)
numpy.random.seed(42)
pytorch_lightning.seed_everything(42)
```

Package versions are fully pinned in `requirements.txt`.
