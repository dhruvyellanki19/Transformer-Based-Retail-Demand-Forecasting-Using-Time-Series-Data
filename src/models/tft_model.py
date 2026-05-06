"""
src/models/tft_model.py
-----------------------
Temporal Fusion Transformer (TFT) — primary model.
Uses pytorch-forecasting's TemporalFusionTransformer with quantile outputs.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import lightning.pytorch as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

MODELS_DIR = Path("outputs/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TIME_VARYING_KNOWN_REALS = [
    "time_idx",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
    "sell_price",
    "is_snap",
    "is_event",
    "is_cultural",
    "is_national",
    "is_religious",
    "is_sporting",
    "price_vs_category_mean",
    "is_price_reduced",
]

TIME_VARYING_UNKNOWN_REALS = [
    "sales",
    "lag_28",
    "lag_35",
    "lag_42",
    "lag_56",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_std_28",
]

STATIC_CATEGORICALS = ["store_id", "dept_id", "cat_id", "state_id"]


def build_timeseries_dataset(
    df: pd.DataFrame,
    max_encoder_length: int = 112,
    min_encoder_length: int = 56,
    max_prediction_length: int = 28,
    min_prediction_length: int = 28,
) -> TimeSeriesDataSet:
    """Build pytorch-forecasting TimeSeriesDataSet from feature DataFrame."""
    dataset = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="sales",
        group_ids=["id"],
        min_encoder_length=min_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=min_prediction_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=STATIC_CATEGORICALS,
        time_varying_known_reals=TIME_VARYING_KNOWN_REALS,
        time_varying_unknown_reals=TIME_VARYING_UNKNOWN_REALS,
        target_normalizer=GroupNormalizer(
            groups=["id"],
            transformation="softplus",
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
    )
    return dataset


def build_tft_model(
    training_dataset: TimeSeriesDataSet,
    hidden_size: int = 64,
    attention_head_size: int = 4,
    dropout: float = 0.1,
    hidden_continuous_size: int = 16,
    learning_rate: float = 3e-3,
    log_interval: int = 10,
) -> TemporalFusionTransformer:
    """Instantiate TFT model from a TimeSeriesDataSet."""
    model = TemporalFusionTransformer.from_dataset(
        training_dataset,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        output_size=7,  # 7 quantiles
        loss=QuantileLoss(),
        learning_rate=learning_rate,
        log_interval=log_interval,
        reduce_on_plateau_patience=4,
    )
    return model


def train_tft(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: Optional[TemporalFusionTransformer] = None,
    training_dataset: Optional[TimeSeriesDataSet] = None,
    max_epochs: int = 50,
    gradient_clip_val: float = 0.1,
) -> TemporalFusionTransformer:
    """Train TFT using PyTorch Lightning with model checkpointing."""
    if model is None and training_dataset is None:
        raise ValueError("Provide either model or training_dataset")
    if model is None:
        model = build_tft_model(training_dataset)

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=str(MODELS_DIR),
            filename="tft_best",
            monitor="val_loss",
            save_top_k=1,
            mode="min",
        ),
        pl.callbacks.EarlyStopping(monitor="val_loss", patience=10, mode="min"),
    ]

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        gradient_clip_val=gradient_clip_val,
        callbacks=callbacks,
        enable_progress_bar=True,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return model
