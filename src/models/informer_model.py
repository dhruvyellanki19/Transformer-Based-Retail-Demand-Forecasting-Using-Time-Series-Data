"""
src/models/informer_model.py
-----------------------------
Informer transformer baseline model.

Two implementations are provided:
1. InformerPlaceholder / train_informer  — original LSTM stand-in kept for
   backward compatibility.
2. InformerModel / InformerDataset / train_informer_full  — proper
   Transformer encoder-decoder with multi-head self-attention and
   convolutional distilling layers (the Informer's distinctive component).
"""

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

MODELS_DIR = Path("outputs/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class InformerConfig:
    """Hyperparameter configuration for the Informer model."""

    enc_in: int = 20  # Number of encoder input features
    dec_in: int = 20  # Number of decoder input features
    c_out: int = 1  # Output size (univariate)
    seq_len: int = 180  # Encoder input sequence length
    label_len: int = 48  # Decoder start token length
    pred_len: int = 28  # Forecast horizon
    d_model: int = 512  # Hidden size
    n_heads: int = 8  # Attention heads
    e_layers: int = 2  # Encoder layers
    d_layers: int = 1  # Decoder layers
    factor: int = 5  # ProbSparse sampling factor
    dropout: float = 0.05
    lr: float = 1e-4
    max_epochs: int = 30
    patience: int = 5


class InformerPlaceholder(nn.Module):
    """
    Placeholder Informer model.
    Replace with full ProbSparse attention implementation from:
    https://github.com/zhouhaoyi/Informer2020
    """

    def __init__(self, config: InformerConfig):
        super().__init__()
        self.config = config
        self.encoder = nn.LSTM(
            input_size=config.enc_in,
            hidden_size=config.d_model,
            num_layers=config.e_layers,
            batch_first=True,
        )
        self.decoder = nn.Linear(config.d_model, config.pred_len)

    def forward(self, enc_x: torch.Tensor, dec_x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        enc_x : Tensor shape (batch, seq_len, enc_in)
        dec_x : Tensor shape (batch, label_len + pred_len, dec_in)

        Returns
        -------
        Tensor shape (batch, pred_len, c_out)
        """
        _, (h_n, _) = self.encoder(enc_x)
        out = self.decoder(h_n[-1])  # (batch, pred_len)
        return out.unsqueeze(-1)  # (batch, pred_len, 1)


def train_informer(
    train_loader,
    val_loader,
    config: InformerConfig = None,
    device: str = None,
) -> InformerPlaceholder:
    """Train the Informer (placeholder) model with early stopping."""
    if config is None:
        config = InformerConfig()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = InformerPlaceholder(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None

    for epoch in range(config.max_epochs):
        model.train()
        for enc_x, dec_x, targets in train_loader:
            enc_x = enc_x.to(device)
            dec_x = dec_x.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            pred = model(enc_x, dec_x).squeeze(-1)
            loss = criterion(pred, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for enc_x, dec_x, targets in val_loader:
                enc_x, dec_x, targets = (
                    enc_x.to(device),
                    dec_x.to(device),
                    targets.to(device),
                )
                pred = model(enc_x, dec_x).squeeze(-1)
                val_losses.append(criterion(pred, targets).item())

        val_loss = sum(val_losses) / len(val_losses)
        logger.info(f"Epoch {epoch+1}: val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(best_state, MODELS_DIR / "informer_best.pt")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# Full Informer Implementation
# ---------------------------------------------------------------------------


class _PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.05):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class _MultiHeadAttention(nn.Module):
    """Scaled dot-product multi-head attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.05):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        B = query.size(0)
        Q = self.q_proj(query).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(key).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(value).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = (
            torch.matmul(attn, V)
            .transpose(1, 2)
            .contiguous()
            .view(B, -1, self.n_heads * self.d_k)
        )
        return self.out_proj(out)


class _EncoderLayer(nn.Module):
    """Informer encoder layer: self-attention + feed-forward + distilling."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.05):
        super().__init__()
        self.attn = _MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        # Convolutional distilling — Informer's distinctive component
        self.distil_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, distil: bool = True) -> torch.Tensor:
        x = self.norm1(x + self.drop(self.attn(x, x, x)))
        x = self.norm2(x + self.drop(self.ff(x)))
        if distil:
            xt = torch.relu(self.distil_conv(x.transpose(1, 2)))
            x = self.pool(xt).transpose(1, 2)
        return x


class _DecoderLayer(nn.Module):
    """Informer decoder layer: masked self-attention + cross-attention + FF."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.05):
        super().__init__()
        self.self_attn = _MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = _MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.drop(self.self_attn(x, x, x)))
        x = self.norm2(x + self.drop(self.cross_attn(x, enc_out, enc_out)))
        x = self.norm3(x + self.drop(self.ff(x)))
        return x


class InformerModel(nn.Module):
    """
    Informer Transformer with convolutional distilling encoder and
    cross-attention decoder.

    Parameters are taken from InformerConfig.
    """

    def __init__(self, config: InformerConfig):
        super().__init__()
        self.config = config
        d_ff = config.d_model * 4

        self.enc_embed = nn.Linear(config.enc_in, config.d_model)
        self.dec_embed = nn.Linear(config.dec_in, config.d_model)
        self.enc_pos = _PositionalEncoding(config.d_model, dropout=config.dropout)
        self.dec_pos = _PositionalEncoding(config.d_model, dropout=config.dropout)

        self.encoder = nn.ModuleList(
            [
                _EncoderLayer(config.d_model, config.n_heads, d_ff, config.dropout)
                for _ in range(config.e_layers)
            ]
        )
        self.decoder = nn.ModuleList(
            [
                _DecoderLayer(config.d_model, config.n_heads, d_ff, config.dropout)
                for _ in range(config.d_layers)
            ]
        )
        self.projection = nn.Linear(config.d_model, config.c_out)

    def forward(self, enc_x: torch.Tensor, dec_x: torch.Tensor) -> torch.Tensor:
        enc_out = self.enc_pos(self.enc_embed(enc_x))
        for i, layer in enumerate(self.encoder):
            enc_out = layer(enc_out, distil=(i < len(self.encoder) - 1))

        dec_out = self.dec_pos(self.dec_embed(dec_x))
        for layer in self.decoder:
            dec_out = layer(dec_out, enc_out)

        return self.projection(dec_out[:, -self.config.pred_len :, :])


class InformerDataset(Dataset):
    """
    Dataset for InformerModel.

    Each sample:
      enc_x  : (seq_len, enc_in)             — full encoder history
      dec_x  : (label_len + pred_len, dec_in) — label context + zero-padded future
      target : (pred_len,)                    — normalised future sales
    """

    # Features whose future values are known (calendar, price signals)
    _KNOWN_COLS: tuple = (
        "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend",
        "is_event", "is_snap", "is_cultural", "is_national", "is_religious",
        "is_sporting", "sell_price", "price_vs_category_mean", "is_price_reduced",
    )

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list,
        scale_dict: dict,
        seq_len: int = 180,
        label_len: int = 48,
        pred_len: int = 28,
        stride: int = 28,
    ):
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.feature_cols = feature_cols

        known_mask = np.array(
            [c in self._KNOWN_COLS for c in feature_cols], dtype=bool
        )
        self._unknown_mask = ~known_mask

        self.index: list = []
        self.series_data: dict = {}

        for sid, grp in df.groupby("id"):
            grp = grp.sort_values("date").reset_index(drop=True)
            feats = grp[feature_cols].fillna(0).values.astype(np.float32)
            sales = grp["sales"].values.astype(np.float32)
            scale = max(float(scale_dict.get(sid, 1.0)), 1.0)
            self.series_data[sid] = (feats, sales / scale)

            n = len(grp)
            max_start = n - seq_len - pred_len
            for start in range(0, max_start + 1, stride):
                self.index.append((sid, start))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        sid, start = self.index[idx]
        feats, sales_norm = self.series_data[sid]

        enc_end = start + self.seq_len
        dec_start = enc_end - self.label_len
        dec_end = enc_end + self.pred_len

        enc_x = torch.tensor(feats[start:enc_end], dtype=torch.float32)

        dec_arr = feats[dec_start:dec_end].copy()
        dec_arr[self.label_len :, self._unknown_mask] = 0.0
        dec_x = torch.tensor(dec_arr, dtype=torch.float32)

        target = torch.tensor(sales_norm[enc_end:dec_end], dtype=torch.float32)
        return enc_x, dec_x, target


def train_informer_full(
    train_loader,
    val_loader,
    config: InformerConfig = None,
    device: str = None,
) -> "InformerModel":
    """Train InformerModel with early stopping. Saves best checkpoint."""
    if config is None:
        config = InformerConfig()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = InformerModel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None

    for epoch in range(config.max_epochs):
        model.train()
        for enc_x, dec_x, targets in train_loader:
            enc_x, dec_x, targets = (
                enc_x.to(device),
                dec_x.to(device),
                targets.to(device),
            )
            optimizer.zero_grad()
            pred = model(enc_x, dec_x).squeeze(-1)
            loss = criterion(pred, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for enc_x, dec_x, targets in val_loader:
                enc_x, dec_x, targets = (
                    enc_x.to(device),
                    dec_x.to(device),
                    targets.to(device),
                )
                pred = model(enc_x, dec_x).squeeze(-1)
                val_losses.append(criterion(pred, targets).item())

        val_loss = float(np.mean(val_losses))
        logger.info(f"Epoch {epoch + 1}: val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(best_state, MODELS_DIR / "informer_best.pt")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model
