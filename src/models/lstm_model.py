"""
src/models/lstm_model.py
------------------------
LSTM baseline model using PyTorch.
2-layer stacked LSTM encoder with Huber loss.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

MODELS_DIR = Path("outputs/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

ENCODER_LENGTH = 90
DECODER_LENGTH = 28


class SlidingWindowDataset(Dataset):
    """Sliding window dataset for LSTM encoder-decoder training."""

    def __init__(self, features, targets):
        """
        Parameters
        ----------
        features : np.ndarray  shape (N, ENCODER_LENGTH, n_features)
        targets  : np.ndarray  shape (N, DECODER_LENGTH)
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class LSTMForecaster(nn.Module):
    """2-layer stacked LSTM encoder → FC decoder."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        forecast_horizon: int = DECODER_LENGTH,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, forecast_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers, batch, hidden_size) — take last layer
        out = self.fc(h_n[-1])  # (batch, forecast_horizon)
        return out


def train_lstm(
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_size: int,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    max_epochs: int = 100,
    patience: int = 10,
    device: Optional[str] = None,
) -> LSTMForecaster:
    """Train LSTM with early stopping. Saves best checkpoint."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LSTMForecaster(input_size, hidden_size, num_layers, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.HuberLoss(delta=1.0)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None

    for epoch in range(max_epochs):
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                pred = model(x_batch)
                val_losses.append(criterion(pred, y_batch).item())

        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)
        scheduler.step(val_loss)
        logger.info(f"Epoch {epoch+1}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(best_state, MODELS_DIR / "lstm_best.pt")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model
