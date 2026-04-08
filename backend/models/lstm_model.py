"""
LSTM Congestion Forecaster.

Detects Case B (abnormally long peak-hour queue) by processing a 60-timestep
rolling window of per-minute traffic metrics and outputting a congestion score (0–1).
"""

import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config

logger = logging.getLogger(__name__)


class LSTMCongestionForecaster(nn.Module):
    """
    2-layer LSTM → dual FC heads → Sigmoid.

    Input shape:  (batch, seq_len=60, features=11)
    Output:
      congestion_score:        (batch, 1) in [0, 1]
      extreme_congestion_risk: (batch, 1) in [0, 1]
    """

    def __init__(
        self,
        num_features: int = config.LSTM_NUM_FEATURES,
        hidden_size: int = config.LSTM_HIDDEN_SIZE,
        num_layers: int = config.LSTM_NUM_LAYERS,
        dropout: float = config.LSTM_DROPOUT,
        fc_hidden: int = config.LSTM_FC_HIDDEN,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Head 1: congestion score (original)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, 1),
            nn.Sigmoid(),
        )

        # Head 2: extreme congestion risk (10-min lookahead)
        self.extreme_risk_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, num_features)
        Returns:
            (congestion_score, extreme_congestion_risk) — each (batch, 1) in [0, 1]
        """
        # lstm_out: (batch, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        # Take the last timestep's output
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)
        congestion_score = self.fc(last_hidden)
        extreme_risk = self.extreme_risk_head(last_hidden)
        return congestion_score, extreme_risk


def load_lstm_model(model_path: Path | None = None, device: str = "cpu") -> LSTMCongestionForecaster:
    """Load a trained LSTM model from disk."""
    if model_path is None:
        model_path = config.MODEL_DIR / "lstm_congestion.pt"

    model = LSTMCongestionForecaster()
    if model_path.exists():
        state = torch.load(model_path, map_location=device, weights_only=True)
        # Handle loading old single-head checkpoints (missing extreme_risk_head keys)
        model.load_state_dict(state, strict=False)
        logger.info("Loaded LSTM model from %s", model_path)
    else:
        logger.warning("No LSTM model found at %s — using untrained model", model_path)

    model.to(device)
    model.eval()
    return model
