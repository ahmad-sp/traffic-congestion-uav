"""
Autoencoder Anomaly Detector.

Detects Case A (off-peak sudden jam) by measuring reconstruction error.
Trained ONLY on normal traffic data — anomalous patterns produce high error.
"""

import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config

logger = logging.getLogger(__name__)


class TrafficAutoencoder(nn.Module):
    """
    Encoder: 10 → 32 → 16 → 8  (bottleneck)
    Decoder: 8 → 16 → 32 → 10

    Input/output shape: (batch, 10)
    """

    def __init__(self, input_dim: int = config.AE_INPUT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-sample MSE reconstruction error."""
        with torch.no_grad():
            reconstructed = self.forward(x)
            # Per-sample MSE: mean across features
            error = ((x - reconstructed) ** 2).mean(dim=1)
        return error


class AnomalyDetector:
    """Wraps autoencoder + threshold for inference."""

    def __init__(self, model: TrafficAutoencoder, threshold: float, device: str = "cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.threshold = threshold
        self.device = device

    def predict(self, features: torch.Tensor) -> tuple[float, bool]:
        """
        Args:
            features: (10,) or (1, 10) tensor
        Returns:
            (reconstruction_error, is_anomaly)
        """
        if features.dim() == 1:
            features = features.unsqueeze(0)
        features = features.to(self.device)
        error = self.model.reconstruction_error(features).item()
        return error, error > self.threshold


def load_autoencoder(
    model_path: Path | None = None,
    threshold_path: Path | None = None,
    device: str = "cpu",
) -> AnomalyDetector:
    """Load trained autoencoder and anomaly threshold."""
    if model_path is None:
        model_path = config.MODEL_DIR / "autoencoder.pt"
    if threshold_path is None:
        threshold_path = config.MODEL_DIR / "ae_threshold.json"

    model = TrafficAutoencoder()

    if model_path.exists():
        state = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        logger.info("Loaded autoencoder from %s", model_path)
    else:
        logger.warning("No autoencoder found at %s — using untrained model", model_path)

    threshold = 1.0  # fallback
    if threshold_path.exists():
        with open(threshold_path) as f:
            data = json.load(f)
        threshold = data["anomaly_threshold"]
        logger.info("Anomaly threshold: %.6f (from %s)", threshold, threshold_path)
    else:
        logger.warning("No threshold file at %s — using default %.2f", threshold_path, threshold)

    return AnomalyDetector(model, threshold, device)
