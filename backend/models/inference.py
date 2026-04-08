"""
Unified inference runner — runs both LSTM and Autoencoder models
on the latest per-minute metrics for each camera.
"""

import logging
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from backend.models.lstm_model import LSTMCongestionForecaster, load_lstm_model
from backend.models.autoencoder import AnomalyDetector, load_autoencoder

logger = logging.getLogger(__name__)

# Feature column order — must match training data exactly
FEATURE_COLUMNS = [
    "VPM",
    "queue_depth",
    "stopped_ratio",
    "occupancy_pct",
    "mean_bbox_area",
    "max_bbox_area",
    "approach_flow",
    "time_sin",
    "time_cos",
    "is_peak_hour",
    "mean_bbox_growth_rate",
]


class InferenceRunner:
    """
    Manages both ML models and provides a single inference entry point.
    Called once per minute per camera.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.lstm_model = load_lstm_model(device=device)
        self.anomaly_detector = load_autoencoder(device=device)

        # Per-camera rolling buffer: camera_id → list of feature vectors (most recent last)
        self._buffers: dict[str, list[np.ndarray]] = {}

    def _get_buffer(self, camera_id: str) -> list[np.ndarray]:
        if camera_id not in self._buffers:
            self._buffers[camera_id] = []
        return self._buffers[camera_id]

    def push_metrics(self, camera_id: str, features: dict[str, float]) -> None:
        """
        Push one minute's metrics into the rolling buffer.
        `features` must contain all keys in FEATURE_COLUMNS.
        """
        vec = np.array([features[col] for col in FEATURE_COLUMNS], dtype=np.float32)
        buf = self._get_buffer(camera_id)
        buf.append(vec)
        # Keep only the last LSTM_SEQUENCE_LENGTH entries
        if len(buf) > config.LSTM_SEQUENCE_LENGTH:
            self._buffers[camera_id] = buf[-config.LSTM_SEQUENCE_LENGTH:]

    def run_inference(self, camera_id: str) -> dict:
        """
        Run both models on the current data for a camera.

        Returns:
            {
                "lstm_score": float (0–1),
                "lstm_ready": bool,
                "anomaly_score": float,
                "is_anomaly": bool,
            }
        """
        buf = self._get_buffer(camera_id)
        result = {
            "lstm_score": 0.0,
            "lstm_ready": False,
            "anomaly_score": 0.0,
            "is_anomaly": False,
            "extreme_congestion_risk": 0.0,
        }

        if not buf:
            return result

        # --- Autoencoder (single timestep) ---
        current_vec = torch.tensor(buf[-1], dtype=torch.float32)
        anomaly_score, is_anomaly = self.anomaly_detector.predict(current_vec)
        result["anomaly_score"] = anomaly_score
        result["is_anomaly"] = is_anomaly

        # --- LSTM (needs full sequence) ---
        if len(buf) >= config.LSTM_SEQUENCE_LENGTH:
            sequence = np.stack(buf[-config.LSTM_SEQUENCE_LENGTH:])
            tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)  # (1, 60, 11)
            tensor = tensor.to(self.device)

            with torch.no_grad():
                congestion_score, extreme_risk = self.lstm_model(tensor)

            result["lstm_score"] = congestion_score.item()
            result["extreme_congestion_risk"] = extreme_risk.item()
            result["lstm_ready"] = True
        else:
            logger.debug(
                "LSTM not ready for %s: %d/%d timesteps",
                camera_id, len(buf), config.LSTM_SEQUENCE_LENGTH,
            )

        return result
