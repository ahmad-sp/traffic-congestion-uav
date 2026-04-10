"""
Unified inference runner — runs both LSTM and Autoencoder models
on the latest per-minute metrics for each camera.
"""

import json
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
    "mean_bbox_growth_rate",  # index 10 — LSTM only, not passed to AE
]


class InferenceRunner:
    """
    Manages both ML models and provides a single inference entry point.
    Called once per minute per camera.

    Both models were trained on z-score normalized features.  Raw metric
    values (e.g. mean_bbox_area ~ 3000+) must be normalised with the stats
    saved during training before being forwarded to either model.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.lstm_model = load_lstm_model(device=device)
        self.anomaly_detector = load_autoencoder(device=device)

        # Per-camera rolling buffer: camera_id → list of raw feature vectors
        self._buffers: dict[str, list[np.ndarray]] = {}

        # Normalization stats — loaded once at startup
        self._ae_means, self._ae_stds = self._load_ae_norm_stats()
        self._lstm_stats, self._lstm_fallback_means, self._lstm_fallback_stds = (
            self._load_lstm_norm_stats()
        )

    # ------------------------------------------------------------------ #
    # Norm-stat loaders                                                    #
    # ------------------------------------------------------------------ #

    def _load_ae_norm_stats(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Load the global z-score stats saved by train_autoencoder.py.
        Returns (means, stds) arrays of length AE_INPUT_DIM (10).
        Falls back to identity (0, 1) so the runner stays functional even
        if the file is missing, though scores will be wrong.
        """
        stats_path = config.MODEL_DIR / "ae_norm_stats.json"
        n = config.AE_INPUT_DIM
        if not stats_path.exists():
            logger.warning("AE norm stats not found at %s — AE inputs will NOT be normalised", stats_path)
            return np.zeros(n, dtype=np.float32), np.ones(n, dtype=np.float32)

        with open(stats_path) as f:
            data = json.load(f)
        means = np.array(data["means"], dtype=np.float32)
        stds = np.array(data["stds"], dtype=np.float32)
        stds[stds == 0] = 1.0
        logger.info("Loaded AE norm stats from %s", stats_path)
        return means, stds

    def _load_lstm_norm_stats(
        self,
    ) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
        """
        Load the per-camera z-score stats saved by train_lstm.py.
        Returns:
          - per_camera: {camera_id: (means, stds)}  each shape (11,)
          - fallback_means, fallback_stds: average across all cameras,
            used for cameras that were not in the training set (e.g. new
            arms added via the admin panel).
        """
        n = len(FEATURE_COLUMNS)
        stats_path = config.MODEL_DIR / "lstm_norm_stats.json"
        if not stats_path.exists():
            logger.warning("LSTM norm stats not found at %s — LSTM inputs will NOT be normalised", stats_path)
            return {}, np.zeros(n, dtype=np.float32), np.ones(n, dtype=np.float32)

        with open(stats_path) as f:
            raw = json.load(f)

        per_camera: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        all_means, all_stds = [], []
        for cam_id, cam_stats in raw.items():
            m = np.array(cam_stats["means"], dtype=np.float32)
            s = np.array(cam_stats["stds"], dtype=np.float32)
            s[s == 0] = 1.0
            per_camera[cam_id] = (m, s)
            all_means.append(m)
            all_stds.append(s)

        if all_means:
            fallback_means = np.stack(all_means).mean(axis=0)
            fallback_stds = np.stack(all_stds).mean(axis=0)
        else:
            fallback_means = np.zeros(n, dtype=np.float32)
            fallback_stds = np.ones(n, dtype=np.float32)

        logger.info("Loaded LSTM norm stats for %d cameras from %s", len(per_camera), stats_path)
        return per_camera, fallback_means, fallback_stds

    def _get_lstm_norm(self, camera_id: str) -> tuple[np.ndarray, np.ndarray]:
        """Return the (means, stds) pair to use for this camera's LSTM input."""
        if camera_id in self._lstm_stats:
            return self._lstm_stats[camera_id]
        logger.debug("No LSTM norm stats for %s — using cross-camera average", camera_id)
        return self._lstm_fallback_means, self._lstm_fallback_stds

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def _get_buffer(self, camera_id: str) -> list[np.ndarray]:
        if camera_id not in self._buffers:
            self._buffers[camera_id] = []
        return self._buffers[camera_id]

    def push_metrics(self, camera_id: str, features: dict[str, float]) -> None:
        """
        Push one minute's raw metrics into the rolling buffer.
        Raw (un-normalised) values are stored; normalisation is applied
        inside run_inference just before each model call.
        """
        vec = np.array([features[col] for col in FEATURE_COLUMNS], dtype=np.float32)
        buf = self._get_buffer(camera_id)
        buf.append(vec)
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
                "extreme_congestion_risk": float (0–1),
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

        # ── Autoencoder (single timestep, 10 features) ─────────────────
        # Slice to AE_INPUT_DIM (10) — mean_bbox_growth_rate at index 10
        # is LSTM-only and was not part of AE training.
        # Apply the global z-score stats saved by train_autoencoder.py.
        ae_raw = buf[-1][:config.AE_INPUT_DIM]
        ae_norm = (ae_raw - self._ae_means) / self._ae_stds
        current_vec = torch.tensor(ae_norm, dtype=torch.float32)
        anomaly_score, is_anomaly = self.anomaly_detector.predict(current_vec)
        result["anomaly_score"] = anomaly_score
        result["is_anomaly"] = is_anomaly

        # ── LSTM (60-timestep sequence, 11 features) ────────────────────
        # Apply per-camera z-score stats saved by train_lstm.py.
        if len(buf) >= config.LSTM_SEQUENCE_LENGTH:
            sequence = np.stack(buf[-config.LSTM_SEQUENCE_LENGTH:])  # (60, 11) raw
            lstm_means, lstm_stds = self._get_lstm_norm(camera_id)
            sequence = (sequence - lstm_means) / lstm_stds             # (60, 11) normalised
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
