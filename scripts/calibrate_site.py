"""
Calibrate threshold artifacts for a new deployment site.

Reads a CSV of collected traffic data (normal + congested) from the new site
and updates three artifact files without retraining the models:

  1. lstm_norm_stats.json  — adds/updates per-camera z-score stats
  2. ae_threshold.json     — recalibrates anomaly threshold from local normal data
  3. hourly_baseline.json  — merges 85th-percentile VPM baseline for new junctions

Usage:
    python -m scripts.calibrate_site --data path/to/new_site.csv

The CSV must have the same columns as all_arms_combined.csv:
    timestamp, camera_id, junction_id, arm_id, label, hour_of_week,
    VPM, queue_depth, stopped_ratio, occupancy_pct, mean_bbox_area,
    max_bbox_area, approach_flow, time_sin, time_cos, is_peak_hour,
    mean_bbox_growth_rate, [extreme_congestion_future optional]

Restart the server after running this script for changes to take effect.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from backend.models.autoencoder import TrafficAutoencoder

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

LSTM_FEATURE_COLUMNS = [
    "VPM", "queue_depth", "stopped_ratio", "occupancy_pct",
    "mean_bbox_area", "max_bbox_area", "approach_flow",
    "time_sin", "time_cos", "is_peak_hour", "mean_bbox_growth_rate",
]

AE_FEATURE_COLUMNS = [
    "VPM", "queue_depth", "stopped_ratio", "occupancy_pct",
    "mean_bbox_area", "max_bbox_area", "approach_flow",
    "time_sin", "time_cos", "is_peak_hour",
]


# ── 1. LSTM norm stats ────────────────────────────────────────────────────────

def calibrate_lstm_norm_stats(df: pd.DataFrame) -> int:
    """
    Compute per-camera z-score stats from new site data and merge into
    lstm_norm_stats.json.  Existing cameras are overwritten; others are kept.
    Returns number of cameras updated.
    """
    stats_path = config.MODEL_DIR / "lstm_norm_stats.json"

    existing: dict = {}
    if stats_path.exists():
        with open(stats_path) as f:
            existing = json.load(f)

    updated = 0
    for camera_id, group in df.groupby("camera_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        features = group[LSTM_FEATURE_COLUMNS].values.astype(np.float32)
        means = features.mean(axis=0)
        stds = features.std(axis=0)
        stds[stds == 0] = 1.0
        existing[camera_id] = {"means": means.tolist(), "stds": stds.tolist()}
        logger.info("LSTM norm stats updated for camera: %s  (n=%d)", camera_id, len(features))
        updated += 1

    with open(stats_path, "w") as f:
        json.dump(existing, f, indent=2)
    logger.info("Saved lstm_norm_stats.json  (%d total cameras)", len(existing))
    return updated


# ── 2. AE threshold ───────────────────────────────────────────────────────────

def calibrate_ae_threshold(df: pd.DataFrame, device: str) -> float:
    """
    Run the existing AE model on NORMAL rows from the new site (using the
    global ae_norm_stats for normalisation — same stats the model was trained
    with) and recompute the anomaly threshold as the 95th percentile of local
    reconstruction errors.
    """
    ae_model_path = config.MODEL_DIR / "autoencoder.pt"
    ae_norm_path = config.MODEL_DIR / "ae_norm_stats.json"
    ae_threshold_path = config.MODEL_DIR / "ae_threshold.json"

    if not ae_model_path.exists():
        logger.error("AE model not found at %s — skipping threshold calibration", ae_model_path)
        return -1.0

    # Load model
    model = TrafficAutoencoder().to(device)
    model.load_state_dict(torch.load(ae_model_path, map_location=device))
    model.eval()

    # Load global norm stats (must match what the model was trained with)
    if ae_norm_path.exists():
        with open(ae_norm_path) as f:
            norm = json.load(f)
        ae_means = np.array(norm["means"], dtype=np.float32)
        ae_stds = np.array(norm["stds"], dtype=np.float32)
        ae_stds[ae_stds == 0] = 1.0
    else:
        logger.warning("ae_norm_stats.json not found — using raw (unnormalised) values for threshold")
        n = config.AE_INPUT_DIM
        ae_means = np.zeros(n, dtype=np.float32)
        ae_stds = np.ones(n, dtype=np.float32)

    # Use NORMAL rows only for threshold computation
    normal_df = df[df["label"] == "NORMAL"].reset_index(drop=True)
    if len(normal_df) < 10:
        logger.error("Too few NORMAL rows (%d) to calibrate AE threshold — need at least 10", len(normal_df))
        return -1.0

    features = normal_df[AE_FEATURE_COLUMNS].values.astype(np.float32)
    features = (features - ae_means) / ae_stds

    dataset = TensorDataset(torch.tensor(features))
    loader = DataLoader(dataset, batch_size=256)
    criterion = torch.nn.MSELoss(reduction="none")

    all_errors = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            recon = model(batch)
            errors = criterion(recon, batch).mean(dim=1)
            all_errors.append(errors.cpu().numpy())

    all_errors = np.concatenate(all_errors)
    threshold = float(np.percentile(all_errors, config.AE_ANOMALY_PERCENTILE))

    threshold_data = {
        "anomaly_threshold": threshold,
        "percentile": config.AE_ANOMALY_PERCENTILE,
        "mean_error": float(all_errors.mean()),
        "std_error": float(all_errors.std()),
        "max_error": float(all_errors.max()),
        "calibrated_from_n_samples": len(all_errors),
    }
    with open(ae_threshold_path, "w") as f:
        json.dump(threshold_data, f, indent=2)

    logger.info(
        "AE threshold calibrated: %.6f  (p%d of %d normal samples)",
        threshold, config.AE_ANOMALY_PERCENTILE, len(all_errors),
    )
    return threshold


# ── 3. Hourly baseline ────────────────────────────────────────────────────────

def calibrate_hourly_baseline(df: pd.DataFrame) -> int:
    """
    Build 85th-percentile VPM baseline per (junction, arm, hour_of_week) from
    NORMAL rows and merge into hourly_baseline.json.
    Existing entries for other junctions are preserved.
    Returns number of (junction, arm) pairs updated.
    """
    baseline_path = config.DATA_DIR / "hourly_baseline.json"

    existing: dict = {}
    if baseline_path.exists():
        with open(baseline_path) as f:
            raw = json.load(f)
        # Keep existing entries; hour keys are strings in JSON
        existing = raw

    normal_df = df[df["label"] == "NORMAL"].reset_index(drop=True)
    if normal_df.empty:
        logger.warning("No NORMAL rows found — hourly baseline not updated")
        return 0

    updated_arms = 0
    for (jid, aid), group in normal_df.groupby(["junction_id", "arm_id"]):
        hours: dict[str, float] = {}
        for how, hw_group in group.groupby("hour_of_week"):
            p85 = float(np.percentile(hw_group["VPM"].values, 85))
            hours[str(int(how))] = round(p85, 2)

        if jid not in existing:
            existing[jid] = {}
        existing[jid][aid] = hours
        logger.info("Baseline updated: %s / %s  (%d hours covered)", jid, aid, len(hours))
        updated_arms += 1

    with open(baseline_path, "w") as f:
        json.dump(existing, f, indent=2)
    logger.info("Saved hourly_baseline.json  (%d junction(s) total)", len(existing))
    return updated_arms


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Calibrate site-specific thresholds")
    parser.add_argument(
        "--data", type=str,
        default=str(config.SYNTHETIC_DATA_DIR / "all_arms_combined.csv"),
        help="Path to CSV with new site traffic data",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Torch device for AE inference (cpu / cuda)",
    )
    parser.add_argument(
        "--skip-lstm", action="store_true",
        help="Skip LSTM norm stats update",
    )
    parser.add_argument(
        "--skip-ae", action="store_true",
        help="Skip AE threshold recalibration",
    )
    parser.add_argument(
        "--skip-baseline", action="store_true",
        help="Skip hourly baseline update",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        logger.error("Data file not found: %s", data_path)
        sys.exit(1)

    logger.info("Loading data from %s", data_path)
    df = pd.read_csv(data_path)

    required_cols = {"camera_id", "junction_id", "arm_id", "label", "hour_of_week", "timestamp"}
    missing = required_cols - set(df.columns)
    if missing:
        logger.error("CSV is missing required columns: %s", missing)
        sys.exit(1)

    logger.info("Loaded %d rows  |  cameras: %s", len(df), sorted(df["camera_id"].unique()))

    if not args.skip_lstm:
        n = calibrate_lstm_norm_stats(df)
        logger.info("Step 1/3 — LSTM norm stats: %d camera(s) updated", n)
    else:
        logger.info("Step 1/3 — LSTM norm stats: skipped")

    if not args.skip_ae:
        t = calibrate_ae_threshold(df, args.device)
        if t >= 0:
            logger.info("Step 2/3 — AE threshold: %.6f", t)
        else:
            logger.warning("Step 2/3 — AE threshold calibration failed")
    else:
        logger.info("Step 2/3 — AE threshold: skipped")

    if not args.skip_baseline:
        n = calibrate_hourly_baseline(df)
        logger.info("Step 3/3 — Hourly baseline: %d arm(s) updated", n)
    else:
        logger.info("Step 3/3 — Hourly baseline: skipped")

    logger.info("Calibration complete. Restart the server for changes to take effect.")


if __name__ == "__main__":
    main()
