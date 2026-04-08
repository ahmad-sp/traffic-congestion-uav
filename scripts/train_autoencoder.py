"""
Train the Autoencoder Anomaly Detector on normal-only traffic data.

Usage:
    python -m scripts.train_autoencoder [--epochs 50] [--data path/to/csv]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from backend.models.autoencoder import TrafficAutoencoder

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "VPM", "queue_depth", "stopped_ratio", "occupancy_pct",
    "mean_bbox_area", "max_bbox_area", "approach_flow",
    "time_sin", "time_cos", "is_peak_hour",
]


def train(data_path: Path, epochs: int, device: str):
    logger.info("Loading normal-only data from %s", data_path)
    df = pd.read_csv(data_path)

    # Filter to NORMAL only (should already be, but safety)
    df = df[df["label"] == "NORMAL"].reset_index(drop=True)
    logger.info("Normal samples: %d", len(df))

    features = df[FEATURE_COLUMNS].values.astype(np.float32)

    # Normalize
    means = features.mean(axis=0)
    stds = features.std(axis=0)
    stds[stds == 0] = 1.0
    features = (features - means) / stds

    # Train/val split (80/20)
    perm = np.random.permutation(len(features))
    features = features[perm]
    split = int(0.8 * len(features))
    X_train, X_val = features[:split], features[split:]

    logger.info("Train: %d, Val: %d", len(X_train), len(X_val))

    train_ds = TensorDataset(torch.tensor(X_train))
    val_ds = TensorDataset(torch.tensor(X_val))
    train_dl = DataLoader(train_ds, batch_size=config.AE_TRAIN_BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=config.AE_TRAIN_BATCH_SIZE)

    model = TrafficAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.AE_TRAIN_LR)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for (batch,) in train_dl:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_dl:
                batch = batch.to(device)
                reconstructed = model(batch)
                val_loss += criterion(reconstructed, batch).item() * len(batch)
        val_loss /= len(val_ds)

        if epoch % 5 == 0 or epoch == 1:
            logger.info("Epoch %d/%d — train_loss=%.6f, val_loss=%.6f",
                        epoch, epochs, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.MODEL_DIR / "autoencoder.pt")

    # Compute anomaly threshold: 95th percentile of training reconstruction errors
    model.eval()
    all_errors = []
    with torch.no_grad():
        for (batch,) in DataLoader(train_ds, batch_size=256):
            batch = batch.to(device)
            errors = model.reconstruction_error(batch)
            all_errors.append(errors.cpu().numpy())

    all_errors = np.concatenate(all_errors)
    threshold = float(np.percentile(all_errors, config.AE_ANOMALY_PERCENTILE))

    threshold_data = {
        "anomaly_threshold": threshold,
        "percentile": config.AE_ANOMALY_PERCENTILE,
        "mean_error": float(all_errors.mean()),
        "std_error": float(all_errors.std()),
        "max_error": float(all_errors.max()),
    }
    threshold_path = config.MODEL_DIR / "ae_threshold.json"
    with open(threshold_path, "w") as f:
        json.dump(threshold_data, f, indent=2)

    # Save normalization stats
    norm_stats = {"means": means.tolist(), "stds": stds.tolist()}
    stats_path = config.MODEL_DIR / "ae_norm_stats.json"
    with open(stats_path, "w") as f:
        json.dump(norm_stats, f, indent=2)

    logger.info("Training complete. Best val_loss=%.6f", best_val_loss)
    logger.info("Anomaly threshold (p%d): %.6f", config.AE_ANOMALY_PERCENTILE, threshold)
    logger.info("Model → %s", config.MODEL_DIR / "autoencoder.pt")
    logger.info("Threshold → %s", threshold_path)
    logger.info("Norm stats → %s", stats_path)


def main():
    parser = argparse.ArgumentParser(description="Train autoencoder anomaly detector")
    parser.add_argument("--data", type=str,
                        default=str(config.SYNTHETIC_DATA_DIR / "normal_only_combined.csv"))
    parser.add_argument("--epochs", type=int, default=config.AE_TRAIN_EPOCHS)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    train(Path(args.data), args.epochs, args.device)


if __name__ == "__main__":
    main()
