"""
Train the LSTM Congestion Forecaster on synthetic data.

Usage:
    python -m scripts.train_lstm [--epochs 50] [--data path/to/csv]
"""

import argparse
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
from backend.models.lstm_model import LSTMCongestionForecaster

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "VPM", "queue_depth", "stopped_ratio", "occupancy_pct",
    "mean_bbox_area", "max_bbox_area", "approach_flow",
    "time_sin", "time_cos", "is_peak_hour",
    "mean_bbox_growth_rate",
]


def create_sequences(df: pd.DataFrame, seq_len: int):
    """
    Create sliding-window sequences from per-minute data.
    Returns X (N, seq_len, features), y_cong (N, 1), y_extreme (N, 1).
    """
    features = df[FEATURE_COLUMNS].values.astype(np.float32)

    # Label 1: congestion (1.0 if any abnormal event in the label column)
    labels_cong = (df["label"] != "NORMAL").astype(np.float32).values

    # Label 2: extreme congestion future (will queue appear in next 10 min?)
    if "extreme_congestion_future" in df.columns:
        labels_extreme = df["extreme_congestion_future"].astype(np.float32).values
    else:
        labels_extreme = np.zeros(len(df), dtype=np.float32)

    # Normalize features per column (store stats for inference)
    means = features.mean(axis=0)
    stds = features.std(axis=0)
    stds[stds == 0] = 1.0  # avoid division by zero
    features = (features - means) / stds

    X_list, y_cong_list, y_ext_list = [], [], []
    for i in range(len(features) - seq_len):
        X_list.append(features[i : i + seq_len])
        y_cong_list.append(labels_cong[i + seq_len - 1])
        y_ext_list.append(labels_extreme[i + seq_len - 1])

    X = np.stack(X_list)
    y_cong = np.array(y_cong_list, dtype=np.float32).reshape(-1, 1)
    y_ext = np.array(y_ext_list, dtype=np.float32).reshape(-1, 1)
    return X, y_cong, y_ext, means, stds


def train(data_path: Path, epochs: int, device: str):
    logger.info("Loading data from %s", data_path)
    df = pd.read_csv(data_path)

    # Group by camera and create sequences per arm
    all_X, all_y_cong, all_y_ext = [], [], []
    norm_stats = {}

    for camera_id, group in df.groupby("camera_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        X, y_cong, y_ext, means, stds = create_sequences(group, config.LSTM_SEQUENCE_LENGTH)
        all_X.append(X)
        all_y_cong.append(y_cong)
        all_y_ext.append(y_ext)
        norm_stats[camera_id] = {"means": means.tolist(), "stds": stds.tolist()}
        logger.info("Camera %s: %d sequences (%.1f%% congested, %.1f%% extreme)",
                     camera_id, len(X), 100 * y_cong.mean(), 100 * y_ext.mean())

    X_all = np.concatenate(all_X)
    y_cong_all = np.concatenate(all_y_cong)
    y_ext_all = np.concatenate(all_y_ext)

    # Shuffle
    perm = np.random.permutation(len(X_all))
    X_all, y_cong_all, y_ext_all = X_all[perm], y_cong_all[perm], y_ext_all[perm]

    # Train/val split (80/20)
    split = int(0.8 * len(X_all))
    X_train, X_val = X_all[:split], X_all[split:]
    y_cong_train, y_cong_val = y_cong_all[:split], y_cong_all[split:]
    y_ext_train, y_ext_val = y_ext_all[:split], y_ext_all[split:]

    logger.info("Train: %d sequences, Val: %d sequences", len(X_train), len(X_val))

    train_ds = TensorDataset(
        torch.tensor(X_train), torch.tensor(y_cong_train), torch.tensor(y_ext_train))
    val_ds = TensorDataset(
        torch.tensor(X_val), torch.tensor(y_cong_val), torch.tensor(y_ext_val))
    train_dl = DataLoader(train_ds, batch_size=config.LSTM_TRAIN_BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=config.LSTM_TRAIN_BATCH_SIZE)

    model = LSTMCongestionForecaster().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LSTM_TRAIN_LR)

    w_cong = config.LSTM_LOSS_WEIGHT_CONGESTION
    w_ext = config.LSTM_LOSS_WEIGHT_EXTREME

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for X_batch, y_cong_batch, y_ext_batch in train_dl:
            X_batch = X_batch.to(device)
            y_cong_batch = y_cong_batch.to(device)
            y_ext_batch = y_ext_batch.to(device)
            optimizer.zero_grad()
            pred_cong, pred_ext = model(X_batch)
            loss_cong = criterion(pred_cong, y_cong_batch)
            loss_ext = criterion(pred_ext, y_ext_batch)
            loss = w_cong * loss_cong + w_ext * loss_ext
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
        train_loss /= len(train_ds)

        # Validate
        model.eval()
        val_loss = 0.0
        correct_cong = 0
        correct_ext = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_cong_batch, y_ext_batch in val_dl:
                X_batch = X_batch.to(device)
                y_cong_batch = y_cong_batch.to(device)
                y_ext_batch = y_ext_batch.to(device)
                pred_cong, pred_ext = model(X_batch)
                loss_cong = criterion(pred_cong, y_cong_batch)
                loss_ext = criterion(pred_ext, y_ext_batch)
                val_loss += (w_cong * loss_cong + w_ext * loss_ext).item() * len(X_batch)
                correct_cong += ((pred_cong >= 0.5).float() == y_cong_batch).sum().item()
                correct_ext += ((pred_ext >= 0.5).float() == y_ext_batch).sum().item()
                total += len(y_cong_batch)
        val_loss /= len(val_ds)
        acc_cong = correct_cong / total if total > 0 else 0
        acc_ext = correct_ext / total if total > 0 else 0

        if epoch % 5 == 0 or epoch == 1:
            logger.info("Epoch %d/%d — train_loss=%.4f, val_loss=%.4f, "
                        "val_acc_cong=%.3f, val_acc_ext=%.3f",
                        epoch, epochs, train_loss, val_loss, acc_cong, acc_ext)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = config.MODEL_DIR / "lstm_congestion.pt"
            torch.save(model.state_dict(), model_path)

    # Save normalization stats
    import json
    stats_path = config.MODEL_DIR / "lstm_norm_stats.json"
    with open(stats_path, "w") as f:
        json.dump(norm_stats, f, indent=2)
    logger.info("Saved normalization stats → %s", stats_path)

    logger.info("Training complete. Best val_loss=%.4f. Model → %s",
                best_val_loss, config.MODEL_DIR / "lstm_congestion.pt")


def main():
    parser = argparse.ArgumentParser(description="Train LSTM congestion forecaster")
    parser.add_argument("--data", type=str,
                        default=str(config.SYNTHETIC_DATA_DIR / "all_arms_combined.csv"))
    parser.add_argument("--epochs", type=int, default=config.LSTM_TRAIN_EPOCHS)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    train(Path(args.data), args.epochs, args.device)


if __name__ == "__main__":
    main()
