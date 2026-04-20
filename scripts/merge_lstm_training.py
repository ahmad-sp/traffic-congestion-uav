"""
Merge real labeled data + synthetic data into a single CSV for LSTM training.

Usage:
    python -m scripts.merge_lstm_training
    python -m scripts.merge_lstm_training --real data/real_combined.csv --synthetic data/synthetic/all_arms_combined.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

REQUIRED_COLS = [
    "timestamp", "camera_id", "junction_id", "arm_id", "label",
    "hour_of_week", "VPM", "queue_depth", "stopped_ratio", "occupancy_pct",
    "mean_bbox_area", "max_bbox_area", "approach_flow",
    "time_sin", "time_cos", "is_peak_hour", "mean_bbox_growth_rate",
]


def merge(real_path: Path, synthetic_path: Path, out_path: Path):
    if not real_path.exists():
        logger.error("Real data not found: %s", real_path)
        sys.exit(1)
    if not synthetic_path.exists():
        logger.error("Synthetic data not found: %s", synthetic_path)
        sys.exit(1)

    real_df = pd.read_csv(real_path)
    synth_df = pd.read_csv(synthetic_path)

    logger.info("Real data:      %d rows", len(real_df))
    logger.info("Synthetic data: %d rows", len(synth_df))

    # Keep only columns needed for training
    keep_cols = [c for c in REQUIRED_COLS if c in real_df.columns and c in synth_df.columns]
    real_df = real_df[keep_cols]
    synth_df = synth_df[keep_cols]

    combined = pd.concat([real_df, synth_df], ignore_index=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)

    logger.info("Combined: %d rows → %s", len(combined), out_path)

    # Label distribution
    label_counts = combined["label"].value_counts()
    for lbl, cnt in label_counts.items():
        logger.info("  %s: %d rows (%.1f%%)", lbl, cnt, 100 * cnt / len(combined))


def main():
    parser = argparse.ArgumentParser(description="Merge real + synthetic data for LSTM training")
    parser.add_argument("--real", type=str,
                        default=str(config.DATA_DIR / "real_combined.csv"))
    parser.add_argument("--synthetic", type=str,
                        default=str(config.SYNTHETIC_DATA_DIR / "all_arms_combined.csv"))
    parser.add_argument("--out", type=str,
                        default=str(config.DATA_DIR / "lstm_training.csv"))
    args = parser.parse_args()

    merge(Path(args.real), Path(args.synthetic), Path(args.out))


if __name__ == "__main__":
    main()
