"""
Merge pre-labeled *_extracted.csv files into a single training-ready CSV.

Each extracted CSV must already have a 'label' column added manually before
running this script. Valid labels: NORMAL, PEAK_EXCESS, OFF_PEAK_JAM.

Usage:
    python -m scripts.merge_extracted
    python -m scripts.merge_extracted --data-dir data --out data/real_combined.csv
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

VALID_LABELS = {"NORMAL", "PEAK_EXCESS", "OFF_PEAK_JAM"}

# Columns expected by training scripts
ORDERED_COLS = [
    "timestamp", "camera_id", "junction_id", "arm_id", "label",
    "hour_of_week", "VPM", "queue_depth", "stopped_ratio", "occupancy_pct",
    "mean_bbox_area", "max_bbox_area", "approach_flow",
    "time_sin", "time_cos", "is_peak_hour", "mean_bbox_growth_rate",
]


def merge(data_dir: Path, out_path: Path):
    csv_files = sorted(data_dir.glob("*_extracted.csv"))
    if not csv_files:
        logger.error("No *_extracted.csv files found in %s", data_dir)
        sys.exit(1)

    frames = []
    for f in csv_files:
        df = pd.read_csv(f)

        # Validate label column exists
        if "label" not in df.columns:
            logger.error("MISSING 'label' column in %s — label it before merging", f.name)
            sys.exit(1)

        # Validate label values
        bad_labels = set(df["label"].unique()) - VALID_LABELS
        if bad_labels:
            logger.error("Invalid labels in %s: %s  (valid: %s)", f.name, bad_labels, VALID_LABELS)
            sys.exit(1)

        normal_n = (df["label"] == "NORMAL").sum()
        congested_n = len(df) - normal_n
        logger.info("  %s: %d rows  (%d NORMAL, %d congested)", f.name, len(df), normal_n, congested_n)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # Drop mean_speed_proxy if present (not used by models)
    combined.drop(columns=["mean_speed_proxy"], errors="ignore", inplace=True)

    # Ensure column order matches training scripts
    for col in ORDERED_COLS:
        if col not in combined.columns:
            combined[col] = 0
    combined = combined[[c for c in ORDERED_COLS if c in combined.columns]]

    # Sort by camera and timestamp for clean sequences
    combined.sort_values(["camera_id", "timestamp"], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)

    logger.info("Merged %d files → %d rows → %s", len(csv_files), len(combined), out_path)
    label_counts = combined["label"].value_counts()
    for lbl, cnt in label_counts.items():
        logger.info("  %s: %d rows (%.1f%%)", lbl, cnt, 100 * cnt / len(combined))


def main():
    parser = argparse.ArgumentParser(description="Merge pre-labeled extracted CSVs")
    parser.add_argument("--data-dir", type=str, default=str(config.DATA_DIR),
                        help="Directory containing labeled *_extracted.csv files")
    parser.add_argument("--out", type=str, default=str(config.DATA_DIR / "real_combined.csv"),
                        help="Output CSV path")
    args = parser.parse_args()

    merge(Path(args.data_dir), Path(args.out))


if __name__ == "__main__":
    main()
