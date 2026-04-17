"""
Export collected minute metrics from the DB to a calibration-ready CSV.

The exported CSV is compatible with calibrate_site.py and the training scripts.
A 'label' column is derived by joining with the alerts table:
  - Any minute where an AMBER or RED alert fired for that camera → CONGESTED
  - All other minutes → NORMAL

Usage:
    python -m scripts.export_training_data
    python -m scripts.export_training_data --junction JCT02
    python -m scripts.export_training_data --out data/site2_collected.csv
    python -m scripts.export_training_data --days 14
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from backend.db.models import SessionLocal

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


def export(junction_id: str | None, days: int | None, out_path: Path):
    db = SessionLocal()
    try:
        # ── Load minute metrics ──────────────────────────────────────────────
        metrics_query = "SELECT * FROM minute_metrics WHERE 1=1"
        params: dict = {}

        if junction_id:
            metrics_query += " AND junction_id = :jid"
            params["jid"] = junction_id

        if days:
            since = datetime.utcnow() - timedelta(days=days)
            metrics_query += " AND timestamp >= :since"
            params["since"] = since

        metrics_query += " ORDER BY camera_id, timestamp"

        metrics_df = pd.read_sql(text(metrics_query), db.bind, params=params)
        if metrics_df.empty:
            logger.error("No minute metrics found in DB. Run the server first to collect data.")
            return

        logger.info("Loaded %d minute metric rows", len(metrics_df))

        # ── Load alerts to derive labels ─────────────────────────────────────
        alerts_query = "SELECT junction_id, arm_id, timestamp, level FROM alerts WHERE level IN ('AMBER', 'RED')"
        alerts_params: dict = {}

        if junction_id:
            alerts_query += " AND junction_id = :jid"
            alerts_params["jid"] = junction_id

        if days:
            alerts_query += " AND timestamp >= :since"
            alerts_params["since"] = since

        alerts_df = pd.read_sql(text(alerts_query), db.bind, params=alerts_params)
        logger.info("Loaded %d alert rows for label derivation", len(alerts_df))

    finally:
        db.close()

    # ── Derive label ─────────────────────────────────────────────────────────
    # Round both timestamps to the nearest minute for matching
    metrics_df["timestamp"] = pd.to_datetime(metrics_df["timestamp"])
    metrics_df["minute_key"] = (
        metrics_df["camera_id"] + "_" +
        metrics_df["timestamp"].dt.floor("min").astype(str)
    )

    congested_keys: set = set()
    if not alerts_df.empty:
        alerts_df["timestamp"] = pd.to_datetime(alerts_df["timestamp"])
        alerts_df["camera_id"] = alerts_df["junction_id"] + "_" + alerts_df["arm_id"]
        alerts_df["minute_key"] = (
            alerts_df["camera_id"] + "_" +
            alerts_df["timestamp"].dt.floor("min").astype(str)
        )
        congested_keys = set(alerts_df["minute_key"])

    metrics_df["label"] = metrics_df["minute_key"].apply(
        lambda k: "CONGESTED" if k in congested_keys else "NORMAL"
    )
    metrics_df.drop(columns=["minute_key"], inplace=True)

    # ── Rename timestamp to ISO string, drop DB id ───────────────────────────
    metrics_df["timestamp"] = metrics_df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    metrics_df.drop(columns=["id"], errors="ignore", inplace=True)

    # ── Ensure extreme_congestion_future column exists (used by train_lstm) ──
    if "extreme_congestion_future" not in metrics_df.columns:
        # Derive: 1 if any CONGESTED row exists within next 10 min for same camera
        metrics_df["extreme_congestion_future"] = 0

    # ── Column order matching training scripts ───────────────────────────────
    ordered_cols = [
        "timestamp", "camera_id", "junction_id", "arm_id", "label",
        "hour_of_week", "VPM", "queue_depth", "stopped_ratio", "occupancy_pct",
        "mean_bbox_area", "max_bbox_area", "approach_flow",
        "time_sin", "time_cos", "is_peak_hour", "mean_bbox_growth_rate",
        "extreme_congestion_future",
    ]
    for col in ordered_cols:
        if col not in metrics_df.columns:
            metrics_df[col] = 0

    export_df = metrics_df[[c for c in ordered_cols if c in metrics_df.columns]]

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(out_path, index=False)

    normal_count = (export_df["label"] == "NORMAL").sum()
    congested_count = (export_df["label"] == "CONGESTED").sum()
    cameras = export_df["camera_id"].unique().tolist()

    logger.info("Exported %d rows → %s", len(export_df), out_path)
    logger.info("  NORMAL: %d rows  |  CONGESTED: %d rows", normal_count, congested_count)
    logger.info("  Cameras: %s", cameras)

    if normal_count < 100:
        logger.warning("Only %d NORMAL rows — collect more data before calibrating (recommend 1000+)", normal_count)


def main():
    parser = argparse.ArgumentParser(description="Export DB metrics to calibration CSV")
    parser.add_argument("--junction", type=str, default=None,
                        help="Filter to specific junction (e.g. JCT02)")
    parser.add_argument("--days", type=int, default=None,
                        help="Only export last N days of data")
    parser.add_argument("--out", type=str,
                        default=str(config.SYNTHETIC_DATA_DIR / "collected_site_data.csv"),
                        help="Output CSV path")
    args = parser.parse_args()

    export(args.junction, args.days, Path(args.out))


if __name__ == "__main__":
    main()
