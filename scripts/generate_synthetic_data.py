"""
Synthetic traffic data generator.

Generates realistic per-minute traffic metrics for training the LSTM and autoencoder.
Produces 30 days of data per arm with configurable abnormal event injection.

Usage:
    python -m scripts.generate_synthetic_data [--days 30] [--inject-anomalies]

Output: CSV files in data/synthetic/ with columns matching the per-minute metrics schema.
"""

import argparse
import logging
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path so config imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# Lookahead window for extreme congestion future label (minutes)
EXTREME_CONGESTION_FORECAST_WINDOW = config.EXTREME_CONGESTION_FORECAST_WINDOW


def time_of_day_hours(minute_of_day: int) -> float:
    """Convert minute-of-day (0–1439) to fractional hours (0.0–24.0)."""
    return minute_of_day / 60.0


def time_encoding(minute_of_day: int) -> tuple[float, float]:
    """Cyclical sin/cos encoding for minute of day."""
    angle = 2 * math.pi * minute_of_day / 1440
    return math.sin(angle), math.cos(angle)


def hour_of_week(dt: datetime) -> int:
    """0 = Monday 00:00, 167 = Sunday 23:00."""
    return dt.weekday() * 24 + dt.hour


def is_weekend(dt: datetime) -> bool:
    return dt.weekday() >= 5


def base_vpm(hour_frac: float, weekend: bool, rng: np.random.Generator) -> float:
    """
    Compute base VPM for a given fractional hour of day.
    Applies weekend scaling and gaussian noise.
    """
    mp_start, mp_end = config.SYNTH_MORNING_PEAK_HOURS
    ep_start, ep_end = config.SYNTH_EVENING_PEAK_HOURS
    dt_start, dt_end = config.SYNTH_DAYTIME_HOURS
    night_start = config.SYNTH_NIGHT_HOURS_START
    night_end = config.SYNTH_NIGHT_HOURS_END

    if mp_start <= hour_frac < mp_end:
        lo, hi = config.SYNTH_MORNING_PEAK_VPM
        vpm = rng.uniform(lo, hi) + rng.normal(0, config.SYNTH_MORNING_PEAK_NOISE_STD)
    elif ep_start <= hour_frac < ep_end:
        lo, hi = config.SYNTH_EVENING_PEAK_VPM
        vpm = rng.uniform(lo, hi) + rng.normal(0, config.SYNTH_EVENING_PEAK_NOISE_STD)
    elif dt_start <= hour_frac < dt_end:
        lo, hi = config.SYNTH_DAYTIME_VPM
        vpm = rng.uniform(lo, hi) + rng.normal(0, 1.5)
    elif hour_frac >= night_start or hour_frac < night_end:
        lo, hi = config.SYNTH_NIGHT_VPM
        vpm = rng.uniform(lo, hi) + rng.normal(0, 0.5)
    else:
        # Shoulder transition periods (06:00-07:30, 19:00-22:00)
        lo, hi = config.SYNTH_DAYTIME_VPM
        vpm = rng.uniform(lo, hi) + rng.normal(0, 2)

    if weekend:
        vpm *= config.SYNTH_WEEKEND_SCALE

    return max(0.0, vpm)


def derive_metrics(vpm: float, is_peak: bool, rng: np.random.Generator,
                   label: str = "NORMAL") -> dict:
    """
    Given VPM and context, derive correlated per-minute metrics.
    Simulates the relationships that would exist in real camera data.
    """
    # Queue depth correlates with high VPM during congestion
    if label == "OFF_PEAK_JAM":
        queue_depth = rng.integers(*config.SYNTH_OFFPEAK_JAM_QUEUE_DEPTH)
        stopped_ratio = config.SYNTH_OFFPEAK_JAM_STOPPED_RATIO + rng.normal(0, 0.03)
        occupancy_pct = rng.uniform(40, 70)
        mean_bbox_area = rng.uniform(8000, 15000)  # large — vehicles close & stopped
        max_bbox_area = rng.uniform(15000, 25000)
        approach_flow = rng.uniform(0, 2)          # almost no throughput
    elif label == "PEAK_EXCESS":
        queue_depth = rng.integers(*config.SYNTH_PEAK_EXCESS_QUEUE_DEPTH)
        stopped_ratio = rng.uniform(0.5, 0.8)
        occupancy_pct = rng.uniform(50, 80)
        mean_bbox_area = rng.uniform(6000, 12000)
        max_bbox_area = rng.uniform(12000, 22000)
        approach_flow = rng.uniform(1, 5)
    else:
        # Normal traffic — metrics vary with VPM
        if vpm < 5:
            queue_depth = 0
            stopped_ratio = rng.uniform(0, 0.05)
            occupancy_pct = rng.uniform(1, 8)
            mean_bbox_area = rng.uniform(1000, 3000)
            max_bbox_area = rng.uniform(2000, 5000)
            approach_flow = vpm * rng.uniform(0.8, 1.2)
        elif vpm < 15:
            queue_depth = rng.choice([0, 0, 0, 1, 1, 2])
            stopped_ratio = rng.uniform(0.02, 0.15)
            occupancy_pct = rng.uniform(8, 25)
            mean_bbox_area = rng.uniform(2000, 5000)
            max_bbox_area = rng.uniform(4000, 8000)
            approach_flow = vpm * rng.uniform(0.7, 1.1)
        else:
            queue_depth = rng.integers(1, 6)
            stopped_ratio = rng.uniform(0.1, 0.35)
            occupancy_pct = rng.uniform(20, 45)
            mean_bbox_area = rng.uniform(3000, 7000)
            max_bbox_area = rng.uniform(6000, 12000)
            approach_flow = vpm * rng.uniform(0.5, 0.9)

    stopped_ratio = float(np.clip(stopped_ratio, 0, 1))
    occupancy_pct = float(np.clip(occupancy_pct, 0, 100))
    approach_flow = max(0.0, approach_flow)

    # Far-zone bbox growth rate (px²/frame)
    if label == "OFF_PEAK_JAM":
        mean_bbox_growth_rate = rng.normal(-5, 3)  # negative = jam
    elif label == "PEAK_EXCESS":
        mean_bbox_growth_rate = rng.normal(-5, 3)  # negative = jam
    else:
        mean_bbox_growth_rate = rng.normal(50, 10)  # positive = normal approach flow

    return {
        "queue_depth": int(queue_depth),
        "stopped_ratio": round(stopped_ratio, 4),
        "occupancy_pct": round(occupancy_pct, 2),
        "mean_bbox_area": round(mean_bbox_area, 1),
        "max_bbox_area": round(max_bbox_area, 1),
        "approach_flow": round(approach_flow, 2),
        "mean_bbox_growth_rate": round(float(mean_bbox_growth_rate), 4),
    }


def inject_offpeak_jam(rows: list[dict], rng: np.random.Generator) -> list[dict]:
    """
    Inject off-peak sudden jam events at random off-peak times.
    Modifies rows in-place and returns the list.
    """
    # Find off-peak minute indices
    offpeak_indices = [
        i for i, r in enumerate(rows)
        if not config.is_peak_hour(r["_dt"].hour)
        and 60 < i < len(rows) - 60  # avoid edges
    ]
    if not offpeak_indices:
        return rows

    # Pick 1–3 jam events per 30-day period
    num_events = rng.integers(1, 4)
    for _ in range(num_events):
        start_idx = rng.choice(offpeak_indices)
        duration = rng.integers(*config.SYNTH_OFFPEAK_JAM_DURATION_MIN)

        for i in range(start_idx, min(start_idx + duration, len(rows))):
            lo, hi = config.SYNTH_OFFPEAK_JAM_VPM
            rows[i]["VPM"] = rng.integers(lo, hi + 1)
            jam_metrics = derive_metrics(rows[i]["VPM"], False, rng, "OFF_PEAK_JAM")
            rows[i].update(jam_metrics)
            rows[i]["label"] = "OFF_PEAK_JAM"

        logger.info(
            "Injected OFF_PEAK_JAM: start_idx=%d, duration=%d min, time=%s",
            start_idx, duration, rows[start_idx]["_dt"].isoformat()
        )

    return rows


def inject_peak_excess(rows: list[dict], rng: np.random.Generator) -> list[dict]:
    """
    Inject abnormally-long peak-hour excess events.
    """
    peak_indices = [
        i for i, r in enumerate(rows)
        if config.is_peak_hour(r["_dt"].hour)
        and 60 < i < len(rows) - 120
    ]
    if not peak_indices:
        return rows

    num_events = rng.integers(1, 4)
    for _ in range(num_events):
        start_idx = rng.choice(peak_indices)
        lo_dur, hi_dur = config.SYNTH_PEAK_EXCESS_DURATION_MIN
        duration = rng.integers(lo_dur, hi_dur + 1)

        for i in range(start_idx, min(start_idx + duration, len(rows))):
            lo, hi = config.SYNTH_PEAK_EXCESS_VPM
            rows[i]["VPM"] = rng.integers(lo, hi + 1)
            excess_metrics = derive_metrics(rows[i]["VPM"], True, rng, "PEAK_EXCESS")
            rows[i].update(excess_metrics)
            rows[i]["label"] = "PEAK_EXCESS"

        logger.info(
            "Injected PEAK_EXCESS: start_idx=%d, duration=%d min, time=%s",
            start_idx, duration, rows[start_idx]["_dt"].isoformat()
        )

    return rows


def _add_pre_jam_slowdown(rows: list[dict], rng: np.random.Generator) -> list[dict]:
    """
    Add a linear bbox growth rate slowdown in the 10 minutes preceding
    any jam event (where queue_depth transitions from 0 to >0).
    """
    ramp_len = EXTREME_CONGESTION_FORECAST_WINDOW
    i = 0
    while i < len(rows):
        # Find jam onset: queue_depth goes from 0 to >0
        if rows[i]["queue_depth"] > 0 and (i == 0 or rows[i - 1]["queue_depth"] == 0):
            ramp_start = max(0, i - ramp_len)
            for j in range(ramp_start, i):
                if rows[j]["label"] != "NORMAL":
                    continue  # don't overwrite jam rows
                progress = (j - ramp_start) / max(1, i - ramp_start)  # 0→1
                # Linearly ramp from normal (~50) down to near-zero (~5)
                target = 50.0 * (1.0 - progress) + 5.0 * progress
                rows[j]["mean_bbox_growth_rate"] = round(
                    rng.normal(target, 5), 4
                )
        i += 1
    return rows


def generate_arm_data(
    junction_id: str,
    arm_id: str,
    days: int,
    inject_anomalies: bool,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate per-minute synthetic data for one arm."""
    rng = np.random.default_rng(seed)

    start_dt = datetime(2026, 3, 1, 0, 0, 0)  # arbitrary start date
    total_minutes = days * 1440
    rows = []

    for m in range(total_minutes):
        dt = start_dt + timedelta(minutes=m)
        minute_of_day = dt.hour * 60 + dt.minute
        hour_frac = time_of_day_hours(minute_of_day)
        weekend = is_weekend(dt)
        is_peak = config.is_peak_hour(dt.hour)

        vpm = base_vpm(hour_frac, weekend, rng)
        vpm_int = max(0, int(round(vpm)))

        time_sin, time_cos = time_encoding(minute_of_day)
        h_of_w = hour_of_week(dt)

        metrics = derive_metrics(vpm_int, is_peak, rng)

        rows.append({
            "timestamp": dt.isoformat(),
            "junction_id": junction_id,
            "arm_id": arm_id,
            "camera_id": f"{junction_id}_{arm_id}",
            "VPM": vpm_int,
            "queue_depth": metrics["queue_depth"],
            "stopped_ratio": metrics["stopped_ratio"],
            "occupancy_pct": metrics["occupancy_pct"],
            "mean_bbox_area": metrics["mean_bbox_area"],
            "max_bbox_area": metrics["max_bbox_area"],
            "approach_flow": metrics["approach_flow"],
            "mean_bbox_growth_rate": metrics["mean_bbox_growth_rate"],
            "time_sin": round(time_sin, 6),
            "time_cos": round(time_cos, 6),
            "is_peak_hour": int(is_peak),
            "hour_of_week": h_of_w,
            "label": "NORMAL",
            "_dt": dt,  # helper field, dropped before export
        })

    if inject_anomalies:
        rows = inject_offpeak_jam(rows, rng)
        rows = inject_peak_excess(rows, rng)

        # Add pre-jam bbox growth rate slowdown ramp (10 min before queue_depth > 0)
        rows = _add_pre_jam_slowdown(rows, rng)

    df = pd.DataFrame(rows)
    df.drop(columns=["_dt"], inplace=True)

    # Compute extreme_congestion_future label (lookahead)
    df["extreme_congestion_future"] = 0
    for i in range(len(df) - EXTREME_CONGESTION_FORECAST_WINDOW):
        if df["queue_depth"].iloc[i + 1 : i + 1 + EXTREME_CONGESTION_FORECAST_WINDOW].max() > 0:
            df.at[i, "extreme_congestion_future"] = 1

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic traffic data")
    parser.add_argument("--days", type=int, default=config.SYNTH_DAYS,
                        help="Number of days to generate")
    parser.add_argument("--inject-anomalies", action="store_true", default=True,
                        help="Inject abnormal events (default: True)")
    parser.add_argument("--no-anomalies", action="store_true",
                        help="Generate only normal data (for autoencoder training)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    inject = not args.no_anomalies

    output_dir = config.SYNTHETIC_DATA_DIR
    all_frames = []

    for jid, jdata in config.JUNCTIONS.items():
        for aid in jdata["arms"]:
            camera_id = f"{jid}_{aid}"
            # Vary seed per arm for diversity
            arm_seed = args.seed + hash(camera_id) % 10000

            logger.info("Generating %d days for %s (inject_anomalies=%s)",
                        args.days, camera_id, inject)

            df = generate_arm_data(jid, aid, args.days, inject, arm_seed)
            all_frames.append(df)

            # Save per-arm file
            arm_path = output_dir / f"{camera_id}.csv"
            df.to_csv(arm_path, index=False)
            logger.info("Saved %d rows → %s", len(df), arm_path)

    # Also save combined file
    combined = pd.concat(all_frames, ignore_index=True)
    combined_path = output_dir / "all_arms_combined.csv"
    combined.to_csv(combined_path, index=False)
    logger.info("Combined dataset: %d rows → %s", len(combined), combined_path)

    # Generate a normal-only version for autoencoder training
    if inject:
        logger.info("Generating normal-only data for autoencoder training...")
        ae_frames = []
        for jid, jdata in config.JUNCTIONS.items():
            for aid in jdata["arms"]:
                camera_id = f"{jid}_{aid}"
                arm_seed = args.seed + hash(camera_id) % 10000
                df_normal = generate_arm_data(jid, aid, args.days, False, arm_seed)
                ae_frames.append(df_normal)

        ae_combined = pd.concat(ae_frames, ignore_index=True)
        ae_path = output_dir / "normal_only_combined.csv"
        ae_combined.to_csv(ae_path, index=False)
        logger.info("Normal-only dataset: %d rows → %s", len(ae_combined), ae_path)

    # Print label distribution
    label_counts = combined["label"].value_counts()
    logger.info("Label distribution:\n%s", label_counts.to_string())


if __name__ == "__main__":
    main()
