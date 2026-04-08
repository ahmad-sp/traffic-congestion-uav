"""
Historical 85th-percentile VPM baseline per hour-of-week.

Used by Warrant 3 to determine if current traffic exceeds historical norms.
hour_of_week: 0 = Monday 00:00, 167 = Sunday 23:00.
"""

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config

logger = logging.getLogger(__name__)

# In-memory baseline cache: {junction_id: {arm_id: {hour_of_week: float}}}
_baseline_cache: dict[str, dict[str, dict[int, float]]] = {}


def build_baseline_from_csv(csv_path: Path) -> dict[str, dict[str, dict[int, float]]]:
    """
    Build 85th-percentile VPM baseline from synthetic/real data CSV.
    Groups by camera (junction_id, arm_id) and hour_of_week.
    """
    df = pd.read_csv(csv_path)
    # Only use NORMAL data for baseline
    df = df[df["label"] == "NORMAL"]

    baseline = defaultdict(lambda: defaultdict(dict))

    for (jid, aid), group in df.groupby(["junction_id", "arm_id"]):
        for how, hw_group in group.groupby("hour_of_week"):
            p85 = float(np.percentile(hw_group["VPM"].values, 85))
            baseline[jid][aid][int(how)] = round(p85, 2)

    logger.info("Built baseline for %d camera(s), 168 hours each",
                sum(len(arms) for arms in baseline.values()))
    return dict(baseline)


def load_baseline(data_path: Path | None = None):
    """Load or build the baseline and cache it in memory."""
    global _baseline_cache

    baseline_json = config.DATA_DIR / "hourly_baseline.json"

    if baseline_json.exists():
        with open(baseline_json) as f:
            raw = json.load(f)
        # Convert string keys back to int for hour_of_week
        _baseline_cache = {
            jid: {
                aid: {int(k): v for k, v in hours.items()}
                for aid, hours in arms.items()
            }
            for jid, arms in raw.items()
        }
        logger.info("Loaded baseline from %s", baseline_json)
    else:
        if data_path is None:
            data_path = config.SYNTHETIC_DATA_DIR / "all_arms_combined.csv"
        if not data_path.exists():
            logger.warning("No data for baseline at %s", data_path)
            return
        _baseline_cache = build_baseline_from_csv(data_path)
        save_baseline()


def save_baseline():
    """Persist current baseline to JSON."""
    baseline_json = config.DATA_DIR / "hourly_baseline.json"
    with open(baseline_json, "w") as f:
        json.dump(_baseline_cache, f, indent=2)
    logger.info("Saved baseline → %s", baseline_json)


def get_baseline_vpm(junction_id: str, arm_id: str, hour_of_week: int) -> float | None:
    """Get the 85th-percentile VPM for a specific arm and hour-of-week."""
    return _baseline_cache.get(junction_id, {}).get(arm_id, {}).get(hour_of_week)


def get_arm_baseline(junction_id: str, arm_id: str) -> dict[int, float]:
    """Get the full 168-hour baseline for an arm."""
    return _baseline_cache.get(junction_id, {}).get(arm_id, {})


def update_baseline_hour(junction_id: str, arm_id: str, hour_of_week: int, new_value: float):
    """Update a single hour's baseline value."""
    if junction_id not in _baseline_cache:
        _baseline_cache[junction_id] = {}
    if arm_id not in _baseline_cache[junction_id]:
        _baseline_cache[junction_id][arm_id] = {}
    _baseline_cache[junction_id][arm_id][hour_of_week] = round(new_value, 2)
