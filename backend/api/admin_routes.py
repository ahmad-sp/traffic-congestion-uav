"""
Admin API endpoints for runtime configuration management.

Allows operators to:
- Edit per-junction peak hours
- Reposition camera markers (drag-drop)
- Create new junctions
- Add new camera arms
- Define road path geometry per arm

All mutations update config.JUNCTIONS in-memory and persist to
traffic_system/data/junction_overrides.json so changes survive restarts.
"""

import json
import logging
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config

logger = logging.getLogger(__name__)

admin_router = APIRouter(prefix="/admin")

OVERRIDES_PATH = config.DATA_DIR / "junction_overrides.json"
_overrides: dict = {}


# ─── Persistence helpers ───

def _save_overrides():
    """Write in-memory overrides to disk."""
    with open(OVERRIDES_PATH, "w") as f:
        json.dump(_overrides, f, indent=2)


def load_admin_overrides():
    """
    Called once at startup to apply persisted changes to config.JUNCTIONS.
    Reads junction_overrides.json and merges into config.JUNCTIONS.
    """
    if not OVERRIDES_PATH.exists():
        return
    try:
        with open(OVERRIDES_PATH) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error("Failed to load admin overrides: %s", e)
        return

    _overrides.update(data)
    for jid, jdata in data.items():
        if jid not in config.JUNCTIONS:
            config.JUNCTIONS[jid] = jdata  # new junction added via admin
        else:
            if "peak_periods" in jdata:
                config.JUNCTIONS[jid]["peak_periods"] = [tuple(p) for p in jdata["peak_periods"]]
            if "name" in jdata:
                config.JUNCTIONS[jid]["name"] = jdata["name"]
            if "type" in jdata:
                config.JUNCTIONS[jid]["type"] = jdata["type"]
            for aid, arm_data in jdata.get("arms", {}).items():
                if aid not in config.JUNCTIONS[jid]["arms"]:
                    config.JUNCTIONS[jid]["arms"][aid] = arm_data
                else:
                    config.JUNCTIONS[jid]["arms"][aid].update(arm_data)

    logger.info("Loaded admin overrides: %d junction(s) modified", len(data))


# ─── Pydantic schemas ───

class PeakPeriodsUpdate(BaseModel):
    peak_periods: list[list[int]]  # e.g. [[7,9],[17,19]]


class ArmLocationUpdate(BaseModel):
    gps_lat: float
    gps_lon: float


class RoadPathUpdate(BaseModel):
    road_path: list[list[float]]  # list of [lat, lon] pairs


class NewArmRequest(BaseModel):
    arm_id: str
    name: str
    gps_lat: float
    gps_lon: float
    rtsp_url: str = ""


class NewJunctionRequest(BaseModel):
    junction_id: str
    name: str
    type: str  # "+", "T", or "L"
    peak_periods: list[list[int]] = [[7, 9], [17, 19]]
    first_arm: NewArmRequest


# ─── Endpoints ───

@admin_router.put("/junctions/{junction_id}/peak_periods")
def update_peak_periods(junction_id: str, req: PeakPeriodsUpdate):
    """Update peak periods for a junction."""
    if junction_id not in config.JUNCTIONS:
        raise HTTPException(404, f"Junction {junction_id} not found")

    for period in req.peak_periods:
        if len(period) != 2:
            raise HTTPException(400, "Each period must be [start_hour, end_hour]")
        start, end = period
        if not (0 <= start < end <= 24):
            raise HTTPException(400, f"Invalid period {period}: must satisfy 0 <= start < end <= 24")

    config.JUNCTIONS[junction_id]["peak_periods"] = [tuple(p) for p in req.peak_periods]

    _overrides.setdefault(junction_id, {})["peak_periods"] = req.peak_periods
    _save_overrides()

    logger.info("Updated peak_periods for %s: %s", junction_id, req.peak_periods)
    return {"junction_id": junction_id, "peak_periods": req.peak_periods}


@admin_router.put("/junctions/{junction_id}/arms/{arm_id}/location")
def update_arm_location(junction_id: str, arm_id: str, req: ArmLocationUpdate):
    """Update GPS position of a camera arm (drag-drop)."""
    if junction_id not in config.JUNCTIONS:
        raise HTTPException(404, f"Junction {junction_id} not found")
    if arm_id not in config.JUNCTIONS[junction_id]["arms"]:
        raise HTTPException(404, f"Arm {arm_id} not found in junction {junction_id}")
    if not (-90 <= req.gps_lat <= 90):
        raise HTTPException(400, f"Invalid latitude {req.gps_lat}")
    if not (-180 <= req.gps_lon <= 180):
        raise HTTPException(400, f"Invalid longitude {req.gps_lon}")

    config.JUNCTIONS[junction_id]["arms"][arm_id]["gps_lat"] = req.gps_lat
    config.JUNCTIONS[junction_id]["arms"][arm_id]["gps_lon"] = req.gps_lon

    _overrides.setdefault(junction_id, {}).setdefault("arms", {}).setdefault(arm_id, {}).update({
        "gps_lat": req.gps_lat,
        "gps_lon": req.gps_lon,
    })
    _save_overrides()

    logger.info("Updated location for %s/%s: (%.6f, %.6f)", junction_id, arm_id, req.gps_lat, req.gps_lon)
    return {"junction_id": junction_id, "arm_id": arm_id, "gps_lat": req.gps_lat, "gps_lon": req.gps_lon}


@admin_router.put("/junctions/{junction_id}/arms/{arm_id}/road_path")
def update_arm_road_path(junction_id: str, arm_id: str, req: RoadPathUpdate):
    """
    Set the road path geometry for a camera arm.
    The last point in the path becomes the arm's GPS anchor (tip closest to junction).
    """
    if junction_id not in config.JUNCTIONS:
        raise HTTPException(404, f"Junction {junction_id} not found")
    if arm_id not in config.JUNCTIONS[junction_id]["arms"]:
        raise HTTPException(404, f"Arm {arm_id} not found in junction {junction_id}")

    for i, coord in enumerate(req.road_path):
        if len(coord) != 2:
            raise HTTPException(400, f"Point {i} must be [lat, lon]")
        lat, lon = coord
        if not (-90 <= lat <= 90):
            raise HTTPException(400, f"Invalid latitude {lat} at point {i}")
        if not (-180 <= lon <= 180):
            raise HTTPException(400, f"Invalid longitude {lon} at point {i}")

    config.JUNCTIONS[junction_id]["arms"][arm_id]["road_path"] = req.road_path

    # Sync GPS anchor to tip of road (last point = closest to junction)
    if req.road_path:
        last = req.road_path[-1]
        config.JUNCTIONS[junction_id]["arms"][arm_id]["gps_lat"] = last[0]
        config.JUNCTIONS[junction_id]["arms"][arm_id]["gps_lon"] = last[1]

    arm_override = _overrides.setdefault(junction_id, {}).setdefault("arms", {}).setdefault(arm_id, {})
    arm_override["road_path"] = req.road_path
    if req.road_path:
        arm_override["gps_lat"] = req.road_path[-1][0]
        arm_override["gps_lon"] = req.road_path[-1][1]
    _save_overrides()

    logger.info("Updated road_path for %s/%s: %d points", junction_id, arm_id, len(req.road_path))
    return {"junction_id": junction_id, "arm_id": arm_id, "road_path": req.road_path}


@admin_router.post("/junctions")
def create_junction(req: NewJunctionRequest):
    """Create a new junction with an initial arm."""
    if req.junction_id in config.JUNCTIONS:
        raise HTTPException(409, f"Junction {req.junction_id} already exists")
    if req.type not in ("+", "T", "L"):
        raise HTTPException(400, f"Invalid junction type '{req.type}'. Must be one of: +, T, L")

    for period in req.peak_periods:
        if len(period) != 2:
            raise HTTPException(400, "Each period must be [start_hour, end_hour]")
        start, end = period
        if not (0 <= start < end <= 24):
            raise HTTPException(400, f"Invalid period {period}")

    first_arm = req.first_arm
    junction_dict = {
        "name": req.name,
        "type": req.type,
        "peak_periods": [tuple(p) for p in req.peak_periods],
        "arms": {
            first_arm.arm_id: {
                "name": first_arm.name,
                "gps_lat": first_arm.gps_lat,
                "gps_lon": first_arm.gps_lon,
                "rtsp_url": first_arm.rtsp_url,
                "counting_line_y": config.COUNTING_LINE_Y_FRACTION,
                "road_path": [],
            }
        },
    }

    config.JUNCTIONS[req.junction_id] = junction_dict

    # Persist — store with list peak_periods (JSON-serializable)
    persist_dict = dict(junction_dict)
    persist_dict["peak_periods"] = req.peak_periods
    persist_dict["arms"] = {
        first_arm.arm_id: dict(junction_dict["arms"][first_arm.arm_id])
    }
    _overrides[req.junction_id] = persist_dict
    _save_overrides()

    logger.info("Created new junction %s (%s)", req.junction_id, req.name)
    return {
        "junction_id": req.junction_id,
        "name": req.name,
        "type": req.type,
        "peak_periods": req.peak_periods,
        "arms": [
            {
                "arm_id": first_arm.arm_id,
                "name": first_arm.name,
                "gps_lat": first_arm.gps_lat,
                "gps_lon": first_arm.gps_lon,
                "road_path": [],
                "alert_level": "GREEN",
            }
        ],
    }


@admin_router.post("/junctions/{junction_id}/arms")
def add_arm(junction_id: str, req: NewArmRequest):
    """Add a new camera arm to an existing junction."""
    if junction_id not in config.JUNCTIONS:
        raise HTTPException(404, f"Junction {junction_id} not found")
    if req.arm_id in config.JUNCTIONS[junction_id]["arms"]:
        raise HTTPException(409, f"Arm {req.arm_id} already exists in junction {junction_id}")

    arm_dict = {
        "name": req.name,
        "gps_lat": req.gps_lat,
        "gps_lon": req.gps_lon,
        "rtsp_url": req.rtsp_url,
        "counting_line_y": config.COUNTING_LINE_Y_FRACTION,
        "road_path": [],
    }

    config.JUNCTIONS[junction_id]["arms"][req.arm_id] = arm_dict

    _overrides.setdefault(junction_id, {}).setdefault("arms", {})[req.arm_id] = arm_dict
    _save_overrides()

    logger.info("Added arm %s to junction %s", req.arm_id, junction_id)
    return {
        "junction_id": junction_id,
        "arm_id": req.arm_id,
        "name": req.name,
        "gps_lat": req.gps_lat,
        "gps_lon": req.gps_lon,
        "road_path": [],
        "alert_level": "GREEN",
    }
