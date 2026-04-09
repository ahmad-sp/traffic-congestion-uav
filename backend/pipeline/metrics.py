"""
Frame-level → per-minute metric aggregation.

Computes per-frame metrics from track states, then aggregates
into per-minute summaries stored to the database.
"""

import logging
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from backend.pipeline.tracking import TrackState
from backend.pipeline.counting_line import CountingLine

logger = logging.getLogger(__name__)


@dataclass
class FrameMetrics:
    """Metrics computed from a single frame."""
    timestamp: float
    vehicle_count: int = 0
    near_zone_count: int = 0
    far_zone_count: int = 0
    stopped_count: int = 0
    mean_bbox_area: float = 0.0
    max_bbox_area: float = 0.0
    occupancy_ratio: float = 0.0
    approach_count: int = 0
    far_zone_bbox_growth_rate: float = 0.0


@dataclass
class MinuteMetrics:
    """Aggregated per-minute metrics — matches the synthetic data schema."""
    timestamp: str  # ISO format
    junction_id: str
    arm_id: str
    camera_id: str
    VPM: int
    queue_depth: int
    stopped_ratio: float
    occupancy_pct: float
    mean_bbox_area: float
    max_bbox_area: float
    approach_flow: float
    time_sin: float
    time_cos: float
    is_peak_hour: int
    hour_of_week: int
    mean_speed_proxy: float = 0.0
    mean_bbox_growth_rate: float = 0.0


class MetricsAggregator:
    """
    Collects per-frame metrics and produces per-minute summaries.
    One instance per camera.
    """

    def __init__(self, junction_id: str, arm_id: str, frame_height: int, frame_width: int,
                 counting_line_y: float = config.COUNTING_LINE_Y_FRACTION,
                 recording_start_dt=None,   # datetime (timezone-aware) — offline mode
                 peak_periods=None):        # list[(int,int)] — per-junction override
        self.junction_id = junction_id
        self.arm_id = arm_id
        self.camera_id = f"{junction_id}_{arm_id}"
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.frame_area = frame_height * frame_width

        self._recording_start_dt = recording_start_dt
        self._offline = recording_start_dt is not None
        self._peak_periods = peak_periods or config.PEAK_PERIODS

        self.counting_line = CountingLine(frame_height, counting_line_y)

        # Accumulation buffers (reset each minute)
        self._frame_metrics: list[FrameMetrics] = []
        self._speed_proxies: list[float] = []
        self._near_zone_timestamps: list[float] = []
        self._far_zone_bbox_growth_rates: list[float] = []
        self._minute_start: float = time.time()
        self._minute_video_ts_start: float = 0.0   # video seconds at start of current minute
        self._last_video_ts: float = 0.0            # updated each frame

    def compute_frame_metrics(self, tracks: list[TrackState], timestamp: float) -> FrameMetrics:
        """Compute metrics from the current set of active tracks."""
        near_y = self.frame_height * config.NEAR_ZONE_Y_FRACTION
        far_y = self.frame_height * config.FAR_ZONE_Y_FRACTION

        fm = FrameMetrics(timestamp=timestamp)

        if not tracks:
            return fm

        fm.vehicle_count = len(tracks)

        areas = []
        for t in tracks:
            areas.append(t.bbox_area)

            if t.centroid_y > near_y:
                fm.near_zone_count += 1
            elif t.centroid_y < far_y:
                fm.far_zone_count += 1

            if t.is_stopped:
                fm.stopped_count += 1

            if t.bbox_area_delta > config.APPROACH_THRESHOLD:
                fm.approach_count += 1

            self._speed_proxies.append(t.speed_proxy)

        fm.mean_bbox_area = float(np.mean(areas))
        fm.max_bbox_area = float(np.max(areas))

        # Far-zone bbox growth rate: early congestion signal
        far_y_upper = self.frame_height * config.FAR_ZONE_UPPER_Y_RATIO
        far_deltas = [t.bbox_area_delta for t in tracks if t.centroid_y < far_y_upper]
        if far_deltas:
            fm.far_zone_bbox_growth_rate = float(np.mean(far_deltas))
            self._far_zone_bbox_growth_rates.append(fm.far_zone_bbox_growth_rate)

        total_area = sum(areas)
        fm.occupancy_ratio = (total_area / self.frame_area) * 100.0 if self.frame_area > 0 else 0.0

        # Track near-zone presence for queue depth
        if fm.near_zone_count > 0:
            self._near_zone_timestamps.append(timestamp)

        # Update counting line
        crossings = self.counting_line.update(tracks)
        active_ids = {t.track_id for t in tracks}
        self.counting_line.cleanup_stale(active_ids)

        self._frame_metrics.append(fm)
        self._last_video_ts = timestamp
        return fm

    def should_aggregate(self) -> bool:
        """Check if 60 seconds have passed since last aggregation."""
        if self._offline:
            return (self._last_video_ts - self._minute_video_ts_start) >= 60.0
        return (time.time() - self._minute_start) >= 60.0

    def aggregate_minute(self) -> MinuteMetrics | None:
        """
        Aggregate accumulated frame metrics into a per-minute summary.
        Returns None if no frames were collected.
        """
        if not self._frame_metrics:
            return None

        if self._offline:
            from datetime import timedelta
            dt = self._recording_start_dt + timedelta(seconds=self._minute_video_ts_start)
        else:
            from datetime import datetime, timezone
            dt = datetime.fromtimestamp(time.time(), tz=timezone.utc)
        minute_of_day = dt.hour * 60 + dt.minute
        time_sin = math.sin(2 * math.pi * minute_of_day / 1440)
        time_cos = math.cos(2 * math.pi * minute_of_day / 1440)
        h_of_w = dt.weekday() * 24 + dt.hour
        is_peak = int(any(s <= dt.hour < e for s, e in self._peak_periods))

        # VPM from counting line
        vpm = self.counting_line.reset_count()

        # Queue depth: near-zone vehicles sustained over threshold
        queue_depth = 0
        if self._near_zone_timestamps:
            duration = self._near_zone_timestamps[-1] - self._near_zone_timestamps[0]
            if duration >= config.QUEUE_DEPTH_SUSTAIN_SECONDS:
                # Average near-zone count
                nz_counts = [fm.near_zone_count for fm in self._frame_metrics if fm.near_zone_count > 0]
                queue_depth = int(np.mean(nz_counts)) if nz_counts else 0

        # Averages across frames
        total_vehicles = sum(fm.vehicle_count for fm in self._frame_metrics)
        total_stopped = sum(fm.stopped_count for fm in self._frame_metrics)
        stopped_ratio = total_stopped / total_vehicles if total_vehicles > 0 else 0.0

        occupancy_vals = [fm.occupancy_ratio for fm in self._frame_metrics]
        occupancy_pct = float(np.mean(occupancy_vals))

        bbox_areas = [fm.mean_bbox_area for fm in self._frame_metrics if fm.mean_bbox_area > 0]
        mean_bbox = float(np.mean(bbox_areas)) if bbox_areas else 0.0

        max_areas = [fm.max_bbox_area for fm in self._frame_metrics]
        max_bbox = float(np.max(max_areas)) if max_areas else 0.0

        approach_flow = sum(fm.approach_count for fm in self._frame_metrics) / max(1, len(self._frame_metrics))

        mean_speed = float(np.mean(self._speed_proxies)) if self._speed_proxies else 0.0

        mean_bbox_growth = float(np.mean(self._far_zone_bbox_growth_rates)) if self._far_zone_bbox_growth_rates else 0.0

        mm = MinuteMetrics(
            timestamp=dt.isoformat(),
            junction_id=self.junction_id,
            arm_id=self.arm_id,
            camera_id=self.camera_id,
            VPM=vpm,
            queue_depth=queue_depth,
            stopped_ratio=round(stopped_ratio, 4),
            occupancy_pct=round(occupancy_pct, 2),
            mean_bbox_area=round(mean_bbox, 1),
            max_bbox_area=round(max_bbox, 1),
            approach_flow=round(approach_flow, 2),
            time_sin=round(time_sin, 6),
            time_cos=round(time_cos, 6),
            is_peak_hour=is_peak,
            hour_of_week=h_of_w,
            mean_speed_proxy=round(mean_speed, 2),
            mean_bbox_growth_rate=round(mean_bbox_growth, 4),
        )

        # Reset buffers
        self._frame_metrics.clear()
        self._speed_proxies.clear()
        self._near_zone_timestamps.clear()
        self._far_zone_bbox_growth_rates.clear()
        self._minute_video_ts_start = self._last_video_ts
        self._minute_start = time.time()  # keep for online-mode fallback

        return mm
