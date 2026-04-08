"""
Warrant Engine — rule-based alert logic.

Implements Warrants 1, 2, 3, and X as specified.
Runs after ML inference each minute and determines alert level.
"""

import logging
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from backend.warrants.baseline import get_baseline_vpm

logger = logging.getLogger(__name__)


@dataclass
class WarrantResult:
    """Result from a single warrant check."""
    name: str
    fired: bool
    details: dict = field(default_factory=dict)


@dataclass
class WarrantEngineOutput:
    """Combined output from all warrant checks for one arm."""
    junction_id: str
    arm_id: str
    timestamp: str
    warrants: list[WarrantResult]
    alert_level: str  # GREEN, AMBER, RED
    congestion_type: str | None  # OFF_PEAK_JAM, PEAK_EXCESS, or None

    @property
    def active_warrant_names(self) -> list[str]:
        return [w.name for w in self.warrants if w.fired]


class WarrantEngine:
    """
    Per-arm warrant engine. Maintains a rolling VPM history
    and evaluates all four warrants each minute.
    """

    def __init__(self, junction_id: str, arm_id: str):
        self.junction_id = junction_id
        self.arm_id = arm_id

        # Rolling VPM history: (timestamp_iso, vpm) for the last 8 hours
        max_minutes = config.W1_HOURS * 60
        self._vpm_history: deque[tuple[str, float]] = deque(maxlen=max_minutes)

    def push_vpm(self, timestamp_iso: str, vpm: float):
        """Record a new per-minute VPM value."""
        self._vpm_history.append((timestamp_iso, vpm))

    def evaluate(
        self,
        current_vpm: float,
        hour_of_week: int,
        lstm_score: float,
        lstm_ready: bool,
        anomaly_score: float,
        is_anomaly: bool,
        queue_depth: int,
    ) -> WarrantEngineOutput:
        """
        Run all warrants and determine alert level.

        Returns WarrantEngineOutput with alert level and active warrants.
        """
        now_iso = datetime.now(timezone.utc).isoformat()

        w1 = self._check_warrant_1()
        w2 = self._check_warrant_2()
        w3 = self._check_warrant_3(current_vpm, hour_of_week)
        wx = self._check_warrant_x(
            lstm_score, lstm_ready, w3.fired,
            anomaly_score, is_anomaly, queue_depth,
        )

        warrants = [w1, w2, w3, wx]

        # Determine alert level and congestion type
        congestion_type = None
        if wx.fired:
            alert_level = "RED"
            congestion_type = wx.details.get("congestion_type")
        elif any(w.fired for w in [w1, w2, w3]):
            alert_level = "AMBER"
        else:
            alert_level = "GREEN"

        output = WarrantEngineOutput(
            junction_id=self.junction_id,
            arm_id=self.arm_id,
            timestamp=now_iso,
            warrants=warrants,
            alert_level=alert_level,
            congestion_type=congestion_type,
        )

        if alert_level != "GREEN":
            logger.info(
                "Warrant evaluation %s_%s: level=%s, active=%s, type=%s",
                self.junction_id, self.arm_id, alert_level,
                output.active_warrant_names, congestion_type,
            )

        return output

    def _check_warrant_1(self) -> WarrantResult:
        """
        WARRANT_1: 8-hour sustained volume.
        Condition: VPM > W1_VOLUME_THRESHOLD for 8 of the last 8 hours.
        """
        if len(self._vpm_history) < config.W1_HOURS * 60:
            return WarrantResult("WARRANT_1", False, {"reason": "insufficient_data"})

        # Check hourly averages for each of the last 8 hours
        history = list(self._vpm_history)
        hours_above = 0
        entries_per_hour = 60

        for h in range(config.W1_HOURS):
            start = len(history) - (config.W1_HOURS - h) * entries_per_hour
            end = start + entries_per_hour
            if start < 0:
                continue
            hour_vpms = [v for _, v in history[start:end]]
            if hour_vpms:
                avg = sum(hour_vpms) / len(hour_vpms)
                if avg > config.W1_VOLUME_THRESHOLD:
                    hours_above += 1

        fired = hours_above >= config.W1_HOURS
        return WarrantResult("WARRANT_1", fired, {
            "hours_above": hours_above,
            "required": config.W1_HOURS,
            "threshold": config.W1_VOLUME_THRESHOLD,
        })

    def _check_warrant_2(self) -> WarrantResult:
        """
        WARRANT_2: 4-hour consecutive elevated volume.
        Condition: VPM > W2_VOLUME_THRESHOLD for 4 consecutive hours.
        """
        needed = config.W2_HOURS * 60
        if len(self._vpm_history) < needed:
            return WarrantResult("WARRANT_2", False, {"reason": "insufficient_data"})

        # Check the last 4 hours — each hour's average must exceed threshold
        history = list(self._vpm_history)
        consecutive = 0
        entries_per_hour = 60

        for h in range(config.W2_HOURS):
            start = len(history) - (config.W2_HOURS - h) * entries_per_hour
            end = start + entries_per_hour
            if start < 0:
                consecutive = 0
                continue
            hour_vpms = [v for _, v in history[start:end]]
            if hour_vpms:
                avg = sum(hour_vpms) / len(hour_vpms)
                if avg > config.W2_VOLUME_THRESHOLD:
                    consecutive += 1
                else:
                    consecutive = 0

        fired = consecutive >= config.W2_HOURS
        return WarrantResult("WARRANT_2", fired, {
            "consecutive_hours": consecutive,
            "required": config.W2_HOURS,
            "threshold": config.W2_VOLUME_THRESHOLD,
        })

    def _check_warrant_3(self, current_vpm: float, hour_of_week: int) -> WarrantResult:
        """
        WARRANT_3: Peak-hour excess.
        Condition: current VPM > baseline_85th[hour_of_week] * W3_PEAK_MULTIPLIER.
        """
        baseline = get_baseline_vpm(self.junction_id, self.arm_id, hour_of_week)
        if baseline is None:
            return WarrantResult("WARRANT_3", False, {"reason": "no_baseline"})

        threshold = baseline * config.W3_PEAK_MULTIPLIER
        fired = current_vpm > threshold

        return WarrantResult("WARRANT_3", fired, {
            "current_vpm": current_vpm,
            "baseline_85th": baseline,
            "multiplier": config.W3_PEAK_MULTIPLIER,
            "effective_threshold": round(threshold, 2),
        })

    def _check_warrant_x(
        self,
        lstm_score: float,
        lstm_ready: bool,
        warrant_3_fired: bool,
        anomaly_score: float,
        is_anomaly: bool,
        queue_depth: int,
    ) -> WarrantResult:
        """
        WARRANT_X: Abnormal congestion — the hard alert trigger.
        Condition A (Case B): LSTM congestion_score >= 0.7 AND WARRANT_3 fires.
        Condition B (Case A): reconstruction_error > threshold AND queue_depth > 0.
        """
        condition_a = (
            lstm_ready
            and lstm_score >= config.LSTM_CONGESTION_THRESHOLD
            and warrant_3_fired
        )
        condition_b = is_anomaly and queue_depth > 0

        fired = condition_a or condition_b

        congestion_type = None
        if condition_b:
            congestion_type = "OFF_PEAK_JAM"
        elif condition_a:
            congestion_type = "PEAK_EXCESS"

        return WarrantResult("WARRANT_X", fired, {
            "condition_a": condition_a,
            "condition_b": condition_b,
            "congestion_type": congestion_type,
            "lstm_score": lstm_score,
            "lstm_ready": lstm_ready,
            "anomaly_score": anomaly_score,
            "is_anomaly": is_anomaly,
            "queue_depth": queue_depth,
        })
