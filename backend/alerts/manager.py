"""
Alert state machine with debounce logic.

Manages per-arm alert levels (GREEN/AMBER/RED) with:
- 5-minute debounce for same-level re-issue
- Immediate escalation (AMBER → RED)
- 2 consecutive clean minutes before de-escalation
"""

import logging
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from backend.warrants.engine import WarrantEngineOutput

logger = logging.getLogger(__name__)

LEVEL_ORDER = {"GREEN": 0, "AMBER": 1, "EARLY_RED": 2, "RED": 3}


@dataclass
class Alert:
    """A single alert record."""
    alert_id: str
    timestamp: str
    junction_id: str
    arm_id: str
    level: str  # GREEN, AMBER, RED
    congestion_type: str | None
    active_warrants: list[str]
    lstm_score: float = 0.0
    anomaly_score: float = 0.0
    current_vpm: int = 0
    queue_depth: int = 0
    confirmed: bool | None = None  # operator feedback
    notes: str = ""


class AlertManager:
    """
    Manages alert state for all arms.
    Handles debounce, escalation, and de-escalation.
    """

    def __init__(self):
        # Per-arm state
        self._current_level: dict[str, str] = {}  # camera_id → level
        self._last_alert_time: dict[str, datetime] = {}
        self._last_early_red_time: dict[str, datetime] = {}  # separate debounce for EARLY_RED
        self._last_red_time: dict[str, datetime] = {}  # cooldown for RED (prevents oscillation drones)
        self._clean_minutes: dict[str, int] = {}  # consecutive GREEN minutes

        # Alert log
        self._alerts: list[Alert] = []

        # Callback for new alerts (set by main app to push to WebSocket / DB)
        self.on_alert: callable | None = None
        # Callback for EARLY_RED events (set by main app to log to DB)
        self.on_early_red: callable | None = None

    def _camera_key(self, junction_id: str, arm_id: str) -> str:
        return f"{junction_id}_{arm_id}"

    def process_warrant_output(
        self,
        output: WarrantEngineOutput,
        lstm_score: float = 0.0,
        anomaly_score: float = 0.0,
        current_vpm: int = 0,
        queue_depth: int = 0,
    ) -> Alert | None:
        """
        Process warrant engine output and decide whether to issue an alert.

        Returns an Alert if one was issued, None if debounced/suppressed.
        """
        key = self._camera_key(output.junction_id, output.arm_id)
        new_level = output.alert_level
        current_level = self._current_level.get(key, "GREEN")
        now = datetime.now(timezone.utc)

        new_order = LEVEL_ORDER[new_level]
        current_order = LEVEL_ORDER[current_level]

        # --- De-escalation logic ---
        if new_order < current_order:
            clean = self._clean_minutes.get(key, 0) + 1
            self._clean_minutes[key] = clean
            if clean < config.ALERT_DEESCALATE_MINUTES:
                logger.debug("%s: de-escalation pending (%d/%d clean minutes)",
                             key, clean, config.ALERT_DEESCALATE_MINUTES)
                return None
            # De-escalation confirmed
            logger.info("%s: de-escalating %s → %s", key, current_level, new_level)
        else:
            self._clean_minutes[key] = 0

        # --- Escalation: immediate, but RED has a cooldown to prevent
        #     oscillation-driven drone spam (GREEN→RED→GREEN→RED…) ---
        if new_order > current_order:
            if new_level == "RED":
                last_red = self._last_red_time.get(key)
                if last_red and (now - last_red) < timedelta(minutes=config.ALERT_DEBOUNCE_MINUTES):
                    logger.debug("%s: RED escalation suppressed — cooldown active", key)
                    return None
            logger.info("%s: ESCALATING %s → %s", key, current_level, new_level)
            return self._issue_alert(key, output, lstm_score, anomaly_score,
                                     current_vpm, queue_depth, now)

        # --- Same level: debounce ---
        if new_level == current_level and new_level != "GREEN":
            last = self._last_alert_time.get(key)
            if last and (now - last) < timedelta(minutes=config.ALERT_DEBOUNCE_MINUTES):
                return None

        # --- Issue alert ---
        if new_level != "GREEN" or (new_level != current_level):
            return self._issue_alert(key, output, lstm_score, anomaly_score,
                                     current_vpm, queue_depth, now)

        return None

    def process_extreme_risk(
        self,
        junction_id: str,
        arm_id: str,
        extreme_congestion_risk: float,
        queue_depth: int,
        mean_bbox_growth_rate: float = 0.0,
        current_vpm: int = 0,
        stopped_ratio: float = 0.0,
        occupancy_pct: float = 0.0,
    ) -> Alert | None:
        """
        Check if EARLY_RED should fire based on extreme congestion risk.

        EARLY_RED triggers when:
          - extreme_congestion_risk >= threshold
          - queue_depth == 0 (queue hasn't arrived yet — this is a prediction)
          - current level < RED

        EARLY_RED does NOT trigger warrants, drones, or modify warrant counters.
        """
        key = self._camera_key(junction_id, arm_id)
        current_level = self._current_level.get(key, "GREEN")
        current_order = LEVEL_ORDER[current_level]
        now = datetime.now(timezone.utc)

        if (
            extreme_congestion_risk >= config.EXTREME_CONGESTION_FORECAST_THRESHOLD
            and queue_depth == 0
            and current_order < LEVEL_ORDER["RED"]
        ):
            # Debounce check — unconditional, regardless of current state
            # Uses its own timer so de-escalation back to GREEN doesn't bypass the window
            last = self._last_early_red_time.get(key)
            if last and (now - last) < timedelta(minutes=config.ALERT_DEBOUNCE_MINUTES):
                return None

            logger.info(
                "EARLY_RED: Early extreme congestion detected "
                "junction_id=%s arm_id=%s risk=%.3f mean_bbox_growth_rate=%.4f "
                "VPM=%d stopped_ratio=%.2f",
                junction_id, arm_id, extreme_congestion_risk,
                mean_bbox_growth_rate, current_vpm, stopped_ratio,
            )

            alert = Alert(
                alert_id=str(uuid.uuid4()),
                timestamp=now.isoformat(),
                junction_id=junction_id,
                arm_id=arm_id,
                level="EARLY_RED",
                congestion_type="EARLY_EXTREME",
                active_warrants=[],
                lstm_score=extreme_congestion_risk,
                anomaly_score=0.0,
                current_vpm=current_vpm,
                queue_depth=queue_depth,
            )

            self._current_level[key] = "EARLY_RED"
            self._last_alert_time[key] = now
            self._last_early_red_time[key] = now
            self._alerts.append(alert)

            if self.on_alert:
                self.on_alert(alert)

            if self.on_early_red:
                self.on_early_red({
                    "event_id": alert.alert_id,
                    "timestamp": alert.timestamp,
                    "junction_id": junction_id,
                    "arm_id": arm_id,
                    "extreme_congestion_risk": extreme_congestion_risk,
                    "mean_bbox_growth_rate": mean_bbox_growth_rate,
                    "VPM": current_vpm,
                    "stopped_ratio": stopped_ratio,
                    "occupancy_pct": occupancy_pct,
                })

            return alert

        return None

    def _issue_alert(
        self,
        key: str,
        output: WarrantEngineOutput,
        lstm_score: float,
        anomaly_score: float,
        current_vpm: int,
        queue_depth: int,
        now: datetime,
    ) -> Alert:
        alert = Alert(
            alert_id=str(uuid.uuid4()),
            timestamp=now.isoformat(),
            junction_id=output.junction_id,
            arm_id=output.arm_id,
            level=output.alert_level,
            congestion_type=output.congestion_type,
            active_warrants=output.active_warrant_names,
            lstm_score=lstm_score,
            anomaly_score=anomaly_score,
            current_vpm=current_vpm,
            queue_depth=queue_depth,
        )

        self._current_level[key] = output.alert_level
        self._last_alert_time[key] = now
        if output.alert_level == "RED":
            self._last_red_time[key] = now
        self._alerts.append(alert)

        logger.info("Alert issued: %s level=%s warrants=%s type=%s",
                     key, alert.level, alert.active_warrants, alert.congestion_type)

        if self.on_alert:
            self.on_alert(alert)

        return alert

    def get_current_level(self, junction_id: str, arm_id: str) -> str:
        key = self._camera_key(junction_id, arm_id)
        return self._current_level.get(key, "GREEN")

    def get_alerts(self, limit: int = 50, level: str | None = None) -> list[Alert]:
        alerts = self._alerts
        if level:
            alerts = [a for a in alerts if a.level == level]
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)[:limit]

    def get_alert_by_id(self, alert_id: str) -> Alert | None:
        for a in self._alerts:
            if a.alert_id == alert_id:
                return a
        return None

    def submit_feedback(self, alert_id: str, confirmed: bool, notes: str = "") -> bool:
        alert = self.get_alert_by_id(alert_id)
        if alert is None:
            return False
        alert.confirmed = confirmed
        alert.notes = notes
        logger.info("Feedback for %s: confirmed=%s notes=%s", alert_id, confirmed, notes)
        return True
