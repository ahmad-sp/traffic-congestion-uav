"""
Drone trigger packet compilation and dispatch.

On every RED alert, compile a drone trigger JSON payload, log it,
POST to webhook, and stream to WebSocket clients.
"""

import json
import logging
import sys
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from backend.alerts.manager import Alert

logger = logging.getLogger(__name__)


@dataclass
class DroneTriggerPacket:
    """Exact schema required by the drone system."""
    trigger_id: str
    timestamp_iso: str
    junction_id: str
    arm_id: str
    gps_lat: float
    gps_lon: float
    congestion_type: str  # OFF_PEAK_JAM or PEAK_EXCESS
    severity_score: float
    lstm_score: float
    anomaly_score: float
    current_VPM: int
    queue_depth: int
    active_warrants: list[str]
    evidence_clip_path: str
    evidence_snapshot_path: str

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class DroneTriggerManager:
    """Compiles and dispatches drone trigger packets."""

    def __init__(self):
        self._triggers: list[DroneTriggerPacket] = []
        self.on_trigger: callable | None = None  # WebSocket callback

    def compile_trigger(self, alert: Alert, evidence_clip: str = "",
                        evidence_snapshot: str = "") -> DroneTriggerPacket:
        """
        Compile a drone trigger packet from a RED alert.
        """
        arm_cfg = config.get_arm_config(alert.junction_id, alert.arm_id)

        # Severity: max of lstm_score and anomaly_score (normalized)
        severity = max(alert.lstm_score, min(1.0, alert.anomaly_score * 100))

        packet = DroneTriggerPacket(
            trigger_id=str(uuid.uuid4()),
            timestamp_iso=datetime.now(timezone.utc).isoformat(),
            junction_id=alert.junction_id,
            arm_id=alert.arm_id,
            gps_lat=arm_cfg["gps_lat"],
            gps_lon=arm_cfg["gps_lon"],
            congestion_type=alert.congestion_type or "PEAK_EXCESS",
            severity_score=round(severity, 3),
            lstm_score=round(alert.lstm_score, 4),
            anomaly_score=round(alert.anomaly_score, 4),
            current_VPM=alert.current_vpm,
            queue_depth=alert.queue_depth,
            active_warrants=alert.active_warrants,
            evidence_clip_path=evidence_clip,
            evidence_snapshot_path=evidence_snapshot,
        )

        self._triggers.append(packet)
        logger.info("Drone trigger compiled: %s for %s_%s type=%s",
                     packet.trigger_id, alert.junction_id, alert.arm_id,
                     packet.congestion_type)

        # Post to webhook
        self._post_webhook(packet)

        # Notify WebSocket clients
        if self.on_trigger:
            self.on_trigger(packet)

        return packet

    def _post_webhook(self, packet: DroneTriggerPacket):
        """POST trigger packet to configured webhook URL."""
        if not config.DRONE_WEBHOOK_URL:
            logger.debug("No DRONE_WEBHOOK_URL configured — skipping webhook POST")
            return

        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(config.DRONE_WEBHOOK_URL, json=packet.to_dict())
                resp.raise_for_status()
            logger.info("Webhook POST success: %s → %d", config.DRONE_WEBHOOK_URL, resp.status_code)
        except Exception as e:
            logger.error("Webhook POST failed: %s", e)

    def get_recent_triggers(self, limit: int = 20) -> list[DroneTriggerPacket]:
        return list(reversed(self._triggers[-limit:]))

    def get_trigger_by_id(self, trigger_id: str) -> DroneTriggerPacket | None:
        for t in self._triggers:
            if t.trigger_id == trigger_id:
                return t
        return None
