"""
Unit tests for drone trigger packet schema validation.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.alerts.drone_trigger import DroneTriggerPacket, DroneTriggerManager
from backend.alerts.manager import Alert


REQUIRED_FIELDS = [
    "trigger_id",
    "timestamp_iso",
    "junction_id",
    "arm_id",
    "gps_lat",
    "gps_lon",
    "congestion_type",
    "severity_score",
    "lstm_score",
    "anomaly_score",
    "current_VPM",
    "queue_depth",
    "active_warrants",
    "evidence_clip_path",
    "evidence_snapshot_path",
]


def make_alert(**overrides) -> Alert:
    defaults = {
        "alert_id": "test-alert-001",
        "timestamp": "2026-03-15T08:30:00+00:00",
        "junction_id": "JCT01",
        "arm_id": "ARM_NORTH",
        "level": "RED",
        "congestion_type": "PEAK_EXCESS",
        "active_warrants": ["WARRANT_3", "WARRANT_X"],
        "lstm_score": 0.85,
        "anomaly_score": 0.002,
        "current_vpm": 35,
        "queue_depth": 12,
    }
    defaults.update(overrides)
    return Alert(**defaults)


class TestDroneTriggerPacketSchema:

    def test_all_required_fields_present(self):
        packet = DroneTriggerPacket(
            trigger_id="abc-123",
            timestamp_iso="2026-03-15T08:30:00Z",
            junction_id="JCT01",
            arm_id="ARM_NORTH",
            gps_lat=51.5074,
            gps_lon=-0.1278,
            congestion_type="PEAK_EXCESS",
            severity_score=0.85,
            lstm_score=0.85,
            anomaly_score=0.002,
            current_VPM=35,
            queue_depth=12,
            active_warrants=["WARRANT_X"],
            evidence_clip_path="/evidence/clip.mp4",
            evidence_snapshot_path="/evidence/snap.jpg",
        )
        d = packet.to_dict()
        for field in REQUIRED_FIELDS:
            assert field in d, f"Missing required field: {field}"

    def test_to_json_is_valid(self):
        packet = DroneTriggerPacket(
            trigger_id="abc-123",
            timestamp_iso="2026-03-15T08:30:00Z",
            junction_id="JCT01",
            arm_id="ARM_NORTH",
            gps_lat=51.5074,
            gps_lon=-0.1278,
            congestion_type="OFF_PEAK_JAM",
            severity_score=0.92,
            lstm_score=0.3,
            anomaly_score=0.015,
            current_VPM=2,
            queue_depth=10,
            active_warrants=["WARRANT_X"],
            evidence_clip_path="",
            evidence_snapshot_path="",
        )
        j = packet.to_json()
        parsed = json.loads(j)
        assert parsed["congestion_type"] == "OFF_PEAK_JAM"
        assert isinstance(parsed["active_warrants"], list)
        assert isinstance(parsed["gps_lat"], float)
        assert isinstance(parsed["current_VPM"], int)

    def test_field_types(self):
        packet = DroneTriggerPacket(
            trigger_id="test",
            timestamp_iso="2026-03-15T08:30:00Z",
            junction_id="JCT01",
            arm_id="ARM_NORTH",
            gps_lat=51.5074,
            gps_lon=-0.1278,
            congestion_type="PEAK_EXCESS",
            severity_score=0.85,
            lstm_score=0.85,
            anomaly_score=0.002,
            current_VPM=35,
            queue_depth=12,
            active_warrants=["WARRANT_3", "WARRANT_X"],
            evidence_clip_path="path.mp4",
            evidence_snapshot_path="path.jpg",
        )
        d = packet.to_dict()
        assert isinstance(d["trigger_id"], str)
        assert isinstance(d["gps_lat"], float)
        assert isinstance(d["gps_lon"], float)
        assert isinstance(d["severity_score"], float)
        assert isinstance(d["current_VPM"], int)
        assert isinstance(d["queue_depth"], int)
        assert isinstance(d["active_warrants"], list)

    def test_congestion_type_values(self):
        """Congestion type must be one of the two valid values."""
        for ct in ["OFF_PEAK_JAM", "PEAK_EXCESS"]:
            packet = DroneTriggerPacket(
                trigger_id="t", timestamp_iso="t", junction_id="J",
                arm_id="A", gps_lat=0.0, gps_lon=0.0,
                congestion_type=ct, severity_score=0.5,
                lstm_score=0.5, anomaly_score=0.5,
                current_VPM=10, queue_depth=5,
                active_warrants=[], evidence_clip_path="",
                evidence_snapshot_path="",
            )
            assert packet.congestion_type in ("OFF_PEAK_JAM", "PEAK_EXCESS")


class TestDroneTriggerManager:

    def test_compile_trigger_from_alert(self):
        manager = DroneTriggerManager()
        alert = make_alert()
        packet = manager.compile_trigger(alert)

        assert packet.junction_id == "JCT01"
        assert packet.arm_id == "ARM_NORTH"
        assert packet.congestion_type == "PEAK_EXCESS"
        assert packet.current_VPM == 35
        assert packet.queue_depth == 12
        assert "WARRANT_X" in packet.active_warrants
        assert len(packet.trigger_id) > 0

    def test_triggers_are_stored(self):
        manager = DroneTriggerManager()
        alert = make_alert()
        manager.compile_trigger(alert)
        manager.compile_trigger(alert)

        recent = manager.get_recent_triggers(limit=10)
        assert len(recent) == 2

    def test_offpeak_jam_trigger(self):
        manager = DroneTriggerManager()
        alert = make_alert(
            congestion_type="OFF_PEAK_JAM",
            lstm_score=0.3,
            anomaly_score=0.015,
            current_vpm=2,
            active_warrants=["WARRANT_X"],
        )
        packet = manager.compile_trigger(alert)
        assert packet.congestion_type == "OFF_PEAK_JAM"
        assert packet.lstm_score == 0.3
