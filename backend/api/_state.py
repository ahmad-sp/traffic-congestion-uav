"""
Shared application state — singleton accessors for the pipeline components.

This module avoids circular imports by providing lazy access to
alert manager, warrant engines, and latest metrics.
"""

import logging

logger = logging.getLogger(__name__)

# Set by main.py at startup
_alert_manager = None
_warrant_engines = {}
_latest_metrics = {}  # {camera_id: dict}
_active_warrants = []  # cached list of active warrants


def set_alert_manager(mgr):
    global _alert_manager
    _alert_manager = mgr


def get_alert_manager():
    return _alert_manager


def set_warrant_engines(engines: dict):
    global _warrant_engines
    _warrant_engines = engines


def get_warrant_engines() -> dict:
    return _warrant_engines


def update_latest_metrics(junction_id: str, arm_id: str, metrics: dict):
    _latest_metrics[f"{junction_id}_{arm_id}"] = metrics


def get_latest_metrics(junction_id: str, arm_id: str) -> dict:
    return _latest_metrics.get(f"{junction_id}_{arm_id}", {})


def set_active_warrants(warrants: list):
    global _active_warrants
    _active_warrants = warrants


def get_active_warrants() -> list:
    return _active_warrants
