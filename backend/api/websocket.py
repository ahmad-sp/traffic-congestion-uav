"""
WebSocket manager — broadcasts live metrics, alerts, and drone triggers
to connected dashboard clients.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

from fastapi import WebSocket

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages active WebSocket connections and broadcasts messages."""

    def __init__(self):
        self._connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self._connections.append(ws)
        logger.info("WebSocket client connected (%d total)", len(self._connections))

    def disconnect(self, ws: WebSocket):
        if ws in self._connections:
            self._connections.remove(ws)
        logger.info("WebSocket client disconnected (%d remaining)", len(self._connections))

    async def broadcast(self, message: dict):
        """Send a message to all connected clients."""
        if not self._connections:
            return

        payload = json.dumps(message)
        disconnected = []

        for ws in self._connections:
            try:
                await ws.send_text(payload)
            except Exception:
                disconnected.append(ws)

        for ws in disconnected:
            self.disconnect(ws)

    async def send_metrics(self, junction_id: str, arm_id: str, data: dict):
        await self.broadcast({
            "type": "metrics",
            "junction_id": junction_id,
            "arm_id": arm_id,
            "data": data,
        })

    async def send_alert(self, junction_id: str, arm_id: str, data: dict):
        await self.broadcast({
            "type": "alert",
            "junction_id": junction_id,
            "arm_id": arm_id,
            "data": data,
        })

    async def send_drone_trigger(self, junction_id: str, arm_id: str, data: dict):
        await self.broadcast({
            "type": "drone_trigger",
            "junction_id": junction_id,
            "arm_id": arm_id,
            "data": data,
        })

    @property
    def client_count(self) -> int:
        return len(self._connections)


# Singleton
ws_manager = ConnectionManager()
