"""
Frame ingestion — reads from RTSP streams or local video files.

Each camera gets its own reader thread that pushes frames into a queue
at the configured FPS rate.
"""

import logging
import queue
import threading
import time
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config

logger = logging.getLogger(__name__)


class FramePacket:
    """A single frame with metadata."""
    __slots__ = ("frame", "camera_id", "junction_id", "arm_id", "timestamp", "frame_number")

    def __init__(self, frame: np.ndarray, camera_id: str, junction_id: str,
                 arm_id: str, timestamp: float, frame_number: int):
        self.frame = frame
        self.camera_id = camera_id
        self.junction_id = junction_id
        self.arm_id = arm_id
        self.timestamp = timestamp
        self.frame_number = frame_number


class CameraReader:
    """
    Reads frames from an RTSP stream or local video file for one camera.
    Drops frames to maintain the target FPS.
    """

    def __init__(self, junction_id: str, arm_id: str, source: str,
                 target_fps: int = config.FRAME_RATE,
                 queue_maxsize: int = config.FRAME_QUEUE_MAXSIZE):
        self.junction_id = junction_id
        self.arm_id = arm_id
        self.camera_id = f"{junction_id}_{arm_id}"
        self.source = source
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps

        self.frame_queue: queue.Queue[FramePacket] = queue.Queue(maxsize=queue_maxsize)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame_count = 0

    def start(self):
        """Start the reader thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._read_loop, daemon=True,
                                         name=f"reader-{self.camera_id}")
        self._thread.start()
        logger.info("Started reader for %s from %s", self.camera_id, self.source)

    def stop(self):
        """Signal the reader to stop and wait."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        logger.info("Stopped reader for %s", self.camera_id)

    def _read_loop(self):
        """Main capture loop — runs in a background thread."""
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            logger.error("Failed to open video source for %s: %s", self.camera_id, self.source)
            return

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_skip = max(1, int(round(source_fps / self.target_fps)))

        logger.info("%s source FPS=%.1f, skip=%d → effective ~%d FPS",
                    self.camera_id, source_fps, frame_skip, self.target_fps)

        frame_idx = 0
        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                # If reading a file, loop back to start (demo mode)
                if not self.source.startswith("rtsp"):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    logger.debug("Looping video for %s", self.camera_id)
                    continue
                else:
                    logger.warning("Lost RTSP stream for %s, retrying...", self.camera_id)
                    time.sleep(2.0)
                    cap.release()
                    cap = cv2.VideoCapture(self.source)
                    continue

            frame_idx += 1
            if frame_idx % frame_skip != 0:
                continue

            self._frame_count += 1
            packet = FramePacket(
                frame=frame,
                camera_id=self.camera_id,
                junction_id=self.junction_id,
                arm_id=self.arm_id,
                timestamp=time.time(),
                frame_number=self._frame_count,
            )

            try:
                self.frame_queue.put_nowait(packet)
            except queue.Full:
                # Drop oldest frame to keep up
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                self.frame_queue.put_nowait(packet)

        cap.release()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


class IngestionManager:
    """Manages camera readers for all configured junctions/arms."""

    def __init__(self):
        self.readers: dict[str, CameraReader] = {}

    def start_all(self):
        """Start readers for all configured cameras."""
        for jid, jdata in config.JUNCTIONS.items():
            for aid, arm_cfg in jdata["arms"].items():
                source = arm_cfg.get("rtsp_url", "")
                if not source and config.DEMO_VIDEO_PATH:
                    source = config.DEMO_VIDEO_PATH
                if not source:
                    logger.warning("No video source for %s_%s — skipping", jid, aid)
                    continue

                camera_id = f"{jid}_{aid}"
                reader = CameraReader(jid, aid, source)
                self.readers[camera_id] = reader
                reader.start()

    def stop_all(self):
        for reader in self.readers.values():
            reader.stop()

    def get_frame(self, camera_id: str, timeout: float = 1.0) -> FramePacket | None:
        """Get next frame for a specific camera. Returns None on timeout."""
        reader = self.readers.get(camera_id)
        if not reader:
            return None
        try:
            return reader.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
