"""
YOLOv8n vehicle detection wrapper.

Runs YOLOv8n on each frame, filtering for vehicle classes only.
Returns structured detections for the tracker.
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single detected vehicle."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    bbox_area: float

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


class VehicleDetector:
    """YOLOv8n-based vehicle detector."""

    def __init__(self, model_name: str = config.YOLO_MODEL_NAME,
                 device: str = config.YOLO_DEVICE,
                 confidence: float = config.YOLO_CONFIDENCE_THRESHOLD):
        self.device = device
        self.confidence = confidence
        self.target_classes = set(config.YOLO_TARGET_CLASSES)
        self._model = None
        self._model_name = model_name

    def _load_model(self):
        """Lazy-load the YOLO model on first use."""
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self._model_name)
            logger.info("Loaded %s on device=%s", self._model_name, self.device)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run detection on a single frame.

        Args:
            frame: BGR numpy array (H, W, 3)
        Returns:
            List of Detection objects for vehicle classes only.
        """
        self._load_model()

        results = self._model(frame, conf=self.confidence, device=self.device, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                if cls_id not in self.target_classes:
                    continue

                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].item())
                area = float((x2 - x1) * (y2 - y1))

                detections.append(Detection(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    confidence=conf,
                    class_id=cls_id,
                    bbox_area=area,
                ))

        return detections

    def detections_to_array(self, detections: list[Detection]) -> np.ndarray:
        """Convert detections to numpy array for ByteTrack: (N, 5) = [x1, y1, x2, y2, score]."""
        if not detections:
            return np.empty((0, 5), dtype=np.float32)
        return np.array(
            [[d.x1, d.y1, d.x2, d.y2, d.confidence] for d in detections],
            dtype=np.float32,
        )
