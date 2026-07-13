from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Iterable

import numpy as np
from ultralytics import YOLO

from .config import ModelConfig


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    center_x: float
    width: float
    height: float
    model: str

    @property
    def direction(self) -> str:
        return "左侧" if self.center_x < 0.33 else "右侧" if self.center_x > 0.67 else "前方"


class GuidanceEngine:
    """Run one or more YOLO models and produce rate-limited guidance cues."""

    def __init__(self, configs: Iterable[ModelConfig], cooldown_seconds: float = 2.0):
        self.configs = list(configs)
        self.models = {config.name: YOLO(str(config.weights)) for config in self.configs}
        self.cooldown_seconds = cooldown_seconds
        self._last_message = ""
        self._last_spoken_at = 0.0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        height, width = frame.shape[:2]
        detections: list[Detection] = []
        for config in self.configs:
            result = self.models[config.name](frame, conf=config.confidence, verbose=False)[0]
            for box in result.boxes:
                class_id = int(box.cls.item())
                label = config.classes.get(class_id, result.names[class_id])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(Detection(label, float(box.conf.item()), ((x1 + x2) / 2) / width,
                                            (x2 - x1) / width, (y2 - y1) / height, config.name))
        return detections

    def guidance(self, detections: Iterable[Detection]) -> str | None:
        """Prioritize obstacles, then tactile paving; monocular boxes provide no metric distance."""
        items = list(detections)
        paving = {"horizontal-directional-tactile", "vertical-directional-tactile", "warning-tactile", "tactile paving"}
        hazards = [item for item in items if item.label not in paving]
        if hazards:
            target = max(hazards, key=lambda item: item.confidence * item.width * item.height)
            message = f"注意，{target.direction}发现{target.label}。"
        elif any(item.label == "warning-tactile" for item in items):
            message = "前方为警示盲道，请减速并确认周边环境。"
        elif any(item.label == "horizontal-directional-tactile" for item in items):
            message = "检测到横向导向盲道，请确认行进方向。"
        elif any(item.label == "vertical-directional-tactile" for item in items):
            message = "检测到纵向导向盲道，可沿盲道继续前行。"
        else:
            return None
        now = monotonic()
        if message == self._last_message and now - self._last_spoken_at < self.cooldown_seconds:
            return None
        self._last_message, self._last_spoken_at = message, now
        return message
