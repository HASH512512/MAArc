from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np


class LoadingPhase(Enum):
    IDLE = "idle"
    LOADING_SEEN = "loading_seen"
    MONITORING = "monitoring"
    TRIGGERED = "triggered"


@dataclass(slots=True)
class LoadingDetectorConfig:
    roi: tuple[float, float, float, float] = (0.50, 0.05, 0.98, 0.95)
    brightness_ratio_threshold: float = 1.25
    min_contour_area_ratio: float = 0.30
    approx_epsilon_ratio: float = 0.012
    min_vertices: int = 3
    min_turns: int = 2
    frame_diff_threshold: int = 20
    frame_diff_ratio_threshold: float = 0.003
    confirm_frames: int = 2


@dataclass(slots=True)
class LoadingDetectorResult:
    phase: LoadingPhase
    triggered: bool = False
    loading_seen: bool = False
    timestamp: float | None = None
    estimated_change_timestamp: float | None = None
    metrics: dict[str, float | int | str] = field(default_factory=dict)


class LoadingEndDetector:
    def __init__(self, config: LoadingDetectorConfig | None = None) -> None:
        self.config = config or LoadingDetectorConfig()
        self.phase = LoadingPhase.IDLE
        self._reference_gray: np.ndarray | None = None
        self._confirm_count = 0
        self._previous_frame_timestamp: float | None = None
        self.loading_seen_timestamp: float | None = None
        self.estimated_change_timestamp: float | None = None

    def update(self, frame: np.ndarray, timestamp: float) -> LoadingDetectorResult:
        roi = self._crop_roi(frame)
        if roi.size == 0:
            self._previous_frame_timestamp = timestamp
            return LoadingDetectorResult(
                phase=self.phase,
                timestamp=timestamp,
                metrics={"error": "empty_roi"},
            )

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        metrics: dict[str, float | int | str] = {}

        if self.phase is LoadingPhase.IDLE:
            found, metrics = self._detect_loading_boundary(gray)
            if found:
                self._confirm_count += 1
                if self._confirm_count >= self.config.confirm_frames:
                    self.phase = LoadingPhase.LOADING_SEEN
                    self.loading_seen_timestamp = timestamp
                    self._reference_gray = gray.copy()
                    self.phase = LoadingPhase.MONITORING
                    self._previous_frame_timestamp = timestamp
                    return LoadingDetectorResult(
                        phase=self.phase,
                        loading_seen=True,
                        timestamp=timestamp,
                        metrics=metrics,
                    )
            else:
                self._confirm_count = 0
            self._previous_frame_timestamp = timestamp
            return LoadingDetectorResult(
                phase=self.phase,
                timestamp=timestamp,
                metrics=metrics,
            )

        if self.phase is LoadingPhase.MONITORING:
            if self._reference_gray is None:
                self._reference_gray = gray.copy()
                self._previous_frame_timestamp = timestamp
                return LoadingDetectorResult(phase=self.phase, timestamp=timestamp)

            changed, metrics = self._detect_frame_change(gray)
            if changed:
                self.phase = LoadingPhase.TRIGGERED
                previous = self._previous_frame_timestamp
                if previous is None:
                    self.estimated_change_timestamp = timestamp
                else:
                    self.estimated_change_timestamp = (previous + timestamp) / 2.0
                self._previous_frame_timestamp = timestamp
                return LoadingDetectorResult(
                    phase=self.phase,
                    triggered=True,
                    timestamp=timestamp,
                    estimated_change_timestamp=self.estimated_change_timestamp,
                    metrics=metrics,
                )

        self._previous_frame_timestamp = timestamp
        return LoadingDetectorResult(
            phase=self.phase,
            triggered=self.phase is LoadingPhase.TRIGGERED,
            timestamp=timestamp,
            estimated_change_timestamp=self.estimated_change_timestamp,
            metrics=metrics,
        )

    def _crop_roi(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        left, top, right, bottom = self.config.roi
        x0 = max(0, min(w, int(w * left)))
        y0 = max(0, min(h, int(h * top)))
        x1 = max(0, min(w, int(w * right)))
        y1 = max(0, min(h, int(h * bottom)))
        if x1 <= x0 or y1 <= y0:
            return frame[0:0, 0:0]
        return frame[y0:y1, x0:x1]

    def _detect_loading_boundary(
        self, gray: np.ndarray
    ) -> tuple[bool, dict[str, float | int | str]]:
        h, w = gray.shape[:2]
        left_mean = float(np.mean(gray[:, : max(1, w // 2)]))
        right_mean = float(np.mean(gray[:, max(1, w // 2) :]))
        dark = max(1.0, min(left_mean, right_mean))
        bright = max(left_mean, right_mean)
        brightness_ratio = bright / dark

        _threshold, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        contours, _hierarchy = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return False, {
                "brightness_ratio": brightness_ratio,
                "contour_count": 0,
            }

        largest = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(largest))
        area_ratio = area / float(max(1, h * w))
        perimeter = max(1.0, float(cv2.arcLength(largest, True)))
        approx = cv2.approxPolyDP(
            largest, self.config.approx_epsilon_ratio * perimeter, True
        )
        vertices = int(len(approx))
        turns = self._count_turns(approx)
        found = (
            brightness_ratio >= self.config.brightness_ratio_threshold
            and area_ratio >= self.config.min_contour_area_ratio
            and vertices >= self.config.min_vertices
            and turns >= self.config.min_turns
        )
        return found, {
            "brightness_ratio": brightness_ratio,
            "contour_count": len(contours),
            "largest_area_ratio": area_ratio,
            "vertices": vertices,
            "turns": turns,
            "found": int(found),
        }

    def _count_turns(self, approx: np.ndarray) -> int:
        points = approx.reshape(-1, 2)
        if len(points) < 3:
            return 0
        turns = 0
        for idx in range(len(points)):
            prev_pt = points[idx - 1].astype(np.float32)
            cur_pt = points[idx].astype(np.float32)
            next_pt = points[(idx + 1) % len(points)].astype(np.float32)
            v1 = prev_pt - cur_pt
            v2 = next_pt - cur_pt
            norm = float(np.linalg.norm(v1) * np.linalg.norm(v2))
            if norm <= 1e-6:
                continue
            cos_angle = float(np.clip(np.dot(v1, v2) / norm, -1.0, 1.0))
            angle = float(np.degrees(np.arccos(cos_angle)))
            if 20.0 <= angle <= 160.0:
                turns += 1
        return turns

    def _detect_frame_change(
        self, gray: np.ndarray
    ) -> tuple[bool, dict[str, float | int | str]]:
        assert self._reference_gray is not None
        ref = self._reference_gray
        if ref.shape != gray.shape:
            gray = cv2.resize(gray, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
        diff = cv2.absdiff(ref, gray)
        changed_mask = diff > self.config.frame_diff_threshold
        changed_ratio = float(np.count_nonzero(changed_mask)) / float(max(1, diff.size))
        changed = changed_ratio >= self.config.frame_diff_ratio_threshold
        return changed, {
            "changed_ratio": changed_ratio,
            "changed": int(changed),
        }
