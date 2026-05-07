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
    brightness_delta_threshold: float = 90.0
    edge_strength_threshold: float = 12.0
    min_edge_coverage: float = 0.45
    min_boundary_x_range_ratio: float = 0.035
    min_boundary_turns: int = 1
    max_work_width: int = 480
    max_work_height: int = 480
    change_pixel_diff_threshold: int = 18
    change_diff_ratio_threshold: float = 0.006
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
        self._confirm_count = 0
        self._previous_frame_timestamp: float | None = None
        self._previous_full_gray: np.ndarray | None = None
        self.loading_seen_timestamp: float | None = None
        self.estimated_change_timestamp: float | None = None

    def update(self, frame: np.ndarray, timestamp: float) -> LoadingDetectorResult:
        full_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        full_gray = self._resize_for_detection(full_gray)
        roi = self._crop_roi(frame)
        if roi.size == 0:
            self._previous_frame_timestamp = timestamp
            self._previous_full_gray = full_gray
            return LoadingDetectorResult(
                phase=self.phase,
                timestamp=timestamp,
                metrics={"error": "empty_roi"},
            )

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        metrics: dict[str, float | int | str] = {}

        is_loading, metrics = self._detect_loading_boundary(gray)

        if self.phase is LoadingPhase.IDLE:
            if is_loading:
                self._confirm_count += 1
                if self._confirm_count >= self.config.confirm_frames:
                    self.phase = LoadingPhase.LOADING_SEEN
                    self.loading_seen_timestamp = timestamp
                    self._previous_full_gray = full_gray.copy()
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
            self._previous_full_gray = full_gray
            return LoadingDetectorResult(
                phase=self.phase,
                timestamp=timestamp,
                metrics=metrics,
            )

        if self.phase is LoadingPhase.MONITORING:
            changed, change_metrics = self._detect_full_frame_change(full_gray)
            metrics.update(change_metrics)
            if not changed:
                self._previous_frame_timestamp = timestamp
                self._previous_full_gray = full_gray
                return LoadingDetectorResult(
                    phase=self.phase,
                    timestamp=timestamp,
                    metrics=metrics,
                )

            self.phase = LoadingPhase.TRIGGERED
            previous = self._previous_frame_timestamp
            if previous is None:
                self.estimated_change_timestamp = timestamp
            else:
                self.estimated_change_timestamp = (previous + timestamp) / 2.0
            self._previous_frame_timestamp = timestamp
            self._previous_full_gray = full_gray
            return LoadingDetectorResult(
                phase=self.phase,
                triggered=True,
                timestamp=timestamp,
                estimated_change_timestamp=self.estimated_change_timestamp,
                metrics=metrics,
            )

        self._previous_frame_timestamp = timestamp
        self._previous_full_gray = full_gray
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
        work = self._resize_for_detection(gray)
        h, w = work.shape[:2]
        if h < 8 or w < 8:
            return False, {"error": "roi_too_small"}

        blurred = cv2.GaussianBlur(work, (5, 5), 0)
        # The loading screen has a strong near-vertical/zigzag bright-dark boundary.
        # Per-row horizontal gradient positions are scale-normalized below.
        gradient = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        abs_gradient = np.abs(gradient)

        boundary_x: list[int] = []
        strengths: list[float] = []
        brightness_deltas: list[float] = []
        sample_radius = max(3, int(w * 0.035))

        for y in range(h):
            row = abs_gradient[y]
            x = int(np.argmax(row))
            strength = float(row[x])
            if strength < self.config.edge_strength_threshold:
                continue
            left_start = max(0, x - sample_radius)
            right_end = min(w, x + sample_radius + 1)
            if x <= left_start or right_end <= x + 1:
                continue
            left_mean = float(np.mean(work[y, left_start:x]))
            right_mean = float(np.mean(work[y, x + 1 : right_end]))
            brightness_delta = abs(left_mean - right_mean)
            if brightness_delta < self.config.brightness_delta_threshold:
                continue
            boundary_x.append(x)
            strengths.append(strength)
            brightness_deltas.append(brightness_delta)

        coverage = len(boundary_x) / float(max(1, h))
        if not boundary_x:
            return False, {
                "edge_coverage": coverage,
                "found": 0,
            }

        x_values = np.asarray(boundary_x, dtype=np.float32)
        smoothed = self._smooth_1d(x_values)
        x_range_ratio = float((np.max(smoothed) - np.min(smoothed)) / max(1, w))
        turns = self._count_polyline_turns(smoothed)
        median_strength = float(np.median(strengths))
        median_delta = float(np.median(brightness_deltas))
        score = (
            min(1.0, coverage / max(1e-6, self.config.min_edge_coverage)) * 0.40
            + min(1.0, x_range_ratio / max(1e-6, self.config.min_boundary_x_range_ratio)) * 0.25
            + min(1.0, turns / max(1, self.config.min_boundary_turns)) * 0.20
            + min(1.0, median_delta / max(1e-6, self.config.brightness_delta_threshold)) * 0.15
        )
        found = (
            coverage >= self.config.min_edge_coverage
            and x_range_ratio >= self.config.min_boundary_x_range_ratio
            and turns >= self.config.min_boundary_turns
            and median_delta >= self.config.brightness_delta_threshold
        )
        return found, {
            "edge_coverage": coverage,
            "boundary_x_range_ratio": x_range_ratio,
            "boundary_turns": turns,
            "median_edge_strength": median_strength,
            "median_brightness_delta": median_delta,
            "loading_score": score,
            "work_width": w,
            "work_height": h,
            "found": int(found),
        }

    def _resize_for_detection(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape[:2]
        scale = min(
            1.0,
            self.config.max_work_width / float(max(1, w)),
            self.config.max_work_height / float(max(1, h)),
        )
        if scale >= 1.0:
            return gray
        return cv2.resize(
            gray,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )

    def _smooth_1d(self, values: np.ndarray) -> np.ndarray:
        if values.size < 5:
            return values
        kernel = np.ones(5, dtype=np.float32) / 5.0
        return np.convolve(values, kernel, mode="same")

    def _count_polyline_turns(self, values: np.ndarray) -> int:
        if values.size < 9:
            return 0
        # Down-sample before counting sign changes to avoid counting noise as bends.
        sample_count = min(32, max(8, values.size // 8))
        indexes = np.linspace(0, values.size - 1, sample_count).astype(np.int32)
        sampled = values[indexes]
        diff = np.diff(sampled)
        threshold = max(1.0, float(np.std(sampled)) * 0.12)
        signs = []
        for delta in diff:
            if abs(float(delta)) < threshold:
                continue
            signs.append(1 if delta > 0 else -1)
        if len(signs) < 2:
            return 0
        return sum(1 for prev, cur in zip(signs, signs[1:]) if prev != cur)

    def _detect_full_frame_change(
        self, gray: np.ndarray
    ) -> tuple[bool, dict[str, float | int | str]]:
        if self._previous_full_gray is None:
            return False, {"change_diff_ratio": 0.0, "changed": 0, "first_monitoring_frame": 1}
        previous = self._previous_full_gray
        if previous.shape != gray.shape:
            gray = cv2.resize(
                gray,
                (previous.shape[1], previous.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        diff = cv2.absdiff(previous, gray)
        changed_mask = diff > self.config.change_pixel_diff_threshold
        diff_ratio = float(np.count_nonzero(changed_mask)) / float(max(1, diff.size))
        changed = diff_ratio >= self.config.change_diff_ratio_threshold
        return changed, {
            "change_diff_ratio": diff_ratio,
            "change_pixel_diff_threshold": self.config.change_pixel_diff_threshold,
            "change_diff_ratio_threshold": self.config.change_diff_ratio_threshold,
            "changed": int(changed),
        }


class FreezeChangePhase(Enum):
    WAITING_STILL = "waiting_still"
    WAITING_CHANGE = "waiting_change"
    TRIGGERED = "triggered"


@dataclass(slots=True)
class FreezeChangeDetectorConfig:
    max_work_width: int = 480
    max_work_height: int = 480
    still_diff_ratio_threshold: float = 0.0015
    change_diff_ratio_threshold: float = 0.010
    pixel_diff_threshold: int = 18
    still_confirm_frames: int = 3
    change_confirm_frames: int = 1


@dataclass(slots=True)
class FreezeChangeDetectorResult:
    phase: FreezeChangePhase
    still_seen: bool = False
    triggered: bool = False
    timestamp: float | None = None
    estimated_change_timestamp: float | None = None
    metrics: dict[str, float | int | str] = field(default_factory=dict)


class FreezeChangeDetector:
    def __init__(self, config: FreezeChangeDetectorConfig | None = None) -> None:
        self.config = config or FreezeChangeDetectorConfig()
        self.phase = FreezeChangePhase.WAITING_STILL
        self._previous_gray: np.ndarray | None = None
        self._previous_timestamp: float | None = None
        self._still_count = 0
        self._change_count = 0
        self.estimated_change_timestamp: float | None = None

    def update(self, frame: np.ndarray, timestamp: float) -> FreezeChangeDetectorResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = self._resize_for_detection(gray)
        if self._previous_gray is None:
            self._previous_gray = gray
            self._previous_timestamp = timestamp
            return FreezeChangeDetectorResult(
                phase=self.phase,
                timestamp=timestamp,
                metrics={"diff_ratio": 0.0, "first_frame": 1},
            )

        diff_ratio = self._diff_ratio(self._previous_gray, gray)
        metrics = {
            "diff_ratio": diff_ratio,
            "still_count": self._still_count,
            "change_count": self._change_count,
        }

        if self.phase is FreezeChangePhase.WAITING_STILL:
            if diff_ratio <= self.config.still_diff_ratio_threshold:
                self._still_count += 1
                if self._still_count >= self.config.still_confirm_frames:
                    self.phase = FreezeChangePhase.WAITING_CHANGE
                    self._change_count = 0
                    self._previous_gray = gray
                    self._previous_timestamp = timestamp
                    metrics["still_seen"] = 1
                    return FreezeChangeDetectorResult(
                        phase=self.phase,
                        still_seen=True,
                        timestamp=timestamp,
                        metrics=metrics,
                    )
            else:
                self._still_count = 0

        elif self.phase is FreezeChangePhase.WAITING_CHANGE:
            if diff_ratio >= self.config.change_diff_ratio_threshold:
                self._change_count += 1
                if self._change_count >= self.config.change_confirm_frames:
                    self.phase = FreezeChangePhase.TRIGGERED
                    previous = self._previous_timestamp
                    self.estimated_change_timestamp = (
                        timestamp if previous is None else (previous + timestamp) / 2.0
                    )
                    self._previous_gray = gray
                    self._previous_timestamp = timestamp
                    metrics["changed"] = 1
                    return FreezeChangeDetectorResult(
                        phase=self.phase,
                        triggered=True,
                        timestamp=timestamp,
                        estimated_change_timestamp=self.estimated_change_timestamp,
                        metrics=metrics,
                    )
            else:
                self._change_count = 0

        self._previous_gray = gray
        self._previous_timestamp = timestamp
        return FreezeChangeDetectorResult(
            phase=self.phase,
            triggered=self.phase is FreezeChangePhase.TRIGGERED,
            timestamp=timestamp,
            estimated_change_timestamp=self.estimated_change_timestamp,
            metrics=metrics,
        )

    def _resize_for_detection(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape[:2]
        scale = min(
            1.0,
            self.config.max_work_width / float(max(1, w)),
            self.config.max_work_height / float(max(1, h)),
        )
        if scale >= 1.0:
            return gray
        return cv2.resize(
            gray,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )

    def _diff_ratio(self, previous: np.ndarray, current: np.ndarray) -> float:
        if previous.shape != current.shape:
            current = cv2.resize(
                current,
                (previous.shape[1], previous.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        diff = cv2.absdiff(previous, current)
        changed = diff > self.config.pixel_diff_threshold
        return float(np.count_nonzero(changed)) / float(max(1, diff.size))
