from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import cv2
import numpy as np

cv2.setUseOptimized(True)


def _imread_unicode(path: Path, flags: int = cv2.IMREAD_COLOR):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


@dataclass(slots=True)
class VisionRuntimeConfig:
    ui_left_roi: tuple[float, float, float, float]
    ground_roi: tuple[float, float, float, float]
    ui_feature_threshold: float
    ground_blue_ratio_threshold: float
    arc_color_ratio_threshold: float
    arc_logic_roi_half_x: float
    arc_logic_roi_half_y: float
    stream_crop_roi: tuple[float, float, float, float]


FEATURE_ROI_MAX_WIDTH = 420


@dataclass(slots=True)
class VisionMetrics:
    ui_left_feature_score: float = 0.0
    ui_left_good_matches: int = 0
    ui_left_inliers: int = 0
    ui_left_keypoints: int = 0
    ui_left_template_count: int = 0
    ui_left_template_keypoints_max: int = 0
    ui_left_roi_shape: tuple[int, int] = (0, 0)
    ui_left_descriptor_ready: bool = False
    ui_left_template_dir: str = ""
    ui_pass: bool = False
    ground_blue_ratio: float = 0.0
    ground_blue_pixels: int = 0
    ground_roi_pixels: int = 0
    ground_band_ratio: float = 0.0
    ground_band_rect: tuple[int, int, int, int] | None = None
    ground_pass: bool = False
    arc_color_ratio: float = 0.0
    arc_color_pixels: int = 0
    arc_roi_pixels: int = 0
    arc_target_distance: float = 0.0
    arc_pass: bool = False
    last_arc_rect: tuple[int, int, int, int] | None = None


@dataclass(slots=True)
class VisionPerfMetrics:
    stage: str = "idle"
    total_ms: float = 0.0
    roi_ms: float = 0.0
    resize_ms: float = 0.0
    gray_ms: float = 0.0
    orb_ms: float = 0.0
    match_ms: float = 0.0
    homography_ms: float = 0.0
    hsv_ms: float = 0.0
    mask_ms: float = 0.0
    count_ms: float = 0.0
    decision_ms: float = 0.0


class VisionDetector:
    def __init__(self, ref_dir: Path, use_cuda: bool = True) -> None:
        self.ref_dir = ref_dir
        self.debug_dump_enabled = os.getenv("VISION_DEBUG_DUMP", "0") == "1"
        self.debug_dump_dir = self.ref_dir.resolve() / "_vision_debug"
        self.metrics = VisionMetrics()
        self.perf = VisionPerfMetrics()
        self._ground_hit_streak = 0
        self._arc_hit_streak = 0
        self.runtime = VisionRuntimeConfig(
            ui_left_roi=(0.005, 0.02, 0.34, 0.20),
            ground_roi=(0.12, 1310 / 1440, 0.88, 1345 / 1440),
            ui_feature_threshold=0.08,
            ground_blue_ratio_threshold=0.03,
            arc_color_ratio_threshold=0.02,
            arc_logic_roi_half_x=0.25,
            arc_logic_roi_half_y=0.25,
            stream_crop_roi=(0.0, 0.0, 1.0, 1.0),
        )

        self.cuda_enabled = False
        self.cuda_canny = None
        self.ui_left_templates = self._load_ui_left_templates()

        self.orb = cv2.ORB_create(
            nfeatures=1100, fastThreshold=10, scaleFactor=1.15, nlevels=8
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.ui_left_template_features: list[
            tuple[list, np.ndarray | None, tuple[int, int]]
        ] = []
        for tpl in self.ui_left_templates:
            gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
            kp, des = self.orb.detectAndCompute(gray, None)
            self.ui_left_template_features.append((kp, des, gray.shape[:2]))
        self.metrics.ui_left_template_count = len(self.ui_left_template_features)
        self.metrics.ui_left_template_keypoints_max = max(
            (len(kp) for kp, _des, _shape in self.ui_left_template_features),
            default=0,
        )
        self.metrics.ui_left_descriptor_ready = bool(self.ui_left_template_features)
        self.metrics.ui_left_template_dir = str(self.ref_dir.resolve())

    def _template_crop_variants(self, image: np.ndarray) -> list[np.ndarray]:
        h, w = image.shape[:2]
        variants = [image]
        if w < 40 or h < 20:
            return variants

        crop_ratios = (0.55, 0.70, 0.85)
        for ratio in crop_ratios:
            crop_w = max(16, int(w * ratio))
            if crop_w >= w:
                continue
            variants.append(image[:, :crop_w])
            variants.append(image[:, w - crop_w :])

        return variants

    def set_runtime(self, runtime: VisionRuntimeConfig) -> None:
        self.runtime = runtime

    def _load_ui_left_templates(self) -> list[np.ndarray]:
        templates: list[np.ndarray] = []
        img = _imread_unicode(self.ref_dir / "uileft.png", cv2.IMREAD_COLOR)
        if img is not None:
            templates.extend(self._template_crop_variants(img))
        return templates

    def _roi_rect(
        self, frame: np.ndarray, roi: tuple[float, float, float, float]
    ) -> tuple[int, int, int, int]:
        h, w = frame.shape[:2]
        crop_x0, crop_y0, crop_x1, crop_y1 = self.runtime.stream_crop_roi
        crop_w = max(1e-6, crop_x1 - crop_x0)
        crop_h = max(1e-6, crop_y1 - crop_y0)
        local_x0 = (roi[0] - crop_x0) / crop_w
        local_y0 = (roi[1] - crop_y0) / crop_h
        local_x1 = (roi[2] - crop_x0) / crop_w
        local_y1 = (roi[3] - crop_y0) / crop_h
        x0 = int(w * local_x0)
        y0 = int(h * local_y0)
        x1 = int(w * local_x1)
        y1 = int(h * local_y1)
        x0, x1 = sorted((max(0, x0), min(w, x1)))
        y0, y1 = sorted((max(0, y0), min(h, y1)))
        return x0, y0, x1, y1

    def _match_feature_score(
        self,
        roi: np.ndarray,
        template_features: list[tuple[list, np.ndarray | None, tuple[int, int]]],
    ) -> tuple[float, int, int, dict[str, float]]:
        perf = {
            "resize_ms": 0.0,
            "gray_ms": 0.0,
            "orb_ms": 0.0,
            "match_ms": 0.0,
            "homography_ms": 0.0,
            "roi_keypoints": 0.0,
        }
        if roi.shape[1] > FEATURE_ROI_MAX_WIDTH:
            t0 = cv2.getTickCount()
            scale = FEATURE_ROI_MAX_WIDTH / float(roi.shape[1])
            roi = cv2.resize(
                roi,
                (FEATURE_ROI_MAX_WIDTH, max(1, int(roi.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
            perf["resize_ms"] += (
                (cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000.0
            )

        t0 = cv2.getTickCount()
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        perf["gray_ms"] += (cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000.0
        t0 = cv2.getTickCount()
        kp_f, des_f = self.orb.detectAndCompute(gray, None)
        perf["orb_ms"] += (cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000.0
        perf["roi_keypoints"] = float(len(kp_f))
        if des_f is None or len(kp_f) < 8:
            return 0.0, 0, 0, perf

        best_score = 0.0
        best_good = 0
        best_inliers = 0
        for kp_t, des_t, _shape in template_features:
            if des_t is None or len(kp_t) < 8:
                continue

            t0 = cv2.getTickCount()
            knn = self.matcher.knnMatch(des_t, des_f, k=2)
            good = []
            for pair in knn:
                if len(pair) != 2:
                    continue
                m, n = pair
                if m.distance < 0.72 * n.distance:
                    good.append(m)
            perf["match_ms"] += (
                (cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000.0
            )

            inliers = 0
            if len(good) >= 8:
                t0 = cv2.getTickCount()
                src = np.float32([kp_t[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst = np.float32([kp_f[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                _h, mask = cv2.findHomography(src, dst, cv2.RANSAC, 2.5)
                if mask is not None:
                    inliers = int(mask.sum())
                perf["homography_ms"] += (
                    (cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000.0
                )

            denom = max(len(kp_t), 1)
            score = (0.55 * len(good) + 0.95 * inliers) / float(denom)
            if score > best_score:
                best_score = score
                best_good = len(good)
                best_inliers = inliers
        return best_score, best_good, best_inliers, perf

    def detect_ui_panel(self, frame: np.ndarray) -> bool:
        total_start = cv2.getTickCount()
        roi_start = cv2.getTickCount()
        self.metrics.arc_pass = False
        self.metrics.arc_color_ratio = 0.0
        self.metrics.arc_color_pixels = 0
        self.metrics.arc_roi_pixels = 0
        self.metrics.last_arc_rect = None
        left_score, left_good, left_inliers = 0.0, 0, 0
        lx0, ly0, lx1, ly1 = self._roi_rect(frame, self.runtime.ui_left_roi)
        left_roi = frame[ly0:ly1, lx0:lx1]
        roi_ms = (cv2.getTickCount() - roi_start) / cv2.getTickFrequency() * 1000.0
        feature_perf = {
            "resize_ms": 0.0,
            "gray_ms": 0.0,
            "orb_ms": 0.0,
            "match_ms": 0.0,
            "homography_ms": 0.0,
            "roi_keypoints": 0.0,
        }
        if left_roi.size > 0 and self.ui_left_template_features:
            left_score, left_good, left_inliers, feature_perf = (
                self._match_feature_score(
                    left_roi,
                    self.ui_left_template_features,
                )
            )

        self.metrics.ui_left_feature_score = left_score
        self.metrics.ui_left_good_matches = left_good
        self.metrics.ui_left_inliers = left_inliers
        self.metrics.ui_left_keypoints = int(feature_perf["roi_keypoints"])
        self.metrics.ui_left_roi_shape = (
            int(left_roi.shape[1]) if left_roi.size > 0 else 0,
            int(left_roi.shape[0]) if left_roi.size > 0 else 0,
        )

        decision_start = cv2.getTickCount()
        passed = left_score >= self.runtime.ui_feature_threshold
        decision_ms = (
            (cv2.getTickCount() - decision_start) / cv2.getTickFrequency() * 1000.0
        )
        self.metrics.ui_pass = passed
        total_ms = (cv2.getTickCount() - total_start) / cv2.getTickFrequency() * 1000.0
        self.perf = VisionPerfMetrics(
            stage="ui_left",
            total_ms=total_ms,
            roi_ms=roi_ms,
            resize_ms=feature_perf["resize_ms"],
            gray_ms=feature_perf["gray_ms"],
            orb_ms=feature_perf["orb_ms"],
            match_ms=feature_perf["match_ms"],
            homography_ms=feature_perf["homography_ms"],
            decision_ms=decision_ms,
        )
        return passed

    def detect_ground_overlap(
        self,
        frame: np.ndarray,
        logic_x: float = 0.5,
        logic_y: float = 0.0,
    ) -> bool:
        total_start = cv2.getTickCount()
        roi_start = cv2.getTickCount()
        self.metrics.last_arc_rect = None
        self.metrics.arc_pass = False
        self.metrics.arc_color_ratio = 0.0
        self.metrics.arc_color_pixels = 0
        self.metrics.arc_roi_pixels = 0
        x0, y0, x1, y1 = self._logic_to_pixel_rect(frame, logic_x, logic_y)
        roi = frame[y0:y1, x0:x1]
        roi_ms = (cv2.getTickCount() - roi_start) / cv2.getTickFrequency() * 1000.0
        if roi.size == 0:
            self.metrics.ground_pass = False
            self.perf = VisionPerfMetrics(stage="ground", roi_ms=roi_ms)
            return False
        resize_ms = 0.0
        self.metrics.ground_band_rect = (x0, y0, x1, y1)
        hsv_start = cv2.getTickCount()
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hsv_ms = (cv2.getTickCount() - hsv_start) / cv2.getTickFrequency() * 1000.0
        mask_start = cv2.getTickCount()
        blue_mask = (
            (hsv[:, :, 0] >= 90)
            & (hsv[:, :, 0] <= 135)
            & (hsv[:, :, 1] >= 70)
            & (hsv[:, :, 2] >= 70)
        ).astype(np.uint8) * 255
        mask_ms = (cv2.getTickCount() - mask_start) / cv2.getTickFrequency() * 1000.0

        count_start = cv2.getTickCount()
        blue_pixels = int(cv2.countNonZero(blue_mask))
        roi_pixels = int(roi.shape[0] * roi.shape[1])
        ratio = blue_pixels / float(max(1, roi_pixels))
        count_ms = (cv2.getTickCount() - count_start) / cv2.getTickFrequency() * 1000.0

        self.metrics.ground_blue_pixels = blue_pixels
        self.metrics.ground_roi_pixels = roi_pixels
        self.metrics.ground_blue_ratio = ratio
        self.metrics.ground_band_ratio = ratio
        decision_start = cv2.getTickCount()
        hit = ratio >= self.runtime.ground_blue_ratio_threshold
        self.metrics.ground_pass = self._continuous_pass(hit, "_ground_hit_streak")
        decision_ms = (
            (cv2.getTickCount() - decision_start) / cv2.getTickFrequency() * 1000.0
        )
        total_ms = (cv2.getTickCount() - total_start) / cv2.getTickFrequency() * 1000.0
        self.perf = VisionPerfMetrics(
            stage="ground",
            total_ms=total_ms,
            roi_ms=roi_ms,
            resize_ms=resize_ms,
            hsv_ms=hsv_ms,
            mask_ms=mask_ms,
            count_ms=count_ms,
            decision_ms=decision_ms,
        )
        return self.metrics.ground_pass

    def _logic_to_pixel_rect(
        self,
        frame: np.ndarray,
        logic_x: float,
        logic_y: float,
    ) -> tuple[int, int, int, int]:
        half_x = self.runtime.arc_logic_roi_half_x
        half_y = self.runtime.arc_logic_roi_half_y
        # Solver logical Y uses 0=ground(bottom), 1=sky(top), while image Y is top-down.
        screen_logic_y = 1.0 - float(logic_y)
        lx0 = max(0.0, min(1.0, logic_x - half_x))
        ly0 = max(0.0, min(1.0, screen_logic_y - half_y))
        lx1 = max(0.0, min(1.0, logic_x + half_x))
        ly1 = max(0.0, min(1.0, screen_logic_y + half_y))
        return self._roi_rect(frame, (lx0, ly0, lx1, ly1))

    def _logic_x_to_norm(self, logic_x: float) -> float:
        return max(0.0, min(1.0, (logic_x + 0.25) / 1.5))

    def _continuous_pass(self, hit: bool, streak_name: str) -> bool:
        current = getattr(self, streak_name)
        current = current + 1 if hit else 0
        setattr(self, streak_name, current)
        return current >= 2

    def detect_arc_overlap(
        self, frame: np.ndarray, logic_x: float, logic_y: float
    ) -> bool:
        total_start = cv2.getTickCount()
        roi_start = cv2.getTickCount()
        x0, y0, x1, y1 = self._logic_to_pixel_rect(frame, logic_x, logic_y)
        self.metrics.last_arc_rect = (x0, y0, x1, y1)
        roi = frame[y0:y1, x0:x1]
        roi_ms = (cv2.getTickCount() - roi_start) / cv2.getTickFrequency() * 1000.0
        if roi.size == 0:
            self.metrics.arc_pass = False
            self.perf = VisionPerfMetrics(stage="arc", roi_ms=roi_ms)
            return False

        resize_ms = 0.0
        hsv_start = cv2.getTickCount()
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hsv_ms = (cv2.getTickCount() - hsv_start) / cv2.getTickFrequency() * 1000.0
        mask_start = cv2.getTickCount()
        blue_mask = (
            (hsv[:, :, 0] >= 90)
            & (hsv[:, :, 0] <= 135)
            & (hsv[:, :, 1] >= 60)
            & (hsv[:, :, 2] >= 70)
        )
        red_mask = (
            ((hsv[:, :, 0] <= 10) | (hsv[:, :, 0] >= 165))
            & (hsv[:, :, 1] >= 60)
            & (hsv[:, :, 2] >= 70)
        )
        purple_white_mask = (
            (hsv[:, :, 0] >= 120)
            & (hsv[:, :, 0] <= 165)
            & (hsv[:, :, 1] >= 20)
            & (hsv[:, :, 2] >= 120)
        )
        arc_mask = (blue_mask | red_mask | purple_white_mask).astype(np.uint8) * 255
        mask_ms = (cv2.getTickCount() - mask_start) / cv2.getTickFrequency() * 1000.0

        count_start = cv2.getTickCount()
        color_pixels = int(cv2.countNonZero(arc_mask))
        roi_pixels = int(roi.shape[0] * roi.shape[1])
        ratio = color_pixels / float(max(1, roi_pixels))
        target_distance = float("inf")
        moments = cv2.moments(arc_mask, binaryImage=True)
        if moments["m00"] > 0.0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            target_distance = (
                (cx - roi.shape[1] / 2.0) ** 2 + (cy - roi.shape[0] / 2.0) ** 2
            ) ** 0.5
        count_ms = (cv2.getTickCount() - count_start) / cv2.getTickFrequency() * 1000.0

        self.metrics.arc_color_pixels = color_pixels
        self.metrics.arc_roi_pixels = roi_pixels
        self.metrics.arc_color_ratio = ratio
        self.metrics.arc_target_distance = (
            0.0 if target_distance == float("inf") else target_distance
        )
        decision_start = cv2.getTickCount()
        distance_threshold = max(2.0, min(roi.shape[:2]) * 0.18)
        hit = (
            ratio >= self.runtime.arc_color_ratio_threshold
            and target_distance <= distance_threshold
        )
        self.metrics.arc_pass = self._continuous_pass(hit, "_arc_hit_streak")
        decision_ms = (
            (cv2.getTickCount() - decision_start) / cv2.getTickFrequency() * 1000.0
        )
        total_ms = (cv2.getTickCount() - total_start) / cv2.getTickFrequency() * 1000.0
        self.perf = VisionPerfMetrics(
            stage="arc",
            total_ms=total_ms,
            roi_ms=roi_ms,
            resize_ms=resize_ms,
            hsv_ms=hsv_ms,
            mask_ms=mask_ms,
            count_ms=count_ms,
            decision_ms=decision_ms,
        )
        return self.metrics.arc_pass

    def render_overlay(
        self, frame: np.ndarray, stage: str, decode_fps: float | None
    ) -> np.ndarray:
        overlay = frame.copy()
        ui_left = self._roi_rect(frame, self.runtime.ui_left_roi)
        ground = self._roi_rect(frame, self.runtime.ground_roi)
        cv2.rectangle(
            overlay,
            (ui_left[0], ui_left[1]),
            (ui_left[2], ui_left[3]),
            (220, 180, 40),
            2,
        )
        cv2.rectangle(
            overlay, (ground[0], ground[1]), (ground[2], ground[3]), (80, 255, 80), 2
        )
        if self.metrics.ground_band_rect is not None:
            gx0, gy0, gx1, gy1 = self.metrics.ground_band_rect
            cv2.rectangle(overlay, (gx0, gy0), (gx1, gy1), (0, 255, 255), 2)

        if self.metrics.last_arc_rect is not None:
            ax0, ay0, ax1, ay1 = self.metrics.last_arc_rect
            cv2.rectangle(
                overlay,
                (ax0, ay0),
                (ax1, ay1),
                (255, 255, 0),
                2,
            )

        lines = [
            f"stage={stage}",
            f"fps={decode_fps:.1f}" if decode_fps is not None else "fps=n/a",
            f"ui pass={self.metrics.ui_pass} left={self.metrics.ui_left_feature_score:.3f}",
            f"ground pass={self.metrics.ground_pass} band={self.metrics.ground_band_ratio:.3f}",
            f"arc pass={self.metrics.arc_pass} color={self.metrics.arc_color_ratio:.3f} dist={self.metrics.arc_target_distance:.1f}",
        ]
        text_metrics = [
            cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.68, 2)[0]
            for text in lines
        ]
        max_w = max((size[0] for size in text_metrics), default=200)
        x = max(10, overlay.shape[1] - max_w - 22)
        y = max(26, overlay.shape[0] - 24 * len(lines) - 14)
        for text in lines:
            cv2.putText(
                overlay,
                text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (30, 250, 255),
                2,
                cv2.LINE_AA,
            )
            y += 24
        return overlay
