from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .detector import VisionDetector, VisionRuntimeConfig


def _parse_roi(text: str) -> tuple[float, float, float, float]:
    if not isinstance(text, str):
        return tuple(text)  # type: ignore[return-value]
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI must be x0,y0,x1,y1")
    return tuple(float(p) for p in parts)  # type: ignore[return-value]


def _collect_images(inputs: list[str]) -> list[Path]:
    result: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
                result.extend(sorted(p.glob(ext)))
        elif any(ch in item for ch in "*?[]"):
            result.extend(sorted(Path().glob(item)))
        else:
            result.append(p)
    dedup = []
    seen = set()
    for p in result:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(p)
    return dedup


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline vision replay tuner")
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Image files, directories, or glob patterns",
    )
    parser.add_argument(
        "--ref-dir", default="ref/opencv", help="Reference material directory"
    )
    parser.add_argument(
        "--ground-roi",
        type=_parse_roi,
        default=(0.12, 1310 / 1440, 0.88, 1345 / 1440),
    )
    parser.add_argument(
        "--ui-left-roi", type=_parse_roi, default=(0.005, 0.02, 0.34, 0.20)
    )
    parser.add_argument("--ui-threshold", type=float, default=0.08)
    parser.add_argument("--ground-threshold", type=float, default=0.03)
    parser.add_argument("--arc-threshold", type=float, default=0.02)
    parser.add_argument("--arc-logic-x", type=float, default=0.25)
    parser.add_argument("--arc-logic-y", type=float, default=0.25)
    parser.add_argument("--arc-center", nargs=2, type=float, default=(0.5, 0.5))
    parser.add_argument(
        "--save-overlay-dir", default="", help="Output folder for debug overlay images"
    )
    parser.add_argument(
        "--no-cuda", action="store_true", help="Disable CUDA even if available"
    )
    args = parser.parse_args()

    images = _collect_images(args.inputs)
    if not images:
        raise SystemExit("No input images found")

    detector = VisionDetector(Path(args.ref_dir), use_cuda=not args.no_cuda)
    detector.set_runtime(
        VisionRuntimeConfig(
            ui_left_roi=args.ui_left_roi,
            ground_roi=args.ground_roi,
            ui_feature_threshold=args.ui_threshold,
            ground_blue_ratio_threshold=args.ground_threshold,
            arc_color_ratio_threshold=args.arc_threshold,
            arc_logic_roi_half_x=args.arc_logic_x,
            arc_logic_roi_half_y=args.arc_logic_y,
            stream_crop_roi=(0.0, 0.0, 1.0, 1.0),
        )
    )

    out_dir = Path(args.save_overlay_dir) if args.save_overlay_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(
        "file,ui_pass,ui_score,ui_good,ui_inliers,ground_pass,ground_overlap,arc_pass,arc_overlap"
    )
    for img_path in images:
        frame = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if frame is None:
            print(f"{img_path},READ_FAIL,0,0,0,0,0,0,0")
            continue

        ui_pass = detector.detect_ui_panel(frame)
        ground_pass = detector.detect_ground_overlap(
            frame,
            float(args.arc_center[0]),
            float(args.arc_center[1]),
        )
        arc_pass = detector.detect_arc_overlap(
            frame,
            float(args.arc_center[0]),
            float(args.arc_center[1]),
        )
        m = detector.metrics

        print(
            f"{img_path},{int(ui_pass)},{m.ui_left_feature_score:.4f},{m.ui_left_good_matches},{m.ui_left_inliers},{int(ground_pass)},{m.ground_blue_ratio:.4f},{int(arc_pass)},{m.arc_color_ratio:.4f}"
        )

        if out_dir is not None:
            overlay = detector.render_overlay(frame, stage="offline", decode_fps=None)
            out_path = out_dir / f"{img_path.stem}_overlay{img_path.suffix}"
            cv2.imwrite(str(out_path), overlay)


if __name__ == "__main__":
    main()
