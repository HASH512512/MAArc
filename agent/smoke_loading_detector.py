from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from loading_detector import LoadingDetectorConfig, LoadingEndDetector


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def _read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise ValueError(f"Empty image file: {path}")
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to decode image: {path}")
    return image


def _collect_images(paths: list[str], recursive: bool) -> list[Path]:
    if not paths:
        paths = ["tests"]
    images: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path.is_dir():
            iterator = path.rglob("*") if recursive else path.iterdir()
            images.extend(
                child
                for child in sorted(iterator)
                if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES
            )
        elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(path)
        else:
            raise ValueError(f"Not an image file or directory: {path}")
    return images


def _config_from_args(args: argparse.Namespace) -> LoadingDetectorConfig:
    return LoadingDetectorConfig(
        roi=tuple(args.roi),
        brightness_delta_threshold=args.brightness_delta_threshold,
        edge_strength_threshold=args.edge_strength_threshold,
        min_edge_coverage=args.min_edge_coverage,
        min_boundary_x_range_ratio=args.min_boundary_x_range_ratio,
        min_boundary_turns=args.min_boundary_turns,
        max_work_width=args.max_work_width,
        max_work_height=args.max_work_height,
        change_pixel_diff_threshold=args.change_pixel_diff_threshold,
        change_diff_ratio_threshold=args.change_diff_ratio_threshold,
        confirm_frames=args.confirm_frames,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MAArc loading-end detector against offline screenshots."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Image files or directories. Defaults to tests/.",
    )
    parser.add_argument(
        "--roi",
        nargs=4,
        type=float,
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
        default=(0.50, 0.05, 0.98, 0.95),
        help="Relative ROI used for loading boundary detection.",
    )
    parser.add_argument("--brightness-delta-threshold", type=float, default=90.0)
    parser.add_argument("--edge-strength-threshold", type=float, default=12.0)
    parser.add_argument("--min-edge-coverage", type=float, default=0.45)
    parser.add_argument("--min-boundary-x-range-ratio", type=float, default=0.035)
    parser.add_argument("--min-boundary-turns", type=int, default=1)
    parser.add_argument("--max-work-width", type=int, default=480)
    parser.add_argument("--max-work-height", type=int, default=480)
    parser.add_argument("--change-pixel-diff-threshold", type=int, default=18)
    parser.add_argument("--change-diff-ratio-threshold", type=float, default=0.006)
    parser.add_argument("--confirm-frames", type=int, default=2)
    parser.add_argument(
        "--timestamp-step",
        type=float,
        default=0.1,
        help="Synthetic seconds between screenshots.",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="Optional output JSONL path.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan directories. Disabled by default to avoid chart jacket images under tests/samples.",
    )
    args = parser.parse_args()

    images = _collect_images(args.paths, recursive=args.recursive)
    if not images:
        raise SystemExit("No images found")

    detector = LoadingEndDetector(_config_from_args(args))
    records = []
    for index, image_path in enumerate(images):
        frame = _read_image(image_path)
        timestamp = index * args.timestamp_step
        result = detector.update(frame, timestamp)
        record = {
            "index": index,
            "path": str(image_path),
            "shape": list(frame.shape[:2]),
            "timestamp": timestamp,
            "phase": result.phase.value,
            "loading_seen": result.loading_seen,
            "triggered": result.triggered,
            "estimated_change_timestamp": result.estimated_change_timestamp,
            "metrics": result.metrics,
        }
        records.append(record)
        print(json.dumps(record, ensure_ascii=False))

    if args.jsonl is not None:
        out_path = args.jsonl
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Wrote {len(records)} records to {out_path}")


if __name__ == "__main__":
    main()
