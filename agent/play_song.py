from __future__ import annotations

import ctypes
import json
import queue
import statistics
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from autoplay.domain.arcaea_ir import ArcIR, HoldIR, TapIR
from autoplay.domain.chart import Chart
from autoplay.runtime.config_store import load_app_config
from autoplay.solver import CoordConv, ProjectiveCoordConv, solve_chart_auto

from loading_detector import FreezeChangeDetector, FreezeChangeDetectorConfig, LoadingEndDetector
from touch_backends import MaaTouchBackend, TouchBackend, create_touch_backend


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_ROOT = PROJECT_ROOT / "assets"
TIMING_LOG_DIR = PROJECT_ROOT / "debug" / "timing"
FRAME_TRACE_DIR = PROJECT_ROOT / "debug" / "frames"
STATE_TRACE_DIR = PROJECT_ROOT / "debug" / "state_trace"
FIRST_TOUCH_DEBUG_DIR = PROJECT_ROOT / "debug" / "first_touch"
LOAD_CHART_DEBUG_DIR = PROJECT_ROOT / "debug" / "load_chart"
AUTO_CALIBRATION_LOGICAL_POINTS = [
    (-0.5, -0.2),
    (0.1, 1.0),
    (0.9, 1.0),
    (1.5, -0.2),
]


@dataclass(slots=True)
class PlayCache:
    events_by_time: dict[int, list] = field(default_factory=dict)
    first_tick_ms: int = 0
    chart_path: Path | None = None
    tick_count: int = 0
    event_count: int = 0
    calibration: dict[str, tuple[int, int]] = field(default_factory=dict)
    screen_resolution: tuple[int, int] | None = None
    calibration_mode: str = "manual"


@dataclass(slots=True)
class TouchBackendCache:
    backend: TouchBackend | None = None
    name: str = ""
    maa_wait_mode: str = ""
    adb_serial: str | None = None
    scrcpy_max_fps: int = 60
    scrcpy_video_bit_rate: int | None = None
    scrcpy_video_crop: tuple[int, int, int, int] | None = None


PLAY_CACHE = PlayCache()
TOUCH_BACKEND_CACHE = TouchBackendCache()


def _parse_param(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        if not value.strip():
            return {}
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"custom_action_param must be dict or JSON object string, got {type(value)!r}")


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    project_path = PROJECT_ROOT / path
    if project_path.exists():
        return project_path
    return ASSETS_ROOT / path


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _calculate_chart_points(width: int, height: int) -> dict[str, tuple[int, int]]:
    w = float(width)
    h = float(height)
    r = w / h
    t = ((16.0 / 9.0) - r) / ((16.0 / 9.0) - (16.0 / 10.0))
    top_y_norm = 0.4037 + 0.0185 * t
    bottom_y_norm = 0.8398 - 0.0207 * t
    if r >= 16.0 / 9.0:
        top_half_norm = 0.0469 + 0.2893 / r
        bottom_half_norm = 0.7407 / r
    elif r >= 16.0 / 10.0:
        top_half_norm = 0.2096 + 0.0140 * t
        bottom_half_norm = 0.4169 + 0.0082 * t
    else:
        u = t - 1.0
        top_half_norm = 0.2236 - 0.0035 * u
        bottom_half_norm = 0.4252 + 0.0047 * u
    return {
        "top_left": (round((0.5 - top_half_norm) * w), round(top_y_norm * h)),
        "bottom_left": (round((0.5 - bottom_half_norm) * w), round(bottom_y_norm * h)),
        "top_right": (round((0.5 + top_half_norm) * w), round(top_y_norm * h)),
        "bottom_right": (round((0.5 + bottom_half_norm) * w), round(bottom_y_norm * h)),
    }


def _resolution_from_value(value: Any) -> tuple[int, int] | None:
    if isinstance(value, dict):
        width = value.get("width") or value.get("w")
        height = value.get("height") or value.get("h")
        if width and height:
            return int(width), int(height)
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return int(value[0]), int(value[1])
    return None


def _get_screen_resolution(context: Context) -> tuple[int, int] | None:
    try:
        resolution = _resolution_from_value(context.tasker.controller.resolution)
        if resolution is not None:
            return resolution
    except Exception:
        pass
    try:
        job = context.tasker.controller.post_screencap()
        job.wait()
        frame = context.tasker.controller.cached_image
        if frame is not None:
            height, width = frame.shape[:2]
            return int(width), int(height)
    except Exception:
        pass
    return None


def _load_chart_events(
    chart_path: Path,
    calibration: dict[str, tuple[int, int]] | None = None,
    screen_resolution: tuple[int, int] | None = None,
    calibration_mode: str = "manual",
) -> PlayCache:
    app_config = load_app_config()
    global_config = app_config.global_config
    content = _read_text(chart_path)
    chart = Chart.loads(content, designant_choice=global_config.designant_choice)
    if calibration is None:
        calibration = {
            "bottom_left": global_config.bottom_left,
            "top_left": global_config.top_left,
            "top_right": global_config.top_right,
            "bottom_right": global_config.bottom_right,
        }
        calibration_mode = "manual"
    if calibration_mode == "auto_resolution":
        converter = ProjectiveCoordConv(
            AUTO_CALIBRATION_LOGICAL_POINTS,
            [
                calibration["bottom_left"],
                calibration["top_left"],
                calibration["top_right"],
                calibration["bottom_right"],
            ],
        )
    else:
        converter = CoordConv(
            calibration["bottom_left"],
            calibration["top_left"],
            calibration["top_right"],
            calibration["bottom_right"],
        )
    events_by_time = solve_chart_auto(chart, converter)
    if not events_by_time:
        raise ValueError("Solver generated no touch events")
    first_tick_ms = min(events_by_time)
    event_count = sum(len(events) for events in events_by_time.values())
    return PlayCache(
        events_by_time=events_by_time,
        first_tick_ms=first_tick_ms,
        chart_path=chart_path,
        tick_count=len(events_by_time),
        event_count=event_count,
        calibration=calibration,
        screen_resolution=screen_resolution,
        calibration_mode=calibration_mode,
    )


def _write_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


class TraceClock:
    def __init__(self, origin: float) -> None:
        self.origin = origin

    def ms(self, timestamp: float | int | None) -> float | None:
        if timestamp is None:
            return None
        return round((float(timestamp) - self.origin) * 1000.0, 2)

    def normalize(self, value: Any, key: str | None = None) -> Any:
        if isinstance(value, dict):
            return {item_key: self.normalize(item_value, item_key) for item_key, item_value in value.items()}
        if isinstance(value, list):
            return [self.normalize(item) for item in value]
        if isinstance(value, tuple):
            return [self.normalize(item) for item in value]
        if isinstance(value, (int, float)) and key is not None and _is_absolute_trace_time_key(key):
            return self.ms(value)
        if isinstance(value, float):
            return round(value, 2)
        return value


def _is_absolute_trace_time_key(key: str) -> bool:
    if key.endswith("_ms"):
        return False
    return (
        key == "timestamp"
        or key.startswith("t_")
        or key.endswith("_timestamp")
        or key in {"target_start", "first_touch_target"}
    )


def _write_trace_jsonl(path: Path, clock: TraceClock, record: dict[str, Any]) -> None:
    _write_jsonl(path, clock.normalize(record))


def _event_debug_payload(event: Any, screen_resolution: tuple[int, int] | None) -> dict[str, Any]:
    x, y = event.pos
    width = height = None
    in_bounds = None
    if screen_resolution is not None:
        width, height = screen_resolution
        in_bounds = 0 <= int(x) < width and 0 <= int(y) < height
    return {
        "pos": [int(x), int(y)],
        "action": getattr(event.action, "name", str(event.action)),
        "pointer": int(event.pointer),
        "source_note_id": event.source_note_id,
        "source_type": event.source_type,
        "logical_tick": event.logical_tick,
        "logical_pos": list(event.logical_pos) if event.logical_pos is not None else None,
        "screen_resolution": [width, height] if screen_resolution is not None else None,
        "in_screen_bounds": in_bounds,
    }


def _write_first_touch_debug(
    output_dir: Path,
    target_start: float,
    input_backend: str,
    maa_wait_mode: str,
) -> Path:
    if not PLAY_CACHE.events_by_time:
        raise ValueError("No touch events loaded")
    first_tick = min(PLAY_CACHE.events_by_time)
    first_events = PLAY_CACHE.events_by_time[first_tick]
    first_touch_target = target_start + first_tick / 1000.0
    payload = {
        "type": "first_note_touch_debug",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "chart_path": str(PLAY_CACHE.chart_path),
        "input_backend": input_backend,
        "maa_wait_mode": maa_wait_mode,
        "screen_resolution": list(PLAY_CACHE.screen_resolution) if PLAY_CACHE.screen_resolution else None,
        "calibration_mode": PLAY_CACHE.calibration_mode,
        "auto_calibration_logical_points": [list(point) for point in AUTO_CALIBRATION_LOGICAL_POINTS],
        "calibration": {key: list(value) for key, value in PLAY_CACHE.calibration.items()},
        "first_tick_ms": first_tick,
        "first_touch_target_perf_counter": first_touch_target,
        "event_count_at_first_tick": len(first_events),
        "events": [_event_debug_payload(event, PLAY_CACHE.screen_resolution) for event in first_events],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "first_note_touch.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _compact_event_debug_payload(event: Any) -> list[Any]:
    x, y = event.pos
    logical_pos = None
    if event.logical_pos is not None:
        logical_pos = [round(float(event.logical_pos[0]), 6), round(float(event.logical_pos[1]), 6)]
    return [
        getattr(event.action, "name", str(event.action)),
        int(event.pointer),
        event.source_note_id,
        event.source_type,
        [int(x), int(y)],
        logical_pos,
    ]


def _build_note_debug_index(chart_path: Path) -> dict[str, dict[str, Any]]:
    app_config = load_app_config()
    content = _read_text(chart_path)
    chart = Chart.loads(content, designant_choice=app_config.global_config.designant_choice)
    chart_ir = chart.ir
    if chart_ir is None:
        return {}

    notes: dict[str, dict[str, Any]] = {}
    for note in chart_ir.notes:
        if isinstance(note, TapIR):
            notes[str(note.note_id)] = {
                "type": "tap",
                "tick": note.tick,
                "lane": note.lane,
                "group_id": note.group_id,
                "group_properties": note.group_properties,
            }
        elif isinstance(note, HoldIR):
            notes[str(note.note_id)] = {
                "type": "hold",
                "start": note.start,
                "end": note.end,
                "lane": note.lane,
                "group_id": note.group_id,
                "group_properties": note.group_properties,
            }
        elif isinstance(note, ArcIR):
            notes[str(note.note_id)] = {
                "type": "trace_arc" if note.trace_arc else "arc",
                "start": note.start,
                "end": note.end,
                "start_pos": [note.start_x, note.start_y],
                "end_pos": [note.end_x, note.end_y],
                "color": note.color,
                "trace_arc": bool(note.trace_arc),
                "arctaps": list(note.taps),
                "group_id": note.group_id,
                "group_properties": note.group_properties,
            }
    return notes


def _write_load_chart_debug(output_dir: Path, cache: PlayCache) -> Path:
    ticks = []
    for tick in sorted(cache.events_by_time):
        events = cache.events_by_time[tick]
        ticks.append(
            {
                "t": tick,
                "e": [_compact_event_debug_payload(event) for event in events],
            }
        )
    payload = {
        "type": "load_chart_touch_events_debug",
        "schema": {
            "ticks[].t": "tick_ms",
            "ticks[].e[]": ["action", "pointer", "source_note_id", "source_type", "pos", "logical_pos"],
            "notes_by_id": "map source_note_id to parsed logical note",
        },
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "chart_path": str(cache.chart_path),
        "tick_count": cache.tick_count,
        "event_count": cache.event_count,
        "first_tick_ms": cache.first_tick_ms,
        "screen_resolution": list(cache.screen_resolution) if cache.screen_resolution else None,
        "calibration_mode": cache.calibration_mode,
        "calibration": {key: list(value) for key, value in cache.calibration.items()},
        "notes_by_id": _build_note_debug_index(cache.chart_path) if cache.chart_path else {},
        "ticks": ticks,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "touch_events.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


class FrameTraceWriter:
    def __init__(
        self,
        output_dir: Path,
        prefix: str,
        clock: TraceClock,
        jpeg_quality: int = 80,
        max_queue: int = 8,
        scale: float = 1.0,
    ) -> None:
        self.output_dir = output_dir
        self.prefix = prefix
        self.clock = clock
        self.jpeg_quality = max(30, min(100, int(jpeg_quality)))
        self.scale = max(0.1, min(1.0, float(scale)))
        self.queue: queue.Queue[tuple[float, str, Any]] = queue.Queue(maxsize=max_queue)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.saved = 0
        self.dropped = 0
        self._thread.start()

    def enqueue(self, capture_ts: float, phase: str, frame) -> None:
        if self._stop_event.is_set():
            return
        try:
            self.queue.put_nowait((capture_ts, phase, frame))
        except queue.Full:
            self.dropped += 1

    def close(self) -> None:
        self._stop_event.set()
        try:
            self.queue.put_nowait((0.0, "stop", None))
        except queue.Full:
            pass
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        while not self._stop_event.is_set():
            try:
                capture_ts, phase, frame = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if frame is None:
                continue
            try:
                work = frame
                if self.scale < 0.999:
                    h, w = frame.shape[:2]
                    work = cv2.resize(
                        frame,
                        (max(1, int(w * self.scale)), max(1, int(h * self.scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                ok, encoded = cv2.imencode(
                    ".jpg",
                    work,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
                )
                if not ok:
                    continue
                safe_phase = phase.replace(" ", "_")
                capture_ms = self.clock.ms(capture_ts)
                name = f"{self.prefix}_{capture_ms:012.2f}ms_{safe_phase}.jpg"
                out_path = self.output_dir / name
                encoded.tofile(str(out_path))
                self.saved += 1
            except Exception:
                continue


class StateTraceRecorder:
    def __init__(
        self,
        output_dir: Path,
        prefix: str,
        clock: TraceClock,
        jpeg_quality: int = 90,
        scale: float = 1.0,
    ) -> None:
        self.output_dir = output_dir
        self.prefix = prefix
        self.clock = clock
        self.jpeg_quality = max(30, min(100, int(jpeg_quality)))
        self.scale = max(0.1, min(1.0, float(scale)))
        self.timeline_path = output_dir / f"{prefix}_timeline.jsonl"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def event(self, name: str, timestamp: float | None = None, **fields: Any) -> None:
        record = {
            "type": "state_event",
            "name": name,
            "timestamp": timestamp if timestamp is not None else time.perf_counter(),
            **fields,
        }
        _write_jsonl(self.timeline_path, self.clock.normalize(record))

    def capture(
        self,
        name: str,
        frame,
        capture_timestamp: float,
        **fields: Any,
    ) -> Path | None:
        safe_name = _safe_trace_name(name)
        capture_ms = self.clock.ms(capture_timestamp)
        filename = f"{self.prefix}_{capture_ms:012.2f}ms_{safe_name}.jpg"
        out_path = self.output_dir / filename
        saved = _save_frame_jpeg(
            frame,
            out_path,
            jpeg_quality=self.jpeg_quality,
            scale=self.scale,
        )
        self.event(
            name,
            timestamp=capture_timestamp,
            screenshot=str(out_path) if saved else None,
            **fields,
        )
        return out_path if saved else None


def _safe_trace_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)


def _save_frame_jpeg(
    frame,
    out_path: Path,
    jpeg_quality: int = 90,
    scale: float = 1.0,
) -> bool:
    try:
        work = frame
        if scale < 0.999:
            h, w = frame.shape[:2]
            work = cv2.resize(
                frame,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        ok, encoded = cv2.imencode(
            ".jpg",
            work,
            [int(cv2.IMWRITE_JPEG_QUALITY), max(30, min(100, int(jpeg_quality)))],
        )
        if not ok:
            return False
        out_path.parent.mkdir(parents=True, exist_ok=True)
        encoded.tofile(str(out_path))
        return True
    except Exception:
        return False


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * percentile))
    return ordered[max(0, min(index, len(ordered) - 1))]


@dataclass(slots=True)
class RecognitionTimingStats:
    previous_frame_timestamp: float | None = None
    frame_intervals_ms: list[float] = field(default_factory=list)
    screencap_wait_ms: list[float] = field(default_factory=list)
    cached_image_ms: list[float] = field(default_factory=list)
    detector_ms: list[float] = field(default_factory=list)

    def update(
        self,
        frame_timestamp: float,
        t_screencap_post_begin: float,
        t_screencap_wait_end: float,
        t_cached_image_read_end: float,
        t_detector_begin: float,
        t_detector_end: float,
    ) -> dict[str, float | None]:
        previous = self.previous_frame_timestamp
        frame_interval_ms = None
        if previous is not None:
            frame_interval_ms = (frame_timestamp - previous) * 1000.0
            self.frame_intervals_ms.append(frame_interval_ms)
        self.previous_frame_timestamp = frame_timestamp

        screencap_wait_ms = (t_screencap_wait_end - t_screencap_post_begin) * 1000.0
        cached_image_ms = (t_cached_image_read_end - t_screencap_wait_end) * 1000.0
        detector_ms = (t_detector_end - t_detector_begin) * 1000.0
        self.screencap_wait_ms.append(screencap_wait_ms)
        self.cached_image_ms.append(cached_image_ms)
        self.detector_ms.append(detector_ms)

        return {
            "previous_frame_timestamp": previous,
            "frame_interval_ms": frame_interval_ms,
            "avg_frame_interval_ms": _avg(self.frame_intervals_ms),
            "avg_screencap_wait_ms": _avg(self.screencap_wait_ms),
            "avg_cached_image_ms": _avg(self.cached_image_ms),
            "avg_detector_ms": _avg(self.detector_ms),
            "current_screencap_wait_ms": screencap_wait_ms,
            "current_cached_image_ms": cached_image_ms,
            "current_detector_ms": detector_ms,
            "max_recognition_induced_delay_ms": self.max_recognition_induced_delay_ms(),
        }

    def trigger_latency(self, trigger_frame_timestamp: float) -> dict[str, float | None]:
        previous = self.previous_frame_timestamp
        max_polling_delay_ms = None
        if previous is not None:
            max_polling_delay_ms = (trigger_frame_timestamp - previous) * 1000.0
        return {
            "trigger_previous_frame_timestamp": previous,
            "trigger_frame_timestamp": trigger_frame_timestamp,
            "trigger_max_polling_delay_ms": max_polling_delay_ms,
            "max_recognition_induced_delay_ms": self.max_recognition_induced_delay_ms(),
            "avg_frame_interval_ms": _avg(self.frame_intervals_ms),
            "avg_screencap_wait_ms": _avg(self.screencap_wait_ms),
            "avg_cached_image_ms": _avg(self.cached_image_ms),
            "avg_detector_ms": _avg(self.detector_ms),
        }

    def max_recognition_induced_delay_ms(self) -> float:
        # Worst-case detection latency is one polling interval plus the current
        # recognition pipeline cost. This is separate from the trigger polling
        # delay, which is logged from the exact previous/current frame pair.
        return (
            _avg(self.frame_intervals_ms)
            + _avg(self.screencap_wait_ms)
            + _avg(self.cached_image_ms)
            + _avg(self.detector_ms)
        )


def _avg(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


@dataclass(slots=True)
class FrameSample:
    frame: Any
    timestamp: float
    t_capture_begin: float
    t_capture_ready: float
    t_frame_read_end: float
    source: str
    seq: int | None = None
    decode_fps: float | None = None


class MaaScreencapFrameSource:
    name = "maa_screencap"

    def __init__(self, context: Context) -> None:
        self.context = context

    def capture(self, timeout: float | None = None) -> FrameSample:
        del timeout
        t_capture_begin = time.perf_counter()
        job = self.context.tasker.controller.post_screencap()
        job.wait()
        t_capture_ready = time.perf_counter()
        frame = self.context.tasker.controller.cached_image
        t_frame_read_end = time.perf_counter()
        return FrameSample(
            frame=frame,
            timestamp=t_frame_read_end,
            t_capture_begin=t_capture_begin,
            t_capture_ready=t_capture_ready,
            t_frame_read_end=t_frame_read_end,
            source=self.name,
        )


class ScrcpyVideoFrameSource:
    name = "scrcpy_video"

    def __init__(self, controller, poll_sleep_s: float = 0.001) -> None:
        self.controller = controller
        self.poll_sleep_s = max(0.0, poll_sleep_s)
        self._last_seq = controller.get_latest_frame_seq()

    def capture(self, timeout: float | None = None) -> FrameSample:
        t_capture_begin = time.perf_counter()
        deadline = None if timeout is None else t_capture_begin + timeout
        while True:
            seq = self.controller.get_latest_frame_seq()
            frame_ts = self.controller.get_latest_frame_timestamp()
            frame = self.controller.get_latest_frame(copy_frame=True)
            if frame is not None and frame_ts is not None and seq != self._last_seq:
                self._last_seq = seq
                t_frame_read_end = time.perf_counter()
                return FrameSample(
                    frame=frame,
                    timestamp=frame_ts,
                    t_capture_begin=t_capture_begin,
                    t_capture_ready=frame_ts,
                    t_frame_read_end=t_frame_read_end,
                    source=self.name,
                    seq=seq,
                    decode_fps=self.controller.get_decode_fps(),
                )
            if deadline is not None and time.perf_counter() >= deadline:
                raise TimeoutError("Timed out waiting for scrcpy video frame")
            if self.poll_sleep_s > 0:
                time.sleep(self.poll_sleep_s)


def _wait_for_loading_end(
    context: Context,
    timeout_ms: int,
    log_path: Path | None,
    trace_clock: TraceClock,
    frame_tracer: FrameTraceWriter | None,
    state_trace: StateTraceRecorder | None,
    poll_sleep_ms: float,
    frame_source: Any | None = None,
) -> tuple[float, dict[str, Any]]:
    if frame_source is None:
        frame_source = MaaScreencapFrameSource(context)
    detector = LoadingEndDetector()
    deadline = time.perf_counter() + timeout_ms / 1000.0
    last_metrics: dict[str, Any] = {}
    loading_seen_logged = False
    monitoring_started_at: float | None = None
    timing_stats = RecognitionTimingStats()
    while time.perf_counter() < deadline:
        sample = frame_source.capture(timeout=max(0.001, deadline - time.perf_counter()))
        frame = sample.frame
        t_screencap_post_begin = sample.t_capture_begin
        t_screencap_wait_end = sample.t_capture_ready
        t_cached_image_read_end = sample.t_frame_read_end
        t_detector_begin = time.perf_counter()
        result = detector.update(frame, sample.timestamp)
        t_detector_end = time.perf_counter()
        if frame_tracer is not None:
            frame_tracer.enqueue(t_cached_image_read_end, result.phase.value, frame)
        timing_metrics = timing_stats.update(
            t_cached_image_read_end,
            t_screencap_post_begin,
            t_screencap_wait_end,
            t_cached_image_read_end,
            t_detector_begin,
            t_detector_end,
        )
        last_metrics = dict(result.metrics)
        if log_path is not None:
            _write_trace_jsonl(
                log_path,
                trace_clock,
                {
                        "type": "loading_sample",
                        "frame_source": sample.source,
                        "frame_seq": sample.seq,
                        "decode_fps": sample.decode_fps,
                        "phase": result.phase.value,
                    "triggered": result.triggered,
                    "loading_seen": result.loading_seen,
                    "t_screencap_post_begin": t_screencap_post_begin,
                    "t_screencap_wait_end": t_screencap_wait_end,
                    "t_cached_image_read_end": t_cached_image_read_end,
                    "t_detector_begin": t_detector_begin,
                    "t_detector_end": t_detector_end,
                    "recognition_timing": timing_metrics,
                    "metrics": result.metrics,
                },
            )
        if result.loading_seen and not loading_seen_logged:
            loading_seen_logged = True
            monitoring_started_at = result.timestamp
            print(f"[INFO] Loading screen detected at {result.timestamp:.6f}")
            print("[INFO] Monitoring until loading zigzag disappears")
            if state_trace is not None:
                state_trace.capture(
                    "loading_screen_detected",
                    frame,
                    sample.timestamp,
                    detector_timestamp=result.timestamp,
                    t_screencap_post_begin=t_screencap_post_begin,
                    t_screencap_wait_end=t_screencap_wait_end,
                    t_detector_begin=t_detector_begin,
                    t_detector_end=t_detector_end,
                    phase=result.phase.value,
                    recognition_timing=timing_metrics,
                    metrics=result.metrics,
                )
        if result.triggered and result.estimated_change_timestamp is not None:
            trigger_latency = timing_stats.trigger_latency(t_cached_image_read_end)
            if state_trace is not None:
                state_trace.capture(
                    "loading_screen_change_detected",
                    frame,
                    sample.timestamp,
                    estimated_change_timestamp=result.estimated_change_timestamp,
                    detector_timestamp=result.timestamp,
                    t_screencap_post_begin=t_screencap_post_begin,
                    t_screencap_wait_end=t_screencap_wait_end,
                    t_detector_begin=t_detector_begin,
                    t_detector_end=t_detector_end,
                    phase=result.phase.value,
                    recognition_timing=timing_metrics,
                    trigger_latency=trigger_latency,
                    metrics=result.metrics,
                )
            return result.estimated_change_timestamp, {
                "phase": result.phase.value,
                "estimated_change_timestamp": result.estimated_change_timestamp,
                "detector_timestamp": result.timestamp,
                "recognition_timing": timing_metrics,
                "trigger_latency": trigger_latency,
                "metrics": result.metrics,
            }
        if poll_sleep_ms > 0:
            time.sleep(poll_sleep_ms / 1000.0)
    elapsed_after_seen = None
    if monitoring_started_at is not None:
        elapsed_after_seen = time.perf_counter() - monitoring_started_at
    raise TimeoutError(
        "Timed out waiting for loading end; "
        f"phase={detector.phase.value}, "
        f"elapsed_after_seen={elapsed_after_seen}, "
        f"last_metrics={last_metrics}"
    )


def _wait_for_freeze_then_change(
    context: Context,
    timeout_ms: int,
    start_delay_ms: int,
    log_path: Path | None,
    trace_clock: TraceClock,
    frame_tracer: FrameTraceWriter | None,
    state_trace: StateTraceRecorder | None,
    poll_sleep_ms: float,
    frame_source: Any | None = None,
    detector_config: FreezeChangeDetectorConfig | None = None,
) -> tuple[float, dict[str, Any]]:
    if frame_source is None:
        frame_source = MaaScreencapFrameSource(context)
    if start_delay_ms > 0:
        time.sleep(start_delay_ms / 1000.0)
    detector = FreezeChangeDetector(detector_config)
    deadline = time.perf_counter() + timeout_ms / 1000.0
    last_metrics: dict[str, Any] = {}
    still_seen_logged = False
    timing_stats = RecognitionTimingStats()
    while time.perf_counter() < deadline:
        sample = frame_source.capture(timeout=max(0.001, deadline - time.perf_counter()))
        frame = sample.frame
        t_screencap_post_begin = sample.t_capture_begin
        t_screencap_wait_end = sample.t_capture_ready
        t_cached_image_read_end = sample.t_frame_read_end
        t_detector_begin = time.perf_counter()
        result = detector.update(frame, sample.timestamp)
        t_detector_end = time.perf_counter()
        if frame_tracer is not None:
            frame_tracer.enqueue(t_cached_image_read_end, result.phase.value, frame)
        timing_metrics = timing_stats.update(
            t_cached_image_read_end,
            t_screencap_post_begin,
            t_screencap_wait_end,
            t_cached_image_read_end,
            t_detector_begin,
            t_detector_end,
        )
        last_metrics = dict(result.metrics)
        if log_path is not None:
            _write_trace_jsonl(
                log_path,
                trace_clock,
                {
                        "type": "freeze_change_sample",
                        "frame_source": sample.source,
                        "frame_seq": sample.seq,
                        "decode_fps": sample.decode_fps,
                        "phase": result.phase.value,
                    "still_seen": result.still_seen,
                    "triggered": result.triggered,
                    "t_screencap_post_begin": t_screencap_post_begin,
                    "t_screencap_wait_end": t_screencap_wait_end,
                    "t_cached_image_read_end": t_cached_image_read_end,
                    "t_detector_begin": t_detector_begin,
                    "t_detector_end": t_detector_end,
                    "recognition_timing": timing_metrics,
                    "metrics": result.metrics,
                },
            )
        if result.still_seen and not still_seen_logged:
            still_seen_logged = True
            print(f"[INFO] Still frame confirmed at {result.timestamp:.6f}")
            if state_trace is not None:
                state_trace.capture(
                    "still_waiting_frame_detected",
                    frame,
                    sample.timestamp,
                    detector_timestamp=result.timestamp,
                    t_screencap_post_begin=t_screencap_post_begin,
                    t_screencap_wait_end=t_screencap_wait_end,
                    t_detector_begin=t_detector_begin,
                    t_detector_end=t_detector_end,
                    phase=result.phase.value,
                    recognition_timing=timing_metrics,
                    metrics=result.metrics,
                )
        if result.triggered and result.estimated_change_timestamp is not None:
            trigger_latency = timing_stats.trigger_latency(t_cached_image_read_end)
            if state_trace is not None:
                state_trace.capture(
                    "still_waiting_frame_change_detected",
                    frame,
                    sample.timestamp,
                    estimated_change_timestamp=result.estimated_change_timestamp,
                    detector_timestamp=result.timestamp,
                    t_screencap_post_begin=t_screencap_post_begin,
                    t_screencap_wait_end=t_screencap_wait_end,
                    t_detector_begin=t_detector_begin,
                    t_detector_end=t_detector_end,
                    phase=result.phase.value,
                    recognition_timing=timing_metrics,
                    trigger_latency=trigger_latency,
                    metrics=result.metrics,
                )
            return result.estimated_change_timestamp, {
                "phase": result.phase.value,
                "estimated_change_timestamp": result.estimated_change_timestamp,
                "detector_timestamp": result.timestamp,
                "recognition_timing": timing_metrics,
                "trigger_latency": trigger_latency,
                "metrics": result.metrics,
            }
        if poll_sleep_ms > 0:
            time.sleep(poll_sleep_ms / 1000.0)
    raise TimeoutError(
        "Timed out waiting for freeze-then-change trigger; "
        f"phase={detector.phase.value}, metrics={last_metrics}"
    )


def result_phase_name(phase: Any) -> str:
    try:
        return str(phase.value)
    except Exception:
        return str(phase)


def _enable_timer_resolution() -> bool:
    try:
        return ctypes.windll.winmm.timeBeginPeriod(1) == 0
    except Exception:
        return False


def _disable_timer_resolution(active: bool) -> None:
    if not active:
        return
    try:
        ctypes.windll.winmm.timeEndPeriod(1)
    except Exception:
        pass


def _set_high_thread_priority() -> None:
    try:
        thread_priority_highest = 2
        ctypes.windll.kernel32.SetThreadPriority(
            ctypes.windll.kernel32.GetCurrentThread(), thread_priority_highest
        )
    except Exception:
        pass


def _dispatch_events(
    backend: TouchBackend,
    target_start: float,
    maa_wait_mode: str,
    log_path: Path | None,
    trace_clock: TraceClock,
) -> None:
    sorted_events = sorted(PLAY_CACHE.events_by_time.items())
    first_touch_logged = False
    timer_active = _enable_timer_resolution()
    _set_high_thread_priority()
    print(
        f"[INFO] Dispatching {PLAY_CACHE.event_count} events in {PLAY_CACHE.tick_count} ticks via {backend.name}"
    )
    try:
        for tick_ms, events in sorted_events:
            due = target_start + tick_ms / 1000.0
            while True:
                now = time.perf_counter()
                remaining = due - now
                if remaining <= 0:
                    break
                time.sleep(0.0005 if remaining < 0.003 else min(0.002, remaining / 2.0))

            for event in events:
                x, y = event.pos
                t_call_begin = time.perf_counter()
                meta = backend.dispatch(event)
                t_call_end = time.perf_counter()
                should_log = log_path is not None and (
                    not first_touch_logged or tick_ms % 500 == 0 or tick_ms == sorted_events[-1][0]
                )
                if should_log:
                    _write_trace_jsonl(
                        log_path,
                        trace_clock,
                        {
                            "type": "touch_dispatch",
                            "tick_ms": tick_ms,
                            "event_count": len(events),
                            "action": event.action.name,
                            "pointer": int(event.pointer),
                            "contact": meta.get("contact"),
                            "x": int(x),
                            "y": int(y),
                            "t_due": due,
                            "t_call_begin": t_call_begin,
                            "t_call_end": t_call_end,
                            "lateness_before_call_ms": (t_call_begin - due) * 1000.0,
                            "call_duration_ms": (t_call_end - t_call_begin) * 1000.0,
                            "backend": meta.get("backend"),
                            "maa_wait_mode": maa_wait_mode,
                        },
                    )
                first_touch_logged = True
            if isinstance(backend, MaaTouchBackend):
                backend.flush_tick()
    finally:
        _disable_timer_resolution(timer_active)


def _dry_run_dispatch(target_start: float, log_path: Path | None, trace_clock: TraceClock) -> None:
    sorted_events = sorted(PLAY_CACHE.events_by_time.items())
    first_tick, first_events = sorted_events[0]
    last_tick, last_events = sorted_events[-1]
    payload = {
        "type": "dry_run_dispatch",
        "target_start": target_start,
        "tick_count": PLAY_CACHE.tick_count,
        "event_count": PLAY_CACHE.event_count,
        "first_tick_ms": first_tick,
        "first_event_count": len(first_events),
        "last_tick_ms": last_tick,
        "last_event_count": len(last_events),
    }
    print(f"[INFO] Dry run dispatch: {payload}")
    if log_path is not None:
        _write_trace_jsonl(log_path, trace_clock, payload)


def _close_cached_touch_backend() -> None:
    if TOUCH_BACKEND_CACHE.backend is not None:
        try:
            TOUCH_BACKEND_CACHE.backend.close()
        finally:
            TOUCH_BACKEND_CACHE.backend = None
            TOUCH_BACKEND_CACHE.name = ""
            TOUCH_BACKEND_CACHE.maa_wait_mode = ""
            TOUCH_BACKEND_CACHE.adb_serial = None
            TOUCH_BACKEND_CACHE.scrcpy_max_fps = 60
            TOUCH_BACKEND_CACHE.scrcpy_video_bit_rate = None
            TOUCH_BACKEND_CACHE.scrcpy_video_crop = None


def _normalize_optional_serial(value: Any) -> str | None:
    if value is None:
        return None
    serial = str(value).strip()
    return serial or None


def _normalize_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return int(value)


def _normalize_optional_crop(value: Any) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("scrcpy_video_crop must be [x, y, width, height]")
    x, y, width, height = (int(value[0]), int(value[1]), int(value[2]), int(value[3]))
    if width <= 0 or height <= 0:
        raise ValueError("scrcpy_video_crop width and height must be positive")
    return x, y, width, height


def _scrcpy_options_from_params(params: dict[str, Any]) -> tuple[int, int | None, tuple[int, int, int, int] | None]:
    max_fps = int(params.get("scrcpy_max_fps", params.get("stream_max_fps", 60)))
    video_bit_rate_mbps = _normalize_optional_int(
        params.get("scrcpy_video_bit_rate_mbps", params.get("stream_bitrate_mbps"))
    )
    video_bit_rate = None if video_bit_rate_mbps is None else video_bit_rate_mbps * 1_000_000
    video_crop = _normalize_optional_crop(params.get("scrcpy_video_crop"))
    return max_fps, video_bit_rate, video_crop


def _default_pipeline_adb_serial() -> str | None:
    pipeline_path = ASSETS_ROOT / "resource" / "pipeline" / "auto_play.json"
    try:
        payload = json.loads(pipeline_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    execute_touch = payload.get("ExecuteTouch")
    if not isinstance(execute_touch, dict):
        return None
    params = execute_touch.get("custom_action_param")
    if not isinstance(params, dict):
        return None
    return _normalize_optional_serial(params.get("adb_serial"))


def _take_cached_touch_backend(
    input_backend: str,
    maa_wait_mode: str,
    adb_serial: str | None = None,
    scrcpy_max_fps: int = 60,
    scrcpy_video_bit_rate: int | None = None,
    scrcpy_video_crop: tuple[int, int, int, int] | None = None,
) -> TouchBackend | None:
    normalized = input_backend.strip().lower()
    if (
        TOUCH_BACKEND_CACHE.backend is not None
        and TOUCH_BACKEND_CACHE.name == normalized
        and TOUCH_BACKEND_CACHE.maa_wait_mode == maa_wait_mode
        and TOUCH_BACKEND_CACHE.adb_serial == adb_serial
        and TOUCH_BACKEND_CACHE.scrcpy_max_fps == scrcpy_max_fps
        and TOUCH_BACKEND_CACHE.scrcpy_video_bit_rate == scrcpy_video_bit_rate
        and TOUCH_BACKEND_CACHE.scrcpy_video_crop == scrcpy_video_crop
    ):
        backend = TOUCH_BACKEND_CACHE.backend
        TOUCH_BACKEND_CACHE.backend = None
        TOUCH_BACKEND_CACHE.name = ""
        TOUCH_BACKEND_CACHE.maa_wait_mode = ""
        TOUCH_BACKEND_CACHE.adb_serial = None
        TOUCH_BACKEND_CACHE.scrcpy_max_fps = 60
        TOUCH_BACKEND_CACHE.scrcpy_video_bit_rate = None
        TOUCH_BACKEND_CACHE.scrcpy_video_crop = None
        return backend
    _close_cached_touch_backend()
    return None


def _prewarm_touch_backend(
    context: Context,
    input_backend: str,
    maa_wait_mode: str,
    adb_serial: str | None = None,
    scrcpy_max_fps: int = 60,
    scrcpy_video_bit_rate: int | None = None,
    scrcpy_video_crop: tuple[int, int, int, int] | None = None,
) -> TouchBackend:
    normalized = input_backend.strip().lower()
    cached = _take_cached_touch_backend(
        normalized,
        maa_wait_mode,
        adb_serial,
        scrcpy_max_fps,
        scrcpy_video_bit_rate,
        scrcpy_video_crop,
    )
    if cached is not None:
        TOUCH_BACKEND_CACHE.backend = cached
        TOUCH_BACKEND_CACHE.name = normalized
        TOUCH_BACKEND_CACHE.maa_wait_mode = maa_wait_mode
        TOUCH_BACKEND_CACHE.adb_serial = adb_serial
        TOUCH_BACKEND_CACHE.scrcpy_max_fps = scrcpy_max_fps
        TOUCH_BACKEND_CACHE.scrcpy_video_bit_rate = scrcpy_video_bit_rate
        TOUCH_BACKEND_CACHE.scrcpy_video_crop = scrcpy_video_crop
        return cached
    backend = create_touch_backend(
        normalized,
        context,
        maa_wait_mode=maa_wait_mode,
        adb_serial=adb_serial,
        scrcpy_max_fps=scrcpy_max_fps,
        scrcpy_video_bit_rate=scrcpy_video_bit_rate,
        scrcpy_video_crop=scrcpy_video_crop,
    )
    TOUCH_BACKEND_CACHE.backend = backend
    TOUCH_BACKEND_CACHE.name = normalized
    TOUCH_BACKEND_CACHE.maa_wait_mode = maa_wait_mode
    TOUCH_BACKEND_CACHE.adb_serial = adb_serial
    TOUCH_BACKEND_CACHE.scrcpy_max_fps = scrcpy_max_fps
    TOUCH_BACKEND_CACHE.scrcpy_video_bit_rate = scrcpy_video_bit_rate
    TOUCH_BACKEND_CACHE.scrcpy_video_crop = scrcpy_video_crop
    return backend


@AgentServer.custom_action("PlaySong.PrewarmTouch")
class PrewarmTouchAction(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        try:
            params = _parse_param(argv.custom_action_param)
            input_backend = str(params.get("input_backend", "scrcpy"))
            maa_wait_mode = str(params.get("maa_wait_mode", "wait_each"))
            adb_serial = _normalize_optional_serial(params.get("adb_serial")) or _default_pipeline_adb_serial()
            scrcpy_max_fps, scrcpy_video_bit_rate, scrcpy_video_crop = _scrcpy_options_from_params(params)
            t_begin = time.perf_counter()
            backend = _prewarm_touch_backend(
                context,
                input_backend,
                maa_wait_mode,
                adb_serial,
                scrcpy_max_fps,
                scrcpy_video_bit_rate,
                scrcpy_video_crop,
            )
            t_ready = time.perf_counter()
            print(
                "[INFO] Touch backend prewarmed: "
                f"backend={backend.name}, adb_serial={adb_serial or '<default>'}, "
                f"scrcpy_max_fps={scrcpy_max_fps}, "
                f"init_duration_ms={(t_ready - t_begin) * 1000.0:.2f}"
            )
            return True
        except Exception as exc:
            print(f"[ERROR] PlaySong.PrewarmTouch failed: {exc}")
            return False


@AgentServer.custom_action("PlaySong.LoadChart")
class LoadChartAction(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        try:
            params = _parse_param(argv.custom_action_param)
            chart_param = params.get("chart_path")
            if not chart_param:
                print("[ERROR] PlaySong.LoadChart requires chart_path")
                return False
            chart_path = _resolve_path(str(chart_param))
            if not chart_path.is_file():
                print(f"[ERROR] Chart file not found: {chart_path}")
                return False
            resolution = _get_screen_resolution(context)
            calibration = _calculate_chart_points(*resolution) if resolution is not None else None
            calibration_mode = "auto_resolution" if calibration is not None else "manual"
            cache = _load_chart_events(
                chart_path,
                calibration=calibration,
                screen_resolution=resolution,
                calibration_mode=calibration_mode,
            )
            PLAY_CACHE.events_by_time = cache.events_by_time
            PLAY_CACHE.first_tick_ms = cache.first_tick_ms
            PLAY_CACHE.chart_path = cache.chart_path
            PLAY_CACHE.tick_count = cache.tick_count
            PLAY_CACHE.event_count = cache.event_count
            PLAY_CACHE.calibration = cache.calibration
            PLAY_CACHE.screen_resolution = cache.screen_resolution
            PLAY_CACHE.calibration_mode = cache.calibration_mode
            trace_prefix = str(params.get("trace_prefix", time.strftime("%Y%m%d_%H%M%S")))
            load_chart_debug_path = _write_load_chart_debug(
                LOAD_CHART_DEBUG_DIR / trace_prefix,
                cache,
            )
            print(
                "[INFO] Chart loaded: "
                f"path={chart_path}, first_tick={cache.first_tick_ms}, "
                f"ticks={cache.tick_count}, events={cache.event_count}, "
                f"resolution={resolution}, calibration_mode={cache.calibration_mode}, "
                f"calibration={cache.calibration}, debug={load_chart_debug_path}"
            )
            return True
        except Exception as exc:
            print(f"[ERROR] PlaySong.LoadChart failed: {exc}")
            return False


@AgentServer.custom_action("PlaySong.ExecuteTouch")
class ExecuteTouchAction(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        t_action_enter = time.perf_counter()
        frame_tracer: FrameTraceWriter | None = None
        state_trace: StateTraceRecorder | None = None
        try:
            if not PLAY_CACHE.events_by_time:
                print("[ERROR] ExecuteTouch requires a successful LoadChart first")
                return False
            params = _parse_param(argv.custom_action_param)
            fixed_delay_ms = int(params.get("fixed_delay_ms", 3620))
            user_offset_ms = int(params.get("user_offset_ms", 0))
            input_backend = str(params.get("input_backend", "scrcpy"))
            maa_wait_mode = str(params.get("maa_wait_mode", "wait_each"))
            adb_serial = _normalize_optional_serial(params.get("adb_serial"))
            scrcpy_max_fps, scrcpy_video_bit_rate, scrcpy_video_crop = _scrcpy_options_from_params(params)
            loading_timeout_ms = int(params.get("loading_timeout_ms", 30000))
            start_detection_delay_ms = int(params.get("start_detection_delay_ms", 0))
            start_detection_mode = str(params.get("start_detection_mode", "zigzag"))
            freeze_change_config = FreezeChangeDetectorConfig(
                still_diff_ratio_threshold=float(params.get("freeze_still_diff_ratio_threshold", 0.0015)),
                anchor_still_diff_ratio_threshold=float(
                    params.get("freeze_anchor_still_diff_ratio_threshold", 0.0025)
                ),
                change_diff_ratio_threshold=float(params.get("freeze_change_diff_ratio_threshold", 0.010)),
                pixel_diff_threshold=int(params.get("freeze_pixel_diff_threshold", 18)),
                still_confirm_frames=int(params.get("freeze_still_confirm_frames", 6)),
                still_confirm_duration_ms=float(params.get("freeze_still_confirm_duration_ms", 150.0)),
                change_confirm_frames=int(params.get("freeze_change_confirm_frames", 2)),
            )
            frame_source_name = str(params.get("frame_source", "maa_screencap")).strip().lower()
            scrcpy_frame_poll_sleep_ms = float(params.get("scrcpy_frame_poll_sleep_ms", 1.0))
            detection_poll_sleep_ms = float(params.get("detection_poll_sleep_ms", 0.0))
            debug_log = bool(params.get("debug_log", False))
            trace_frames = bool(params.get("trace_frames", False))
            trace_frame_scale = float(params.get("trace_frame_scale", 0.5))
            trace_jpeg_quality = int(params.get("trace_jpeg_quality", 80))
            trace_queue_size = int(params.get("trace_queue_size", 8))
            trace_state_snapshots = bool(params.get("trace_state_snapshots", True))
            state_trace_scale = float(params.get("state_trace_scale", 1.0))
            state_trace_jpeg_quality = int(params.get("state_trace_jpeg_quality", 92))
            dry_run = bool(params.get("dry_run", False))
            skip_loading_detection = bool(params.get("skip_loading_detection", False))
            trace_prefix = time.strftime("%Y%m%d_%H%M%S") + "_execute_touch"
            trace_clock = TraceClock(t_action_enter)
            run_frame_trace_dir = FRAME_TRACE_DIR / trace_prefix
            run_state_trace_dir = STATE_TRACE_DIR / trace_prefix
            run_first_touch_debug_dir = FIRST_TOUCH_DEBUG_DIR / trace_prefix
            log_path = None
            if debug_log:
                log_name = trace_prefix + ".jsonl"
                log_path = TIMING_LOG_DIR / log_name
                _write_trace_jsonl(
                    log_path,
                    trace_clock,
                    {
                        "type": "action_enter",
                        "t_action_enter": t_action_enter,
                        "origin_perf_counter": t_action_enter,
                        "start_detection_mode": start_detection_mode,
                        "frame_source": frame_source_name,
                        "adb_serial": adb_serial,
                        "scrcpy_max_fps": scrcpy_max_fps,
                        "scrcpy_video_bit_rate": scrcpy_video_bit_rate,
                        "scrcpy_video_crop": scrcpy_video_crop,
                        "start_detection_delay_ms": start_detection_delay_ms,
                        "detection_poll_sleep_ms": detection_poll_sleep_ms,
                        "freeze_change_config": {
                            "still_diff_ratio_threshold": freeze_change_config.still_diff_ratio_threshold,
                            "anchor_still_diff_ratio_threshold": freeze_change_config.anchor_still_diff_ratio_threshold,
                            "change_diff_ratio_threshold": freeze_change_config.change_diff_ratio_threshold,
                            "pixel_diff_threshold": freeze_change_config.pixel_diff_threshold,
                            "still_confirm_frames": freeze_change_config.still_confirm_frames,
                            "still_confirm_duration_ms": freeze_change_config.still_confirm_duration_ms,
                            "change_confirm_frames": freeze_change_config.change_confirm_frames,
                        },
                        "trace_prefix": trace_prefix,
                        "frame_trace_dir": str(run_frame_trace_dir),
                        "state_trace_dir": str(run_state_trace_dir),
                        "first_touch_debug_dir": str(run_first_touch_debug_dir),
                    },
                )
            if trace_state_snapshots:
                state_trace = StateTraceRecorder(
                    output_dir=run_state_trace_dir,
                    prefix=trace_prefix,
                    clock=trace_clock,
                    jpeg_quality=state_trace_jpeg_quality,
                    scale=state_trace_scale,
                )
                state_trace.event(
                    "execute_touch_enter",
                    timestamp=t_action_enter,
                    chart_path=str(PLAY_CACHE.chart_path),
                    calibration=PLAY_CACHE.calibration,
                    screen_resolution=PLAY_CACHE.screen_resolution,
                    calibration_mode=PLAY_CACHE.calibration_mode,
                    auto_calibration_logical_points=AUTO_CALIBRATION_LOGICAL_POINTS,
                    first_tick_ms=PLAY_CACHE.first_tick_ms,
                    tick_count=PLAY_CACHE.tick_count,
                    event_count=PLAY_CACHE.event_count,
                    start_detection_mode=start_detection_mode,
                    frame_source=frame_source_name,
                    start_detection_delay_ms=start_detection_delay_ms,
                    detection_poll_sleep_ms=detection_poll_sleep_ms,
                    freeze_change_config={
                        "still_diff_ratio_threshold": freeze_change_config.still_diff_ratio_threshold,
                        "anchor_still_diff_ratio_threshold": freeze_change_config.anchor_still_diff_ratio_threshold,
                        "change_diff_ratio_threshold": freeze_change_config.change_diff_ratio_threshold,
                        "pixel_diff_threshold": freeze_change_config.pixel_diff_threshold,
                        "still_confirm_frames": freeze_change_config.still_confirm_frames,
                        "still_confirm_duration_ms": freeze_change_config.still_confirm_duration_ms,
                        "change_confirm_frames": freeze_change_config.change_confirm_frames,
                    },
                    dry_run=dry_run,
                    input_backend=input_backend,
                    maa_wait_mode=maa_wait_mode,
                    adb_serial=adb_serial,
                    scrcpy_max_fps=scrcpy_max_fps,
                    scrcpy_video_bit_rate=scrcpy_video_bit_rate,
                    scrcpy_video_crop=scrcpy_video_crop,
                )
            if trace_frames:
                frame_tracer = FrameTraceWriter(
                    output_dir=run_frame_trace_dir,
                    prefix=trace_prefix,
                    clock=trace_clock,
                    jpeg_quality=trace_jpeg_quality,
                    max_queue=trace_queue_size,
                    scale=trace_frame_scale,
                )

            detection_frame_source = MaaScreencapFrameSource(context)
            prewarmed_detection_backend: TouchBackend | None = None
            if frame_source_name == "scrcpy_video":
                if input_backend.strip().lower() != "scrcpy" and not dry_run:
                    raise ValueError("frame_source=scrcpy_video requires input_backend=scrcpy for this implementation")
                prewarmed_detection_backend = _prewarm_touch_backend(
                    context,
                    "scrcpy",
                    maa_wait_mode,
                    adb_serial,
                    scrcpy_max_fps,
                    scrcpy_video_bit_rate,
                    scrcpy_video_crop,
                )
                controller = getattr(prewarmed_detection_backend, "controller", None)
                if controller is None:
                    raise RuntimeError("scrcpy_video frame source requires a scrcpy controller")
                detection_frame_source = ScrcpyVideoFrameSource(
                    controller,
                    poll_sleep_s=scrcpy_frame_poll_sleep_ms / 1000.0,
                )
                print("[INFO] Using scrcpy video frame source for start detection")
                if state_trace is not None:
                    state_trace.event(
                        "scrcpy_video_frame_source_ready",
                        adb_serial=adb_serial,
                        scrcpy_max_fps=scrcpy_max_fps,
                        scrcpy_video_bit_rate=scrcpy_video_bit_rate,
                        scrcpy_video_crop=scrcpy_video_crop,
                        frame_seq=controller.get_latest_frame_seq(),
                        decode_fps=controller.get_decode_fps(),
                    )
            elif frame_source_name == "maa_screencap":
                print("[INFO] Using Maa screencap frame source for start detection")
            else:
                raise ValueError(f"Unsupported frame_source: {frame_source_name}")

            if skip_loading_detection:
                estimated_change = time.perf_counter()
                detail = {
                    "phase": "skipped",
                    "estimated_change_timestamp": estimated_change,
                    "metrics": {},
                }
                print("[INFO] Loading detection skipped for dry/smoke test")
                if state_trace is not None:
                    state_trace.event(
                        "loading_detection_skipped",
                        timestamp=estimated_change,
                        reason="skip_loading_detection",
                    )
            else:
                if start_detection_mode == "freeze_change":
                    print("[INFO] Waiting for still frame, then first screen change")
                    if state_trace is not None:
                        state_trace.event(
                        "waiting_for_still_frame_started",
                        timeout_ms=loading_timeout_ms,
                        start_delay_ms=start_detection_delay_ms,
                        poll_sleep_ms=detection_poll_sleep_ms,
                        freeze_change_config={
                            "still_diff_ratio_threshold": freeze_change_config.still_diff_ratio_threshold,
                            "anchor_still_diff_ratio_threshold": freeze_change_config.anchor_still_diff_ratio_threshold,
                            "change_diff_ratio_threshold": freeze_change_config.change_diff_ratio_threshold,
                            "pixel_diff_threshold": freeze_change_config.pixel_diff_threshold,
                            "still_confirm_frames": freeze_change_config.still_confirm_frames,
                            "still_confirm_duration_ms": freeze_change_config.still_confirm_duration_ms,
                            "change_confirm_frames": freeze_change_config.change_confirm_frames,
                        },
                    )
                    estimated_change, detail = _wait_for_freeze_then_change(
                        context,
                        timeout_ms=loading_timeout_ms,
                        start_delay_ms=start_detection_delay_ms,
                        log_path=log_path,
                        trace_clock=trace_clock,
                        frame_tracer=frame_tracer,
                        state_trace=state_trace,
                        poll_sleep_ms=detection_poll_sleep_ms,
                        frame_source=detection_frame_source,
                        detector_config=freeze_change_config,
                    )
                elif start_detection_mode == "zigzag":
                    print("[INFO] Waiting for loading zigzag to disappear")
                    if state_trace is not None:
                        state_trace.event(
                            "waiting_for_loading_screen_started",
                            timeout_ms=loading_timeout_ms,
                            poll_sleep_ms=detection_poll_sleep_ms,
                        )
                    estimated_change, detail = _wait_for_loading_end(
                        context,
                        timeout_ms=loading_timeout_ms,
                        log_path=log_path,
                        trace_clock=trace_clock,
                        frame_tracer=frame_tracer,
                        state_trace=state_trace,
                        poll_sleep_ms=detection_poll_sleep_ms,
                        frame_source=detection_frame_source,
                    )
                else:
                    raise ValueError(f"Unsupported start_detection_mode: {start_detection_mode}")
            # Event ticks are absolute chart milliseconds. The scheduler zero point must
            # only include the post-loading fixed delay; the first tick is added later
            # when dispatching each event.
            delay_s = (fixed_delay_ms + user_offset_ms) / 1000.0
            target_start = estimated_change + delay_s
            t_delay_sleep_begin = time.perf_counter()
            remaining = target_start - t_delay_sleep_begin
            if state_trace is not None:
                state_trace.event(
                    "touch_schedule_computed",
                    timestamp=t_delay_sleep_begin,
                    loading_detail=detail,
                    fixed_delay_ms=fixed_delay_ms,
                    user_offset_ms=user_offset_ms,
                    first_tick_ms=PLAY_CACHE.first_tick_ms,
                    target_start=target_start,
                    first_touch_target=target_start + PLAY_CACHE.first_tick_ms / 1000.0,
                    remaining_ms=remaining * 1000.0,
                )
            if log_path is not None:
                _write_trace_jsonl(
                    log_path,
                    trace_clock,
                    {
                        "type": "schedule",
                        "loading_detail": detail,
                        "first_tick_ms": PLAY_CACHE.first_tick_ms,
                        "fixed_delay_ms": fixed_delay_ms,
                        "user_offset_ms": user_offset_ms,
                        "first_touch_target": target_start + PLAY_CACHE.first_tick_ms / 1000.0,
                        "target_start": target_start,
                        "t_delay_sleep_begin": t_delay_sleep_begin,
                        "remaining_ms": remaining * 1000.0,
                    },
                )
            first_touch_debug_path = _write_first_touch_debug(
                run_first_touch_debug_dir,
                target_start,
                input_backend,
                maa_wait_mode,
            )
            print(f"[INFO] First note touch debug saved: {first_touch_debug_path}")
            if log_path is not None:
                _write_trace_jsonl(
                    log_path,
                    trace_clock,
                    {
                        "type": "first_note_touch_debug_written",
                        "path": str(first_touch_debug_path),
                        "screen_resolution": PLAY_CACHE.screen_resolution,
                        "first_tick_ms": PLAY_CACHE.first_tick_ms,
                    },
                )
            if state_trace is not None:
                state_trace.event(
                    "first_note_touch_debug_written",
                    path=str(first_touch_debug_path),
                    screen_resolution=PLAY_CACHE.screen_resolution,
                    first_tick_ms=PLAY_CACHE.first_tick_ms,
                )
            backend: TouchBackend | None = None
            if not dry_run:
                t_backend_init_begin = time.perf_counter()
                backend = _take_cached_touch_backend(
                    input_backend,
                    maa_wait_mode,
                    adb_serial,
                    scrcpy_max_fps,
                    scrcpy_video_bit_rate,
                    scrcpy_video_crop,
                )
                if backend is prewarmed_detection_backend:
                    prewarmed_detection_backend = None
                if backend is None and prewarmed_detection_backend is not None:
                    backend = prewarmed_detection_backend
                    prewarmed_detection_backend = None
                if log_path is not None:
                    _write_trace_jsonl(
                        log_path,
                        trace_clock,
                        {
                            "type": "touch_backend_init_begin",
                            "backend": input_backend,
                            "cached": backend is not None,
                            "t_backend_init_begin": t_backend_init_begin,
                            "adb_serial": adb_serial,
                            "scrcpy_max_fps": scrcpy_max_fps,
                        },
                    )
                if state_trace is not None:
                    state_trace.event(
                        "touch_backend_init_begin",
                        timestamp=t_backend_init_begin,
                        backend=input_backend,
                        cached=backend is not None,
                        adb_serial=adb_serial,
                        scrcpy_max_fps=scrcpy_max_fps,
                    )
                if backend is None:
                    backend = create_touch_backend(
                        input_backend,
                        context,
                        maa_wait_mode=maa_wait_mode,
                        adb_serial=adb_serial,
                        scrcpy_max_fps=scrcpy_max_fps,
                        scrcpy_video_bit_rate=scrcpy_video_bit_rate,
                        scrcpy_video_crop=scrcpy_video_crop,
                    )
                t_backend_ready = time.perf_counter()
                first_touch_target = target_start + PLAY_CACHE.first_tick_ms / 1000.0
                remaining_after_backend = target_start - t_backend_ready
                first_touch_remaining_after_backend = first_touch_target - t_backend_ready
                backend_ready_payload = {
                    "type": "touch_backend_ready",
                    "backend": backend.name,
                    "adb_serial": adb_serial,
                    "scrcpy_max_fps": scrcpy_max_fps,
                    "scrcpy_video_bit_rate": scrcpy_video_bit_rate,
                    "scrcpy_video_crop": scrcpy_video_crop,
                    "cached": (t_backend_ready - t_backend_init_begin) < 0.001,
                    "t_backend_init_begin": t_backend_init_begin,
                    "t_backend_ready": t_backend_ready,
                    "init_duration_ms": (t_backend_ready - t_backend_init_begin) * 1000.0,
                    "remaining_ms": remaining_after_backend * 1000.0,
                    "first_touch_remaining_ms": first_touch_remaining_after_backend * 1000.0,
                }
                if log_path is not None:
                    _write_trace_jsonl(log_path, trace_clock, backend_ready_payload)
                if state_trace is not None:
                    state_trace.event(
                        "touch_backend_ready",
                        timestamp=t_backend_ready,
                        backend=backend.name,
                        adb_serial=adb_serial,
                        scrcpy_max_fps=scrcpy_max_fps,
                        init_duration_ms=backend_ready_payload["init_duration_ms"],
                        remaining_ms=backend_ready_payload["remaining_ms"],
                        first_touch_remaining_ms=backend_ready_payload["first_touch_remaining_ms"],
                    )
                remaining = remaining_after_backend

            if remaining > 0:
                time.sleep(remaining)
            t_delay_sleep_end = time.perf_counter()
            if log_path is not None:
                _write_trace_jsonl(log_path, trace_clock, {"type": "delay_sleep_end", "t_delay_sleep_end": t_delay_sleep_end})
            if state_trace is not None:
                state_trace.event("delay_sleep_end", timestamp=t_delay_sleep_end)

            if dry_run:
                _dry_run_dispatch(target_start, log_path, trace_clock)
                if state_trace is not None:
                    state_trace.event("dry_run_dispatch_end")
            else:
                if backend is None:
                    raise RuntimeError("Touch backend was not initialized")
                if state_trace is not None:
                    state_trace.event("touch_dispatch_started")
                _dispatch_events(backend, target_start, maa_wait_mode, log_path, trace_clock)
                if state_trace is not None:
                    state_trace.event("touch_dispatch_finished")
            if frame_tracer is not None:
                frame_tracer.close()
                print(
                    "[INFO] Frame trace saved: "
                    f"saved={frame_tracer.saved}, dropped={frame_tracer.dropped}, "
                    f"dir={run_frame_trace_dir}"
                )
            print(f"[INFO] ExecuteTouch completed; timing_log={log_path}")
            if state_trace is not None:
                state_trace.event("execute_touch_completed", timing_log=str(log_path) if log_path else None)
            return True
        except Exception as exc:
            print(f"[ERROR] PlaySong.ExecuteTouch failed: {exc}")
            if state_trace is not None:
                state_trace.event("execute_touch_failed", error=str(exc))
            return False
        finally:
            if "backend" in locals() and backend is not None:
                backend.close()
            if "prewarmed_detection_backend" in locals() and prewarmed_detection_backend is not None:
                prewarmed_detection_backend.close()
            if frame_tracer is not None:
                frame_tracer.close()


@AgentServer.custom_action("PlaySong.BenchmarkScreencap")
class BenchmarkScreencapAction(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        try:
            params = _parse_param(argv.custom_action_param)
            sample_count = int(params.get("sample_count", 200))
            warmup_count = int(params.get("warmup_count", 10))
            sleep_ms = int(params.get("sleep_ms", 0))
            debug_log = bool(params.get("debug_log", True))
            if sample_count <= 0:
                print("[ERROR] sample_count must be positive")
                return False

            log_path = None
            if debug_log:
                log_name = time.strftime("%Y%m%d_%H%M%S") + "_benchmark_screencap.jsonl"
                log_path = TIMING_LOG_DIR / log_name

            try:
                controller_info = context.tasker.controller.info
            except Exception as exc:
                controller_info = {"error": str(exc)}
            try:
                resolution = context.tasker.controller.resolution
            except Exception as exc:
                resolution = {"error": str(exc)}

            if log_path is not None:
                _write_jsonl(
                    log_path,
                    {
                        "type": "benchmark_start",
                        "sample_count": sample_count,
                        "warmup_count": warmup_count,
                        "sleep_ms": sleep_ms,
                        "controller_info": controller_info,
                        "resolution": resolution,
                    },
                )

            rows: list[dict[str, Any]] = []
            last_post_begin: float | None = None
            total_count = warmup_count + sample_count
            for index in range(total_count):
                t_post_begin = time.perf_counter()
                job = context.tasker.controller.post_screencap()
                job.wait()
                t_wait_end = time.perf_counter()
                frame = context.tasker.controller.cached_image
                t_cached_end = time.perf_counter()
                period_ms = None
                if last_post_begin is not None:
                    period_ms = (t_post_begin - last_post_begin) * 1000.0
                last_post_begin = t_post_begin
                record = {
                    "type": "screencap_sample",
                    "index": index,
                    "warmup": index < warmup_count,
                    "t_post_begin": t_post_begin,
                    "t_wait_end": t_wait_end,
                    "t_cached_end": t_cached_end,
                    "screencap_wait_ms": (t_wait_end - t_post_begin) * 1000.0,
                    "cached_image_ms": (t_cached_end - t_wait_end) * 1000.0,
                    "total_ms": (t_cached_end - t_post_begin) * 1000.0,
                    "period_ms": period_ms,
                    "shape": list(frame.shape[:2]),
                }
                if index >= warmup_count:
                    rows.append(record)
                if log_path is not None:
                    _write_jsonl(log_path, record)
                if sleep_ms > 0:
                    time.sleep(sleep_ms / 1000.0)

            wait_values = [float(row["screencap_wait_ms"]) for row in rows]
            cached_values = [float(row["cached_image_ms"]) for row in rows]
            total_values = [float(row["total_ms"]) for row in rows]
            period_values = [
                float(row["period_ms"])
                for row in rows
                if row.get("period_ms") is not None
            ]
            summary = {
                "type": "benchmark_summary",
                "samples": len(rows),
                "controller_info": controller_info,
                "resolution": resolution,
                "screencap_wait_ms": _summary_stats(wait_values),
                "cached_image_ms": _summary_stats(cached_values),
                "total_ms": _summary_stats(total_values),
                "period_ms": _summary_stats(period_values),
                "effective_fps_avg": (
                    1000.0 / statistics.mean(period_values) if period_values else 0.0
                ),
            }
            if log_path is not None:
                _write_jsonl(log_path, summary)
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            print(f"[INFO] BenchmarkScreencap completed; timing_log={log_path}")
            return True
        except Exception as exc:
            print(f"[ERROR] PlaySong.BenchmarkScreencap failed: {exc}")
            return False


def _summary_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"avg": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "avg": statistics.mean(values),
        "min": min(values),
        "max": max(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
    }
