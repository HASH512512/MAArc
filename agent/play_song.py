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

from autoplay.domain.chart import Chart
from autoplay.runtime.config_store import load_app_config
from autoplay.solver import CoordConv, solve_chart_auto

from loading_detector import FreezeChangeDetector, LoadingEndDetector
from touch_backends import MaaTouchBackend, create_touch_backend


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_ROOT = PROJECT_ROOT / "assets"
TIMING_LOG_DIR = PROJECT_ROOT / "debug" / "timing"
FRAME_TRACE_DIR = PROJECT_ROOT / "debug" / "frames"
STATE_TRACE_DIR = PROJECT_ROOT / "debug" / "state_trace"


@dataclass(slots=True)
class PlayCache:
    events_by_time: dict[int, list] = field(default_factory=dict)
    first_tick_ms: int = 0
    chart_path: Path | None = None
    tick_count: int = 0
    event_count: int = 0


PLAY_CACHE = PlayCache()


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


def _load_chart_events(chart_path: Path) -> PlayCache:
    app_config = load_app_config()
    global_config = app_config.global_config
    content = _read_text(chart_path)
    chart = Chart.loads(content, designant_choice=global_config.designant_choice)
    converter = CoordConv(
        global_config.bottom_left,
        global_config.top_left,
        global_config.top_right,
        global_config.bottom_right,
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


def _wait_for_loading_end(
    context: Context,
    timeout_ms: int,
    log_path: Path | None,
    trace_clock: TraceClock,
    frame_tracer: FrameTraceWriter | None,
    state_trace: StateTraceRecorder | None,
    poll_sleep_ms: float,
) -> tuple[float, dict[str, Any]]:
    detector = LoadingEndDetector()
    deadline = time.perf_counter() + timeout_ms / 1000.0
    last_metrics: dict[str, Any] = {}
    loading_seen_logged = False
    monitoring_started_at: float | None = None
    timing_stats = RecognitionTimingStats()
    while time.perf_counter() < deadline:
        t_screencap_post_begin = time.perf_counter()
        job = context.tasker.controller.post_screencap()
        job.wait()
        t_screencap_wait_end = time.perf_counter()
        frame = context.tasker.controller.cached_image
        t_cached_image_read_end = time.perf_counter()
        t_detector_begin = time.perf_counter()
        result = detector.update(frame, t_cached_image_read_end)
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
                    t_cached_image_read_end,
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
                    t_cached_image_read_end,
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
) -> tuple[float, dict[str, Any]]:
    if start_delay_ms > 0:
        time.sleep(start_delay_ms / 1000.0)
    detector = FreezeChangeDetector()
    deadline = time.perf_counter() + timeout_ms / 1000.0
    last_metrics: dict[str, Any] = {}
    still_seen_logged = False
    timing_stats = RecognitionTimingStats()
    while time.perf_counter() < deadline:
        t_screencap_post_begin = time.perf_counter()
        job = context.tasker.controller.post_screencap()
        job.wait()
        t_screencap_wait_end = time.perf_counter()
        frame = context.tasker.controller.cached_image
        t_cached_image_read_end = time.perf_counter()
        t_detector_begin = time.perf_counter()
        result = detector.update(frame, t_cached_image_read_end)
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
                    t_cached_image_read_end,
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
                    t_cached_image_read_end,
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
    context: Context,
    target_start: float,
    input_backend: str,
    maa_wait_mode: str,
    log_path: Path | None,
    trace_clock: TraceClock,
) -> None:
    backend = create_touch_backend(input_backend, context, maa_wait_mode=maa_wait_mode)
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
        backend.close()
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


@AgentServer.custom_action("PlaySong.LoadChart")
class LoadChartAction(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        del context
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
            cache = _load_chart_events(chart_path)
            PLAY_CACHE.events_by_time = cache.events_by_time
            PLAY_CACHE.first_tick_ms = cache.first_tick_ms
            PLAY_CACHE.chart_path = cache.chart_path
            PLAY_CACHE.tick_count = cache.tick_count
            PLAY_CACHE.event_count = cache.event_count
            print(
                "[INFO] Chart loaded: "
                f"path={chart_path}, first_tick={cache.first_tick_ms}, "
                f"ticks={cache.tick_count}, events={cache.event_count}"
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
            loading_timeout_ms = int(params.get("loading_timeout_ms", 30000))
            start_detection_delay_ms = int(params.get("start_detection_delay_ms", 100))
            start_detection_mode = str(params.get("start_detection_mode", "zigzag"))
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
                        "start_detection_delay_ms": start_detection_delay_ms,
                        "detection_poll_sleep_ms": detection_poll_sleep_ms,
                        "trace_prefix": trace_prefix,
                        "frame_trace_dir": str(run_frame_trace_dir),
                        "state_trace_dir": str(run_state_trace_dir),
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
                    first_tick_ms=PLAY_CACHE.first_tick_ms,
                    tick_count=PLAY_CACHE.tick_count,
                    event_count=PLAY_CACHE.event_count,
                    start_detection_mode=start_detection_mode,
                    start_detection_delay_ms=start_detection_delay_ms,
                    detection_poll_sleep_ms=detection_poll_sleep_ms,
                    dry_run=dry_run,
                    input_backend=input_backend,
                    maa_wait_mode=maa_wait_mode,
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
                if state_trace is not None:
                    state_trace.event("touch_dispatch_started")
                _dispatch_events(context, target_start, input_backend, maa_wait_mode, log_path, trace_clock)
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
