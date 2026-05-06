from __future__ import annotations

import ctypes
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from autoplay.domain.chart import Chart
from autoplay.runtime.config_store import load_app_config
from autoplay.solver import CoordConv, solve_chart_auto

from loading_detector import LoadingEndDetector
from touch_backends import MaaTouchBackend, create_touch_backend


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_ROOT = PROJECT_ROOT / "assets"
TIMING_LOG_DIR = PROJECT_ROOT / "debug" / "timing"


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


def _wait_for_loading_end(
    context: Context,
    timeout_ms: int,
    log_path: Path | None,
) -> tuple[float, dict[str, Any]]:
    detector = LoadingEndDetector()
    deadline = time.perf_counter() + timeout_ms / 1000.0
    last_metrics: dict[str, Any] = {}
    loading_seen_logged = False
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
        last_metrics = dict(result.metrics)
        if log_path is not None:
            _write_jsonl(
                log_path,
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
                    "metrics": result.metrics,
                },
            )
        if result.loading_seen and not loading_seen_logged:
            loading_seen_logged = True
            print(f"[INFO] Loading screen detected at {result.timestamp:.6f}")
        if result.triggered and result.estimated_change_timestamp is not None:
            return result.estimated_change_timestamp, {
                "phase": result.phase.value,
                "estimated_change_timestamp": result.estimated_change_timestamp,
                "detector_timestamp": result.timestamp,
                "metrics": result.metrics,
            }
        time.sleep(0.05)
    raise TimeoutError(
        f"Timed out waiting for loading end; phase={detector.phase.value}, metrics={last_metrics}"
    )


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
                    _write_jsonl(
                        log_path,
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


def _dry_run_dispatch(target_start: float, log_path: Path | None) -> None:
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
        _write_jsonl(log_path, payload)


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
            debug_log = bool(params.get("debug_log", False))
            dry_run = bool(params.get("dry_run", False))
            skip_loading_detection = bool(params.get("skip_loading_detection", False))
            log_path = None
            if debug_log:
                log_name = time.strftime("%Y%m%d_%H%M%S") + "_execute_touch.jsonl"
                log_path = TIMING_LOG_DIR / log_name
                _write_jsonl(log_path, {"type": "action_enter", "t_action_enter": t_action_enter})

            if skip_loading_detection:
                estimated_change = time.perf_counter()
                detail = {
                    "phase": "skipped",
                    "estimated_change_timestamp": estimated_change,
                    "metrics": {},
                }
                print("[INFO] Loading detection skipped for dry/smoke test")
            else:
                print("[INFO] Waiting for loading screen to end")
                estimated_change, detail = _wait_for_loading_end(
                    context, loading_timeout_ms=loading_timeout_ms, log_path=log_path
                )
            # Event ticks are absolute chart milliseconds. The scheduler zero point must
            # only include the post-loading fixed delay; the first tick is added later
            # when dispatching each event.
            delay_s = (fixed_delay_ms + user_offset_ms) / 1000.0
            target_start = estimated_change + delay_s
            t_delay_sleep_begin = time.perf_counter()
            remaining = target_start - t_delay_sleep_begin
            if log_path is not None:
                _write_jsonl(
                    log_path,
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
                _write_jsonl(log_path, {"type": "delay_sleep_end", "t_delay_sleep_end": t_delay_sleep_end})

            if dry_run:
                _dry_run_dispatch(target_start, log_path)
            else:
                _dispatch_events(context, target_start, input_backend, maa_wait_mode, log_path)
            print(f"[INFO] ExecuteTouch completed; timing_log={log_path}")
            return True
        except Exception as exc:
            print(f"[ERROR] PlaySong.ExecuteTouch failed: {exc}")
            return False
