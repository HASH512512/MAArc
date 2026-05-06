from __future__ import annotations

import threading
import time
import queue
import ctypes
from collections.abc import Callable

from algo.algo_base import TouchAction
from control import DeviceController


class FineTuneState:
    def __init__(self, step_ms: int) -> None:
        self.step_ms = step_ms
        self.time_offset = 0.0
        self.time_lock = threading.Lock()
        self.input_listener_active = False
        self.automation_started = False

    def increment(self) -> float:
        with self.time_lock:
            self.time_offset += self.step_ms / 1000.0
            return self.time_offset

    def decrement(self) -> float:
        with self.time_lock:
            self.time_offset -= self.step_ms / 1000.0
            return self.time_offset

    def reset(self) -> float:
        with self.time_lock:
            self.time_offset = 0.0
            return self.time_offset

    def current_offset(self) -> float:
        with self.time_lock:
            return self.time_offset


def _read_hotkey_edges(
    get_async_key_state,
    previous_state: dict[int, bool],
    vk_to_command: dict[int, str],
) -> list[str]:
    commands: list[str] = []
    for vk, command in vk_to_command.items():
        is_down = bool(get_async_key_state(vk) & 0x8000)
        was_down = previous_state[vk]
        previous_state[vk] = is_down
        if is_down and not was_down:
            commands.append(command)
    return commands


def start_input_listener(state: FineTuneState, on_command) -> threading.Thread:
    command_queue: queue.Queue[str] = queue.Queue()

    def stdin_reader() -> None:
        while state.input_listener_active:
            try:
                text = input()
            except (EOFError, KeyboardInterrupt):
                break

            command = text.strip().lower()[:1]
            if command in {"z", "x", "r"}:
                command_queue.put(command)

    def input_listener() -> None:
        while state.input_listener_active:
            try:
                try:
                    command = command_queue.get_nowait()
                except queue.Empty:
                    time.sleep(0.01)
                    continue

                on_command(command)
            except (EOFError, KeyboardInterrupt, SystemExit):
                break
            except Exception as exc:
                print(f"[Input listener error] {exc}")
                break

    stdin_thread = threading.Thread(target=stdin_reader, daemon=True)
    stdin_thread.start()

    listener_thread = threading.Thread(target=input_listener, daemon=True)
    listener_thread.start()
    return listener_thread


def prepare_device_controller(
    max_fps: int = 60,
    video_bit_rate: int | None = None,
    video_crop: tuple[int, int, int, int] | None = None,
) -> DeviceController:
    return DeviceController(
        max_fps=max_fps,
        video_bit_rate=video_bit_rate,
        video_crop=video_crop,
    )


def run_touch_events(
    events_by_time: dict[int, list],
    base_delay: float,
    state: FineTuneState,
    controller: DeviceController | None = None,
    log: Callable[[str], None] | None = None,
    on_progress: Callable[[int, list, int | None, list | None], None] | None = None,
    on_first_dispatch: Callable[[dict[str, float]], None] | None = None,
    start_signal: threading.Event | None = None,
    debug: bool = False,
    optimize_high_priority: bool = False,
    optimize_timer_resolution: bool = False,
) -> None:
    def _log(message: str) -> None:
        if log is not None:
            log(message)
        else:
            print(message)

    sorted_events = sorted(events_by_time.items())
    if not sorted_events:
        _log("[Error] No touch events generated")
        return

    if controller is None:
        controller = prepare_device_controller()
    event_iter = iter(sorted_events)
    active_pointers: dict[int, tuple[int, int]] = {}

    try:
        ms, events = next(event_iter)
    except StopIteration:
        _log("[Warning] Event sequence terminated unexpectedly")
        return

    state.automation_started = True

    timer_boost_active = False
    if optimize_timer_resolution:
        try:
            if ctypes.windll.winmm.timeBeginPeriod(1) == 0:
                timer_boost_active = True
                if debug:
                    _log("[DEBUG] High-resolution timer enabled (1ms)")
        except Exception as exc:
            if debug:
                _log(f"[DEBUG] Failed to enable high-resolution timer: {exc}")

    if optimize_high_priority:
        try:
            THREAD_PRIORITY_HIGHEST = 2
            ctypes.windll.kernel32.SetThreadPriority(
                ctypes.windll.kernel32.GetCurrentThread(), THREAD_PRIORITY_HIGHEST
            )
            if debug:
                _log("[DEBUG] Playback thread priority set to HIGHEST")
        except Exception as exc:
            if debug:
                _log(f"[DEBUG] Failed to set playback thread priority: {exc}")

    if start_signal is not None:
        if debug:
            _log("[DEBUG] Waiting for external start signal")
        while state.input_listener_active and not start_signal.wait(0.001):
            pass
        if not state.input_listener_active:
            return

    start_time = time.perf_counter() + base_delay
    first_dispatch_sent = False
    if debug:
        _log(
            f"[DEBUG] Scheduler armed: base_delay={base_delay:.6f}s, first_tick={ms}, queue_ticks={len(sorted_events)}"
        )
    _log("[INFO] Auto play started")

    try:
        while state.input_listener_active:
            now = (time.perf_counter() - start_time + state.current_offset()) * 1000
            if now >= ms:
                if debug:
                    _log(
                        f"[DEBUG] Dispatch tick={ms}, lateness={now - ms:.3f}ms, events={len(events)}"
                    )

                for event in events:
                    x, y = event.pos
                    dispatch_started_at = time.perf_counter()
                    controller.touch(x, y, event.action, event.pointer)
                    if not first_dispatch_sent and on_first_dispatch is not None:
                        first_dispatch_sent = True
                        on_first_dispatch(
                            {
                                "scheduled_tick_ms": float(ms),
                                "dispatch_loop_time_ms": now,
                                "lateness_ms": float(now - ms),
                                "touch_send_call_ms": (
                                    time.perf_counter() - dispatch_started_at
                                )
                                * 1000.0,
                            }
                        )
                    if event.action in {
                        TouchAction.DOWN,
                        TouchAction.MOVE,
                        TouchAction.POINTER_DOWN,
                    }:
                        active_pointers[event.pointer] = (x, y)
                    elif event.action in {
                        TouchAction.UP,
                        TouchAction.POINTER_UP,
                        TouchAction.CANCEL,
                    }:
                        active_pointers.pop(event.pointer, None)

                try:
                    peek_ms, peek_events = next(event_iter)
                    if on_progress is not None:
                        on_progress(ms, events, peek_ms, peek_events)
                    ms, events = peek_ms, peek_events
                except StopIteration:
                    if on_progress is not None:
                        on_progress(ms, events, None, None)
                    break
            else:
                time.sleep(0.0005)
    except (KeyboardInterrupt, SystemExit):
        _log("[INFO] User interrupted execution")
    except Exception as exc:
        _log(f"[ERROR] Execution error: {exc}")
    finally:
        if active_pointers:
            if debug:
                _log(
                    f"[DEBUG] Releasing active pointers: {sorted(active_pointers.keys())}"
                )
            for pointer, (x, y) in list(active_pointers.items()):
                try:
                    controller.touch(x, y, TouchAction.UP, pointer)
                except Exception as exc:
                    if debug:
                        _log(f"[DEBUG] Failed to release pointer {pointer}: {exc}")

        if timer_boost_active:
            try:
                ctypes.windll.winmm.timeEndPeriod(1)
                if debug:
                    _log("[DEBUG] High-resolution timer released")
            except Exception:
                pass
        state.input_listener_active = False
        state.automation_started = False
