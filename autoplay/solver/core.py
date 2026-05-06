from __future__ import annotations

import math
from dataclasses import dataclass

from algo.algo_base import TouchAction
from autoplay.analyzer.mode_analyzer import ArcaeaTimelineAnalyzer
from autoplay.domain.arcaea_ir import (
    ArcIR,
    ArcaeaChartIR,
    HoldIR,
    SceneControlIR,
    TapIR,
)
from autoplay.domain.chart import Chart
from autoplay.solver.events import TouchEvent


@dataclass(slots=True)
class LogicalTouchEvent:
    tick: int
    x: float
    y: float
    action: TouchAction
    pointer: int
    source_note_id: int
    source_type: str


ACTION_PRIORITY = {
    TouchAction.DOWN: 0,
    TouchAction.MOVE: 1,
    TouchAction.UP: 2,
}


@dataclass(slots=True)
class LaneProfile:
    name: str

    def map_lane_to_x(self, lane: float) -> float:
        # AFF float lane coordinate maps to arc x by: x = -0.5 + lane * 2
        if not float(lane).is_integer():
            return -0.5 + lane * 2.0

        lane_i = int(lane)
        if self.name == "6k":
            # Raw 6k lane centers in Arcaea note space are:
            # lane 0..5 -> -0.75, -0.25, 0.25, 0.75, 1.25, 1.75
            return -0.75 + lane_i * 0.5

        # 4k default: lane 1..4 are the standard playable lanes.
        if lane_i <= 0:
            return -0.75
        if lane_i >= 5:
            return 1.75
        return -0.25 + (lane_i - 1) * 0.5


PROFILE_4K = LaneProfile(name="4k")
PROFILE_6K = LaneProfile(name="6k")
ARC_POINTER_BASE = 5000
SKY_WIDEN_Y_SCALE = 1.61


def _rotate_point(x: float, y: float, anglex: int, angley: int) -> tuple[float, float]:
    ax = math.radians(anglex / 10)
    ay = math.radians(angley / 10)

    y_rot = y * math.cos(ax) - math.sin(ax)
    z_rot = y * math.sin(ax) + math.cos(ax)
    x_rot = x * math.cos(ay) + z_rot * math.sin(ay)
    return x_rot, y_rot


def _sample_arc_ticks(start: int, end: int, smoothness: float | None) -> list[int]:
    delta = end - start
    if delta <= 0:
        return [start]

    base_step = 10
    if smoothness is not None and smoothness > 1:
        base_step = max(3, int(base_step / smoothness))

    steps = max(2, math.ceil(delta / base_step))
    return [start + int(idx * delta / steps) for idx in range(steps + 1)]


def _merge_and_sort_ticks(base_ticks: list[int], extra_ticks: set[int]) -> list[int]:
    merged = set(base_ticks)
    merged.update(extra_ticks)
    return sorted(merged)


def _project_ground_lane_x_for_mode(lane: float, lane_widen_ratio: float) -> float:
    ratio = max(0.0, min(1.0, lane_widen_ratio))
    x_4k = PROFILE_4K.map_lane_to_x(lane)
    x_6k = PROFILE_6K.map_lane_to_x(lane)
    return x_4k + (x_6k - x_4k) * ratio


def _project_ground_logical_coord_for_mode(
    lane: float,
    tick: int,
    timeline: ArcaeaTimelineAnalyzer,
) -> tuple[float, float]:
    lane_ratio = timeline.lane_widen_ratio_at(tick)
    raw_x = _project_ground_lane_x_for_mode(lane, lane_ratio)
    sky_ratio = timeline.sky_widen_ratio_at(tick)
    return _project_arc_logical_coord_for_mode(raw_x, 0.0, sky_ratio)


def _arc_pointer_from_color(color: int) -> int:
    if color < 0:
        color = 0
    return ARC_POINTER_BASE + color


def _project_arc_logical_coord_for_mode(
    x: float, y: float, sky_widen_ratio: float
) -> tuple[float, float]:
    ratio = max(0.0, min(1.0, sky_widen_ratio))
    y_max = 1.0 + (SKY_WIDEN_Y_SCALE - 1.0) * ratio
    if y_max <= 0:
        return x, y

    v = y / y_max

    left_bottom = -0.5 - 0.5 * ratio
    right_bottom = 1.5 + 0.5 * ratio
    left_top = 0.0 - 0.25 * ratio
    right_top = 1.0 + 0.25 * ratio

    left = left_bottom + (left_top - left_bottom) * v
    right = right_bottom + (right_top - right_bottom) * v
    width = right - left
    if abs(width) < 1e-9:
        u = 0.5
    else:
        u = (x - left) / width

    projected_y = v
    target_left = -0.5 + 0.5 * projected_y
    target_right = 1.5 - 0.5 * projected_y
    projected_x = target_left + (target_right - target_left) * u
    return projected_x, projected_y


def _is_same_logical_point(
    left: LogicalTouchEvent, right: LogicalTouchEvent, eps: float = 1e-4
) -> bool:
    return abs(left.x - right.x) <= eps and abs(left.y - right.y) <= eps


def _resolve_same_tick_arc_head_arctap_conflicts(
    events: list[LogicalTouchEvent],
) -> list[LogicalTouchEvent]:
    by_tick: dict[int, list[int]] = {}
    for idx, event in enumerate(events):
        if event.action is not TouchAction.DOWN:
            continue
        if event.source_type not in {"arc", "arctap"}:
            continue
        by_tick.setdefault(event.tick, []).append(idx)

    remove_indices: set[int] = set()

    for tick, down_indices in by_tick.items():
        arc_down_indices = [
            idx for idx in down_indices if events[idx].source_type == "arc"
        ]
        arctap_down_indices = [
            idx for idx in down_indices if events[idx].source_type == "arctap"
        ]
        if not arc_down_indices or not arctap_down_indices:
            continue

        for arctap_idx in arctap_down_indices:
            arctap_down = events[arctap_idx]
            has_overlap_arc_head = any(
                _is_same_logical_point(arctap_down, events[arc_idx])
                for arc_idx in arc_down_indices
            )
            if not has_overlap_arc_head:
                continue

            remove_indices.add(arctap_idx)
            for idx, event in enumerate(events):
                if idx in remove_indices:
                    continue
                if event.tick != tick + 12:
                    continue
                if event.action is not TouchAction.UP:
                    continue
                if event.source_type != "arctap":
                    continue
                if event.pointer != arctap_down.pointer:
                    continue
                if event.source_note_id != arctap_down.source_note_id:
                    continue
                remove_indices.add(idx)
                break

    if not remove_indices:
        return events
    return [event for idx, event in enumerate(events) if idx not in remove_indices]


def _resolve_connected_same_color_arc_boundaries(
    events: list[LogicalTouchEvent],
) -> list[LogicalTouchEvent]:
    by_pointer: dict[int, list[int]] = {}
    for idx, event in enumerate(events):
        if event.source_type != "arc":
            continue
        if event.action not in {TouchAction.DOWN, TouchAction.UP}:
            continue
        by_pointer.setdefault(event.pointer, []).append(idx)

    remove_indices: set[int] = set()
    inserted_moves: list[LogicalTouchEvent] = []

    for _, indices in by_pointer.items():
        down_indices = [
            idx for idx in indices if events[idx].action is TouchAction.DOWN
        ]
        up_indices = [idx for idx in indices if events[idx].action is TouchAction.UP]
        if not down_indices or not up_indices:
            continue

        down_indices.sort(key=lambda idx: events[idx].tick)
        up_indices.sort(key=lambda idx: events[idx].tick)

        used_down: set[int] = set()
        for up_idx in up_indices:
            up_event = events[up_idx]
            for down_idx in down_indices:
                if down_idx in used_down:
                    continue
                down_event = events[down_idx]
                if up_event.source_note_id == down_event.source_note_id:
                    continue
                if abs(up_event.tick - down_event.tick) > 3:
                    continue

                has_boundary_move = any(
                    event.tick == up_event.tick
                    and event.pointer == up_event.pointer
                    and event.action is TouchAction.MOVE
                    and idx not in remove_indices
                    for idx, event in enumerate(events)
                )
                if not has_boundary_move:
                    inserted_moves.append(
                        LogicalTouchEvent(
                            tick=up_event.tick,
                            x=down_event.x,
                            y=down_event.y,
                            action=TouchAction.MOVE,
                            pointer=up_event.pointer,
                            source_note_id=down_event.source_note_id,
                            source_type="arc",
                        )
                    )

                remove_indices.add(up_idx)
                remove_indices.add(down_idx)
                used_down.add(down_idx)
                break

    if not remove_indices and not inserted_moves:
        return events
    resolved = [event for idx, event in enumerate(events) if idx not in remove_indices]
    resolved.extend(inserted_moves)
    resolved.sort(
        key=lambda item: (item.tick, item.pointer, ACTION_PRIORITY.get(item.action, 99))
    )
    return resolved


def _build_logical_events(
    chart_ir: ArcaeaChartIR, timeline: ArcaeaTimelineAnalyzer
) -> list[LogicalTouchEvent]:
    events: list[LogicalTouchEvent] = []
    arctap_pointer = 1000

    def append_event(
        tick: int,
        x: float,
        y: float,
        action: TouchAction,
        pointer: int,
        source_note_id: int,
        source_type: str,
    ) -> None:
        events.append(
            LogicalTouchEvent(
                tick=tick,
                x=x,
                y=y,
                action=action,
                pointer=pointer,
                source_note_id=source_note_id,
                source_type=source_type,
            )
        )

    for note in chart_ir.notes:
        if note.noinput:
            continue

        anglex = int(note.group_properties.get("anglex", 0))
        angley = int(note.group_properties.get("angley", 0))

        if isinstance(note, TapIR):
            x, y = _project_ground_logical_coord_for_mode(note.lane, note.tick, timeline)
            append_event(
                note.tick,
                x,
                y,
                TouchAction.DOWN,
                int(round(note.lane)),
                note.note_id,
                "tap",
            )
            append_event(
                note.tick + 20,
                x,
                y,
                TouchAction.UP,
                int(round(note.lane)),
                note.note_id,
                "tap",
            )
            continue

        if isinstance(note, HoldIR):
            x_start, y_start = _project_ground_logical_coord_for_mode(
                note.lane, note.start, timeline
            )
            x_end, y_end = _project_ground_logical_coord_for_mode(
                note.lane, note.end, timeline
            )
            pointer = int(round(note.lane)) + 100
            append_event(
                note.start,
                x_start,
                y_start,
                TouchAction.DOWN,
                pointer,
                note.note_id,
                "hold",
            )
            if x_start != x_end and note.end > note.start:
                mid_tick = note.start + (note.end - note.start) // 2
                x_mid, y_mid = _project_ground_logical_coord_for_mode(
                    note.lane, mid_tick, timeline
                )
                append_event(
                    mid_tick,
                    x_mid,
                    y_mid,
                    TouchAction.MOVE,
                    pointer,
                    note.note_id,
                    "hold",
                )
            append_event(
                note.end, x_end, y_end, TouchAction.UP, pointer, note.note_id, "hold"
            )
            continue

        if not isinstance(note, ArcIR):
            continue

        if note.trace_arc:
            delta = note.end - note.start if note.end != note.start else 1
            start = (note.start_x, note.start_y, 1)
            end = (note.end_x, note.end_y, 1)
            for tap_tick in note.taps:
                t = max(0.0, min(1.0, (tap_tick - note.start) / delta))
                px, py, _ = note.easing.value(start, end, t)
                px, py = _rotate_point(px, py, anglex, angley)
                sky_ratio = timeline.sky_widen_ratio_at(tap_tick)
                px, py = _project_arc_logical_coord_for_mode(px, py, sky_ratio)
                append_event(
                    tap_tick,
                    px,
                    py,
                    TouchAction.DOWN,
                    arctap_pointer,
                    note.note_id,
                    "arctap",
                )
                append_event(
                    tap_tick + 12,
                    px,
                    py,
                    TouchAction.UP,
                    arctap_pointer,
                    note.note_id,
                    "arctap",
                )
                arctap_pointer += 1
                if arctap_pointer > 2000:
                    arctap_pointer = 1000
            continue

        pointer = _arc_pointer_from_color(note.color)
        if note.start == note.end:
            pointer = ARC_POINTER_BASE + note.note_id
            px, py = _rotate_point(note.start_x, note.start_y, anglex, angley)
            sky_ratio = timeline.sky_widen_ratio_at(note.start)
            px, py = _project_arc_logical_coord_for_mode(px, py, sky_ratio)
            append_event(
                note.start, px, py, TouchAction.DOWN, pointer, note.note_id, "zero_arc"
            )
            append_event(
                note.start + 12,
                px,
                py,
                TouchAction.UP,
                pointer,
                note.note_id,
                "zero_arc",
            )
            continue

        sample_ticks = _sample_arc_ticks(note.start, note.end, note.smoothness)
        transition_ticks = {
            tick
            for tick in timeline.sky_transition_ticks()
            if note.start <= tick <= note.end
        }
        sample_ticks = _merge_and_sort_ticks(sample_ticks, transition_ticks)
        start = (note.start_x, note.start_y, 1)
        end = (note.end_x, note.end_y, 1)
        delta = note.end - note.start

        for idx, tick in enumerate(sample_ticks):
            t = max(0.0, min(1.0, (tick - note.start) / delta))
            px, py, _ = note.easing.value(start, end, t)
            px, py = _rotate_point(px, py, anglex, angley)
            sky_ratio = timeline.sky_widen_ratio_at(tick)
            px, py = _project_arc_logical_coord_for_mode(px, py, sky_ratio)
            if idx == 0:
                action = TouchAction.DOWN
            elif idx == len(sample_ticks) - 1:
                action = TouchAction.UP
            else:
                action = TouchAction.MOVE
                px = round(px, 4)
                py = round(py, 4)
            append_event(tick, px, py, action, pointer, note.note_id, "arc")

        for tap_tick in note.taps:
            t = max(0.0, min(1.0, (tap_tick - note.start) / delta))
            px, py, _ = note.easing.value(start, end, t)
            px, py = _rotate_point(px, py, anglex, angley)
            sky_ratio = timeline.sky_widen_ratio_at(tap_tick)
            px, py = _project_arc_logical_coord_for_mode(px, py, sky_ratio)
            append_event(
                tap_tick,
                px,
                py,
                TouchAction.DOWN,
                arctap_pointer,
                note.note_id,
                "arctap",
            )
            append_event(
                tap_tick + 12,
                px,
                py,
                TouchAction.UP,
                arctap_pointer,
                note.note_id,
                "arctap",
            )
            arctap_pointer += 1
            if arctap_pointer > 2000:
                arctap_pointer = 1000

    events = _resolve_same_tick_arc_head_arctap_conflicts(events)

    events.sort(
        key=lambda item: (item.tick, item.pointer, ACTION_PRIORITY.get(item.action, 99))
    )
    events = _resolve_connected_same_color_arc_boundaries(events)

    compacted: list[LogicalTouchEvent] = []
    for event in events:
        if compacted:
            prev = compacted[-1]
            same_point = prev.x == event.x and prev.y == event.y
            if (
                prev.tick == event.tick
                and prev.pointer == event.pointer
                and prev.action == event.action
                and same_point
            ):
                continue
        compacted.append(event)
    return compacted


def _project_to_touch_events(
    logical_events: list[LogicalTouchEvent], converter
) -> dict[int, list[TouchEvent]]:
    result: dict[int, list[TouchEvent]] = {}
    for logical_event in logical_events:
        px, py = converter(logical_event.x, logical_event.y)
        touch_event = TouchEvent(
            (round(px), round(py)),
            logical_event.action,
            logical_event.pointer,
            source_note_id=logical_event.source_note_id,
            source_type=logical_event.source_type,
            logical_tick=logical_event.tick,
            logical_pos=(logical_event.x, logical_event.y),
        )
        result.setdefault(logical_event.tick, []).append(touch_event)
    return result


def solve_chart_auto(chart: Chart, converter) -> dict[int, list[TouchEvent]]:
    chart_ir = chart.ir
    if chart_ir is None:
        return {}
    timeline = ArcaeaTimelineAnalyzer()
    timeline.build(chart_ir)
    logical_events = _build_logical_events(chart_ir, timeline)
    return _project_to_touch_events(logical_events, converter)


def build_logical_events_for_chart(
    chart: Chart,
    lane_mode: str | None = None,
    sky_mode: str | None = None,
) -> list[LogicalTouchEvent]:
    chart_ir = chart.ir
    if chart_ir is None:
        return []

    timeline = ArcaeaTimelineAnalyzer()
    if lane_mode is None and sky_mode is None:
        timeline.build(chart_ir)
    else:
        lane = lane_mode or "4k"
        sky = sky_mode or lane
        timeline.build(_filter_chart_to_mode(chart_ir, lane, sky))

    return _build_logical_events(chart_ir, timeline)


def _filter_chart_to_mode(
    chart_ir: ArcaeaChartIR, lane_mode: str, sky_mode: str
) -> ArcaeaChartIR:
    # Compatibility shim: keep full IR and force timeline via synthetic scenecontrol,
    # preserving previous solve_4k/solve_6k function signatures.
    filtered = ArcaeaChartIR(
        options=dict(chart_ir.options),
        notes=list(chart_ir.notes),
        timings=list(chart_ir.timings),
        scene_controls=[],
    )
    if lane_mode == "6k":
        filtered.scene_controls.append(
            SceneControlIR(tick=0, control_type="enwidenlanes", param1=0.0, param2=1)
        )
    if sky_mode == "6k":
        filtered.scene_controls.append(
            SceneControlIR(tick=0, control_type="enwidencamera", param1=0.0, param2=1)
        )
    return filtered


def solve_4k(chart: Chart, converter) -> dict[int, list[TouchEvent]]:
    chart_ir = chart.ir
    if chart_ir is None:
        return {}
    timeline = ArcaeaTimelineAnalyzer()
    timeline.build(_filter_chart_to_mode(chart_ir, "4k", "4k"))
    logical_events = _build_logical_events(chart_ir, timeline)
    return _project_to_touch_events(logical_events, converter)


def solve_6k(chart: Chart, converter) -> dict[int, list[TouchEvent]]:
    chart_ir = chart.ir
    if chart_ir is None:
        return {}
    timeline = ArcaeaTimelineAnalyzer()
    timeline.build(_filter_chart_to_mode(chart_ir, "6k", "6k"))
    logical_events = _build_logical_events(chart_ir, timeline)
    return _project_to_touch_events(logical_events, converter)


__all__ = [
    "solve_4k",
    "solve_6k",
    "solve_chart_auto",
    "build_logical_events_for_chart",
    "LogicalTouchEvent",
]
