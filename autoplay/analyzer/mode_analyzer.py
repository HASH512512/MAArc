from __future__ import annotations

from dataclasses import dataclass

from autoplay.domain.arcaea_ir import ArcaeaChartIR


@dataclass(slots=True)
class ModeSegment:
    start: int
    end: int
    mode: str


@dataclass(slots=True)
class ProjectionState:
    lane_mode: str
    sky_mode: str
    lane_widen_ratio: float = 0.0
    sky_widen_ratio: float = 0.0


class ArcaeaTimelineAnalyzer:
    def __init__(self) -> None:
        self._lane_events: list[tuple[int, float, int]] = []
        self._sky_events: list[tuple[int, float, int]] = []

    def build(self, chart_ir: ArcaeaChartIR) -> None:
        self._lane_events = []
        self._sky_events = []

        for control in chart_ir.scene_controls:
            duration = float(control.param1) if control.param1 is not None else 0.0
            switch = int(control.param2) if control.param2 is not None else 0
            if control.control_type == "enwidenlanes":
                self._lane_events.append((control.tick, duration, switch))
            elif control.control_type == "enwidencamera":
                self._sky_events.append((control.tick, duration, switch))

        self._lane_events.sort(key=lambda item: item[0])
        self._sky_events.sort(key=lambda item: item[0])

    def _build_segments(
        self, events: list[tuple[int, float, int]], max_tick: int
    ) -> list[ModeSegment]:
        if max_tick <= 0:
            return []

        if not events:
            return [ModeSegment(0, max_tick, "4k")]

        segments: list[ModeSegment] = []
        current_mode = "4k"
        cursor = 0

        for tick, duration, switch in events:
            if tick > cursor:
                segments.append(ModeSegment(cursor, tick, current_mode))
                cursor = tick

            half = tick + int(duration / 2)
            if half > cursor:
                segments.append(ModeSegment(cursor, half, current_mode))
                cursor = half

            current_mode = "6k" if switch == 1 else "4k"

        if cursor < max_tick:
            segments.append(ModeSegment(cursor, max_tick, current_mode))
        return segments

    def lane_segments(self, max_tick: int) -> list[ModeSegment]:
        return self._build_segments(self._lane_events, max_tick)

    def sky_segments(self, max_tick: int) -> list[ModeSegment]:
        return self._build_segments(self._sky_events, max_tick)

    def lane_mode_at(self, tick: int) -> str:
        return "6k" if self.lane_widen_ratio_at(tick) >= 0.5 else "4k"

    def sky_mode_at(self, tick: int) -> str:
        return "6k" if self.sky_widen_ratio_at(tick) >= 0.5 else "4k"

    @staticmethod
    def _widen_ratio_at(events: list[tuple[int, float, int]], tick: int) -> float:
        ratio = 0.0
        for event_tick, duration, switch in events:
            target = 1.0 if switch == 1 else 0.0
            if tick < event_tick:
                break

            if duration <= 0:
                ratio = target
                continue

            end_tick = event_tick + int(duration)
            if tick >= end_tick:
                ratio = target
                continue

            progress = (tick - event_tick) / duration
            return ratio + (target - ratio) * progress
        return ratio

    def lane_widen_ratio_at(self, tick: int) -> float:
        return self._widen_ratio_at(self._lane_events, tick)

    def sky_widen_ratio_at(self, tick: int) -> float:
        return self._widen_ratio_at(self._sky_events, tick)

    def lane_transition_ticks(self) -> set[int]:
        ticks: set[int] = set()
        for event_tick, duration, _ in self._lane_events:
            ticks.add(event_tick)
            if duration > 0:
                ticks.add(event_tick + int(duration))
        return ticks

    def sky_transition_ticks(self) -> set[int]:
        ticks: set[int] = set()
        for event_tick, duration, _ in self._sky_events:
            ticks.add(event_tick)
            if duration > 0:
                ticks.add(event_tick + int(duration))
        return ticks

    def projection_state_at(self, tick: int) -> ProjectionState:
        return ProjectionState(
            lane_mode=self.lane_mode_at(tick),
            sky_mode=self.sky_mode_at(tick),
            lane_widen_ratio=self.lane_widen_ratio_at(tick),
            sky_widen_ratio=self.sky_widen_ratio_at(tick),
        )


class ModeAnalyzer:
    """Compatibility wrapper exposing the previous API surface."""

    def __init__(self) -> None:
        self.timeline = ArcaeaTimelineAnalyzer()
        self._max_tick = 0

    def analyze_chart_for_6k(
        self,
        chart_content: str,
        chart=None,
        chart_ir: ArcaeaChartIR | None = None,
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
        if chart_ir is None and chart is not None:
            chart_ir = getattr(chart, "ir", None)
        if chart_ir is None:
            return [], [], 0

        self.timeline.build(chart_ir)
        self._max_tick = chart_ir.max_tick()

        sky = [
            (segment.start, segment.end)
            for segment in self.timeline.sky_segments(self._max_tick)
        ]
        lane = [
            (segment.start, segment.end)
            for segment in self.timeline.lane_segments(self._max_tick)
        ]
        return sky, lane, self._max_tick

    def get_sky_segments(self) -> list[tuple[int, int, str]]:
        return [
            (segment.start, segment.end, segment.mode)
            for segment in self.timeline.sky_segments(self._max_tick)
        ]

    def get_ground_segments(self) -> list[tuple[int, int, str]]:
        return [
            (segment.start, segment.end, segment.mode)
            for segment in self.timeline.lane_segments(self._max_tick)
        ]

    def projection_state_at(self, tick: int) -> ProjectionState:
        return self.timeline.projection_state_at(tick)

    def split_and_solve_chart(
        self, chart, converter, solve_4k, solve_6k
    ) -> dict[int, list]:
        chart_ir = getattr(chart, "ir", None)
        if chart_ir is None:
            return {}

        self.timeline.build(chart_ir)

        from autoplay.solver.core import solve_chart_auto

        return solve_chart_auto(chart, converter)
