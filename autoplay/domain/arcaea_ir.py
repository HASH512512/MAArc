from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from easing import Easing


@dataclass(slots=True)
class SceneControlIR:
    tick: int
    control_type: str
    param1: float | None = None
    param2: int | None = None
    group_id: int = 0


@dataclass(slots=True)
class TimingIR:
    tick: int
    bpm: float
    beats_per_measure: float
    group_id: int = 0


@dataclass(slots=True)
class NoteIRBase:
    note_id: int
    group_id: int = 0
    group_properties: dict[str, Any] = field(default_factory=dict)

    @property
    def noinput(self) -> bool:
        return bool(self.group_properties.get("noinput", False))


@dataclass(slots=True)
class TapIR(NoteIRBase):
    tick: int = 0
    lane: float = 0.0


@dataclass(slots=True)
class HoldIR(NoteIRBase):
    start: int = 0
    end: int = 0
    lane: float = 0.0


@dataclass(slots=True)
class ArcIR(NoteIRBase):
    start: int = 0
    end: int = 0
    start_x: float = 0.0
    end_x: float = 0.0
    easing: Easing = Easing.Linear
    start_y: float = 0.0
    end_y: float = 0.0
    color: int = 0
    hitsound: str | None = None
    trace_arc: bool | str = False
    taps: list[int] = field(default_factory=list)
    smoothness: float | None = None


ArcaeaNoteIR = TapIR | HoldIR | ArcIR


@dataclass(slots=True)
class ArcaeaChartIR:
    options: dict[str, Any] = field(default_factory=dict)
    notes: list[ArcaeaNoteIR] = field(default_factory=list)
    timings: list[TimingIR] = field(default_factory=list)
    scene_controls: list[SceneControlIR] = field(default_factory=list)

    def max_tick(self) -> int:
        max_time = 0
        for timing in self.timings:
            max_time = max(max_time, timing.tick)
        for note in self.notes:
            if isinstance(note, TapIR):
                max_time = max(max_time, note.tick)
            elif isinstance(note, HoldIR):
                max_time = max(max_time, note.end)
            elif isinstance(note, ArcIR):
                max_time = max(max_time, max(note.start, note.end))
                for tap_tick in note.taps:
                    max_time = max(max_time, tap_tick)
        for control in self.scene_controls:
            max_time = max(max_time, control.tick)
            if control.param1 is not None:
                max_time = max(max_time, control.tick + int(control.param1))
        return max_time
