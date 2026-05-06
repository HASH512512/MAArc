from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from easing import Easing


@dataclass(slots=True)
class ArcTap:
    tick: int

    def __str__(self) -> str:
        return f"arctap(tick={self.tick})"


@dataclass(slots=True)
class Arc:
    start: int
    end: int
    start_x: float
    end_x: float
    easing: Easing
    start_y: float
    end_y: float
    color: int
    unknown: Any
    trace_arc: bool | str
    taps: list[ArcTap] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.trace_arc, str):
            lower = self.trace_arc.lower()
            if lower == "true":
                self.trace_arc = True
            elif lower == "false":
                self.trace_arc = False
        else:
            self.trace_arc = bool(self.trace_arc)

    def __getitem__(self, taps: ArcTap | tuple[ArcTap, ...]) -> "Arc":
        if isinstance(taps, tuple):
            self.taps = list(taps)
        else:
            self.taps = [taps]
        return self

    def __str__(self) -> str:
        return (
            "arc(("
            f"{self.start_x:.02f}, {self.start_y:.02f})@{self.start} -> "
            f"({self.end_x:.02f}, {self.end_y:.02f})@{self.end} using {self.easing}, "
            f"color={self.color}, trace_arc={self.trace_arc}){self.taps}"
        )


@dataclass(slots=True)
class Tap:
    tick: int
    track: int

    def __str__(self) -> str:
        return f"tap(tick={self.tick}, track={self.track})"


@dataclass(slots=True)
class Hold:
    start: int
    end: int
    track: int

    def __str__(self) -> str:
        return f"hold(start={self.start}, end={self.end}, track={self.track})"


@dataclass(slots=True)
class Timing:
    tick: int
    bpm: float
    beats_per_measure: float

    def __str__(self) -> str:
        return f"timing(tick={self.tick}, bpm={self.bpm}, beats_per_measure={self.beats_per_measure})"


@dataclass(slots=True)
class TimingGroup:
    properties: dict[str, Any]
    notes: list["TimingGroup | Timing | Tap | Hold | Arc"]

    def __str__(self) -> str:
        return f"timinggroup({self.properties}, notes={self.notes})"


@dataclass(slots=True)
class Chart:
    notes: list[Timing | Tap | Hold | Arc | TimingGroup]
    options: dict[str, Any] | None = None
    ir: Any | None = None

    @classmethod
    def loads(cls, content: str, designant_choice: bool | None = None) -> "Chart":
        from autoplay.parser.aff_parser import parse_aff_chart

        return parse_aff_chart(content, designant_choice=designant_choice)
