from __future__ import annotations

from autoplay.domain.arcaea_ir import ArcIR, HoldIR, TapIR
from autoplay.domain.chart import Arc, ArcTap, Chart, Hold, Tap, Timing
from autoplay.domain.errors import MissingDesignantChoiceError
from autoplay.parser.aff_ir_parser import parse_aff_ir


def parse_aff_chart(content: str, designant_choice: bool | None = None) -> Chart:
    ir = parse_aff_ir(content, designant_choice=designant_choice)

    notes: list = []
    for timing in ir.timings:
        notes.append(Timing(timing.tick, timing.bpm, timing.beats_per_measure))

    for note in ir.notes:
        if isinstance(note, TapIR):
            notes.append(Tap(note.tick, int(round(note.lane))))
            continue
        if isinstance(note, HoldIR):
            notes.append(Hold(note.start, note.end, int(round(note.lane))))
            continue
        if isinstance(note, ArcIR):
            arc = Arc(
                note.start,
                note.end,
                note.start_x,
                note.end_x,
                note.easing,
                note.start_y,
                note.end_y,
                note.color,
                note.hitsound,
                note.trace_arc,
            )
            arc.taps = [ArcTap(tick) for tick in note.taps]
            notes.append(arc)

    chart = Chart(notes, ir.options)
    chart.ir = ir  # type: ignore[attr-defined]
    return chart


__all__ = ["MissingDesignantChoiceError", "parse_aff_chart"]
