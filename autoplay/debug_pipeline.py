from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from autoplay.analyzer.mode_analyzer import ArcaeaTimelineAnalyzer
from autoplay.parser import parse_aff_chart
from autoplay.solver import CoordConv, build_logical_events_for_chart, solve_chart_auto


def _to_serializable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _to_serializable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_serializable(v) for v in value]
    if isinstance(value, Enum):
        return value.name
    return value


def _build_debug_snapshot(
    chart_content: str,
    designant_choice: bool,
    converter_points: tuple[
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
    ] | None = None,
) -> dict[str, Any]:
    chart = parse_aff_chart(chart_content, designant_choice=designant_choice)
    chart_ir = chart.ir
    if chart_ir is None:
        return {"error": "chart.ir is None"}

    timeline = ArcaeaTimelineAnalyzer()
    timeline.build(chart_ir)

    max_tick = chart_ir.max_tick()
    lane_segments = [asdict(segment) for segment in timeline.lane_segments(max_tick)]
    sky_segments = [asdict(segment) for segment in timeline.sky_segments(max_tick)]

    if converter_points is None:
        converter_points = ((171, 1350), (171, 300), (2376, 300), (2376, 1350))
    converter = CoordConv(*converter_points)
    logical_events = build_logical_events_for_chart(chart)
    touch_events = solve_chart_auto(chart, converter)

    touch_events_flat: list[dict[str, Any]] = []
    for tick in sorted(touch_events.keys()):
        for event in touch_events[tick]:
            touch_events_flat.append(
                {
                    "tick": tick,
                    "pointer": event.pointer,
                    "action": event.action.name,
                    "position": list(event.position),
                    "source_note_id": event.source_note_id,
                    "source_type": event.source_type,
                    "logical_tick": event.logical_tick,
                    "logical_pos": (
                        list(event.logical_pos)
                        if event.logical_pos is not None
                        else None
                    ),
                }
            )

    return {
        "stats": {
            "timings": len(chart_ir.timings),
            "notes": len(chart_ir.notes),
            "scene_controls": len(chart_ir.scene_controls),
            "logical_events": len(logical_events),
            "touch_event_ticks": len(touch_events),
            "touch_events_total": len(touch_events_flat),
        },
        "notes": _to_serializable(chart.notes),
        "chart_ir": _to_serializable(chart_ir),
        "timeline": {
            "max_tick": max_tick,
            "lane_segments": lane_segments,
            "sky_segments": sky_segments,
        },
        "logical_events": _to_serializable(logical_events),
        "touch_events": touch_events_flat,
    }


def _write_markdown_report(data: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# Autoplay Debug Report")
    lines.append("")
    lines.append("## Stats")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(data.get("stats", {}), ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(data.get("notes", []), ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    lines.append("## Chart IR")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(data.get("chart_ir", {}), ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    lines.append("## Timeline")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(data.get("timeline", {}), ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    lines.append("## Logical Events")
    lines.append("")
    lines.append("```json")
    lines.append(
        json.dumps(data.get("logical_events", []), ensure_ascii=False, indent=2)
    )
    lines.append("```")
    lines.append("")

    lines.append("## Touch Events")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(data.get("touch_events", []), ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def generate_debug_artifacts(
    chart_path: Path,
    designant_choice: bool,
    out_json: Path,
    out_md: Path,
    converter_points: tuple[
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
    ] | None = None,
) -> dict[str, Any]:
    if not chart_path.exists():
        raise FileNotFoundError(f"Chart file not found: {chart_path}")

    content = chart_path.read_text(encoding="utf-8")
    snapshot = _build_debug_snapshot(
        content,
        designant_choice=designant_choice,
        converter_points=converter_points,
    )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _write_markdown_report(snapshot, out_md)
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate autoplay pipeline debug artifacts without ADB/scrcpy."
    )
    parser.add_argument(
        "--chart",
        default="tests/samples/basic_6k_scenecontrol.aff",
        help="Path to input AFF chart file",
    )
    parser.add_argument(
        "--designant",
        choices=["true", "false"],
        default="true",
        help="Whether designant notes are enabled",
    )
    parser.add_argument(
        "--out-json",
        default="debug/debug_pipeline_snapshot.json",
        help="Path to output JSON snapshot file",
    )
    parser.add_argument(
        "--out-md",
        default="debug/debug_pipeline_report.md",
        help="Path to output markdown report file",
    )
    args = parser.parse_args()

    chart_path = Path(args.chart)
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)

    generate_debug_artifacts(
        chart_path=chart_path,
        designant_choice=(args.designant == "true"),
        out_json=out_json,
        out_md=out_md,
    )

    print(f"Debug JSON written: {out_json}")
    print(f"Debug Markdown written: {out_md}")


if __name__ == "__main__":
    main()
