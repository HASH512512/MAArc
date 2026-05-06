from __future__ import annotations

import re

from easing import Easing

from autoplay.domain.arcaea_ir import (
    ArcIR,
    ArcaeaChartIR,
    HoldIR,
    SceneControlIR,
    TapIR,
    TimingIR,
)
from autoplay.domain.errors import MissingDesignantChoiceError


TIMING_RE = re.compile(r"^timing\(([^)]*)\)$", flags=re.IGNORECASE)
TAP_RE = re.compile(r"^\(([^)]*)\)$")
HOLD_RE = re.compile(r"^hold\(([^)]*)\)$", flags=re.IGNORECASE)
SCENECONTROL_RE = re.compile(r"^scenecontrol\(([^)]*)\)$", flags=re.IGNORECASE)
ARCTAP_RE = re.compile(r"arctap\((\d+)\)", flags=re.IGNORECASE)


EASING_MAP: dict[str, Easing] = {
    "s": Easing.Linear,
    "b": Easing.CubicBezier,
    "si": Easing.Si,
    "so": Easing.So,
    "sisi": Easing.SiSi,
    "soso": Easing.SoSo,
    "siso": Easing.SiSo,
    "sosi": Easing.SoSi,
}


def _parse_group_properties(properties_str: str) -> dict[str, object]:
    properties: dict[str, object] = {}
    for item in properties_str.split("_"):
        token = item.strip()
        if not token:
            continue
        if token.startswith("anglex"):
            suffix = token[7:] if token.startswith("anglex=") else token[6:]
            try:
                properties["anglex"] = int(suffix)
            except ValueError:
                properties[token] = True
            continue
        if token.startswith("angley"):
            suffix = token[7:] if token.startswith("angley=") else token[6:]
            try:
                properties["angley"] = int(suffix)
            except ValueError:
                properties[token] = True
            continue
        properties[token] = True
    return properties


def _split_csv(raw: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for ch in raw:
        if ch == "[":
            depth += 1
            current.append(ch)
            continue
        if ch == "]":
            depth = max(0, depth - 1)
            current.append(ch)
            continue
        if ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(ch)
    parts.append("".join(current).strip())
    return parts


def parse_aff_ir(content: str, designant_choice: bool | None = None) -> ArcaeaChartIR:
    lines = content.splitlines()
    options: dict[str, str] = {}

    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line == "-":
            index += 1
            break
        if ":" not in line:
            break
        key, value = line.split(":", 1)
        options[key.strip()] = value.strip()
        index += 1

    ir = ArcaeaChartIR(options=options)
    note_id = 1

    group_stack: list[tuple[int, dict[str, object]]] = [(0, {})]
    next_group_id = 1

    for raw_line in lines[index:]:
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue

        if line.startswith("timinggroup"):
            attr_start = line.find("(")
            attr_end = line.rfind(")")
            properties_str = ""
            if attr_start != -1 and attr_end != -1 and attr_end > attr_start:
                properties_str = line[attr_start + 1 : attr_end]
            properties = _parse_group_properties(properties_str)
            group_stack.append((next_group_id, properties))
            next_group_id += 1
            continue

        if line.startswith("};") or line == "}":
            if len(group_stack) > 1:
                group_stack.pop()
            continue

        if line.endswith(";"):
            line = line[:-1].strip()

        group_id, group_properties = group_stack[-1]

        timing_match = TIMING_RE.match(line)
        if timing_match:
            parts = _split_csv(timing_match.group(1))
            if len(parts) >= 3:
                try:
                    ir.timings.append(
                        TimingIR(
                            tick=int(float(parts[0])),
                            bpm=float(parts[1]),
                            beats_per_measure=float(parts[2]),
                            group_id=group_id,
                        )
                    )
                except ValueError:
                    pass
            continue

        scene_match = SCENECONTROL_RE.match(line)
        if scene_match:
            parts = _split_csv(scene_match.group(1))
            if len(parts) >= 2:
                try:
                    tick = int(float(parts[0]))
                except ValueError:
                    continue
                control_type = parts[1].strip().lower()
                param1 = None
                param2 = None
                if len(parts) >= 4:
                    try:
                        param1 = float(parts[2])
                    except ValueError:
                        param1 = None
                    try:
                        param2 = int(float(parts[3]))
                    except ValueError:
                        param2 = None
                ir.scene_controls.append(
                    SceneControlIR(
                        tick=tick,
                        control_type=control_type,
                        param1=param1,
                        param2=param2,
                        group_id=group_id,
                    )
                )
            continue

        tap_match = TAP_RE.match(line)
        if tap_match:
            parts = _split_csv(tap_match.group(1))
            if len(parts) >= 2:
                try:
                    tick = int(float(parts[0]))
                    lane = float(parts[1])
                    ir.notes.append(
                        TapIR(
                            note_id=note_id,
                            group_id=group_id,
                            group_properties=dict(group_properties),
                            tick=tick,
                            lane=lane,
                        )
                    )
                    note_id += 1
                except ValueError:
                    pass
            continue

        hold_match = HOLD_RE.match(line)
        if hold_match:
            parts = _split_csv(hold_match.group(1))
            if len(parts) >= 3:
                try:
                    start = int(float(parts[0]))
                    end = int(float(parts[1]))
                    lane = float(parts[2])
                    ir.notes.append(
                        HoldIR(
                            note_id=note_id,
                            group_id=group_id,
                            group_properties=dict(group_properties),
                            start=start,
                            end=end,
                            lane=lane,
                        )
                    )
                    note_id += 1
                except ValueError:
                    pass
            continue

        if line.lower().startswith("arc("):
            arc_tap_ticks = [int(match.group(1)) for match in ARCTAP_RE.finditer(line)]

            arc_body = line
            bracket_index = arc_body.find("[")
            if bracket_index != -1:
                arc_body = arc_body[:bracket_index]
            if not arc_body.endswith(")"):
                continue
            inner = arc_body[4:-1]
            parts = _split_csv(inner)
            if len(parts) < 10:
                continue

            try:
                start = int(float(parts[0]))
                end = int(float(parts[1]))
                start_x = float(parts[2])
                end_x = float(parts[3])
                easing_key = parts[4].strip().lower()
                easing = EASING_MAP.get(easing_key, Easing.Linear)
                start_y = float(parts[5])
                end_y = float(parts[6])
                color = int(float(parts[7]))
                hitsound_raw = parts[8].strip()
                hitsound = None if hitsound_raw.lower() == "none" else hitsound_raw
                trace_token = parts[9].strip()

                if trace_token.lower() == "designant":
                    if designant_choice is None:
                        raise MissingDesignantChoiceError()
                    if not designant_choice:
                        continue
                    trace_arc: bool | str = True
                elif trace_token.lower() in {"true", "false"}:
                    trace_arc = trace_token.lower() == "true"
                else:
                    trace_arc = trace_token

                smoothness = None
                if len(parts) > 10:
                    try:
                        smoothness = float(parts[10])
                    except ValueError:
                        smoothness = None

                ir.notes.append(
                    ArcIR(
                        note_id=note_id,
                        group_id=group_id,
                        group_properties=dict(group_properties),
                        start=start,
                        end=end,
                        start_x=start_x,
                        end_x=end_x,
                        easing=easing,
                        start_y=start_y,
                        end_y=end_y,
                        color=color,
                        hitsound=hitsound,
                        trace_arc=trace_arc,
                        taps=arc_tap_ticks,
                        smoothness=smoothness,
                    )
                )
                note_id += 1
            except MissingDesignantChoiceError:
                raise
            except ValueError:
                continue
            continue

    ir.timings.sort(key=lambda item: item.tick)
    ir.scene_controls.sort(key=lambda item: item.tick)
    return ir


__all__ = ["parse_aff_ir"]
