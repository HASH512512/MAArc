from __future__ import annotations

import re


def has_designant_notes(content: str) -> bool:
    return bool(re.search(r"arc\([^)]*designant[^)]*\)", content, flags=re.IGNORECASE))


def extract_delay_from_aff_content(content: str) -> float | None:
    earliest_time: int | None = None

    for raw_line in content.splitlines():
        line = raw_line.strip()

        if line.startswith("(") and line.endswith(");"):
            parts = line[1:-2].split(",")
            if parts:
                try:
                    time_ms = int(parts[0])
                except (ValueError, IndexError):
                    continue
                earliest_time = time_ms if earliest_time is None else min(earliest_time, time_ms)
            continue

        if line.startswith("hold(") and line.endswith(");"):
            parts = line[5:-2].split(",")
            if parts:
                try:
                    time_ms = int(parts[0])
                except (ValueError, IndexError):
                    continue
                earliest_time = time_ms if earliest_time is None else min(earliest_time, time_ms)
            continue

        if line.startswith("arc(") and line.endswith(");"):
            parts = [part.strip() for part in line[4:-2].split(",")]
            if len(parts) >= 10 and parts[-1].lower() != "true":
                try:
                    time_ms = int(parts[0])
                except (ValueError, IndexError):
                    continue
                earliest_time = time_ms if earliest_time is None else min(earliest_time, time_ms)
            continue

        if "arctap(" in line:
            match = re.search(r"arctap\((\d+)\)", line)
            if not match:
                continue
            try:
                time_ms = int(match.group(1))
            except ValueError:
                continue
            earliest_time = time_ms if earliest_time is None else min(earliest_time, time_ms)

    if earliest_time is None:
        return None
    return -earliest_time / 1000
