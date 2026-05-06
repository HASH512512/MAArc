from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from autoplay.domain.chart import Chart
from autoplay.runtime.config_store import load_app_config
from autoplay.solver import CoordConv, solve_chart_auto


def main() -> None:
    chart_arg = sys.argv[1] if len(sys.argv) > 1 else "tests/samples/test_steganography.aff"
    chart_path = Path(chart_arg)
    if not chart_path.is_absolute():
        chart_path = PROJECT_ROOT / chart_path
    if not chart_path.is_file():
        raise SystemExit(f"Chart file not found: {chart_path}")

    app_config = load_app_config()
    cfg = app_config.global_config
    chart = Chart.loads(
        chart_path.read_text(encoding="utf-8"),
        designant_choice=cfg.designant_choice,
    )
    converter = CoordConv(cfg.bottom_left, cfg.top_left, cfg.top_right, cfg.bottom_right)
    events = solve_chart_auto(chart, converter)
    payload = {
        "chart_path": str(chart_path),
        "tick_count": len(events),
        "event_count": sum(len(items) for items in events.values()),
        "first_tick_ms": min(events) if events else None,
        "last_tick_ms": max(events) if events else None,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not events:
        raise SystemExit("Solver generated no events")


if __name__ == "__main__":
    main()
