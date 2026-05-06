from __future__ import annotations

from dataclasses import dataclass, field


SCHEMA_VERSION = 2


DEFAULT_BOTTOM_LEFT = (171, 1350)
DEFAULT_TOP_LEFT = (171, 300)
DEFAULT_TOP_RIGHT = (2376, 300)
DEFAULT_BOTTOM_RIGHT = (2376, 1350)


@dataclass(slots=True)
class GlobalConfig:
    bottom_left: tuple[int, int] = DEFAULT_BOTTOM_LEFT
    top_left: tuple[int, int] = DEFAULT_TOP_LEFT
    top_right: tuple[int, int] = DEFAULT_TOP_RIGHT
    bottom_right: tuple[int, int] = DEFAULT_BOTTOM_RIGHT
    chart_path: str = ""
    fine_tune_step: int = 10
    designant_choice: bool | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "GlobalConfig":
        return cls(
            bottom_left=tuple(data.get("bottom_left", DEFAULT_BOTTOM_LEFT)),
            top_left=tuple(data.get("top_left", DEFAULT_TOP_LEFT)),
            top_right=tuple(data.get("top_right", DEFAULT_TOP_RIGHT)),
            bottom_right=tuple(data.get("bottom_right", DEFAULT_BOTTOM_RIGHT)),
            chart_path=str(data.get("chart_path", "")),
            fine_tune_step=int(data.get("fine_tune_step", 10)),
            designant_choice=data.get("designant_choice"),
        )

    def to_dict(self) -> dict:
        return {
            "bottom_left": list(self.bottom_left),
            "top_left": list(self.top_left),
            "top_right": list(self.top_right),
            "bottom_right": list(self.bottom_right),
            "chart_path": self.chart_path,
            "fine_tune_step": self.fine_tune_step,
            "designant_choice": self.designant_choice,
        }


@dataclass(slots=True)
class VisionConfig:
    ui_left_roi: tuple[float, float, float, float] = (0.005, 0.02, 0.34, 0.20)
    ground_roi: tuple[float, float, float, float] = (
        0.12,
        1310 / 1440,
        0.88,
        1345 / 1440,
    )
    ui_template_threshold: float = 0.42
    ground_blue_ratio_threshold: float = 0.03
    arc_color_ratio_threshold: float = 0.02
    arc_logic_roi_half_x: float = 0.25
    arc_logic_roi_half_y: float = 0.25
    stream_max_fps: int = 60
    stream_bitrate_enabled: bool = False
    stream_bitrate_mbps: int = 8
    overlay_enabled: bool = False
    overlay_detached: bool = True
    perf_log_on_success_only: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "VisionConfig":
        return cls(
            ui_left_roi=tuple(data.get("ui_left_roi", (0.005, 0.02, 0.34, 0.20))),
            ground_roi=tuple(
                data.get("ground_roi", (0.12, 1310 / 1440, 0.88, 1345 / 1440))
            ),
            ui_template_threshold=float(data.get("ui_template_threshold", 0.42)),
            ground_blue_ratio_threshold=float(
                data.get(
                    "ground_blue_ratio_threshold",
                    data.get("ground_note_ratio", 0.03),
                )
            ),
            arc_color_ratio_threshold=float(
                data.get(
                    "arc_color_ratio_threshold",
                    data.get("arc_cap_threshold", 0.02),
                )
            ),
            arc_logic_roi_half_x=float(data.get("arc_logic_roi_half_x", 0.25)),
            arc_logic_roi_half_y=float(data.get("arc_logic_roi_half_y", 0.25)),
            stream_max_fps=int(data.get("stream_max_fps", 60)),
            stream_bitrate_enabled=bool(data.get("stream_bitrate_enabled", False)),
            stream_bitrate_mbps=int(data.get("stream_bitrate_mbps", 8)),
            overlay_enabled=bool(data.get("overlay_enabled", False)),
            overlay_detached=bool(data.get("overlay_detached", True)),
            perf_log_on_success_only=bool(data.get("perf_log_on_success_only", True)),
        )

    def to_dict(self) -> dict:
        return {
            "ui_left_roi": list(self.ui_left_roi),
            "ground_roi": list(self.ground_roi),
            "ui_template_threshold": self.ui_template_threshold,
            "ground_blue_ratio_threshold": self.ground_blue_ratio_threshold,
            "arc_color_ratio_threshold": self.arc_color_ratio_threshold,
            "arc_logic_roi_half_x": self.arc_logic_roi_half_x,
            "arc_logic_roi_half_y": self.arc_logic_roi_half_y,
            "stream_max_fps": self.stream_max_fps,
            "stream_bitrate_enabled": self.stream_bitrate_enabled,
            "stream_bitrate_mbps": self.stream_bitrate_mbps,
            "overlay_enabled": self.overlay_enabled,
            "overlay_detached": self.overlay_detached,
            "perf_log_on_success_only": self.perf_log_on_success_only,
        }


@dataclass(slots=True)
class AppConfig:
    global_config: GlobalConfig
    delay: float = 0.0
    vision: VisionConfig = field(default_factory=VisionConfig)

    @classmethod
    def default(cls) -> "AppConfig":
        return cls(global_config=GlobalConfig(), delay=0.0, vision=VisionConfig())

    @classmethod
    def from_dict(cls, data: dict) -> "AppConfig":
        if "global" in data or "delay" in data:
            global_data = data.get("global", {})
            delay = float(data.get("delay", 0.0))
            vision_data = data.get("vision", {})
        else:
            calibration = data.get("calibration", {})
            chart = data.get("chart", {})
            playback = data.get("playback", {})
            vision_root = data.get("vision", {})
            roi = vision_root.get("roi", {})
            stream = vision_root.get("stream", {})
            debug = vision_root.get("debug", {})

            global_data = {
                "bottom_left": calibration.get("bottom_left", DEFAULT_BOTTOM_LEFT),
                "top_left": calibration.get("top_left", DEFAULT_TOP_LEFT),
                "top_right": calibration.get("top_right", DEFAULT_TOP_RIGHT),
                "bottom_right": calibration.get("bottom_right", DEFAULT_BOTTOM_RIGHT),
                "chart_path": chart.get("path", ""),
                "fine_tune_step": playback.get("fine_tune_step", 10),
                "designant_choice": chart.get("designant_choice"),
            }
            delay = float(playback.get("delay", 0.0))
            vision_data = {
                "ui_left_roi": roi.get("ui_left", (0.005, 0.02, 0.34, 0.20)),
                "ground_roi": roi.get("ground", (0.12, 1310 / 1440, 0.88, 1345 / 1440)),
                "ui_template_threshold": vision_root.get("ui_template_threshold", 0.42),
                "ground_blue_ratio_threshold": vision_root.get(
                    "ground_blue_ratio_threshold", 0.03
                ),
                "arc_color_ratio_threshold": vision_root.get(
                    "arc_color_ratio_threshold", 0.02
                ),
                "arc_logic_roi_half_x": vision_root.get("arc_logic_roi_half_x", 0.25),
                "arc_logic_roi_half_y": vision_root.get("arc_logic_roi_half_y", 0.25),
                "stream_max_fps": stream.get("max_fps", 60),
                "stream_bitrate_enabled": stream.get("bitrate_enabled", False),
                "stream_bitrate_mbps": stream.get("bitrate_mbps", 8),
                "overlay_enabled": debug.get("overlay_enabled", False),
                "overlay_detached": debug.get("overlay_detached", True),
                "perf_log_on_success_only": debug.get("perf_log_on_success_only", True),
            }
        return cls(
            global_config=GlobalConfig.from_dict(global_data),
            delay=delay,
            vision=VisionConfig.from_dict(vision_data),
        )

    def to_dict(self) -> dict:
        global_cfg = self.global_config
        vision = self.vision
        return {
            "schema_version": SCHEMA_VERSION,
            "chart": {
                "path": global_cfg.chart_path,
                "designant_choice": global_cfg.designant_choice,
            },
            "playback": {
                "delay": self.delay,
                "fine_tune_step": global_cfg.fine_tune_step,
            },
            "calibration": {
                "bottom_left": list(global_cfg.bottom_left),
                "top_left": list(global_cfg.top_left),
                "top_right": list(global_cfg.top_right),
                "bottom_right": list(global_cfg.bottom_right),
            },
            "vision": {
                "ui_template_threshold": vision.ui_template_threshold,
                "ground_blue_ratio_threshold": vision.ground_blue_ratio_threshold,
                "arc_color_ratio_threshold": vision.arc_color_ratio_threshold,
                "arc_logic_roi_half_x": vision.arc_logic_roi_half_x,
                "arc_logic_roi_half_y": vision.arc_logic_roi_half_y,
                "roi": {
                    "ui_left": list(vision.ui_left_roi),
                    "ground": list(vision.ground_roi),
                },
                "stream": {
                    "max_fps": vision.stream_max_fps,
                    "bitrate_enabled": vision.stream_bitrate_enabled,
                    "bitrate_mbps": vision.stream_bitrate_mbps,
                },
                "debug": {
                    "overlay_enabled": vision.overlay_enabled,
                    "overlay_detached": vision.overlay_detached,
                    "perf_log_on_success_only": vision.perf_log_on_success_only,
                },
            },
        }
