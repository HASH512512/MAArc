from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from PySide6.QtCore import QThread, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QScrollArea,
    QTabWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from algo.algo_base import TouchAction
from autoplay.analyzer import ModeAnalyzer
from autoplay.domain.arcaea_ir import ArcIR, HoldIR, TapIR
from autoplay.parser import (
    extract_delay_from_aff_content,
    has_designant_notes,
    parse_aff_chart,
)
from autoplay.runtime import (
    load_app_config,
    prepare_device_controller,
    run_touch_events,
    save_app_config,
)
from autoplay.runtime.player import FineTuneState
from autoplay.solver import CoordConv, build_logical_events_for_chart, solve_chart_auto
from autoplay.vision import VisionDetector, VisionRuntimeConfig


LEFT_MIN_WIDTH = 460
RIGHT_MIN_WIDTH = 440
WINDOW_MIN_WIDTH = LEFT_MIN_WIDTH + RIGHT_MIN_WIDTH
WINDOW_MIN_HEIGHT = 600
REPO_ROOT = Path(__file__).resolve().parents[2]
REF_OPENCV_DIR = REPO_ROOT / "ref" / "opencv"
DEBUG_DIR = REPO_ROOT / "debug"


def _to_serializable(value):
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
    if hasattr(value, "__dict__"):
        return {
            key: _to_serializable(val)
            for key, val in vars(value).items()
            if not key.startswith("_")
        }
    return value


def _read_image_unicode(path: Path) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


TEXT = {
    "zh": {
        "window_title": "Arcaea Auto Play GUI",
        "language": "语言",
        "tab_status": "状态",
        "tab_settings": "设置",
        "chart_path": "谱面路径",
        "browse": "浏览",
        "top_left": "左上坐标",
        "top_right": "右上坐标",
        "bottom_left": "左下坐标",
        "bottom_right": "右下坐标",
        "fine_tune_step": "微调步长",
        "designant": "蚂蚁异象触控",
        "unset": "未设置",
        "enabled": "启用",
        "disabled": "禁用",
        "save_config": "保存配置",
        "reload_config": "重载配置",
        "prepare": "预编译谱面",
        "start": "开始",
        "stop": "停止",
        "step_plus": "+步进 (Z)",
        "step_minus": "-步进 (X)",
        "reset": "重置 (R)",
        "run_state": "运行状态",
        "controller": "控制通道",
        "prepare_state": "谱面准备",
        "offset": "微调偏移",
        "delay": "延迟",
        "idle": "空闲",
        "warming": "预热中",
        "ready": "就绪",
        "running": "运行中",
        "error": "错误",
        "summary_current": "当前执行",
        "summary_next": "下一执行",
        "none": "无",
        "select_aff": "选择 AFF 谱面文件",
        "config_error": "配置错误",
        "missing_chart_title": "缺少谱面",
        "missing_chart": "请先选择谱面文件",
        "chart_not_found": "谱面文件不存在: {path}",
        "read_error": "读取谱面失败: {error}",
        "designant_title": "蚂蚁异象",
        "designant_question": "当前谱面包含蚂蚁异象 note，是否启用触控？",
        "controller_not_ready": "控制通道尚未就绪，请等待预热完成。",
        "prepare_not_ready": "谱面尚未准备完成，请先点击“预编译谱面”或等待自动准备。",
        "prepare_mismatch": "设置已变更，正在重新预编译。请等待准备完成后再开始。",
        "log_warm_start": "[信息] 正在预热 ADB/scrcpy 通道...",
        "log_warm_ok": "[信息] ADB/scrcpy 预热完成",
        "log_warm_fail": "[错误] ADB/scrcpy 预热失败: {error}",
        "log_prepare_start": "[信息] 开始预编译谱面...",
        "log_prepare_ok": "[信息] 预编译完成，事件组数={count}，delay={delay:.3f}s",
        "log_prepare_fail": "[错误] 预编译失败: {error}",
        "log_config_saved": "[信息] 配置已保存",
        "log_play_start": "[信息] 播放任务已启动",
        "log_play_finish": "[信息] 播放完成",
        "log_stop": "[信息] 已请求停止",
        "detail_curr_note": "当前 Note 详情",
        "detail_curr_event": "当前 TouchEvent 详情",
        "detail_next_note": "下一 Note 详情",
        "detail_next_event": "下一 TouchEvent 详情",
        "log_clear": "清除日志",
        "log_limit": "日志最大条数",
        "debug_verbose": "调试模式",
        "debug_verbose_hint": "开启后会输出详细调度日志，并在每次开始自动输入时将触摸事件快照写入 debug 文件夹。",
        "opt_high_prio": "播放线程高优先级",
        "auto_start_cv": "视觉自动开始（单次）",
        "auto_start_hint": "勾选后仅生效一次：视觉触发完成或手动停止后会自动关闭。",
        "auto_start_wait": "[信息] 视觉自动开始：等待游戏界面...",
        "auto_start_ready": "[信息] 已检测到游戏界面，等待首 note 触发区域命中。",
        "auto_start_fire": "[信息] 自动开始条件满足，已触发播放",
        "auto_start_off": "[信息] 视觉自动开始已关闭（单次已完成或已中断）",
        "stream_fps": "流 FPS 上限",
        "stream_bitrate_enable": "启用码率限制",
        "stream_bitrate": "码率上限 (Mbps)",
        "vision_overlay": "Vision Overlay",
        "vision_perf_success_only": "仅成功时输出性能日志",
        "vision_debug_group": "视觉识别 Debug",
        "vision_debug_source": "输入源",
        "vision_debug_input": "图片/视频路径",
        "vision_debug_browse": "浏览",
        "vision_debug_stages": "识别步骤",
        "vision_debug_stage_ui": "ui_stage",
        "vision_debug_stage_ground": "ground",
        "vision_debug_stage_arc": "arc_pass",
        "vision_debug_logic_x": "ground/arc logic_x",
        "vision_debug_logic_y": "arc logic_y",
        "vision_debug_start": "启动视觉 Debug",
        "vision_debug_stop": "停止视觉 Debug",
        "vision_debug_hint": "只做识别与指标输出，不触发自动输入。图片/视频会持续循环。",
        "vision_debug_idle": "Vision Debug: idle",
        "vision_debug_running": "Vision Debug: running ({source})",
        "vision_debug_no_stage": "[VISION-DEBUG] 至少选择一个识别步骤（ui_stage/ground/arc_pass）",
        "vision_debug_source_not_ready": "[VISION-DEBUG] scrcpy 输入源不可用：控制通道未就绪",
        "vision_debug_invalid_path": "[VISION-DEBUG] 输入路径无效: {path}",
        "vision_debug_open_fail": "[VISION-DEBUG] 无法打开输入源: {path}",
        "vision_debug_started": "[VISION-DEBUG] 已启动，source={source}",
        "vision_debug_stopped": "[VISION-DEBUG] 已停止",
        "ground_blue_ratio_threshold": "地键蓝色占比阈值",
        "arc_color_ratio_threshold": "天键颜色占比阈值",
        "arc_logic_roi_half_x": "天键逻辑ROI半宽",
        "arc_logic_roi_half_y": "天键逻辑ROI半高",
    },
    "en": {
        "window_title": "Arcaea Auto Play GUI",
        "language": "Language",
        "tab_status": "Status",
        "tab_settings": "Settings",
        "chart_path": "Chart Path",
        "browse": "Browse",
        "top_left": "top_left",
        "top_right": "top_right",
        "bottom_left": "bottom_left",
        "bottom_right": "bottom_right",
        "fine_tune_step": "Fine-tune Step",
        "designant": "Designant Touch",
        "unset": "Unset",
        "enabled": "Enabled",
        "disabled": "Disabled",
        "save_config": "Save Config",
        "reload_config": "Reload Config",
        "prepare": "Prepare Chart",
        "start": "Start",
        "stop": "Stop",
        "step_plus": "+step (Z)",
        "step_minus": "-step (X)",
        "reset": "Reset (R)",
        "run_state": "Run State",
        "controller": "Controller",
        "prepare_state": "Chart Ready",
        "offset": "Fine Offset",
        "delay": "Delay",
        "idle": "Idle",
        "warming": "Warming",
        "ready": "Ready",
        "running": "Running",
        "error": "Error",
        "summary_current": "Current",
        "summary_next": "Next",
        "none": "None",
        "select_aff": "Select AFF Chart File",
        "config_error": "Config Error",
        "missing_chart_title": "Missing Chart",
        "missing_chart": "Please select a chart file first",
        "chart_not_found": "Chart file not found: {path}",
        "read_error": "Failed to read chart: {error}",
        "designant_title": "Designant",
        "designant_question": "Current chart contains designant notes. Enable designant touch?",
        "controller_not_ready": "Controller is not ready yet. Wait until warmup is complete.",
        "prepare_not_ready": "Chart data is not prepared yet. Click Prepare Chart first or wait for auto preparation.",
        "prepare_mismatch": "Settings changed, re-preparing chart now. Start again after preparation completes.",
        "log_warm_start": "[INFO] Preheating ADB/scrcpy channel...",
        "log_warm_ok": "[INFO] ADB/scrcpy preheat completed",
        "log_warm_fail": "[ERROR] ADB/scrcpy preheat failed: {error}",
        "log_prepare_start": "[INFO] Preparing chart pipeline...",
        "log_prepare_ok": "[INFO] Prepare completed, groups={count}, delay={delay:.3f}s",
        "log_prepare_fail": "[ERROR] Prepare failed: {error}",
        "log_config_saved": "[INFO] Config saved",
        "log_play_start": "[INFO] Playback task started",
        "log_play_finish": "[INFO] Playback finished",
        "log_stop": "[INFO] Stop requested",
        "detail_curr_note": "Current Note Detail",
        "detail_curr_event": "Current TouchEvent Detail",
        "detail_next_note": "Next Note Detail",
        "detail_next_event": "Next TouchEvent Detail",
        "log_clear": "Clear Logs",
        "log_limit": "Log Max Entries",
        "debug_verbose": "Debug mode",
        "debug_verbose_hint": "When enabled, emits verbose scheduler logs and writes a touch-event snapshot to the debug folder on each playback start.",
        "opt_high_prio": "High thread priority",
        "auto_start_cv": "Vision auto start (one-shot)",
        "auto_start_hint": "Arms once only. It auto-disables after trigger fire or manual stop.",
        "auto_start_wait": "[INFO] Vision auto-start: waiting gameplay screen...",
        "auto_start_ready": "[INFO] Gameplay screen detected, waiting first-note trigger area...",
        "auto_start_fire": "[INFO] Auto-start condition met, playback triggered",
        "auto_start_off": "[INFO] Vision auto-start disabled (one-shot completed or interrupted)",
        "stream_fps": "Stream FPS Limit",
        "stream_bitrate_enable": "Enable bitrate limit",
        "stream_bitrate": "Bitrate limit (Mbps)",
        "vision_overlay": "Vision Overlay",
        "vision_perf_success_only": "Log perf only on success",
        "vision_debug_group": "Vision Debug",
        "vision_debug_source": "Input Source",
        "vision_debug_input": "Image/Video Path",
        "vision_debug_browse": "Browse",
        "vision_debug_stages": "Detection Stages",
        "vision_debug_stage_ui": "ui_stage",
        "vision_debug_stage_ground": "ground",
        "vision_debug_stage_arc": "arc_pass",
        "vision_debug_logic_x": "ground/arc logic_x",
        "vision_debug_logic_y": "arc logic_y",
        "vision_debug_start": "Start Vision Debug",
        "vision_debug_stop": "Stop Vision Debug",
        "vision_debug_hint": "Recognition only. No automation input is started. Image/video sources keep looping.",
        "vision_debug_idle": "Vision Debug: idle",
        "vision_debug_running": "Vision Debug: running ({source})",
        "vision_debug_no_stage": "[VISION-DEBUG] Select at least one stage (ui_stage/ground/arc_pass)",
        "vision_debug_source_not_ready": "[VISION-DEBUG] scrcpy source unavailable: controller is not ready",
        "vision_debug_invalid_path": "[VISION-DEBUG] Invalid input path: {path}",
        "vision_debug_open_fail": "[VISION-DEBUG] Failed to open input source: {path}",
        "vision_debug_started": "[VISION-DEBUG] Started, source={source}",
        "vision_debug_stopped": "[VISION-DEBUG] Stopped",
        "ground_blue_ratio_threshold": "Ground blue ratio threshold",
        "arc_color_ratio_threshold": "Arc color ratio threshold",
        "arc_logic_roi_half_x": "Arc logic ROI half width",
        "arc_logic_roi_half_y": "Arc logic ROI half height",
    },
}


@dataclass(slots=True)
class RunConfig:
    chart_path: str
    bottom_left: tuple[int, int]
    top_left: tuple[int, int]
    top_right: tuple[int, int]
    bottom_right: tuple[int, int]
    fine_tune_step: int
    designant_choice: bool | None


@dataclass(slots=True)
class PreparedRunData:
    config_key: str
    run_config: RunConfig
    delay: float
    events_by_time: dict[int, list]
    note_meta: dict[int, dict[str, object]]
    first_ground_tick: int | None
    first_ground_logic_x: float | None
    first_ground_logic_pos: tuple[float, float] | None
    first_note_types: tuple[str, ...]
    first_note_logic_pos: tuple[float, float] | None


class StartupPipelineState(Enum):
    IDLE = "idle"
    WARMUP = "warmup"
    PREPARE = "prepare"
    READY = "ready"
    ERROR = "error"


def _coord_to_text(coord: tuple[int, int]) -> str:
    return f"{coord[0]},{coord[1]}"


def _parse_coord(text: str, label: str) -> tuple[int, int]:
    parts = text.strip().replace("，", ",").split(",")
    if len(parts) != 2:
        raise ValueError(f"{label} format must be x,y")
    return int(parts[0].strip()), int(parts[1].strip())


def _build_config_key(cfg: RunConfig) -> str:
    return "|".join(
        [
            cfg.chart_path,
            str(cfg.top_left),
            str(cfg.top_right),
            str(cfg.bottom_left),
            str(cfg.bottom_right),
            str(cfg.designant_choice),
        ]
    )


def _build_note_meta(chart) -> dict[int, dict[str, object]]:
    result: dict[int, dict[str, object]] = {}
    chart_ir = chart.ir
    if chart_ir is None:
        return result

    logical_events = build_logical_events_for_chart(chart)
    logical_by_note: dict[int, list] = {}
    for event in logical_events:
        logical_by_note.setdefault(event.source_note_id, []).append(event)

    def _logical_span(
        note_id: int,
    ) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
        events = logical_by_note.get(note_id, [])
        if not events:
            return None, None
        ordered = sorted(
            events, key=lambda item: (item.tick, item.action.value, item.pointer)
        )
        start = (float(ordered[0].x), float(ordered[0].y))
        end = (float(ordered[-1].x), float(ordered[-1].y))
        return start, end

    for note in chart_ir.notes:
        if isinstance(note, TapIR):
            logical_start, logical_end = _logical_span(note.note_id)
            result[note.note_id] = {
                "type": "tap",
                "tick": note.tick,
                "start": logical_start
                if logical_start is not None
                else (note.lane, 0.0),
                "end": logical_end if logical_end is not None else (note.lane, 0.0),
                "raw_start": (note.lane, 0.0),
                "raw_end": (note.lane, 0.0),
            }
        elif isinstance(note, HoldIR):
            logical_start, logical_end = _logical_span(note.note_id)
            result[note.note_id] = {
                "type": "hold",
                "tick": note.start,
                "start": logical_start
                if logical_start is not None
                else (note.lane, 0.0),
                "end": logical_end if logical_end is not None else (note.lane, 0.0),
                "raw_start": (note.lane, 0.0),
                "raw_end": (note.lane, 0.0),
                "end_tick": note.end,
            }
        elif isinstance(note, ArcIR):
            logical_start, logical_end = _logical_span(note.note_id)
            result[note.note_id] = {
                "type": "arc" if not note.trace_arc else "trace_arc",
                "tick": note.start,
                "start": logical_start
                if logical_start is not None
                else (note.start_x, note.start_y),
                "end": logical_end
                if logical_end is not None
                else (note.end_x, note.end_y),
                "raw_start": (note.start_x, note.start_y),
                "raw_end": (note.end_x, note.end_y),
                "end_tick": note.end,
            }
    return result


class CollapsibleSection(QWidget):
    def __init__(self, title: str) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.button = QToolButton()
        self.button.setText(title)
        self.button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.button.setArrowType(Qt.RightArrow)
        self.button.setCheckable(True)
        self.button.toggled.connect(self._toggle)

        self.content = QTextEdit()
        self.content.setReadOnly(True)
        self.content.setVisible(False)
        self.content.setMinimumHeight(88)

        layout.addWidget(self.button)
        layout.addWidget(self.content)

    def _toggle(self, checked: bool) -> None:
        self.button.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self.content.setVisible(checked)

    def set_title(self, title: str) -> None:
        self.button.setText(title)

    def set_text(self, text: str) -> None:
        self.content.setPlainText(text)


class ControllerWarmupWorker(QThread):
    started_warmup = Signal()
    warmup_ok = Signal(object)
    warmup_fail = Signal(str)

    def __init__(
        self,
        max_fps: int,
        video_bit_rate: int | None,
        video_crop: tuple[int, int, int, int] | None = None,
    ) -> None:
        super().__init__()
        self.max_fps = max_fps
        self.video_bit_rate = video_bit_rate
        self.video_crop = video_crop

    def run(self) -> None:
        self.started_warmup.emit()
        try:
            controller = prepare_device_controller(
                max_fps=self.max_fps,
                video_bit_rate=self.video_bit_rate,
                video_crop=self.video_crop,
            )
        except Exception as exc:
            self.warmup_fail.emit(str(exc))
            return
        self.warmup_ok.emit(controller)


class PrepareWorker(QThread):
    started_prepare = Signal()
    prepared_ok = Signal(object)
    prepared_fail = Signal(str)

    def __init__(self, run_config: RunConfig) -> None:
        super().__init__()
        self.run_config = run_config

    def run(self) -> None:
        self.started_prepare.emit()
        cfg = self.run_config
        try:
            chart_content = Path(cfg.chart_path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            self.prepared_fail.emit(f"read_error:{exc}")
            return

        try:
            chart = parse_aff_chart(
                chart_content, designant_choice=cfg.designant_choice
            )
        except Exception as exc:
            self.prepared_fail.emit(f"parse_error:{exc}")
            return

        analyzer = ModeAnalyzer()
        analyzer.analyze_chart_for_6k(chart_content, chart)

        delay = extract_delay_from_aff_content(chart_content)
        if delay is None:
            self.prepared_fail.emit("delay_error")
            return

        converter = CoordConv(
            cfg.bottom_left, cfg.top_left, cfg.top_right, cfg.bottom_right
        )
        events = solve_chart_auto(chart, converter)
        if not events:
            self.prepared_fail.emit("event_error")
            return

        first_ground_tick: int | None = None
        first_ground_logic_x: float | None = None
        first_ground_logic_pos: tuple[float, float] | None = None
        first_tick = min(events.keys())
        first_types = {str(event.source_type) for event in events[first_tick]}
        first_note_logic_pos: tuple[float, float] | None = None
        for event in events[first_tick]:
            logical_pos = getattr(event, "logical_pos", None)
            if (
                isinstance(logical_pos, tuple)
                and len(logical_pos) == 2
                and all(isinstance(v, (int, float)) for v in logical_pos)
            ):
                first_note_logic_pos = (float(logical_pos[0]), float(logical_pos[1]))
                break
        for tick in sorted(events.keys()):
            for event in events[tick]:
                if event.source_type in {"tap", "hold"}:
                    first_ground_tick = tick
                    logical_pos = getattr(event, "logical_pos", None)
                    if (
                        isinstance(logical_pos, tuple)
                        and len(logical_pos) == 2
                        and isinstance(logical_pos[0], (int, float))
                    ):
                        first_ground_logic_x = float(logical_pos[0])
                        if isinstance(logical_pos[1], (int, float)):
                            first_ground_logic_pos = (
                                float(logical_pos[0]),
                                float(logical_pos[1]),
                            )
                    break
            if first_ground_tick is not None:
                break

        payload = PreparedRunData(
            config_key=_build_config_key(cfg),
            run_config=cfg,
            delay=delay,
            events_by_time=events,
            note_meta=_build_note_meta(chart),
            first_ground_tick=first_ground_tick,
            first_ground_logic_x=first_ground_logic_x,
            first_ground_logic_pos=first_ground_logic_pos,
            first_note_types=tuple(sorted(first_types)),
            first_note_logic_pos=first_note_logic_pos,
        )
        self.prepared_ok.emit(payload)


class PlaybackWorker(QThread):
    log_message = Signal(str)
    started_playback = Signal()
    finished_playback = Signal(bool, str)
    progress = Signal(object)
    first_dispatch_metrics = Signal(object)

    def __init__(
        self,
        prepared: PreparedRunData,
        controller,
        debug_verbose: bool,
        optimize_high_priority: bool,
        optimize_timer_resolution: bool,
        start_signal: threading.Event | None = None,
    ) -> None:
        super().__init__()
        self.prepared = prepared
        self.controller = controller
        self.debug_verbose = debug_verbose
        self.optimize_high_priority = optimize_high_priority
        self.optimize_timer_resolution = optimize_timer_resolution
        self.start_signal = start_signal
        self.state: FineTuneState | None = None

    def current_offset(self) -> float:
        if self.state is None:
            return 0.0
        return self.state.current_offset()

    def nudge_plus(self) -> None:
        if self.state is None:
            return
        offset = self.state.increment()
        self.log_message.emit(
            f"[Fine-tune] Advance {self.prepared.run_config.fine_tune_step}ms, current offset: {offset:.3f}s"
        )

    def nudge_minus(self) -> None:
        if self.state is None:
            return
        offset = self.state.decrement()
        self.log_message.emit(
            f"[Fine-tune] Delay {self.prepared.run_config.fine_tune_step}ms, current offset: {offset:.3f}s"
        )

    def reset_offset(self) -> None:
        if self.state is None:
            return
        offset = self.state.reset()
        self.log_message.emit(f"[Fine-tune] Offset reset: {offset:.3f}s")

    def stop_playback(self) -> None:
        if self.state is not None:
            self.state.input_listener_active = False

    def _on_progress(
        self,
        curr_tick: int,
        curr_events: list,
        next_tick: int | None,
        next_events: list | None,
    ) -> None:
        curr_event = curr_events[0] if curr_events else None
        next_event = next_events[0] if next_events else None
        self.progress.emit(
            {
                "curr_tick": curr_tick,
                "curr_event": curr_event,
                "curr_events": list(curr_events),
                "next_tick": next_tick,
                "next_event": next_event,
                "next_events": list(next_events or []),
                "curr_size": len(curr_events),
                "next_size": len(next_events or []),
                "note_meta": self.prepared.note_meta,
            }
        )

    def _on_first_dispatch_metrics(self, payload: dict[str, float]) -> None:
        self.first_dispatch_metrics.emit(payload)

    def run(self) -> None:
        self.state = FineTuneState(self.prepared.run_config.fine_tune_step)
        self.state.input_listener_active = True
        self.started_playback.emit()

        run_touch_events(
            self.prepared.events_by_time,
            self.prepared.delay,
            self.state,
            controller=self.controller,
            log=self.log_message.emit,
            on_progress=self._on_progress,
            on_first_dispatch=self._on_first_dispatch_metrics,
            start_signal=self.start_signal,
            debug=self.debug_verbose,
            optimize_high_priority=self.optimize_high_priority,
            optimize_timer_resolution=self.optimize_timer_resolution,
        )
        self.finished_playback.emit(True, "ok")


class AutoPlayWindow(QMainWindow):
    auto_start_frame_ready = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.locale = "zh"
        self.app_config = load_app_config()

        self.controller = None
        self.controller_ready = False
        self.vision_controller = None
        self.vision_controller_ready = False
        self.prepared: PreparedRunData | None = None

        self.worker: PlaybackWorker | None = None
        self.prepare_worker: PrepareWorker | None = None
        self.warmup_worker: ControllerWarmupWorker | None = None
        self.vision_warmup_worker: ControllerWarmupWorker | None = None

        self.log_lines: list[str] = []
        self.log_limit = 500
        self._start_click_time: float | None = None
        self._first_dispatch_logged = False
        self._auto_start_detection_started_at: float | None = None
        self._auto_start_triggered_at: float | None = None
        self._auto_start_frame_timestamp: float | None = None
        self._auto_start_vision_total_ms: float = 0.0
        self._vision_perf_success_only = True
        self._auto_start_last_frame_seq = -1
        self._vision_last_frame_seq = -1
        self._auto_start_playback_armed = False
        self._auto_start_start_signal: threading.Event | None = None
        self._auto_start_shutdown_logged = False
        self._vision_debug_running = False
        self._vision_debug_capture: cv2.VideoCapture | None = None
        self._vision_debug_static_frame: np.ndarray | None = None
        self._vision_debug_last_result = ""
        self._vision_debug_last_log_time = 0.0
        self._vision_debug_overlay_stage = "debug-idle"
        self.vision_detector = VisionDetector(REF_OPENCV_DIR, use_cuda=True)
        self.stream_max_fps = 60
        self.stream_bitrate_enabled = False
        self.stream_bitrate_mbps = 8
        self._native_device_size: tuple[int, int] | None = None
        self._startup_state = StartupPipelineState.IDLE
        self._prepare_requested_after_warmup = False
        self._warmup_restart_requested = False
        self._queued_pipeline_reason: str | None = None
        self._queued_pipeline_prepare_after_warmup = False
        self._queued_pipeline_force_touch_restart = False
        self._active_touch_points: dict[int, tuple[int, int]] = {}
        self._manual_stop_release_points: dict[int, tuple[int, int]] = {}
        self._manual_stop_release_pending = False

        self._build_ui()
        self._load_config_to_form()
        self._apply_texts()
        self._report_missing_vision_templates()
        self.auto_start_frame_ready.connect(self._poll_auto_start)

        self.offset_timer = QTimer(self)
        self.offset_timer.setInterval(80)
        self.offset_timer.timeout.connect(self._refresh_offset)

        self._auto_start_stage = "idle"

        self.overlay_timer = QTimer(self)
        self.overlay_timer.setInterval(90)
        self.overlay_timer.timeout.connect(self._poll_overlay)
        self.overlay_timer.start()

        self._start_startup_pipeline(
            "startup",
            prepare_after_warmup=True,
            force_touch_restart=True,
        )

    def _t(self, key: str, **kwargs) -> str:
        return TEXT[self.locale][key].format(**kwargs)

    def _build_ui(self) -> None:
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        self.resize(1360, 840)

        root = QWidget(self)
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)

        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter)

        left = QWidget()
        left.setMinimumWidth(LEFT_MIN_WIDTH)
        left_layout = QVBoxLayout(left)

        top_row = QHBoxLayout()
        self.language_label = QLabel()
        self.language_combo = QComboBox()
        self.language_combo.addItem("中文", "zh")
        self.language_combo.addItem("English", "en")
        self.language_combo.currentIndexChanged.connect(self._on_language_changed)
        top_row.addStretch()
        top_row.addWidget(self.language_label)
        top_row.addWidget(self.language_combo)
        left_layout.addLayout(top_row)

        self.tabs = QTabWidget()
        self.status_page = QWidget()
        self.settings_page = QWidget()
        self.tabs.addTab(self.status_page, "")
        self.tabs.addTab(self.settings_page, "")
        left_layout.addWidget(self.tabs)

        self._build_status_page()
        self._build_settings_page()

        self.control_group = QGroupBox()
        control_layout = QGridLayout(self.control_group)
        self.start_btn = QPushButton()
        self.stop_btn = QPushButton()
        self.plus_btn = QPushButton()
        self.minus_btn = QPushButton()
        self.reset_btn = QPushButton()

        self.plus_btn.setShortcut("Z")
        self.minus_btn.setShortcut("X")
        self.reset_btn.setShortcut("R")

        self.start_btn.clicked.connect(self._start_playback)
        self.stop_btn.clicked.connect(self._stop_playback)
        self.plus_btn.clicked.connect(self._fine_tune_plus)
        self.minus_btn.clicked.connect(self._fine_tune_minus)
        self.reset_btn.clicked.connect(self._fine_tune_reset)

        control_layout.addWidget(self.start_btn, 0, 0)
        control_layout.addWidget(self.stop_btn, 0, 1)
        control_layout.addWidget(self.plus_btn, 1, 0)
        control_layout.addWidget(self.minus_btn, 1, 1)
        control_layout.addWidget(self.reset_btn, 1, 2)
        left_layout.addWidget(self.control_group)

        right = QWidget()
        right.setMinimumWidth(RIGHT_MIN_WIDTH)
        right_layout = QVBoxLayout(right)
        log_top = QHBoxLayout()
        log_top.addStretch()
        self.clear_log_btn = QPushButton()
        self.clear_log_btn.clicked.connect(self._clear_logs)
        log_top.addWidget(self.clear_log_btn)
        right_layout.addLayout(log_top)
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        right_layout.addWidget(self.log_output)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        self._set_running_ui(False)

    def _build_status_page(self) -> None:
        layout = QVBoxLayout(self.status_page)

        summary_group = QGroupBox()
        summary_form = QFormLayout(summary_group)
        self.run_state_key = QLabel()
        self.controller_key = QLabel()
        self.prepare_key = QLabel()
        self.offset_key = QLabel()
        self.delay_key = QLabel()

        self.run_state_label = QLabel()
        self.controller_state_label = QLabel()
        self.prepare_state_label = QLabel()
        self.offset_label = QLabel("0.000s")
        self.delay_label = QLabel(f"{self.app_config.delay:.3f}s")
        summary_form.addRow(self.run_state_key, self.run_state_label)
        summary_form.addRow(self.controller_key, self.controller_state_label)
        summary_form.addRow(self.prepare_key, self.prepare_state_label)
        summary_form.addRow(self.offset_key, self.offset_label)
        summary_form.addRow(self.delay_key, self.delay_label)

        monitor_group = QGroupBox()
        monitor_layout = QVBoxLayout(monitor_group)
        self.curr_title = QLabel()
        self.curr_line1 = QLabel("-")
        self.curr_line2 = QLabel("-")
        self.next_title = QLabel()
        self.next_line1 = QLabel("-")
        self.next_line2 = QLabel("-")

        monitor_layout.addWidget(self.curr_title)
        monitor_layout.addWidget(self.curr_line1)
        monitor_layout.addWidget(self.curr_line2)
        monitor_layout.addSpacing(8)
        monitor_layout.addWidget(self.next_title)
        monitor_layout.addWidget(self.next_line1)
        monitor_layout.addWidget(self.next_line2)

        self.curr_note_detail = CollapsibleSection("")
        self.curr_event_detail = CollapsibleSection("")
        self.next_note_detail = CollapsibleSection("")
        self.next_event_detail = CollapsibleSection("")
        monitor_layout.addWidget(self.curr_note_detail)
        monitor_layout.addWidget(self.curr_event_detail)
        monitor_layout.addWidget(self.next_note_detail)
        monitor_layout.addWidget(self.next_event_detail)

        layout.addWidget(summary_group)
        layout.addWidget(monitor_group)
        layout.addStretch()

    def _build_settings_page(self) -> None:
        layout = QVBoxLayout(self.settings_page)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        self.settings_grid = QGridLayout()

        self.settings_basic_group = QGroupBox()
        basic_form = QFormLayout(self.settings_basic_group)
        basic_form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        chart_row = QHBoxLayout()
        self.chart_path_edit = QLineEdit()
        self.browse_btn = QPushButton()
        self.browse_btn.clicked.connect(self._choose_chart_file)
        chart_row.addWidget(self.chart_path_edit)
        chart_row.addWidget(self.browse_btn)
        chart_widget = QWidget()
        chart_widget.setLayout(chart_row)

        self.chart_path_label = QLabel()
        basic_form.addRow(self.chart_path_label, chart_widget)

        coord_group = QGroupBox()
        self.coord_grid = QGridLayout(coord_group)
        self.top_left_label = QLabel()
        self.top_right_label = QLabel()
        self.bottom_left_label = QLabel()
        self.bottom_right_label = QLabel()
        self.top_left_edit = QLineEdit()
        self.top_right_edit = QLineEdit()
        self.bottom_left_edit = QLineEdit()
        self.bottom_right_edit = QLineEdit()
        basic_form.addRow(coord_group)

        mixed_row = QHBoxLayout()
        self.fine_tune_label = QLabel()
        self.step_spin = QSpinBox()
        self.step_spin.setRange(1, 1000)
        self.step_spin.setSuffix(" ms")
        self.designant_label = QLabel()
        self.designant_combo = QComboBox()
        self.designant_combo.addItems(["", "", ""])
        mixed_row.addWidget(self.fine_tune_label)
        mixed_row.addWidget(self.step_spin)
        mixed_row.addSpacing(20)
        mixed_row.addWidget(self.designant_label)
        mixed_row.addWidget(self.designant_combo)
        mixed_widget = QWidget()
        mixed_widget.setLayout(mixed_row)
        basic_form.addRow(mixed_widget)

        self.save_btn = QPushButton()
        self.reload_btn = QPushButton()
        self.prepare_btn = QPushButton()
        self.save_btn.clicked.connect(self._on_save_clicked)
        self.reload_btn.clicked.connect(self._on_reload_clicked)
        self.prepare_btn.clicked.connect(lambda: self._request_prepare())
        button_row = QHBoxLayout()
        button_row.addWidget(self.save_btn)
        button_row.addWidget(self.reload_btn)
        button_row.addWidget(self.prepare_btn)
        button_widget = QWidget()
        button_widget.setLayout(button_row)
        basic_form.addRow(button_widget)

        stream_group = QGroupBox("scrcpy")
        stream_form = QFormLayout(stream_group)
        stream_form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self.stream_fps_label = QLabel("Stream FPS")
        self.stream_fps_spin = QSpinBox()
        self.stream_fps_spin.setRange(30, 200)
        self.stream_fps_spin.setValue(self.stream_max_fps)
        self.stream_bitrate_enable_check = QCheckBox()
        self.stream_bitrate_enable_check.setChecked(False)
        self.stream_bitrate_spin = QSpinBox()
        self.stream_bitrate_spin.setRange(1, 100)
        self.stream_bitrate_spin.setSuffix(" Mbps")
        self.stream_bitrate_spin.setValue(self.stream_bitrate_mbps)
        self.stream_bitrate_spin.setEnabled(False)
        self.stream_bitrate_enable_check.toggled.connect(
            self.stream_bitrate_spin.setEnabled
        )
        stream_form.addRow(self.stream_fps_label, self.stream_fps_spin)
        stream_form.addRow(self.stream_bitrate_enable_check, self.stream_bitrate_spin)
        basic_form.addRow(stream_group)

        self.log_limit_label = QLabel()
        self.log_limit_spin = QSpinBox()
        self.log_limit_spin.setRange(50, 5000)
        self.log_limit_spin.setValue(self.log_limit)
        basic_form.addRow(self.log_limit_label, self.log_limit_spin)

        self.debug_verbose_check = QCheckBox()
        self.debug_verbose_hint = QLabel()
        self.debug_verbose_hint.setWordWrap(True)
        basic_form.addRow(self.debug_verbose_check)
        basic_form.addRow(self.debug_verbose_hint)

        self.opt_high_prio_check = QCheckBox()
        self.opt_high_prio_check.setChecked(False)
        self.auto_start_cv_check = QCheckBox()
        self.auto_start_cv_check.setChecked(False)
        self.auto_start_cv_check.toggled.connect(self._on_auto_start_toggled)
        opt_row = QHBoxLayout()
        opt_row.addWidget(self.opt_high_prio_check)
        opt_row.addWidget(self.auto_start_cv_check)
        opt_widget = QWidget()
        opt_widget.setLayout(opt_row)
        basic_form.addRow(opt_widget)

        self.auto_start_hint = QLabel()
        self.auto_start_hint.setWordWrap(True)
        basic_form.addRow(self.auto_start_hint)

        self.settings_vision_group = QGroupBox("Vision")
        vision_form = QFormLayout(self.settings_vision_group)
        vision_form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self.ui_template_threshold_label = QLabel("UI template threshold")
        self.ui_template_threshold_spin = QDoubleSpinBox()
        self.ui_template_threshold_spin.setRange(0.00, 0.95)
        self.ui_template_threshold_spin.setSingleStep(0.01)
        self.ui_template_threshold_spin.setValue(0.42)

        self.ground_blue_ratio_threshold_label = QLabel("Ground blue ratio")
        self.ground_blue_ratio_threshold_spin = QDoubleSpinBox()
        self.ground_blue_ratio_threshold_spin.setRange(0.001, 0.50)
        self.ground_blue_ratio_threshold_spin.setSingleStep(0.005)
        self.ground_blue_ratio_threshold_spin.setValue(0.03)

        self.arc_color_ratio_threshold_label = QLabel("Arc color ratio")
        self.arc_color_ratio_threshold_spin = QDoubleSpinBox()
        self.arc_color_ratio_threshold_spin.setRange(0.001, 0.50)
        self.arc_color_ratio_threshold_spin.setSingleStep(0.005)
        self.arc_color_ratio_threshold_spin.setValue(0.02)

        self.arc_logic_roi_half_x_label = QLabel("Arc logic ROI half width")
        self.arc_logic_roi_half_x_spin = QDoubleSpinBox()
        self.arc_logic_roi_half_x_spin.setRange(0.01, 1.00)
        self.arc_logic_roi_half_x_spin.setSingleStep(0.01)
        self.arc_logic_roi_half_x_spin.setValue(0.25)

        self.arc_logic_roi_half_y_label = QLabel("Arc logic ROI half height")
        self.arc_logic_roi_half_y_spin = QDoubleSpinBox()
        self.arc_logic_roi_half_y_spin.setRange(0.01, 1.00)
        self.arc_logic_roi_half_y_spin.setSingleStep(0.01)
        self.arc_logic_roi_half_y_spin.setValue(0.25)

        self.overlay_debug_check = QCheckBox("Vision overlay")
        self.overlay_debug_check.setChecked(False)
        self.overlay_debug_check.toggled.connect(self._on_overlay_toggled)
        self.vision_perf_success_only_check = QCheckBox()
        self.vision_perf_success_only_check.setChecked(True)

        self.overlay_window = QWidget(None)
        self.overlay_window.setWindowFlag(Qt.Window, True)
        self.overlay_window.setWindowTitle("Vision Overlay")
        overlay_layout = QVBoxLayout(self.overlay_window)
        self.overlay_window_label = QLabel("preview")
        self.overlay_window_label.setMinimumSize(720, 420)
        self.overlay_window_label.setStyleSheet("background:#000; color:#bbb;")
        self.overlay_window_label.setAlignment(Qt.AlignCenter)
        overlay_layout.addWidget(self.overlay_window_label)
        self.overlay_window.resize(920, 560)
        self.overlay_window.hide()

        self.ui_left_x0_spin = QDoubleSpinBox()
        self.ui_left_y0_spin = QDoubleSpinBox()
        self.ui_left_x1_spin = QDoubleSpinBox()
        self.ui_left_y1_spin = QDoubleSpinBox()
        self.ground_x0_spin = QDoubleSpinBox()
        self.ground_y0_spin = QDoubleSpinBox()
        self.ground_x1_spin = QDoubleSpinBox()
        self.ground_y1_spin = QDoubleSpinBox()

        for spin in (
            self.ui_left_x0_spin,
            self.ui_left_y0_spin,
            self.ui_left_x1_spin,
            self.ui_left_y1_spin,
            self.ground_x0_spin,
            self.ground_y0_spin,
            self.ground_x1_spin,
            self.ground_y1_spin,
        ):
            spin.setRange(0.0, 1.0)
            spin.setSingleStep(0.005)
            spin.setDecimals(3)

        self.ui_left_x0_spin.setValue(0.005)
        self.ui_left_y0_spin.setValue(0.02)
        self.ui_left_x1_spin.setValue(0.34)
        self.ui_left_y1_spin.setValue(0.20)
        self.ground_x0_spin.setValue(0.12)
        self.ground_y0_spin.setValue(1310 / 1440)
        self.ground_x1_spin.setValue(0.88)
        self.ground_y1_spin.setValue(1345 / 1440)

        roi_group = QGroupBox("ROI")
        roi_grid = QGridLayout(roi_group)
        roi_grid.addWidget(QLabel("UI-L x0,y0,x1,y1"), 0, 0)
        roi_grid.addWidget(self.ui_left_x0_spin, 0, 1)
        roi_grid.addWidget(self.ui_left_y0_spin, 0, 2)
        roi_grid.addWidget(self.ui_left_x1_spin, 0, 3)
        roi_grid.addWidget(self.ui_left_y1_spin, 0, 4)
        roi_grid.addWidget(QLabel("Ground x0,y0,x1,y1"), 1, 0)
        roi_grid.addWidget(self.ground_x0_spin, 1, 1)
        roi_grid.addWidget(self.ground_y0_spin, 1, 2)
        roi_grid.addWidget(self.ground_x1_spin, 1, 3)
        roi_grid.addWidget(self.ground_y1_spin, 1, 4)

        self.roi_values_label = QLabel("roi_values")
        self.roi_values_label.setWordWrap(True)

        self.vision_debug_group = QGroupBox("Vision Debug")
        vision_debug_form = QFormLayout(self.vision_debug_group)
        vision_debug_form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self.vision_debug_source_label = QLabel("Input Source")
        self.vision_debug_source_combo = QComboBox()
        self.vision_debug_source_combo.addItem("scrcpy live", "scrcpy")
        self.vision_debug_source_combo.addItem("image file", "image")
        self.vision_debug_source_combo.addItem("video file", "video")
        self.vision_debug_source_combo.currentIndexChanged.connect(
            self._on_vision_debug_source_changed
        )
        self.vision_debug_input_label = QLabel("Image/Video Path")
        self.vision_debug_path_edit = QLineEdit()
        self.vision_debug_browse_btn = QPushButton("Browse")
        self.vision_debug_browse_btn.clicked.connect(self._browse_vision_debug_input)
        source_path_row = QHBoxLayout()
        source_path_row.addWidget(self.vision_debug_path_edit)
        source_path_row.addWidget(self.vision_debug_browse_btn)
        source_path_widget = QWidget()
        source_path_widget.setLayout(source_path_row)

        self.vision_debug_stage_label = QLabel("Detection Stages")
        self.vision_debug_ui_check = QCheckBox("ui_stage")
        self.vision_debug_ui_check.setChecked(True)
        self.vision_debug_ground_check = QCheckBox("ground")
        self.vision_debug_ground_check.setChecked(True)
        self.vision_debug_arc_check = QCheckBox("arc_pass")
        self.vision_debug_arc_check.setChecked(True)
        stage_row = QHBoxLayout()
        stage_row.addWidget(self.vision_debug_ui_check)
        stage_row.addWidget(self.vision_debug_ground_check)
        stage_row.addWidget(self.vision_debug_arc_check)
        stage_row.addStretch()
        stage_widget = QWidget()
        stage_widget.setLayout(stage_row)

        self.vision_debug_logic_x_label = QLabel("ground/arc logic_x")
        self.vision_debug_logic_x_spin = QDoubleSpinBox()
        self.vision_debug_logic_x_spin.setRange(-0.25, 1.25)
        self.vision_debug_logic_x_spin.setDecimals(3)
        self.vision_debug_logic_x_spin.setSingleStep(0.01)
        self.vision_debug_logic_x_spin.setValue(0.50)

        self.vision_debug_logic_y_label = QLabel("arc logic_y")
        self.vision_debug_logic_y_spin = QDoubleSpinBox()
        self.vision_debug_logic_y_spin.setRange(-0.25, 1.25)
        self.vision_debug_logic_y_spin.setDecimals(3)
        self.vision_debug_logic_y_spin.setSingleStep(0.01)
        self.vision_debug_logic_y_spin.setValue(0.50)

        self.vision_debug_toggle_btn = QPushButton("Start Vision Debug")
        self.vision_debug_toggle_btn.clicked.connect(self._toggle_vision_debug)
        self.vision_debug_hint_label = QLabel(
            "Recognition only. No automation input is started. Image/video sources keep looping."
        )
        self.vision_debug_hint_label.setWordWrap(True)
        self.vision_debug_result_label = QLabel("Vision Debug: idle")
        self.vision_debug_result_label.setWordWrap(True)
        self.vision_debug_result_label.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )

        vision_debug_form.addRow(
            self.vision_debug_source_label, self.vision_debug_source_combo
        )
        vision_debug_form.addRow(self.vision_debug_input_label, source_path_widget)
        vision_debug_form.addRow(self.vision_debug_stage_label, stage_widget)
        vision_debug_form.addRow(
            self.vision_debug_logic_x_label, self.vision_debug_logic_x_spin
        )
        vision_debug_form.addRow(
            self.vision_debug_logic_y_label, self.vision_debug_logic_y_spin
        )
        vision_debug_form.addRow(self.vision_debug_toggle_btn)
        vision_debug_form.addRow(self.vision_debug_hint_label)
        vision_debug_form.addRow(self.vision_debug_result_label)

        vision_form.addRow(
            self.ui_template_threshold_label, self.ui_template_threshold_spin
        )
        vision_form.addRow(
            self.ground_blue_ratio_threshold_label,
            self.ground_blue_ratio_threshold_spin,
        )
        vision_form.addRow(
            self.arc_color_ratio_threshold_label,
            self.arc_color_ratio_threshold_spin,
        )
        vision_form.addRow(
            self.arc_logic_roi_half_x_label,
            self.arc_logic_roi_half_x_spin,
        )
        vision_form.addRow(
            self.arc_logic_roi_half_y_label,
            self.arc_logic_roi_half_y_spin,
        )
        vision_form.addRow(self.vision_perf_success_only_check)
        vision_form.addRow(self.overlay_debug_check)
        vision_form.addRow(roi_group)
        vision_form.addRow(self.roi_values_label)
        vision_form.addRow(self.vision_debug_group)

        self.settings_grid.addWidget(self.settings_basic_group, 0, 0)
        self.settings_grid.addWidget(self.settings_vision_group, 0, 1)
        scroll_layout.addLayout(self.settings_grid)
        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        self._reflow_coord_grid(False)
        self._reflow_settings_layout()
        self._on_vision_debug_source_changed()

    def _apply_texts(self) -> None:
        self.setWindowTitle(self._t("window_title"))
        self.language_label.setText(self._t("language"))
        self.tabs.setTabText(0, self._t("tab_status"))
        self.tabs.setTabText(1, self._t("tab_settings"))

        self.control_group.setTitle("")

        self.run_state_key.setText(self._t("run_state"))
        self.controller_key.setText(self._t("controller"))
        self.prepare_key.setText(self._t("prepare_state"))
        self.offset_key.setText(self._t("offset"))
        self.delay_key.setText(self._t("delay"))

        self.curr_title.setText(self._t("summary_current"))
        self.next_title.setText(self._t("summary_next"))
        self.curr_note_detail.set_title(self._t("detail_curr_note"))
        self.curr_event_detail.set_title(self._t("detail_curr_event"))
        self.next_note_detail.set_title(self._t("detail_next_note"))
        self.next_event_detail.set_title(self._t("detail_next_event"))

        self.start_btn.setText(self._t("start"))
        self.stop_btn.setText(self._t("stop"))
        self.plus_btn.setText(self._t("step_plus"))
        self.minus_btn.setText(self._t("step_minus"))
        self.reset_btn.setText(self._t("reset"))

        self.chart_path_label.setText(self._t("chart_path"))
        self.browse_btn.setText(self._t("browse"))
        self.top_left_label.setText(self._t("top_left"))
        self.top_right_label.setText(self._t("top_right"))
        self.bottom_left_label.setText(self._t("bottom_left"))
        self.bottom_right_label.setText(self._t("bottom_right"))
        self.fine_tune_label.setText(self._t("fine_tune_step"))
        self.designant_label.setText(self._t("designant"))
        self.designant_combo.setItemText(0, self._t("unset"))
        self.designant_combo.setItemText(1, self._t("enabled"))
        self.designant_combo.setItemText(2, self._t("disabled"))
        self.save_btn.setText(self._t("save_config"))
        self.reload_btn.setText(self._t("reload_config"))
        self.prepare_btn.setText(self._t("prepare"))

        self.clear_log_btn.setText(self._t("log_clear"))
        self.stream_fps_label.setText(self._t("stream_fps"))
        self.stream_bitrate_enable_check.setText(self._t("stream_bitrate_enable"))
        self.stream_bitrate_spin.setSuffix(
            f" {self._t('stream_bitrate').split('(')[-1].rstrip(')')}"
        )
        self.log_limit_label.setText(self._t("log_limit"))
        self.debug_verbose_check.setText(self._t("debug_verbose"))
        self.debug_verbose_hint.setText(self._t("debug_verbose_hint"))
        self.opt_high_prio_check.setText(self._t("opt_high_prio"))
        self.auto_start_cv_check.setText(self._t("auto_start_cv"))
        self.auto_start_hint.setText(self._t("auto_start_hint"))
        self.ui_template_threshold_label.setText("UI template threshold")
        self.ground_blue_ratio_threshold_label.setText(
            self._t("ground_blue_ratio_threshold")
        )
        self.arc_color_ratio_threshold_label.setText(
            self._t("arc_color_ratio_threshold")
        )
        self.arc_logic_roi_half_x_label.setText(self._t("arc_logic_roi_half_x"))
        self.arc_logic_roi_half_y_label.setText(self._t("arc_logic_roi_half_y"))
        self.vision_perf_success_only_check.setText(self._t("vision_perf_success_only"))
        self.overlay_debug_check.setText(self._t("vision_overlay"))
        self.vision_debug_group.setTitle(self._t("vision_debug_group"))
        self.vision_debug_source_label.setText(self._t("vision_debug_source"))
        self.vision_debug_input_label.setText(self._t("vision_debug_input"))
        self.vision_debug_browse_btn.setText(self._t("vision_debug_browse"))
        self.vision_debug_stage_label.setText(self._t("vision_debug_stages"))
        self.vision_debug_ui_check.setText(self._t("vision_debug_stage_ui"))
        self.vision_debug_ground_check.setText(self._t("vision_debug_stage_ground"))
        self.vision_debug_arc_check.setText(self._t("vision_debug_stage_arc"))
        self.vision_debug_logic_x_label.setText(self._t("vision_debug_logic_x"))
        self.vision_debug_logic_y_label.setText(self._t("vision_debug_logic_y"))
        self.vision_debug_toggle_btn.setText(
            self._t("vision_debug_stop")
            if self._vision_debug_running
            else self._t("vision_debug_start")
        )
        self.vision_debug_hint_label.setText(self._t("vision_debug_hint"))
        if not self._vision_debug_running:
            self.vision_debug_result_label.setText(self._t("vision_debug_idle"))

        self.run_state_label.setText(self._t("idle"))
        self.controller_state_label.setText(
            self._t("ready") if self.controller_ready else self._t("warming")
        )
        self.prepare_state_label.setText(
            self._t("ready") if self.prepared is not None else self._t("warming")
        )

    def _append_log(self, text: str) -> None:
        self.log_limit = (
            int(self.log_limit_spin.value())
            if hasattr(self, "log_limit_spin")
            else self.log_limit
        )
        self.log_lines.append(text)
        if len(self.log_lines) > self.log_limit:
            overflow = len(self.log_lines) - self.log_limit
            self.log_lines = self.log_lines[overflow:]
        self.log_output.setPlainText("\n".join(self.log_lines))
        self.log_output.verticalScrollBar().setValue(
            self.log_output.verticalScrollBar().maximum()
        )

    def _write_touch_event_snapshot(self) -> Path | None:
        if self.prepared is None:
            return None
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        chart_name = Path(self.prepared.run_config.chart_path).stem or "chart"
        out_json = DEBUG_DIR / f"gui_touch_snapshot_{chart_name}_{timestamp}.json"

        touch_events_flat: list[dict[str, object]] = []
        for tick in sorted(self.prepared.events_by_time.keys()):
            for event in self.prepared.events_by_time[tick]:
                touch_events_flat.append(
                    {
                        "tick": tick,
                        "pointer": event.pointer,
                        "action": event.action.name,
                        "position": list(event.pos),
                        "source_note_id": event.source_note_id,
                        "source_type": event.source_type,
                        "logical_tick": event.logical_tick,
                        "logical_pos": list(event.logical_pos)
                        if event.logical_pos is not None
                        else None,
                    }
                )

        snapshot = {
            "stats": {
                "touch_event_ticks": len(self.prepared.events_by_time),
                "touch_events_total": len(touch_events_flat),
                "first_note_types": list(self.prepared.first_note_types),
                "first_ground_tick": self.prepared.first_ground_tick,
                "first_ground_logic_pos": _to_serializable(
                    self.prepared.first_ground_logic_pos
                ),
                "first_note_logic_pos": _to_serializable(
                    self.prepared.first_note_logic_pos
                ),
            },
            "run_config": _to_serializable(self.prepared.run_config),
            "note_meta": _to_serializable(self.prepared.note_meta),
            "touch_events": touch_events_flat,
        }
        out_json.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return out_json

    def _report_missing_vision_templates(self) -> None:
        metrics = self.vision_detector.metrics
        if metrics.ui_left_template_count > 0 and metrics.ui_left_descriptor_ready:
            return
        self._append_log(
            "[VISION][ERROR] UI-left templates not loaded. template_dir={} template_count={} descriptor_ready={}".format(
                metrics.ui_left_template_dir,
                metrics.ui_left_template_count,
                metrics.ui_left_descriptor_ready,
            )
        )
        expected = ["uileft.png", "uileft_2.png", "pause.png"]
        for name in expected:
            path = Path(metrics.ui_left_template_dir) / name
            self._append_log(
                f"[VISION][ERROR] expected template: {path} exists={path.exists()}"
            )

    def _clear_logs(self) -> None:
        self.log_lines.clear()
        self.log_output.clear()

    def _format_note(
        self, event, note_meta: dict[int, dict[str, object]]
    ) -> tuple[str, str, str]:
        if event is None:
            return self._t("none"), self._t("none"), self._t("none")
        source_note_id = getattr(event, "source_note_id", None)
        meta = (
            note_meta.get(source_note_id, {}) if isinstance(source_note_id, int) else {}
        )
        note_type = meta.get("type", getattr(event, "source_type", "unknown"))
        tick = meta.get("tick", getattr(event, "logical_tick", "?"))
        line1 = f"note_type={note_type}; note_tick={tick}"
        line2 = (
            f"note_start={meta.get('start', 'n/a')}; note_end={meta.get('end', 'n/a')}"
        )
        detail = "\n".join(
            [
                f"note_id: {source_note_id}",
                f"note_type: {note_type}",
                f"tick: {tick}",
                f"start_coord: {meta.get('start', 'n/a')}",
                f"end_coord: {meta.get('end', 'n/a')}",
                f"end_tick: {meta.get('end_tick', 'n/a')}",
            ]
        )
        return line1, line2, detail

    def _format_event(
        self, event, tick: int | None, group_size: int
    ) -> tuple[str, str, str]:
        if event is None:
            return self._t("none"), self._t("none"), self._t("none")
        line1 = f"event_tick={tick}; event_pointer={event.pointer}; event_action={event.action.name}"
        line2 = f"event_position={event.pos}; event_group_size={group_size}"
        detail = "\n".join(
            [
                f"tick: {tick}",
                f"pointer: {event.pointer}",
                f"action: {event.action.name}",
                f"position: {event.pos}",
                f"logical_tick: {event.logical_tick}",
                f"logical_pos: {event.logical_pos}",
                f"source_note_id: {event.source_note_id}",
                f"source_type: {event.source_type}",
            ]
        )
        return line1, line2, detail

    def _track_active_touch_events(self, events: list) -> None:
        for event in events:
            action = getattr(event, "action", None)
            pointer = getattr(event, "pointer", None)
            pos = getattr(event, "pos", None)
            if not isinstance(pointer, int):
                continue
            if not isinstance(pos, tuple) or len(pos) != 2:
                continue
            x = int(pos[0])
            y = int(pos[1])
            if action in {TouchAction.DOWN, TouchAction.MOVE, TouchAction.POINTER_DOWN}:
                self._active_touch_points[pointer] = (x, y)
            elif action in {TouchAction.UP, TouchAction.POINTER_UP, TouchAction.CANCEL}:
                self._active_touch_points.pop(pointer, None)

    def _force_release_touch_points(self, points: dict[int, tuple[int, int]]) -> None:
        if not points:
            return
        if self.controller is None or not self.controller_ready:
            self._append_log(
                "[WARN] Manual stop touch reset skipped: touch controller is not ready."
            )
            return
        released = 0
        for pointer, (x, y) in sorted(points.items()):
            try:
                self.controller.touch(int(x), int(y), TouchAction.UP, int(pointer))
                released += 1
            except Exception as exc:
                self._append_log(
                    f"[WARN] Manual stop touch reset failed for pointer={pointer}: {exc}"
                )
        if released > 0:
            self._append_log(
                f"[INFO] Manual stop touch reset completed: released {released} pointer(s) after 200ms."
            )

    def _on_progress(self, payload: dict) -> None:
        if not self._first_dispatch_logged and self._start_click_time is not None:
            elapsed_ms = (time.perf_counter() - self._start_click_time) * 1000
            self._append_log(
                f"[DEBUG] Start->first dispatch latency: {elapsed_ms:.2f}ms"
            )
            self._first_dispatch_logged = True

        note_meta = payload.get("note_meta", {})
        curr_events = payload.get("curr_events") or []
        if curr_events:
            self._track_active_touch_events(curr_events)
        curr_event = payload.get("curr_event")
        next_event = payload.get("next_event")

        curr_note_l1, curr_note_l2, curr_note_detail = self._format_note(
            curr_event, note_meta
        )
        curr_evt_l1, curr_evt_l2, curr_evt_detail = self._format_event(
            curr_event,
            payload.get("curr_tick"),
            payload.get("curr_size", 0),
        )
        next_note_l1, next_note_l2, next_note_detail = self._format_note(
            next_event, note_meta
        )
        next_evt_l1, next_evt_l2, next_evt_detail = self._format_event(
            next_event,
            payload.get("next_tick"),
            payload.get("next_size", 0),
        )

        self.curr_line1.setText(curr_note_l1)
        self.curr_line2.setText(curr_evt_l1)
        self.next_line1.setText(next_note_l1)
        self.next_line2.setText(next_evt_l1)

        self.curr_note_detail.set_text(curr_note_detail + "\n" + curr_note_l2)
        self.curr_event_detail.set_text(curr_evt_detail)
        self.next_note_detail.set_text(next_note_detail + "\n" + next_note_l2)
        self.next_event_detail.set_text(next_evt_detail)

    def _log_vision_perf(self, label: str, passed: bool) -> None:
        if self._vision_perf_success_only and not passed:
            return
        perf = self.vision_detector.perf
        metrics = self.vision_detector.metrics
        self._append_log(
            "[VISION] {} total={:.3f}ms roi={:.3f}ms resize={:.3f}ms gray={:.3f}ms orb={:.3f}ms match={:.3f}ms homography={:.3f}ms hsv={:.3f}ms mask={:.3f}ms count={:.3f}ms decision={:.3f}ms ui_kp={} ui_tpl={} ui_tpl_kp_max={} ui_roi={} desc_ready={}".format(
                label,
                perf.total_ms,
                perf.roi_ms,
                perf.resize_ms,
                perf.gray_ms,
                perf.orb_ms,
                perf.match_ms,
                perf.homography_ms,
                perf.hsv_ms,
                perf.mask_ms,
                perf.count_ms,
                perf.decision_ms,
                metrics.ui_left_keypoints,
                metrics.ui_left_template_count,
                metrics.ui_left_template_keypoints_max,
                metrics.ui_left_roi_shape,
                metrics.ui_left_descriptor_ready,
            )
        )

    def _on_first_dispatch_metrics(self, payload: dict[str, float]) -> None:
        if self._auto_start_triggered_at is None:
            return
        trigger_to_dispatch_ms = (
            time.perf_counter() - self._auto_start_triggered_at
        ) * 1000.0
        frame_to_dispatch_ms = None
        if self._auto_start_frame_timestamp is not None:
            frame_to_dispatch_ms = (
                time.perf_counter() - self._auto_start_frame_timestamp
            ) * 1000.0
        detect_to_dispatch_ms = None
        if self._auto_start_detection_started_at is not None:
            detect_to_dispatch_ms = (
                time.perf_counter() - self._auto_start_detection_started_at
            ) * 1000.0
        message = "[VISION] end-to-end trigger->dispatch={:.2f}ms detect->dispatch={:.2f}ms vision_total={:.2f}ms scheduler_lateness={:.2f}ms touch_send={:.3f}ms".format(
            trigger_to_dispatch_ms,
            detect_to_dispatch_ms if detect_to_dispatch_ms is not None else -1.0,
            self._auto_start_vision_total_ms,
            float(payload.get("lateness_ms", 0.0)),
            float(payload.get("touch_send_call_ms", 0.0)),
        )
        if frame_to_dispatch_ms is not None:
            message += " frame->dispatch={:.2f}ms".format(frame_to_dispatch_ms)
        self._append_log(message)
        self._auto_start_detection_started_at = None
        self._auto_start_triggered_at = None
        self._auto_start_frame_timestamp = None
        self._auto_start_vision_total_ms = 0.0

    @staticmethod
    def _clamp_roi(
        roi: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = roi
        x0 = max(0.0, min(1.0, float(x0)))
        y0 = max(0.0, min(1.0, float(y0)))
        x1 = max(0.0, min(1.0, float(x1)))
        y1 = max(0.0, min(1.0, float(y1)))
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        return x0, y0, x1, y1

    def _roi_to_pixel_rect(
        self,
        frame: np.ndarray,
        roi: tuple[float, float, float, float],
    ) -> tuple[int, int, int, int]:
        h, w = frame.shape[:2]
        x0 = max(0, min(w - 1, int(round(roi[0] * w))))
        y0 = max(0, min(h - 1, int(round(roi[1] * h))))
        x1 = max(x0 + 1, min(w, int(round(roi[2] * w))))
        y1 = max(y0 + 1, min(h, int(round(roi[3] * h))))
        return x0, y0, x1, y1

    def _logic_rect_roi(self, logic_x: float, logic_y: float) -> tuple[float, float, float, float]:
        half_x = float(self.arc_logic_roi_half_x_spin.value())
        half_y = float(self.arc_logic_roi_half_y_spin.value())
        screen_logic_y = 1.0 - float(logic_y)
        return self._clamp_roi(
            (
                logic_x - half_x,
                screen_logic_y - half_y,
                logic_x + half_x,
                screen_logic_y + half_y,
            )
        )

    def _compute_pretransport_crop_roi(self) -> tuple[float, float, float, float] | None:
        rects: list[tuple[float, float, float, float]] = [
            self._clamp_roi(
                (
                    float(self.ui_left_x0_spin.value()),
                    float(self.ui_left_y0_spin.value()),
                    float(self.ui_left_x1_spin.value()),
                    float(self.ui_left_y1_spin.value()),
                )
            )
        ]

        if self.prepared is None:
            if (
                self._vision_debug_running
                and str(self.vision_debug_source_combo.currentData()) == "scrcpy"
            ):
                rects.append(
                    self._logic_rect_roi(
                        float(self.vision_debug_logic_x_spin.value()),
                        float(self.vision_debug_logic_y_spin.value()),
                    )
                )
        elif self.prepared.first_note_logic_pos is not None:
            rects.append(
                self._logic_rect_roi(
                    float(self.prepared.first_note_logic_pos[0]),
                    float(self.prepared.first_note_logic_pos[1]),
                )
            )

        if self.prepared is not None and self.prepared.first_ground_logic_pos is not None:
            rects.append(
                self._logic_rect_roi(
                    float(self.prepared.first_ground_logic_pos[0]),
                    float(self.prepared.first_ground_logic_pos[1]),
                )
            )
        elif self.prepared is not None and self.prepared.first_ground_logic_x is not None:
            ground_center_screen_y = (
                float(self.ground_y0_spin.value()) + float(self.ground_y1_spin.value())
            ) * 0.5
            ground_center_logic_y = 1.0 - ground_center_screen_y
            rects.append(
                self._logic_rect_roi(
                    float(self.prepared.first_ground_logic_x),
                    ground_center_logic_y,
                )
            )

        if not rects:
            return None

        x0 = min(rect[0] for rect in rects)
        y0 = min(rect[1] for rect in rects)
        x1 = max(rect[2] for rect in rects)
        y1 = max(rect[3] for rect in rects)

        # Keep a small margin to absorb frame jitter near the crop border.
        margin_x = 0.02
        margin_y = 0.02
        crop = self._clamp_roi((x0 - margin_x, y0 - margin_y, x1 + margin_x, y1 + margin_y))

        if (crop[2] - crop[0]) >= 0.98 and (crop[3] - crop[1]) >= 0.98:
            return None
        return crop

    def _compute_pretransport_crop_pixels(
        self,
    ) -> tuple[int, int, int, int] | None:
        roi = self._compute_pretransport_crop_roi()
        if roi is None:
            return None

        native_size = self._native_device_size
        if native_size is None and self.controller is not None:
            native_size = (
                int(getattr(self.controller, "device_width", 0)),
                int(getattr(self.controller, "device_height", 0)),
            )
        if native_size is None:
            return None
        width, height = native_size
        if width <= 0 or height <= 0:
            return None

        x0 = max(0, min(width - 1, int(round(roi[0] * width))))
        y0 = max(0, min(height - 1, int(round(roi[1] * height))))
        x1 = max(x0 + 1, min(width, int(round(roi[2] * width))))
        y1 = max(y0 + 1, min(height, int(round(roi[3] * height))))
        return (x0, y0, x1 - x0, y1 - y0)

    def _get_active_stream_crop_roi(self) -> tuple[float, float, float, float]:
        if self.vision_controller is None or not hasattr(
            self.vision_controller, "get_stream_crop_rect"
        ):
            return (0.0, 0.0, 1.0, 1.0)
        try:
            crop_rect = tuple(self.vision_controller.get_stream_crop_rect())
        except Exception:
            return (0.0, 0.0, 1.0, 1.0)
        if len(crop_rect) != 4:
            return (0.0, 0.0, 1.0, 1.0)
        if self._native_device_size is not None:
            full_w, full_h = self._native_device_size
        else:
            full_w = int(getattr(self.vision_controller, "device_width", 0))
            full_h = int(getattr(self.vision_controller, "device_height", 0))
        if full_w <= 0 or full_h <= 0:
            return (0.0, 0.0, 1.0, 1.0)
        x, y, w, h = (
            int(crop_rect[0]),
            int(crop_rect[1]),
            int(crop_rect[2]),
            int(crop_rect[3]),
        )
        return self._clamp_roi(
            (
                x / float(full_w),
                y / float(full_h),
                (x + w) / float(full_w),
                (y + h) / float(full_h),
            )
        )

    def _is_vision_stream_required(self) -> bool:
        auto_start_on = bool(self.auto_start_cv_check.isChecked())
        debug_scrcpy_on = bool(
            self._vision_debug_running
            and str(self.vision_debug_source_combo.currentData()) == "scrcpy"
        )
        return auto_start_on or debug_scrcpy_on

    def _stop_vision_stream(self) -> None:
        if self.vision_controller is not None:
            try:
                self.vision_controller.close()
            except Exception:
                pass
        self.vision_controller = None
        self.vision_controller_ready = False
        self._vision_last_frame_seq = -1
        self.vision_warmup_worker = None

    def _start_vision_warmup(self, video_crop: tuple[int, int, int, int]) -> None:
        if self.vision_warmup_worker is not None and self.vision_warmup_worker.isRunning():
            return
        bit_rate = None
        if self.stream_bitrate_enabled:
            bit_rate = int(self.stream_bitrate_mbps) * 1_000_000
        self.vision_warmup_worker = ControllerWarmupWorker(
            max_fps=self.stream_max_fps,
            video_bit_rate=bit_rate,
            video_crop=video_crop,
        )
        self.vision_warmup_worker.started_warmup.connect(self._on_vision_warmup_start)
        self.vision_warmup_worker.warmup_ok.connect(self._on_vision_warmup_ok)
        self.vision_warmup_worker.warmup_fail.connect(self._on_vision_warmup_fail)
        self.vision_warmup_worker.start()

    def _restart_vision_stream_if_needed(self, reason: str, *, force: bool = False) -> None:
        if not self._is_vision_stream_required():
            self._stop_vision_stream()
            return
        if not self.controller_ready or self.controller is None:
            return
        debug_scrcpy_on = bool(
            self._vision_debug_running
            and str(self.vision_debug_source_combo.currentData()) == "scrcpy"
        )
        if self.prepared is None and not debug_scrcpy_on:
            return

        desired_crop = self._compute_pretransport_crop_pixels()
        if desired_crop is None:
            self._stop_vision_stream()
            return

        current_crop: tuple[int, int, int, int] | None = None
        if self.vision_controller is not None and hasattr(
            self.vision_controller, "get_stream_crop_rect"
        ):
            try:
                raw_crop = tuple(self.vision_controller.get_stream_crop_rect())
                if len(raw_crop) == 4:
                    current_crop = (
                        int(raw_crop[0]),
                        int(raw_crop[1]),
                        int(raw_crop[2]),
                        int(raw_crop[3]),
                    )
            except Exception:
                current_crop = None
        if not force and current_crop == desired_crop and self.vision_controller_ready:
            return
        if (
            not force
            and self.vision_warmup_worker is not None
            and self.vision_warmup_worker.isRunning()
        ):
            return

        self._append_log(
            "[INFO] Restarting vision stream ({}) pre-crop {}:{}:{}:{} (w:h:x:y), fps={}, bitrate={}".format(
                reason,
                desired_crop[2],
                desired_crop[3],
                desired_crop[0],
                desired_crop[1],
                self.stream_max_fps,
                "off"
                if not self.stream_bitrate_enabled
                else f"{self.stream_bitrate_mbps}Mbps",
            )
        )
        self._stop_vision_stream()
        self._start_vision_warmup(desired_crop)

    def _read_vision_live_frame(self) -> tuple[np.ndarray | None, float | None]:
        if self.vision_controller is None or not self.vision_controller_ready:
            return None, None
        frame = self.vision_controller.get_latest_frame(copy_frame=True)
        fps = self.vision_controller.get_decode_fps()
        return frame, fps

    def _build_vision_runtime(
        self,
        stream_crop_roi: tuple[float, float, float, float] | None = None,
    ) -> VisionRuntimeConfig:
        return VisionRuntimeConfig(
            ui_left_roi=(
                float(self.ui_left_x0_spin.value()),
                float(self.ui_left_y0_spin.value()),
                float(self.ui_left_x1_spin.value()),
                float(self.ui_left_y1_spin.value()),
            ),
            ground_roi=(
                float(self.ground_x0_spin.value()),
                float(self.ground_y0_spin.value()),
                float(self.ground_x1_spin.value()),
                float(self.ground_y1_spin.value()),
            ),
            ui_feature_threshold=float(self.ui_template_threshold_spin.value()),
            ground_blue_ratio_threshold=float(
                self.ground_blue_ratio_threshold_spin.value()
            ),
            arc_color_ratio_threshold=float(
                self.arc_color_ratio_threshold_spin.value()
            ),
            arc_logic_roi_half_x=float(self.arc_logic_roi_half_x_spin.value()),
            arc_logic_roi_half_y=float(self.arc_logic_roi_half_y_spin.value()),
            stream_crop_roi=(
                self._get_active_stream_crop_roi()
                if stream_crop_roi is None
                else self._clamp_roi(stream_crop_roi)
            ),
        )

    def _is_gameplay_screen(self, frame: np.ndarray) -> bool:
        self.vision_detector.set_runtime(self._build_vision_runtime())
        passed = self.vision_detector.detect_ui_panel(frame)
        self._log_vision_perf("ui_left", passed)
        return passed

    def _is_first_ground_note_on_judgment(self, frame: np.ndarray) -> bool:
        if (
            self.prepared is None
            or self.prepared.first_ground_tick is None
        ):
            return False
        if self.prepared.first_ground_logic_pos is not None:
            logic_x, logic_y = self.prepared.first_ground_logic_pos
        elif self.prepared.first_ground_logic_x is not None:
            logic_x = self.prepared.first_ground_logic_x
            ground_center_screen_y = (
                float(self.ground_y0_spin.value()) + float(self.ground_y1_spin.value())
            ) * 0.5
            logic_y = 1.0 - ground_center_screen_y
        else:
            return False
        self.vision_detector.set_runtime(self._build_vision_runtime())
        passed = self.vision_detector.detect_ground_overlap(
            frame, logic_x, logic_y
        )
        self._log_vision_perf("ground", passed)
        return passed

    def _is_arc_cap_triggered(self, frame: np.ndarray) -> bool:
        if self.prepared is None or self.prepared.first_note_logic_pos is None:
            return False
        self.vision_detector.set_runtime(self._build_vision_runtime())
        logic_x, logic_y = self.prepared.first_note_logic_pos
        passed = self.vision_detector.detect_arc_overlap(frame, logic_x, logic_y)
        self._log_vision_perf("arc", passed)
        return passed

    def _update_overlay_preview(
        self,
        frame: np.ndarray,
        stage: str | None = None,
        *,
        force_show: bool = False,
        run_detection: bool = True,
        decode_fps: float | None = None,
    ) -> None:
        if not force_show and not self.overlay_debug_check.isChecked():
            self.overlay_window.hide()
            return

        render_stage = stage or self._auto_start_stage
        if run_detection:
            runtime_crop: tuple[float, float, float, float] | None = None
            if (
                render_stage.startswith("debug")
                and str(self.vision_debug_source_combo.currentData()) != "scrcpy"
            ):
                runtime_crop = (0.0, 0.0, 1.0, 1.0)
            self.vision_detector.set_runtime(
                self._build_vision_runtime(stream_crop_roi=runtime_crop)
            )
            if render_stage == "wait_ui":
                self.vision_detector.detect_ui_panel(frame)
            elif render_stage == "wait_ground":
                logic_x = 0.5
                ground_center_screen_y = (
                    float(self.ground_y0_spin.value()) + float(self.ground_y1_spin.value())
                ) * 0.5
                logic_y = 1.0 - ground_center_screen_y
                if (
                    self.prepared is not None
                    and self.prepared.first_ground_logic_pos is not None
                ):
                    logic_x = float(self.prepared.first_ground_logic_pos[0])
                    logic_y = float(self.prepared.first_ground_logic_pos[1])
                self.vision_detector.detect_ground_overlap(frame, logic_x, logic_y)
            elif render_stage == "wait_arc":
                if (
                    self.prepared is not None
                    and self.prepared.first_note_logic_pos is not None
                ):
                    logic_x, logic_y = self.prepared.first_note_logic_pos
                    self.vision_detector.detect_arc_overlap(frame, logic_x, logic_y)

        overlay = frame.copy()
        runtime_draw = self._build_vision_runtime(stream_crop_roi=(0.0, 0.0, 1.0, 1.0))

        ui_rect = self._roi_to_pixel_rect(overlay, runtime_draw.ui_left_roi)
        cv2.rectangle(overlay, (ui_rect[0], ui_rect[1]), (ui_rect[2], ui_rect[3]), (220, 180, 40), 2)

        crop_roi = self._get_active_stream_crop_roi()
        if crop_roi != (0.0, 0.0, 1.0, 1.0):
            cx0, cy0, cx1, cy1 = self._roi_to_pixel_rect(overlay, crop_roi)
            cv2.rectangle(overlay, (cx0, cy0), (cx1, cy1), (200, 90, 255), 2)

        debug_mode = render_stage.startswith("debug")
        ground_logic_x = None
        ground_logic_y = None
        arc_logic_x = None
        arc_logic_y = None
        if debug_mode:
            ground_logic_x = float(self.vision_debug_logic_x_spin.value())
            ground_logic_y = float(self.vision_debug_logic_y_spin.value())
            arc_logic_x = float(self.vision_debug_logic_x_spin.value())
            arc_logic_y = float(self.vision_debug_logic_y_spin.value())
        elif self.prepared is not None:
            if self.prepared.first_ground_logic_pos is not None:
                ground_logic_x = float(self.prepared.first_ground_logic_pos[0])
                ground_logic_y = float(self.prepared.first_ground_logic_pos[1])
            elif self.prepared.first_ground_logic_x is not None:
                ground_logic_x = float(self.prepared.first_ground_logic_x)
                ground_logic_y = 1.0 - (
                    float(self.ground_y0_spin.value()) + float(self.ground_y1_spin.value())
                ) * 0.5
            if self.prepared.first_note_logic_pos is not None:
                arc_logic_x = float(self.prepared.first_note_logic_pos[0])
                arc_logic_y = float(self.prepared.first_note_logic_pos[1])

        draw_ground = render_stage == "wait_ground" or "ground" in render_stage
        draw_arc = render_stage == "wait_arc" or "arc" in render_stage
        if draw_ground and ground_logic_x is not None and ground_logic_y is not None:
            gx0, gy0, gx1, gy1 = self._roi_to_pixel_rect(
                overlay,
                self._logic_rect_roi(ground_logic_x, ground_logic_y),
            )
            cv2.rectangle(overlay, (gx0, gy0), (gx1, gy1), (80, 255, 80), 2)
        if draw_arc and arc_logic_x is not None and arc_logic_y is not None:
            ax0, ay0, ax1, ay1 = self._roi_to_pixel_rect(
                overlay,
                self._logic_rect_roi(arc_logic_x, arc_logic_y),
            )
            cv2.rectangle(overlay, (ax0, ay0), (ax1, ay1), (255, 255, 0), 2)

        metrics = self.vision_detector.metrics
        fps_value = (
            decode_fps
            if decode_fps is not None
            else (self.controller.get_decode_fps() if self.controller is not None else None)
        )
        lines = [
            f"stage={render_stage}",
            f"fps={fps_value:.1f}" if fps_value is not None else "fps=n/a",
            f"ui={metrics.ui_left_feature_score:.3f} ground={metrics.ground_blue_ratio:.3f} arc={metrics.arc_color_ratio:.3f}",
        ]
        base_y = 28
        for text in lines:
            cv2.putText(
                overlay,
                text,
                (14, base_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                text,
                (14, base_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (40, 245, 255),
                2,
                cv2.LINE_AA,
            )
            base_y += 24

        rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        qimg = QImage(
            rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888
        )

        self.roi_values_label.setText(
            "UI-L=({:.3f},{:.3f},{:.3f},{:.3f}) Ground=({:.3f},{:.3f},{:.3f},{:.3f}) ArcLogicHalf=({:.3f},{:.3f})".format(
                float(self.ui_left_x0_spin.value()),
                float(self.ui_left_y0_spin.value()),
                float(self.ui_left_x1_spin.value()),
                float(self.ui_left_y1_spin.value()),
                float(self.ground_x0_spin.value()),
                float(self.ground_y0_spin.value()),
                float(self.ground_x1_spin.value()),
                float(self.ground_y1_spin.value()),
                float(self.arc_logic_roi_half_x_spin.value()),
                float(self.arc_logic_roi_half_y_spin.value()),
            )
        )
        self.roi_values_label.setText(
            self.roi_values_label.text()
            + " | crop=({:.3f},{:.3f},{:.3f},{:.3f}) ui_left={:.3f} good={} inliers={} ground_blue={:.3f} arc_color={:.3f}".format(
                crop_roi[0],
                crop_roi[1],
                crop_roi[2],
                crop_roi[3],
                metrics.ui_left_feature_score,
                metrics.ui_left_good_matches,
                metrics.ui_left_inliers,
                metrics.ground_blue_ratio,
                metrics.arc_color_ratio,
            )
        )

        self.overlay_window.show()
        pix = QPixmap.fromImage(qimg).scaled(
            self.overlay_window_label.width(),
            self.overlay_window_label.height(),
            Qt.KeepAspectRatio,
            Qt.FastTransformation,
        )
        self.overlay_window_label.setPixmap(pix)

    def _on_overlay_toggled(self, checked: bool) -> None:
        if not checked:
            self.overlay_window.hide()
            return
        self.overlay_window.show()

    def _reflow_coord_grid(self, compact: bool) -> None:
        while self.coord_grid.count():
            item = self.coord_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

        if compact:
            self.coord_grid.addWidget(self.top_left_label, 0, 0)
            self.coord_grid.addWidget(self.top_left_edit, 0, 1)
            self.coord_grid.addWidget(self.top_right_label, 1, 0)
            self.coord_grid.addWidget(self.top_right_edit, 1, 1)
            self.coord_grid.addWidget(self.bottom_left_label, 2, 0)
            self.coord_grid.addWidget(self.bottom_left_edit, 2, 1)
            self.coord_grid.addWidget(self.bottom_right_label, 3, 0)
            self.coord_grid.addWidget(self.bottom_right_edit, 3, 1)
            return

        self.coord_grid.addWidget(self.top_left_label, 0, 0)
        self.coord_grid.addWidget(self.top_left_edit, 0, 1)
        self.coord_grid.addWidget(self.top_right_label, 0, 2)
        self.coord_grid.addWidget(self.top_right_edit, 0, 3)
        self.coord_grid.addWidget(self.bottom_left_label, 1, 0)
        self.coord_grid.addWidget(self.bottom_left_edit, 1, 1)
        self.coord_grid.addWidget(self.bottom_right_label, 1, 2)
        self.coord_grid.addWidget(self.bottom_right_edit, 1, 3)

    def _reflow_settings_layout(self) -> None:
        if not hasattr(self, "settings_grid"):
            return

        width = self.settings_page.width()
        compact = width < 900

        self.settings_grid.removeWidget(self.settings_basic_group)
        self.settings_grid.removeWidget(self.settings_vision_group)
        if compact:
            self.settings_grid.addWidget(self.settings_basic_group, 0, 0)
            self.settings_grid.addWidget(self.settings_vision_group, 1, 0)
        else:
            self.settings_grid.addWidget(self.settings_basic_group, 0, 0)
            self.settings_grid.addWidget(self.settings_vision_group, 0, 1)

        self.settings_grid.setColumnStretch(0, 1)
        self.settings_grid.setColumnStretch(1, 1 if not compact else 0)
        self._reflow_coord_grid(compact)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._reflow_settings_layout()

    def closeEvent(self, event) -> None:
        try:
            self.overlay_window.hide()
        except Exception:
            pass
        release_points = dict(self._active_touch_points)
        for pointer, pos in self._manual_stop_release_points.items():
            release_points.setdefault(pointer, pos)
        self._active_touch_points.clear()
        self._manual_stop_release_points.clear()
        self._manual_stop_release_pending = False
        self._force_release_touch_points(release_points)
        self._stop_vision_debug(log_message=False)
        self._stop_vision_stream()
        if self.controller is not None:
            try:
                self.controller.close()
            except Exception:
                pass
            self.controller = None
            self.controller_ready = False
        super().closeEvent(event)

    def _reset_auto_start_state(self, *, clear_timing: bool = True) -> None:
        self._auto_start_stage = "idle"
        self._auto_start_playback_armed = False
        if clear_timing:
            self._auto_start_detection_started_at = None
            self._auto_start_triggered_at = None
            self._auto_start_frame_timestamp = None
            self._auto_start_vision_total_ms = 0.0
        if self._auto_start_start_signal is not None:
            self._auto_start_start_signal.set()
        self._auto_start_start_signal = None

    def _disable_auto_start_one_shot(self, emit_log: bool, *, clear_timing: bool) -> None:
        was_checked = self.auto_start_cv_check.isChecked()
        self.auto_start_cv_check.blockSignals(True)
        self.auto_start_cv_check.setChecked(False)
        self.auto_start_cv_check.blockSignals(False)
        self._reset_auto_start_state(clear_timing=clear_timing)
        self._restart_vision_stream_if_needed("auto_start_disable")
        if emit_log and was_checked and not self._auto_start_shutdown_logged:
            self._append_log(self._t("auto_start_off"))
        self._auto_start_shutdown_logged = True

    def _on_vision_debug_source_changed(self, _index: int | None = None) -> None:
        source = str(self.vision_debug_source_combo.currentData())
        requires_path = source in {"image", "video"}
        self.vision_debug_path_edit.setEnabled(requires_path)
        self.vision_debug_browse_btn.setEnabled(requires_path)
        if self._vision_debug_running:
            self._restart_vision_stream_if_needed("debug_source_changed")

    def _browse_vision_debug_input(self) -> None:
        source = str(self.vision_debug_source_combo.currentData())
        if source == "image":
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                self._t("vision_debug_input"),
                "",
                "Image Files (*.png *.jpg *.jpeg *.bmp *.webp);;All Files (*.*)",
            )
        elif source == "video":
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                self._t("vision_debug_input"),
                "",
                "Video Files (*.mp4 *.mkv *.avi *.mov *.webm);;All Files (*.*)",
            )
        else:
            return
        if file_path:
            self.vision_debug_path_edit.setText(file_path)

    def _toggle_vision_debug(self, _checked: bool = False) -> None:
        if self._vision_debug_running:
            self._stop_vision_debug(log_message=True)
            return
        self._start_vision_debug()

    def _start_vision_debug(self) -> None:
        if not (
            self.vision_debug_ui_check.isChecked()
            or self.vision_debug_ground_check.isChecked()
            or self.vision_debug_arc_check.isChecked()
        ):
            self._append_log(self._t("vision_debug_no_stage"))
            return
        if self.auto_start_cv_check.isChecked():
            self._disable_auto_start_one_shot(
                emit_log=True,
                clear_timing=True,
            )

        source = str(self.vision_debug_source_combo.currentData())
        source_name = self.vision_debug_source_combo.currentText()
        self._stop_vision_debug(log_message=False)

        if source == "scrcpy":
            if not self.controller_ready or self.controller is None:
                self._append_log(self._t("vision_debug_source_not_ready"))
                return
            self._vision_debug_running = True
            self._restart_vision_stream_if_needed("debug_start")
        elif source == "image":
            path = Path(self.vision_debug_path_edit.text().strip())
            if not path.exists() or not path.is_file():
                self._append_log(self._t("vision_debug_invalid_path", path=str(path)))
                return
            frame = _read_image_unicode(path)
            if frame is None:
                self._append_log(self._t("vision_debug_open_fail", path=str(path)))
                return
            self._vision_debug_static_frame = frame
        elif source == "video":
            path = Path(self.vision_debug_path_edit.text().strip())
            if not path.exists() or not path.is_file():
                self._append_log(self._t("vision_debug_invalid_path", path=str(path)))
                return
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                self._append_log(self._t("vision_debug_open_fail", path=str(path)))
                try:
                    capture.release()
                except Exception:
                    pass
                return
            self._vision_debug_capture = capture

        self._vision_debug_running = True
        self._vision_debug_last_log_time = 0.0
        self._vision_debug_last_result = self._t(
            "vision_debug_running",
            source=source_name,
        )
        self.vision_debug_result_label.setText(self._vision_debug_last_result)
        self.vision_debug_toggle_btn.setText(self._t("vision_debug_stop"))
        self._append_log(self._t("vision_debug_started", source=source_name))

    def _stop_vision_debug(self, *, log_message: bool) -> None:
        running = self._vision_debug_running
        self._vision_debug_running = False
        self._vision_debug_overlay_stage = "debug-idle"
        self._vision_debug_static_frame = None
        if self._vision_debug_capture is not None:
            try:
                self._vision_debug_capture.release()
            except Exception:
                pass
        self._vision_debug_capture = None
        if hasattr(self, "vision_debug_toggle_btn"):
            self.vision_debug_toggle_btn.setText(self._t("vision_debug_start"))
        if hasattr(self, "vision_debug_result_label"):
            self.vision_debug_result_label.setText(self._t("vision_debug_idle"))
        if log_message and running:
            self._append_log(self._t("vision_debug_stopped"))
        self._restart_vision_stream_if_needed("debug_stop")
        if not self.overlay_debug_check.isChecked():
            self.overlay_window.hide()

    def _read_vision_debug_frame(
        self, controller_frame: np.ndarray | None
    ) -> tuple[np.ndarray | None, float | None]:
        if not self._vision_debug_running:
            return None, None

        source = str(self.vision_debug_source_combo.currentData())
        if source == "scrcpy":
            frame, fps = self._read_vision_live_frame()
            return frame, fps

        if source == "image":
            if self._vision_debug_static_frame is None:
                return None, None
            return self._vision_debug_static_frame.copy(), None

        if self._vision_debug_capture is None:
            return None, None
        ok, frame = self._vision_debug_capture.read()
        if not ok or frame is None:
            self._vision_debug_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._vision_debug_capture.read()
        if not ok or frame is None:
            return None, None
        return frame, None

    def _run_vision_debug_detection(self, frame: np.ndarray) -> None:
        runtime = self._build_vision_runtime()
        if str(self.vision_debug_source_combo.currentData()) != "scrcpy":
            runtime = self._build_vision_runtime(stream_crop_roi=(0.0, 0.0, 1.0, 1.0))
        self.vision_detector.set_runtime(runtime)
        stages: list[str] = []
        parts: list[str] = []

        if self.vision_debug_ui_check.isChecked():
            ui_pass = self.vision_detector.detect_ui_panel(frame)
            m = self.vision_detector.metrics
            p = self.vision_detector.perf
            stages.append("ui_stage")
            parts.append(
                "ui pass={} thr={:.3f} score={:.3f} good={} inliers={} ms={:.2f}".format(
                    int(ui_pass),
                    runtime.ui_feature_threshold,
                    m.ui_left_feature_score,
                    m.ui_left_good_matches,
                    m.ui_left_inliers,
                    p.total_ms,
                )
            )

        if self.vision_debug_ground_check.isChecked():
            logic_x = float(self.vision_debug_logic_x_spin.value())
            logic_y = float(self.vision_debug_logic_y_spin.value())
            ground_pass = self.vision_detector.detect_ground_overlap(
                frame, logic_x, logic_y
            )
            m = self.vision_detector.metrics
            p = self.vision_detector.perf
            stages.append("ground")
            parts.append(
                "ground pass={} thr={:.3f} ratio={:.3f} px={}/{} logic_x={:.3f} ms={:.2f}".format(
                    int(ground_pass),
                    runtime.ground_blue_ratio_threshold,
                    m.ground_blue_ratio,
                    m.ground_blue_pixels,
                    m.ground_roi_pixels,
                    logic_x,
                    p.total_ms,
                )
            )

        if self.vision_debug_arc_check.isChecked():
            logic_x = float(self.vision_debug_logic_x_spin.value())
            logic_y = float(self.vision_debug_logic_y_spin.value())
            arc_pass = self.vision_detector.detect_arc_overlap(frame, logic_x, logic_y)
            m = self.vision_detector.metrics
            p = self.vision_detector.perf
            stages.append("arc_pass")
            parts.append(
                "arc pass={} thr={:.3f} ratio={:.3f} dist={:.1f} logic=({:.3f},{:.3f}) ms={:.2f}".format(
                    int(arc_pass),
                    runtime.arc_color_ratio_threshold,
                    m.arc_color_ratio,
                    m.arc_target_distance,
                    logic_x,
                    logic_y,
                    p.total_ms,
                )
            )

        self._vision_debug_overlay_stage = (
            "debug:" + "+".join(stages) if stages else "debug:none"
        )
        result_text = " | ".join(parts) if parts else self._t("vision_debug_no_stage")
        self._vision_debug_last_result = result_text
        self.vision_debug_result_label.setText(result_text)

        now = time.perf_counter()
        if now - self._vision_debug_last_log_time >= 1.0:
            self._append_log(f"[VISION-DEBUG] {result_text}")
            self._vision_debug_last_log_time = now

    def _poll_auto_start(self) -> None:
        if not self.auto_start_cv_check.isChecked():
            self._reset_auto_start_state()
            return
        if self.prepared is None:
            return
        if not self.vision_controller_ready or self.vision_controller is None:
            self._restart_vision_stream_if_needed("auto_start_poll")
            return
        if (
            self.worker is not None
            and self.worker.isRunning()
            and not self._auto_start_playback_armed
        ):
            return

        frame = self.vision_controller.get_latest_frame(copy_frame=True)
        if frame is None:
            return
        frame_timestamp = None
        if hasattr(self.vision_controller, "get_latest_frame_timestamp"):
            frame_timestamp = self.vision_controller.get_latest_frame_timestamp()
        self._auto_start_frame_timestamp = frame_timestamp

        if self._auto_start_stage == "idle":
            self._append_log(self._t("auto_start_wait"))
            self._auto_start_stage = "wait_ui"
            self._arm_auto_start_playback()

        if self._auto_start_stage == "wait_ui":
            self._auto_start_detection_started_at = time.perf_counter()
            if self._is_gameplay_screen(frame):
                first_types = set(self.prepared.first_note_types)
                if first_types and first_types.issubset({"arc", "arctap", "zero_arc"}):
                    self._append_log(self._t("auto_start_ready"))
                    self._auto_start_stage = "wait_arc"
                    return
                self._append_log(self._t("auto_start_ready"))
                self._auto_start_stage = "wait_ground"
            return

        if self._auto_start_stage == "wait_arc":
            self._auto_start_detection_started_at = time.perf_counter()
            if self._is_arc_cap_triggered(frame):
                self._auto_start_vision_total_ms = self.vision_detector.perf.total_ms
                self._auto_start_triggered_at = time.perf_counter()
                self._append_log(self._t("auto_start_fire"))
                self._release_auto_start_playback()
                self._disable_auto_start_one_shot(
                    emit_log=True,
                    clear_timing=False,
                )
            return

        if self._auto_start_stage == "wait_ground":
            self._auto_start_detection_started_at = time.perf_counter()
            if self._is_first_ground_note_on_judgment(frame):
                self._auto_start_vision_total_ms = self.vision_detector.perf.total_ms
                self._auto_start_triggered_at = time.perf_counter()
                self._append_log(self._t("auto_start_fire"))
                self._release_auto_start_playback()
                self._disable_auto_start_one_shot(
                    emit_log=True,
                    clear_timing=False,
                )

    def _on_auto_start_toggled(self, checked: bool) -> None:
        if not checked:
            self._reset_auto_start_state()
            self._restart_vision_stream_if_needed("auto_start_off")
            return
        self._auto_start_shutdown_logged = False
        self._restart_vision_stream_if_needed("auto_start_on")

    def _poll_overlay(self) -> None:
        has_new_controller_frame = False
        controller_frame = None
        controller_fps = None
        if self.controller is not None:
            latest_seq = (
                self.controller.get_latest_frame_seq()
                if hasattr(self.controller, "get_latest_frame_seq")
                else self._auto_start_last_frame_seq
            )
            if latest_seq != self._auto_start_last_frame_seq:
                self._auto_start_last_frame_seq = latest_seq
                controller_frame = self.controller.get_latest_frame(copy_frame=True)
                has_new_controller_frame = controller_frame is not None
            controller_fps = self.controller.get_decode_fps()

        has_new_vision_frame = False
        if self.vision_controller is not None and self.vision_controller_ready:
            latest_vision_seq = (
                self.vision_controller.get_latest_frame_seq()
                if hasattr(self.vision_controller, "get_latest_frame_seq")
                else self._vision_last_frame_seq
            )
            if latest_vision_seq != self._vision_last_frame_seq:
                self._vision_last_frame_seq = latest_vision_seq
                has_new_vision_frame = True

        if (
            has_new_vision_frame
            and not self._vision_debug_running
            and self.auto_start_cv_check.isChecked()
            and self.prepared is not None
        ):
            self.auto_start_frame_ready.emit()
        elif self.auto_start_cv_check.isChecked() and self.prepared is not None:
            self._restart_vision_stream_if_needed("overlay_auto_start")

        if self._vision_debug_running:
            debug_frame, debug_fps = self._read_vision_debug_frame(controller_frame)
            if debug_frame is not None:
                self._run_vision_debug_detection(debug_frame)
                overlay_frame = debug_frame
                overlay_fps = debug_fps
                if str(self.vision_debug_source_combo.currentData()) == "scrcpy":
                    if controller_frame is not None:
                        overlay_frame = controller_frame
                    overlay_fps = controller_fps
                self._update_overlay_preview(
                    overlay_frame,
                    stage=self._vision_debug_overlay_stage,
                    force_show=True,
                    run_detection=False,
                    decode_fps=overlay_fps,
                )
            return

        if not self.overlay_debug_check.isChecked():
            return
        if controller_frame is None:
            return
        self._update_overlay_preview(
            controller_frame,
            stage=self._auto_start_stage,
            force_show=False,
            run_detection=False,
            decode_fps=controller_fps,
        )

    def _on_language_changed(self) -> None:
        self.locale = self.language_combo.currentData()
        self._apply_texts()

    def _choose_chart_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, self._t("select_aff"), "", "All Files (*.*)"
        )
        if file_path:
            self.chart_path_edit.setText(file_path)
            self._on_save_clicked()

    def _load_config_to_form(self) -> None:
        self.app_config = load_app_config()
        cfg = self.app_config.global_config
        vision = self.app_config.vision
        self.chart_path_edit.setText(cfg.chart_path)
        self.top_left_edit.setText(_coord_to_text(cfg.top_left))
        self.top_right_edit.setText(_coord_to_text(cfg.top_right))
        self.bottom_left_edit.setText(_coord_to_text(cfg.bottom_left))
        self.bottom_right_edit.setText(_coord_to_text(cfg.bottom_right))
        self.step_spin.setValue(cfg.fine_tune_step)
        if cfg.designant_choice is None:
            self.designant_combo.setCurrentIndex(0)
        elif cfg.designant_choice:
            self.designant_combo.setCurrentIndex(1)
        else:
            self.designant_combo.setCurrentIndex(2)
        self.delay_label.setText(f"{self.app_config.delay:.3f}s")

        self.stream_max_fps = int(vision.stream_max_fps)
        self.stream_bitrate_enabled = bool(vision.stream_bitrate_enabled)
        self.stream_bitrate_mbps = int(vision.stream_bitrate_mbps)
        self._vision_perf_success_only = bool(vision.perf_log_on_success_only)
        self.stream_fps_spin.setValue(self.stream_max_fps)
        self.stream_bitrate_enable_check.setChecked(self.stream_bitrate_enabled)
        self.stream_bitrate_spin.setValue(self.stream_bitrate_mbps)
        self.overlay_debug_check.setChecked(bool(vision.overlay_enabled))
        self.vision_perf_success_only_check.setChecked(
            bool(vision.perf_log_on_success_only)
        )
        self.ui_template_threshold_spin.setValue(float(vision.ui_template_threshold))
        self.ground_blue_ratio_threshold_spin.setValue(
            float(vision.ground_blue_ratio_threshold)
        )
        self.arc_color_ratio_threshold_spin.setValue(
            float(vision.arc_color_ratio_threshold)
        )
        self.arc_logic_roi_half_x_spin.setValue(float(vision.arc_logic_roi_half_x))
        self.arc_logic_roi_half_y_spin.setValue(float(vision.arc_logic_roi_half_y))
        self.ui_left_x0_spin.setValue(float(vision.ui_left_roi[0]))
        self.ui_left_y0_spin.setValue(float(vision.ui_left_roi[1]))
        self.ui_left_x1_spin.setValue(float(vision.ui_left_roi[2]))
        self.ui_left_y1_spin.setValue(float(vision.ui_left_roi[3]))
        self.ground_x0_spin.setValue(float(vision.ground_roi[0]))
        self.ground_y0_spin.setValue(float(vision.ground_roi[1]))
        self.ground_x1_spin.setValue(float(vision.ground_roi[2]))
        self.ground_y1_spin.setValue(float(vision.ground_roi[3]))
        self._on_vision_debug_source_changed()

    def _save_form_to_config(self, *, emit_log: bool = True) -> bool:
        cfg = self.app_config.global_config
        vision = self.app_config.vision
        try:
            cfg.chart_path = self.chart_path_edit.text().strip()
            cfg.top_left = _parse_coord(self.top_left_edit.text(), "top_left")
            cfg.top_right = _parse_coord(self.top_right_edit.text(), "top_right")
            cfg.bottom_left = _parse_coord(self.bottom_left_edit.text(), "bottom_left")
            cfg.bottom_right = _parse_coord(
                self.bottom_right_edit.text(), "bottom_right"
            )
            cfg.fine_tune_step = int(self.step_spin.value())

            vision.stream_max_fps = int(self.stream_fps_spin.value())
            vision.stream_bitrate_enabled = bool(
                self.stream_bitrate_enable_check.isChecked()
            )
            vision.stream_bitrate_mbps = int(self.stream_bitrate_spin.value())
            vision.overlay_enabled = bool(self.overlay_debug_check.isChecked())
            vision.perf_log_on_success_only = bool(
                self.vision_perf_success_only_check.isChecked()
            )
            self._vision_perf_success_only = vision.perf_log_on_success_only
            vision.ui_template_threshold = float(
                self.ui_template_threshold_spin.value()
            )
            vision.ground_blue_ratio_threshold = float(
                self.ground_blue_ratio_threshold_spin.value()
            )
            vision.arc_color_ratio_threshold = float(
                self.arc_color_ratio_threshold_spin.value()
            )
            vision.arc_logic_roi_half_x = float(self.arc_logic_roi_half_x_spin.value())
            vision.arc_logic_roi_half_y = float(self.arc_logic_roi_half_y_spin.value())
            vision.overlay_detached = True
            vision.ui_left_roi = (
                float(self.ui_left_x0_spin.value()),
                float(self.ui_left_y0_spin.value()),
                float(self.ui_left_x1_spin.value()),
                float(self.ui_left_y1_spin.value()),
            )
            vision.ground_roi = (
                float(self.ground_x0_spin.value()),
                float(self.ground_y0_spin.value()),
                float(self.ground_x1_spin.value()),
                float(self.ground_y1_spin.value()),
            )
        except ValueError as exc:
            QMessageBox.critical(self, self._t("config_error"), str(exc))
            return False

        cfg.designant_choice = {0: None, 1: True, 2: False}[
            self.designant_combo.currentIndex()
        ]
        save_app_config(self.app_config)
        if emit_log:
            self._append_log(self._t("log_config_saved"))
        return True

    def _collect_run_config(self, *, save_form: bool = True) -> RunConfig | None:
        if save_form and not self._save_form_to_config(emit_log=False):
            return None
        cfg = self.app_config.global_config
        chart_path = cfg.chart_path.strip()
        if not chart_path:
            QMessageBox.warning(
                self, self._t("missing_chart_title"), self._t("missing_chart")
            )
            return None
        if not Path(chart_path).exists():
            QMessageBox.warning(
                self,
                self._t("missing_chart_title"),
                self._t("chart_not_found", path=chart_path),
            )
            return None

        try:
            chart_content = Path(chart_path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            QMessageBox.critical(
                self, self._t("config_error"), self._t("read_error", error=exc)
            )
            return None

        designant_choice = cfg.designant_choice
        if has_designant_notes(chart_content) and designant_choice is None:
            answer = QMessageBox.question(
                self,
                self._t("designant_title"),
                self._t("designant_question"),
                QMessageBox.Yes | QMessageBox.No,
            )
            designant_choice = answer == QMessageBox.Yes
            cfg.designant_choice = designant_choice
            save_app_config(self.app_config)

        return RunConfig(
            chart_path=chart_path,
            bottom_left=cfg.bottom_left,
            top_left=cfg.top_left,
            top_right=cfg.top_right,
            bottom_right=cfg.bottom_right,
            fine_tune_step=cfg.fine_tune_step,
            designant_choice=designant_choice,
        )

    def _collect_form_config_fast(self) -> RunConfig | None:
        try:
            chart_path = self.chart_path_edit.text().strip()
            top_left = _parse_coord(self.top_left_edit.text(), "top_left")
            top_right = _parse_coord(self.top_right_edit.text(), "top_right")
            bottom_left = _parse_coord(self.bottom_left_edit.text(), "bottom_left")
            bottom_right = _parse_coord(self.bottom_right_edit.text(), "bottom_right")
            step = int(self.step_spin.value())
        except ValueError as exc:
            QMessageBox.critical(self, self._t("config_error"), str(exc))
            return None
        designant_choice = {0: None, 1: True, 2: False}[
            self.designant_combo.currentIndex()
        ]
        return RunConfig(
            chart_path=chart_path,
            bottom_left=bottom_left,
            top_left=top_left,
            top_right=top_right,
            bottom_right=bottom_right,
            fine_tune_step=step,
            designant_choice=designant_choice,
        )

    def _set_startup_state(self, state: StartupPipelineState) -> None:
        self._startup_state = state

    def _sync_stream_settings_from_form(self) -> None:
        self.stream_max_fps = int(self.stream_fps_spin.value())
        self.stream_bitrate_enabled = bool(self.stream_bitrate_enable_check.isChecked())
        self.stream_bitrate_mbps = int(self.stream_bitrate_spin.value())

    def _start_startup_pipeline(
        self,
        reason: str,
        *,
        prepare_after_warmup: bool,
        force_touch_restart: bool,
    ) -> None:
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(
                self,
                self._t("error"),
                "Stop playback before restarting ADB/scrcpy channels.",
            )
            return
        if self.prepare_worker is not None and self.prepare_worker.isRunning():
            self._queued_pipeline_reason = reason
            self._queued_pipeline_prepare_after_warmup = prepare_after_warmup
            self._queued_pipeline_force_touch_restart = force_touch_restart
            self._append_log(
                "[INFO] Prepare is running; queued startup pipeline restart after prepare finishes."
            )
            return
        self._prepare_requested_after_warmup = prepare_after_warmup
        self._sync_stream_settings_from_form()
        self._start_warmup(force_restart=force_touch_restart, reason=reason)

    def _run_queued_startup_pipeline_if_needed(self) -> bool:
        if self._queued_pipeline_reason is None:
            return False
        reason = self._queued_pipeline_reason
        prepare_after_warmup = self._queued_pipeline_prepare_after_warmup
        force_touch_restart = self._queued_pipeline_force_touch_restart
        self._queued_pipeline_reason = None
        self._queued_pipeline_prepare_after_warmup = False
        self._queued_pipeline_force_touch_restart = False
        self._append_log(f"[INFO] Running queued startup pipeline restart ({reason}).")
        self._start_startup_pipeline(
            reason,
            prepare_after_warmup=prepare_after_warmup,
            force_touch_restart=force_touch_restart,
        )
        return True

    def _start_warmup(self, *, force_restart: bool, reason: str) -> None:
        if self.warmup_worker is not None and self.warmup_worker.isRunning():
            if force_restart:
                self._warmup_restart_requested = True
                self._append_log(
                    f"[INFO] Warmup in progress; queued another warmup restart ({reason})."
                )
            return
        if force_restart:
            self._stop_vision_stream()
            if self.controller is not None:
                try:
                    self.controller.close()
                except Exception:
                    pass
            self.controller = None
            self.controller_ready = False
            self.controller_state_label.setText(self._t("warming"))
        self._set_startup_state(StartupPipelineState.WARMUP)
        bit_rate = None
        if self.stream_bitrate_enabled:
            bit_rate = int(self.stream_bitrate_mbps) * 1_000_000
        self.warmup_worker = ControllerWarmupWorker(
            max_fps=self.stream_max_fps,
            video_bit_rate=bit_rate,
            video_crop=None,
        )
        self.warmup_worker.started_warmup.connect(self._on_warmup_start)
        self.warmup_worker.warmup_ok.connect(self._on_warmup_ok)
        self.warmup_worker.warmup_fail.connect(self._on_warmup_fail)
        self.warmup_worker.start()

    def _request_prepare(self, *, save_form: bool = True) -> None:
        if self.prepare_worker is not None and self.prepare_worker.isRunning():
            return
        run_config = self._collect_run_config(save_form=save_form)
        if run_config is None:
            self._set_startup_state(StartupPipelineState.ERROR)
            self.prepare_state_label.setText(self._t("error"))
            return

        self._set_startup_state(StartupPipelineState.PREPARE)
        self.prepare_worker = PrepareWorker(run_config)
        self.prepare_worker.started_prepare.connect(self._on_prepare_start)
        self.prepare_worker.prepared_ok.connect(self._on_prepare_ok)
        self.prepare_worker.prepared_fail.connect(self._on_prepare_fail)
        self.prepare_worker.start()

    def _on_save_clicked(self) -> None:
        if not self._save_form_to_config(emit_log=True):
            return
        self._append_log(
            "[INFO] Config saved; restarting warmup pipeline (ADB warmup -> chart prepare)."
        )
        self._start_startup_pipeline(
            "save",
            prepare_after_warmup=True,
            force_touch_restart=True,
        )

    def _on_reload_clicked(self) -> None:
        self._load_config_to_form()
        self._append_log(
            "[INFO] Config reloaded; restarting warmup pipeline (ADB warmup -> chart prepare)."
        )
        self._start_startup_pipeline(
            "reload",
            prepare_after_warmup=True,
            force_touch_restart=True,
        )

    def _set_running_ui(self, running: bool) -> None:
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.plus_btn.setEnabled(running)
        self.minus_btn.setEnabled(running)
        self.reset_btn.setEnabled(running)

    def _refresh_offset(self) -> None:
        if self.worker is None:
            self.offset_label.setText("0.000s")
            return
        self.offset_label.setText(f"{self.worker.current_offset():.3f}s")

    def _start_playback(self) -> None:
        self._start_playback_internal(start_armed=False)

    def _start_playback_internal(self, start_armed: bool) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        if not self.controller_ready or self.controller is None:
            QMessageBox.warning(self, self._t("error"), self._t("controller_not_ready"))
            return

        current_cfg = self._collect_form_config_fast()
        if current_cfg is None:
            return
        if self.prepared is None:
            QMessageBox.warning(self, self._t("error"), self._t("prepare_not_ready"))
            return
        if self.prepared.config_key != _build_config_key(current_cfg):
            self._request_prepare()
            QMessageBox.information(self, self._t("error"), self._t("prepare_mismatch"))
            return

        start_signal = None
        if start_armed:
            start_signal = threading.Event()
            self._auto_start_start_signal = start_signal

        self.worker = PlaybackWorker(
            self.prepared,
            self.controller,
            debug_verbose=self.debug_verbose_check.isChecked(),
            optimize_high_priority=self.opt_high_prio_check.isChecked(),
            optimize_timer_resolution=False,
            start_signal=start_signal,
        )
        self.worker.log_message.connect(self._append_log)
        self.worker.started_playback.connect(self._on_play_started)
        self.worker.finished_playback.connect(self._on_play_finished)
        self.worker.progress.connect(self._on_progress)
        self.worker.first_dispatch_metrics.connect(self._on_first_dispatch_metrics)

        self._append_log(self._t("log_play_start"))
        if self.debug_verbose_check.isChecked():
            snapshot_path = self._write_touch_event_snapshot()
            if snapshot_path is not None:
                self._append_log(f"[DEBUG] Touch snapshot written: {snapshot_path}")
        self._active_touch_points.clear()
        self._manual_stop_release_points.clear()
        self._manual_stop_release_pending = False
        self._start_click_time = time.perf_counter()
        self._first_dispatch_logged = False
        self.worker.start()

    def _arm_auto_start_playback(self) -> None:
        if self._auto_start_playback_armed:
            return
        self._auto_start_playback_armed = True
        self._start_playback_internal(start_armed=True)

    def _release_auto_start_playback(self) -> None:
        if self._auto_start_start_signal is not None:
            self._auto_start_start_signal.set()
        self._auto_start_start_signal = None
        self._auto_start_playback_armed = False

    def _stop_playback(self) -> None:
        if self.worker is None:
            return
        self._append_log(self._t("log_stop"))
        if self._auto_start_start_signal is not None:
            self._auto_start_start_signal.set()
        if self.auto_start_cv_check.isChecked():
            self._disable_auto_start_one_shot(
                emit_log=True,
                clear_timing=True,
            )
        self._manual_stop_release_points = dict(self._active_touch_points)
        self._manual_stop_release_pending = bool(self._manual_stop_release_points)
        if self._manual_stop_release_pending:
            self._append_log(
                f"[INFO] Manual stop captured {len(self._manual_stop_release_points)} active pointer(s); reset UP events will be sent after playback exits."
            )
        self.worker.stop_playback()

    def _fine_tune_plus(self) -> None:
        if self.worker is not None:
            self.worker.nudge_plus()

    def _fine_tune_minus(self) -> None:
        if self.worker is not None:
            self.worker.nudge_minus()

    def _fine_tune_reset(self) -> None:
        if self.worker is not None:
            self.worker.reset_offset()

    def _on_play_started(self) -> None:
        self.run_state_label.setText(self._t("running"))
        self._set_running_ui(True)
        self.offset_timer.start()

    def _on_play_finished(self, success: bool, _token: str) -> None:
        self.worker = None
        self.offset_timer.stop()
        self._set_running_ui(False)
        self.run_state_label.setText(self._t("idle") if success else self._t("error"))
        if success:
            self._append_log(self._t("log_play_finish"))
        if self._manual_stop_release_pending:
            release_points = dict(self._manual_stop_release_points)
            self._manual_stop_release_pending = False
            self._manual_stop_release_points.clear()
            QTimer.singleShot(
                200,
                lambda points=release_points: self._force_release_touch_points(points),
            )
        else:
            self._manual_stop_release_points.clear()
        self._active_touch_points.clear()
        self._reset_auto_start_state()

    def _on_warmup_start(self) -> None:
        self._set_startup_state(StartupPipelineState.WARMUP)
        self.controller_state_label.setText(self._t("warming"))
        self._append_log(self._t("log_warm_start"))

    def _on_warmup_ok(self, controller) -> None:
        self.warmup_worker = None
        if self._warmup_restart_requested:
            self._warmup_restart_requested = False
            try:
                controller.close()
            except Exception:
                pass
            self._append_log(
                "[INFO] Applying queued warmup restart with latest stream parameters."
            )
            self._start_warmup(force_restart=True, reason="queued_restart")
            return
        self.controller = controller
        self.controller_ready = True
        reported_size = (
            int(getattr(controller, "device_width", 0)),
            int(getattr(controller, "device_height", 0)),
        )
        if reported_size[0] > 0 and reported_size[1] > 0:
            self._native_device_size = reported_size
        self.controller_state_label.setText(self._t("ready"))
        self._append_log(self._t("log_warm_ok"))
        need_prepare = self._prepare_requested_after_warmup or self.prepared is None
        self._prepare_requested_after_warmup = False
        if need_prepare:
            self._request_prepare(save_form=False)
        else:
            self._set_startup_state(StartupPipelineState.READY)
            self._restart_vision_stream_if_needed("touch_warmup")

    def _on_warmup_fail(self, error: str) -> None:
        self.controller = None
        self.controller_ready = False
        self.warmup_worker = None
        self._prepare_requested_after_warmup = False
        self._warmup_restart_requested = False
        self._set_startup_state(StartupPipelineState.ERROR)
        self._stop_vision_stream()
        self.controller_state_label.setText(self._t("error"))
        self._append_log(self._t("log_warm_fail", error=error))

    def _on_vision_warmup_start(self) -> None:
        self._append_log("[INFO] Starting vision scrcpy stream...")

    def _on_vision_warmup_ok(self, controller) -> None:
        self.vision_controller = controller
        self.vision_controller_ready = True
        self._vision_last_frame_seq = -1
        self.vision_warmup_worker = None
        self._append_log("[INFO] Vision scrcpy stream ready")

    def _on_vision_warmup_fail(self, error: str) -> None:
        self.vision_controller = None
        self.vision_controller_ready = False
        self._vision_last_frame_seq = -1
        self.vision_warmup_worker = None
        self._append_log(f"[ERROR] Vision scrcpy stream failed: {error}")

    def _on_prepare_start(self) -> None:
        self._set_startup_state(StartupPipelineState.PREPARE)
        self.prepare_state_label.setText(self._t("warming"))
        self._append_log(self._t("log_prepare_start"))

    def _on_prepare_ok(self, prepared: PreparedRunData) -> None:
        self.prepare_worker = None
        self.prepared = prepared
        self.app_config.delay = prepared.delay
        save_app_config(self.app_config)
        if prepared.first_note_logic_pos is not None:
            self.vision_debug_logic_x_spin.setValue(
                float(prepared.first_note_logic_pos[0])
            )
            self.vision_debug_logic_y_spin.setValue(
                float(prepared.first_note_logic_pos[1])
            )
        elif prepared.first_ground_logic_pos is not None:
            self.vision_debug_logic_x_spin.setValue(
                float(prepared.first_ground_logic_pos[0])
            )
            self.vision_debug_logic_y_spin.setValue(
                float(prepared.first_ground_logic_pos[1])
            )
        elif prepared.first_ground_logic_x is not None:
            self.vision_debug_logic_x_spin.setValue(float(prepared.first_ground_logic_x))
        self.delay_label.setText(f"{prepared.delay:.3f}s")
        self.prepare_state_label.setText(self._t("ready"))
        self._append_log(
            self._t(
                "log_prepare_ok",
                count=len(prepared.events_by_time),
                delay=prepared.delay,
            )
        )
        self._set_startup_state(StartupPipelineState.READY)
        if self._run_queued_startup_pipeline_if_needed():
            return
        self._restart_vision_stream_if_needed("prepare")

    def _on_prepare_fail(self, token: str) -> None:
        self.prepare_worker = None
        self.prepared = None
        self._set_startup_state(StartupPipelineState.ERROR)
        self.prepare_state_label.setText(self._t("error"))
        if token.startswith("read_error:"):
            msg = self._t("read_error", error=token.split(":", 1)[1])
        elif token.startswith("parse_error:"):
            msg = token.split(":", 1)[1]
        elif token == "delay_error":
            msg = "delay detect failed"
        elif token == "event_error":
            msg = "event build failed"
        else:
            msg = token
        self._append_log(self._t("log_prepare_fail", error=msg))
        if self._run_queued_startup_pipeline_if_needed():
            return
        self._restart_vision_stream_if_needed("prepare_fail")


def run_gui() -> None:
    app = QApplication.instance() or QApplication([])
    window = AutoPlayWindow()
    window.show()
    app.exec()
