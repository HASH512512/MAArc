from __future__ import annotations

import json
from pathlib import Path

from autoplay.domain.config import AppConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
CONFIG_FILE = CONFIG_DIR / "app.json"
CHART_CONFIG_FILE = CONFIG_DIR / "chart.json"
PLAYBACK_CONFIG_FILE = CONFIG_DIR / "playback.json"
CALIBRATION_CONFIG_FILE = CONFIG_DIR / "calibration.json"
VISION_CONFIG_FILE = CONFIG_DIR / "vision.json"
LEGACY_CONFIG_FILE = REPO_ROOT / "auto_arcaea_config.json"


def _resolve_config_path(config_file: str | Path = CONFIG_FILE) -> Path:
    path = Path(config_file)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _load_json_file(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain an object: {path}")
    return payload


def _split_app_config(config: AppConfig) -> dict[str, dict]:
    payload = config.to_dict()
    return {
        "app": {"schema_version": payload.get("schema_version", 2)},
        "chart": payload.get("chart", {}),
        "playback": payload.get("playback", {}),
        "calibration": payload.get("calibration", {}),
        "vision": payload.get("vision", {}),
    }


def _compose_split_payload(parts: dict[str, dict]) -> dict:
    return {
        "schema_version": parts.get("app", {}).get("schema_version", 2),
        "chart": parts.get("chart", {}),
        "playback": parts.get("playback", {}),
        "calibration": parts.get("calibration", {}),
        "vision": parts.get("vision", {}),
    }


def _load_split_config() -> AppConfig:
    parts = {
        "app": _load_json_file(CONFIG_FILE),
        "chart": _load_json_file(CHART_CONFIG_FILE),
        "playback": _load_json_file(PLAYBACK_CONFIG_FILE),
        "calibration": _load_json_file(CALIBRATION_CONFIG_FILE),
        "vision": _load_json_file(VISION_CONFIG_FILE),
    }
    return AppConfig.from_dict(_compose_split_payload(parts))


def _migrate_legacy_if_needed() -> None:
    if CONFIG_FILE.exists():
        return
    if not LEGACY_CONFIG_FILE.exists():
        return
    try:
        payload = _load_json_file(LEGACY_CONFIG_FILE)
        config = AppConfig.from_dict(payload)
        save_app_config(config)
    except Exception:
        return


def load_app_config(config_file: str | Path = CONFIG_FILE) -> AppConfig:
    config_path = _resolve_config_path(config_file)
    try:
        if config_path == CONFIG_FILE:
            _migrate_legacy_if_needed()
            if CONFIG_FILE.exists():
                return _load_split_config()
            return AppConfig.default()
        payload = _load_json_file(config_path)
        return AppConfig.from_dict(payload)
    except FileNotFoundError:
        return AppConfig.default()
    except (json.JSONDecodeError, OSError, ValueError, TypeError):
        return AppConfig.default()


def save_app_config(config: AppConfig, config_file: str | Path = CONFIG_FILE) -> None:
    config_path = _resolve_config_path(config_file)
    if config_path == CONFIG_FILE:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        parts = _split_app_config(config)
        targets = {
            CONFIG_FILE: parts["app"],
            CHART_CONFIG_FILE: parts["chart"],
            PLAYBACK_CONFIG_FILE: parts["playback"],
            CALIBRATION_CONFIG_FILE: parts["calibration"],
            VISION_CONFIG_FILE: parts["vision"],
        }
        for path, payload in targets.items():
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
        return
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2, ensure_ascii=False)
