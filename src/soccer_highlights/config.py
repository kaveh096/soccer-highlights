"""Configuration loading: YAML file + SOCCER_HL__ env var overrides."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

ENV_PREFIX = "SOCCER_HL__"


@dataclass
class InputConfig:
    source_dir: str = "D:/DCIM/DJI_001"
    use_lrf_for_detection: bool = True


@dataclass
class AudioConfig:
    sample_rate: int = 22050
    mono: bool = True


@dataclass
class RmsEnergyConfig:
    window_seconds: float = 0.5
    hop_seconds: float = 0.1
    baseline_window_seconds: float = 60.0
    threshold_sigma: float = 3.0
    min_absolute_dbfs: float = -40.0
    # Minimum required excess (in dB) above the adaptive threshold for a run
    # to count as an event. The adaptive threshold alone is too sensitive:
    # across a couple thousand correlated frames, ordinary noise fluctuations
    # cross a 3-sigma threshold by chance often enough to produce spurious
    # near-zero-excess "events". Real acoustic events (cheers, strikes) clear
    # this by a wide margin, so this floor filters statistical noise without
    # touching genuine detections.
    min_score_dbfs: float = 3.0


@dataclass
class OnsetFluxConfig:
    window_seconds: float = 0.05
    hop_seconds: float = 0.01
    baseline_window_seconds: float = 30.0
    threshold_sigma: float = 2.5
    # Same purpose as RmsEnergyConfig.min_score_dbfs, but flux has no fixed
    # physical unit -- tune this relative to the flux magnitudes your own
    # recordings produce (check the debug plot).
    min_score: float = 0.0


@dataclass
class DetectionConfig:
    strategy: str = "rms_energy"
    rms_energy: RmsEnergyConfig = field(default_factory=RmsEnergyConfig)
    onset_flux: OnsetFluxConfig = field(default_factory=OnsetFluxConfig)


@dataclass
class TimelineConfig:
    lookback_seconds: float = 45.0
    post_peak_seconds: float = 8.0
    min_gap_seconds: float = 5.0
    min_interval_seconds: float = 5.0
    # Ignore any peak before this point in the whole session -- camera
    # handling/setup noise at recording start isn't a real event.
    warmup_seconds: float = 10.0


@dataclass
class OutputConfig:
    mode: str = "clips"
    dir: str = "output"
    force_reencode_on_concat: bool = False


@dataclass
class MetadataConfig:
    events_path: str = "output/events.json"
    debug_plot_path: str = "output/debug_audio.png"


@dataclass
class ReviewConfig:
    # Review clips are always sourced from the small .LRF proxy, never the
    # full-res source -- far cheaper to decode/re-encode, and plenty for
    # judging whether a detection was a true or false positive.
    output_root: str = "output/review"
    max_width: int = 640
    fps: float = 15.0
    crf: int = 30
    # "ultrafast" despite the name is the LIGHTEST x264 preset on the CPU
    # (worse compression, more CPU time saved) -- picked deliberately after
    # this machine crashed twice overnight under sustained encode load.
    # Compression ratio doesn't matter for small low-res review clips.
    preset: str = "ultrafast"
    # Cap thread count so encoding doesn't pin every core at once; reduces
    # sustained thermal/power load on an older laptop running unattended.
    threads: int = 2
    audio_bitrate_kbps: int = 64
    # Negative-space clips: gaps not covered by ANY strategy's candidate
    # intervals, chunked for review so nothing is silently missed.
    max_negative_clip_seconds: float = 120.0
    min_negative_clip_seconds: float = 8.0


@dataclass
class Config:
    input: InputConfig = field(default_factory=InputConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    timeline: TimelineConfig = field(default_factory=TimelineConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
    review: ReviewConfig = field(default_factory=ReviewConfig)


def _apply_dict(obj: Any, data: dict[str, Any]) -> None:
    """Recursively overlay a nested dict onto a dataclass instance in place."""
    for key, value in data.items():
        if not hasattr(obj, key):
            raise ValueError(f"Unknown config key: {key!r}")
        current = getattr(obj, key)
        if isinstance(value, dict) and is_dataclass(current):
            _apply_dict(current, value)
        else:
            setattr(obj, key, value)


def _apply_env_overrides(cfg: Config) -> None:
    """Apply SOCCER_HL__SECTION__FIELD=value overrides from the environment."""
    for env_key, raw_value in os.environ.items():
        if not env_key.startswith(ENV_PREFIX):
            continue
        path = env_key[len(ENV_PREFIX) :].lower().split("__")
        obj: Any = cfg
        for part in path[:-1]:
            obj = getattr(obj, part)
        leaf = path[-1]
        current_value = getattr(obj, leaf)
        setattr(obj, leaf, _coerce(raw_value, type(current_value)))


def _coerce(raw_value: str, target_type: type) -> Any:
    if target_type is bool:
        return raw_value.strip().lower() in {"1", "true", "yes", "on"}
    if target_type in (int, float, str):
        return target_type(raw_value)
    return raw_value


def load_config(path: str | Path | None = None) -> Config:
    """Load config from a YAML file (defaulting to config/default.yaml),
    then apply any SOCCER_HL__ environment variable overrides."""
    cfg = Config()
    yaml_path = Path(path) if path else Path(__file__).resolve().parents[2] / "config" / "default.yaml"
    if yaml_path.exists():
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        _apply_dict(cfg, data)
    _apply_env_overrides(cfg)
    return cfg


def load_strategy_configs(base_cfg: Config, path: str | Path | None = None) -> dict[str, Config]:
    """Load config/strategies.yaml, applying each named strategy's overrides
    on top of a deep copy of base_cfg."""
    yaml_path = Path(path) if path else Path(__file__).resolve().parents[2] / "config" / "strategies.yaml"
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    strategies: dict[str, Config] = {}
    for name, overrides in data.get("strategies", {}).items():
        cfg = copy.deepcopy(base_cfg)
        _apply_dict(cfg, overrides or {})
        strategies[name] = cfg
    return strategies
