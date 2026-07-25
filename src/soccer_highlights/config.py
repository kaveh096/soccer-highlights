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
    # Round 3: best F1 found in a golden-set sweep -- see config/strategies.yaml.
    min_score_dbfs: float = 3.5


@dataclass
class OnsetFluxConfig:
    window_seconds: float = 0.05
    hop_seconds: float = 0.01
    baseline_window_seconds: float = 30.0
    threshold_sigma: float = 2.0
    # Same purpose as RmsEnergyConfig.min_score_dbfs, but flux has no fixed
    # physical unit -- tune this relative to the flux magnitudes your own
    # recordings produce (check the debug plot). Round 3: largest value
    # that still catches every "must-catch" golden event -- see
    # config/strategies.yaml's strike_loose comment.
    min_score: float = 0.55


@dataclass
class CombinedConfig:
    # An onset_flux transient only counts as a confirmed event if an
    # rms_energy swell (crowd reaction) also fires within this window
    # around it. Crowd reaction to real action typically lags the impact
    # sound by a couple seconds, occasionally leads it slightly
    # (anticipatory noise), rarely coincides exactly -- hence separate
    # before/after tolerances rather than one symmetric window.
    window_before_seconds: float = 2.0
    window_after_seconds: float = 12.0


@dataclass
class DetectionConfig:
    # Round 3: onset_flux alone beat rms_energy and combined fusion by a
    # wide margin against real ground truth -- see README's Round 3 results.
    strategy: str = "onset_flux"
    rms_energy: RmsEnergyConfig = field(default_factory=RmsEnergyConfig)
    onset_flux: OnsetFluxConfig = field(default_factory=OnsetFluxConfig)
    combined: CombinedConfig = field(default_factory=CombinedConfig)


@dataclass
class TimelineConfig:
    lookback_seconds: float = 6.0
    post_peak_seconds: float = 5.0
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
    # Round 1 used 640x360/15fps/ultrafast to fit an unattended overnight
    # batch on tight disk space and avoid overloading the CPU (this machine
    # crashed twice under sustained encode load). Round 2 clips are much
    # shorter (social-media-length, not 45-135s) and disk space is no
    # longer tight, so quality is bumped back up; `threads` stays capped
    # rather than unbounded since this still isn't dedicated render hardware.
    max_width: int = 1280
    fps: float = 30.0
    crf: int = 23
    preset: str = "veryfast"
    threads: int = 4
    audio_bitrate_kbps: int = 128
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
