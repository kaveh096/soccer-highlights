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
    threshold_sigma: float = 2.5
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
    # Stereo -- mono was measured to save no CPU/time (audio encode is
    # negligible next to video), so no reason to downgrade it.
    mono_audio: bool = False
    # Negative-space clips: gaps not covered by ANY strategy's candidate
    # intervals, chunked for review so nothing is silently missed.
    max_negative_clip_seconds: float = 120.0
    min_negative_clip_seconds: float = 8.0


@dataclass
class ExportConfig:
    # Re-encoded (not stream-copied) delivery clips for sharing -- source
    # footage here is 4K HEVC Main10, which struggles to play on modest
    # hardware and takes forever to upload. Re-encoding to a lower,
    # standard frame rate and 8-bit H.264 at a quality-targeted (not
    # fixed) bitrate fixes that without a visible quality drop.
    # 2560px, not the source's native 3840: benchmarked on real 4K/10-bit
    # HEVC footage on this laptop (2026-07-28) -- decoding the source
    # dominates cost regardless of output width (2560px: ~10x realtime,
    # 35MB/12s; 1920px: ~10.4x, near-identical since decode-bound; 3840px:
    # ~21x realtime, 96MB/12s). 2K gives the same crf-18 quality target
    # for social media at roughly half the time and disk of full 4K.
    dir: str = "output/export"
    max_width: int = 2560
    fps: float = 30.0
    # x264 CRF is a quality target, not a fixed bitrate -- output bitrate
    # adapts per scene. 18 is the standard "visually lossless" reference
    # value for x264.
    crf: int = 18
    preset: str = "medium"
    threads: int = 0  # 0 = let ffmpeg use all available cores
    audio_bitrate_kbps: int = 192
    # Preserve the source's original channel count -- this is the sharing
    # deliverable, not a cheap review clip, so it should NOT be downmixed
    # to mono (a bug in the first cut of this config: it was forced mono
    # unconditionally, same as review clips, until caught 2026-07-25).
    mono_audio: bool = False


@dataclass
class VisionConfig:
    # Phase 2: off by default -- audio-only behavior is unchanged unless a
    # caller explicitly opts in (see cli.py's `golden-score --vision`).
    enabled: bool = False
    model: str = "claude-sonnet-5"
    api_key_env: str = "ANTHROPIC_API_KEY"
    frames_per_window: int = 5
    # Frames are always pulled from the .LRF proxy, never the full-res
    # source -- same reasoning as audio detection: cheap, fast to decode
    # on old hardware, and plenty of detail for a classification call.
    frame_max_width: int = 640
    # Confirm pass: frames are sampled from a window this wide, CENTERED ON
    # THE INTERVAL'S PEAK TIME -- not spread across the whole (lookback +
    # post_peak) clip. A real shot/strike is a sub-second transient; the
    # first real-footage test (2026-07-25) showed even spacing across a
    # 12-20s clip usually lands all sampled frames just before/after the
    # actual moment, so the model confidently (0.75-0.85) misreads a real
    # event as a practice shot -- recall collapsed 0.77->0.23. Anchoring on
    # the known peak timestamp (already detected by audio) fixes this.
    peak_window_seconds: float = 4.0
    # Where `vision-highlights` writes its kept/pruned clips.
    highlights_dir: str = "output/vision_highlights"
    # Confirm pass: an audio-flagged interval is only DROPPED if vision
    # reports a false positive at or above this confidence. Recall-first
    # per the project's standing priority -- an uncertain or errored call
    # keeps the interval rather than discarding a possibly-real event.
    drop_confidence_threshold: float = 0.75
    # Scan pass: a negative-space gap only gets a new synthesized interval
    # if vision reports an event at or above this confidence.
    add_confidence_threshold: float = 0.75
    scan_chunk_max_seconds: float = 120.0
    scan_chunk_min_seconds: float = 8.0
    request_timeout_seconds: float = 60.0
    max_retries: int = 2


@dataclass
class GeminiConfig:
    # Gemini native-video counterpart to VisionConfig's Claude stills-based
    # confirm pass -- for a direct comparison of "continuous video" vs.
    # "discrete extracted frames" on the identical candidate set. Google's
    # own docs confirm Gemini also subsamples video at 1 FPS by default, so
    # this isn't a free win over frame extraction; it's a real experiment,
    # not an assumed upgrade (see README's Vision AI section, 2026-07-25).
    enabled: bool = False
    model: str = "gemini-flash-latest"
    api_key_env: str = "GEMINI_API_KEY"
    # Same concept as VisionConfig.peak_window_seconds -- the short video
    # clip sent to Gemini is cut from this window, centered on the
    # interval's detected peak, so both providers see the identical time
    # range for a fair comparison.
    peak_window_seconds: float = 4.0
    clip_max_width: int = 640
    drop_confidence_threshold: float = 0.75
    add_confidence_threshold: float = 0.75
    request_timeout_seconds: float = 60.0
    max_retries: int = 2


@dataclass
class LabelAuditConfig:
    # Re-checks the existing human-labeled Round 2 dataset (output/review/*)
    # against a Gemini-generated free-text scene description, judged by
    # Claude for agreement -- not another detection-tuning pass, an audit
    # of whether the LABELS themselves hold up. See README's Vision AI
    # section (Label Audit) for why this was worth doing: three straight
    # rounds of prompt tuning against the existing golden set all failed
    # to clearly improve on audio alone, raising the question of whether
    # the ground truth itself needs a second look before tuning further.
    review_root: str = "output/review"
    output_dir: str = "output/label_audit"
    # No separate render tier here anymore (2026-07-28) -- flagged rows
    # are copied straight from the already-rendered ReviewConfig clip
    # (the same file the human labeled and Gemini scored), not re-encoded
    # at a third resolution. Benchmarked at ~0.33x realtime for
    # ReviewConfig's 1280px/veryfast/crf23 on real LRF footage, so a full
    # audit-scale batch (~50 min of source footage) renders in ~15-30
    # minutes -- cheap enough that a separate lower-quality tier bought
    # nothing but a second lossy re-encode of the same clip.
    # A row gets a rendered clip for human review if the judge's agreement
    # isn't "consistent" or its distance_score is at least this high.
    flag_distance_threshold: float = 0.5


@dataclass
class TelegramConfig:
    # Posts final, hand-picked export clips to a Telegram group (Bot API
    # sendVideo, direct HTTP call -- no SDK, matching this project's existing
    # style of calling providers' REST APIs directly, see vision_gemini.py).
    # Token/chat ID are read from env vars, never stored in config -- see
    # README's Vision AI section's sibling setup note for how to create a
    # bot via @BotFather and find a group's chat ID.
    bot_token_env: str = "TELEGRAM_BOT_TOKEN"
    chat_id_env: str = "TELEGRAM_CHAT_ID"
    # Bot API's hard per-file limit for bot-uploaded video, regardless of
    # method -- checked client-side before attempting an upload so a
    # too-large file fails fast with a clear message instead of a confusing
    # HTTP error partway through a slow upload.
    max_file_size_mb: float = 50.0
    request_timeout_seconds: float = 120.0
    max_retries: int = 2


@dataclass
class Config:
    input: InputConfig = field(default_factory=InputConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    timeline: TimelineConfig = field(default_factory=TimelineConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
    review: ReviewConfig = field(default_factory=ReviewConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    label_audit: LabelAuditConfig = field(default_factory=LabelAuditConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)


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
