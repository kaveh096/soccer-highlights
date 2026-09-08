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
    # See config/default.yaml's timeline block for the rationale/history
    # behind these two -- raised 2026-08-31 (Kaveh's explicit request after
    # the Aug-30 game), not re-swept against golden_events.
    lookback_seconds: float = 9.0
    post_peak_seconds: float = 5.0
    min_gap_seconds: float = 10.0
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
    #
    # Re-benchmarked 2026-08-23 with source .MP4s on the Google Drive path
    # (not a local copy) and the numbers are much worse than the 2026-07-28
    # figures above: 5s of footage at 2560px/crf18/preset medium took 188s
    # end to end == ~37.6x realtime. Isolating the stages: decode alone is
    # ~20x realtime (100s) and is an irreducible floor; preset medium adds
    # ~88s of encode, preset veryfast only ~36s. Budget ~37x realtime, i.e.
    # roughly 2.2 hours for a 10-clip / ~208s batch, and note that any clip
    # over ~15s of footage cannot finish inside a 10-minute command timeout
    # -- see scripts/seg_render.py for the segmented, resumable workaround.
    #
    # Output size varies ~5x with scene motion at a fixed CRF (measured
    # 1.6-3.5 MB/s at crf18 across one game's clips), so a single global CRF
    # either busts Telegram's 50MB cap on the longest clip or wastes quality
    # on the rest. Pick CRF per clip from its duration, and re-measure rather
    # than extrapolating: the usual "+6 CRF halves the bitrate" rule was badly
    # wrong on static wide-shot footage, where the real factor was ~1.25x per
    # single CRF step (9.8MB at crf30 -> 88.7MB at crf20 on the same clip).
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
    # Burn the real time of day into the shared clip (2026-08-31, Kaveh's
    # request): viewers can tell WHEN in the game a moment happened, not just
    # how long the clip is. Export path only -- review/pre-label clips are
    # deliberately left clean (they're private triage, not shared output).
    # The timestamp is anchored per source chunk's own filename timestamp, NOT
    # by adding elapsed seconds to a session start, because the global
    # timeline assumes continuous recording and drifts from real time by
    # however long the camera was stopped between chunks (41.5 min across one
    # 110.3 min game) -- see discovery.slice_start_epoch.
    burn_in_time: bool = True
    # strftime format for the overlay. MUST NOT contain a literal ':' --
    # ffmpeg's drawtext `%{pts:gmtime:...}` expansion splits its own arguments
    # on colons with a hard 3-argument cap, so any colon here (escaped or not)
    # fails the whole render with "%{pts} requires at most 3 arguments".
    # Verified against ffmpeg 8.1.2 on 2026-08-31; '.' separators sidestep it.
    burn_in_time_format: str = "%b %d %I.%M.%S %p"
    # An explicit font FILE, never a `font=<family>` name: this ffmpeg build
    # has fontconfig compiled in but no fontconfig config file on Windows, so
    # asking it to resolve a family name segfaults the whole ffmpeg process
    # (reproduced with `font=Arial`, 2026-08-31) rather than erroring cleanly.
    burn_in_font_path: str = "C:/Windows/Fonts/arial.ttf"
    burn_in_font_size: int = 42
    burn_in_margin_px: int = 24
    # Running goal counter (2026-09-05), drawn under the clock. Needs
    # watch-mark data: with no marks the review sheet's score columns are
    # blank and no counter is drawn, so an audio-only game is unaffected.
    # Team labels are the shirt colours the game is actually played in.
    # A literal ':' would need escaping in the filter string, so the score
    # separator is ' - ' -- same reason the clock uses dots.
    burn_in_score: bool = True
    score_home_label: str = "White"
    score_away_label: str = "Black"
    # Scoreboard "bug" (design signed off 2026-09-05, see soccer_highlights.
    # scoreboard): a dark panel with a team pill either side of the score and
    # the clock on a gold-divided tail. Geometry is expressed for a 2560px
    # frame and scaled by the actual output width; this multiplier is on top
    # of that, for taste rather than for resolution.
    scoreboard_scale: float = 1.15
    # Time only inside the scoreboard, no date: that is what the approved
    # design shows, and the date is redundant next to a score for a clip
    # everyone knows the date of. It also costs real width -- carrying
    # "Sep 06 " would push the board from 37% to ~43% of frame width. The
    # standalone clock (a game with no marks) keeps burn_in_time_format's
    # date, since that one can be seen with no other context.
    # Same no-colon rule applies -- see burn_in_time_format.
    scoreboard_time_format: str = "%I.%M.%S %p"
    # Bold for the score, condensed bold for the team pills. Both are given as
    # explicit file paths for the same reason burn_in_font_path is: fontconfig
    # has no config file on this box and resolving a family name segfaults
    # ffmpeg. Pillow needs real paths regardless.
    burn_in_font_path_bold: str = "C:/Windows/Fonts/arialbd.ttf"
    scoreboard_font_narrow: str = "C:/Windows/Fonts/ARIALNB.TTF"


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
class MarksConfig:
    # ingest-marks: live-tagged marks (watch presses, or any other source
    # emitting wall-clock timestamps + a category) are trailing -- pressed
    # AFTER the event, not during it -- unlike an audio peak, which IS the
    # transient. Lag varies by category: a goal has a natural dead-ball
    # pause (retrieve ball, walk back, restart); a mid-flow "moment" has no
    # stoppage, so later and less reliable.
    #
    # These are the FALLBACK clip windows -- used only when no audio peak is
    # found to anchor on. Wide is the safe direction here: with no peak, a
    # too-long clip is reviewable, a too-short one has missed the event.
    #
    # 60s (2026-09-05, pre-game) was a provisional, untested guess. Revised
    # to 15s (2026-09-07) after reviewing a full real game's worth of
    # fixed-window clips on Sep-06 -- 60s produced consistently oversized,
    # hard-to-review clips (several 65s fallbacks where the real action was
    # in the first third). Evidence-based now, not a guess -- revisit only
    # with new evidence, not another guess.
    goal_lookback_seconds: float = 15.0
    goal_lookahead_seconds: float = 5.0
    moment_lookback_seconds: float = 15.0
    moment_lookahead_seconds: float = 5.0
    # The peak-SEARCH window is deliberately much tighter than the fallback
    # window above, and must stay that way (2026-09-02). Widening the search
    # does NOT make snapping safer -- it makes it worse: resolve_marks anchors
    # on the loudest peak in the window, and across a full minute of a soccer
    # game the loudest peak is quite likely some unrelated shout/whistle/shot
    # rather than the marked event, which would silently cut the wrong clip
    # (worse than the honest wide fallback, because it looks precise).
    #
    # Sized from Kaveh's own Tallies bench test (2026-09-02): he estimates a
    # ~5-10s press delay. This is his ESTIMATE from tapping at a desk, not a
    # measured in-game figure -- the original design brief had assumed +10 to
    # +45s for goals, so the two disagree by a lot and this has never been
    # checked against a real game. The clap-sync ritual at the next game is
    # what would actually measure it. Treat these as provisional.
    snap_lookback_seconds: float = 15.0
    snap_lookahead_seconds: float = 2.0
    # Score-counter burn-in (2026-09-05): the burned-in score flips at the
    # TAP, not at the audio peak -- a broadcast score graphic also updates a
    # beat after the goal, and anchoring on the tap behaves identically
    # whether or not audio found a peak. But a peak-anchored clip ends only
    # timeline.post_peak_seconds (5s) after the peak, while the tap lands
    # later than that, so the clip has to be stretched to give the flip
    # somewhere to happen. This caps that stretch, measured from the PEAK:
    # past it the clip is left alone and the flip falls back to the clip
    # midpoint (marks.score_for_interval). 10s is Kaveh's own estimate of
    # his worst realistic press delay -- provisional until a real game
    # measures it, like the snap windows above.
    score_flip_cap_seconds: float = 10.0
    # A tail after the tap so the updated score is readable instead of
    # flashing for a few frames before the cut.
    score_flip_tail_seconds: float = 2.0
    # Clap-sync search half-width (see marks.measure_clock_offset). Kaveh
    # synced the camera RTC to his phone via DJI Mimo on 2026-09-02 and
    # reports it agrees within ~1s, and the watch takes its clock from the
    # same phone -- so the expected offset is ~0 and this window only has to
    # be wide enough to prove that. It is deliberately much wider than that
    # expectation so a FAILED sync still gets found rather than silently
    # reported as "no clap detected"; note DJI filename timestamps are
    # themselves only 1-second-resolution, so ~+/-2s is the realistic
    # noise floor and an offset inside that is not worth correcting.
    sync_window_seconds: float = 30.0


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
    # Covers the whole upload, not just connect -- so it has to be sized for
    # the largest file the 50MB cap allows on a home upstream link, not for a
    # quick API round-trip. 120s was too low and failed mid-upload on a 45MB
    # clip (2026-08-23) after successfully sending a 24MB one, leaving a
    # partial batch. 900s is deliberately generous: a stalled upload costs
    # only wall-clock time, whereas a timeout costs a re-upload of everything
    # already pushed for that clip.
    request_timeout_seconds: float = 900.0
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
    marks: MarksConfig = field(default_factory=MarksConfig)
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
