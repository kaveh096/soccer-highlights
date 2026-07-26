"""Vision refinement pass over audio-detected candidate windows and audio's
negative-space gaps, using Claude's vision API against frames sampled from
the .LRF proxy (never the full-res source -- same reasoning as audio
detection: cheap and fast to decode on old hardware).

Two passes, one prompt each, both aimed at Phase 1's two confirmed,
human-labeled failure modes (see README's Limitations section):

- **Confirm**: for every audio-flagged interval, sample frames from a
  tight window centered on the interval's detected peak (see
  `_peak_anchor_time` -- NOT spread across the whole clip; a real
  shot/strike is a sub-second transient that even spacing across a
  12-20s clip reliably misses, confirmed against real footage
  2026-07-25) and ask whether they show real in-play action or something
  that sounds similar on audio but isn't (a practice shot during a
  break, crowd chatter), plus a short factual caption. Only drops an
  interval on a confident false-positive verdict -- recall-first, per
  the project's standing tuning priority, so an uncertain or errored
  call keeps the interval rather than discarding a possibly-real event.
- **Scan**: for every gap no audio strategy flagged, ask whether a real
  event (e.g. a quiet, far-side goal) is visible anyway. Only adds a new
  interval on a confident event verdict, windowed the same
  lookback/post_peak way a real audio peak would be.

Pure decision logic (`_evenly_spaced_offsets`, `_parse_verdict_json`,
`decide_confirm`, `decide_scan`, `refine_with_vision`) takes injected
verdicts and has no I/O, so it's unit-tested directly. Frame extraction
(ffmpeg) and the actual API call are not unit-tested -- validate those
against real footage once an API key exists, same as audio.py/clipping.py/
render.py.
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from soccer_highlights.config import TimelineConfig, VisionConfig
from soccer_highlights.discovery import Chunk
from soccer_highlights.timeline import (
    ChunkSlice,
    GlobalPeak,
    Interval,
    invert_intervals,
    map_interval_to_chunks,
    merge_intervals,
    peaks_to_raw_intervals,
)

_CONFIRM_PROMPT = """You are reviewing frames from a Sunday recreational soccer game (16-player, half-court). An audio-based detector flagged a possible shot on target or goal; these frames are sampled from a {duration:.1f}-second window CENTERED ON THE MOMENT OF LOUDEST AUDIO ACTIVITY (the likely instant of ball contact or crowd reaction) -- they are not spread across the whole highlight clip, just the moment itself.

Decide whether the frames show REAL in-play action (an actual shot, save, goal, or the crowd's live reaction to one) versus something that can sound similar on audio but isn't: a practice/drill shot during a stoppage, players standing around or chatting during a break, warm-up, etc. The actual moment can be brief and easy to miss between frames -- if the frames are ambiguous or don't clearly rule out a real event, prefer is_event: true. It's worse to discard a real highlight than to keep one extra false positive.

Also write a short, factual one-line caption suitable for use in a filename, describing only what's visibly confirmable (e.g. "Shot on goal saved by keeper", "Goal celebration near far post"). Do not invent player names, jersey numbers, or the exact score unless clearly legible.

Respond with ONLY a single JSON object, no other text, no code fence:
{{"is_event": true or false, "confidence": 0.0-1.0, "caption": "short factual caption", "rationale": "one short sentence"}}"""

_SCAN_PROMPT = """You are reviewing frames from a Sunday recreational soccer game (16-player, half-court), sampled evenly across a {duration:.1f}-second window that NO audio-based detector flagged as interesting.

Look for a real shot on target, goal, or celebration that audio may have missed because it happened quietly (e.g. a shot on the far side of the field from the camera/microphone). Frames are numbered 0 to {last_index} in chronological order.

Respond with ONLY a single JSON object, no other text, no code fence:
{{"is_event": true or false, "frame_index": <int index of the frame closest to the moment, or null>, "confidence": 0.0-1.0, "rationale": "one short sentence"}}"""


@dataclass
class VisionVerdict:
    is_event: bool
    confidence: float
    frame_index: int | None = None
    rationale: str = ""
    caption: str = ""


@dataclass
class VisionLogEntry:
    kind: str  # "confirm" or "scan"
    window_start_seconds: float
    window_end_seconds: float
    verdict: VisionVerdict | None
    kept: bool  # confirm: audio interval survived; scan: a new interval was added


@dataclass
class VisionRunLog:
    entries: list[VisionLogEntry] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pure logic -- unit tested, no I/O.
# ---------------------------------------------------------------------------


def _evenly_spaced_offsets(duration_seconds: float, count: int) -> list[float]:
    """Midpoints of `count` equal-length segments spanning `duration_seconds`,
    e.g. count=3 over a 12s window -> [2.0, 6.0, 10.0]."""
    if count <= 0:
        return []
    if duration_seconds <= 0:
        return [0.0] * count
    step = duration_seconds / count
    return [step * (i + 0.5) for i in range(count)]


def _parse_verdict_json(raw_text: str, mode: str) -> VisionVerdict:
    """Parse a model response into a VisionVerdict. Raises on malformed or
    missing-field output -- callers catch broadly and fail open (treat as
    "no verdict") rather than trust a half-parsed result."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    data = json.loads(text)
    is_event = bool(data["is_event"])
    confidence = float(data["confidence"])
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence out of [0, 1] range: {confidence}")

    frame_index = None
    if mode == "scan":
        raw_index = data.get("frame_index")
        frame_index = int(raw_index) if raw_index is not None else None

    return VisionVerdict(
        is_event=is_event,
        confidence=confidence,
        frame_index=frame_index,
        rationale=str(data.get("rationale", "")),
        caption=str(data.get("caption", "")),
    )


class _HasDropThreshold(Protocol):
    drop_confidence_threshold: float


def decide_confirm(interval: Interval, verdict: VisionVerdict | None, cfg: _HasDropThreshold) -> bool:
    """Whether an audio-flagged interval survives the confirm pass.
    Fail-open: no verdict (API/parse error), an event verdict, or a
    false-positive verdict below the drop-confidence bar all keep it."""
    if verdict is None:
        return True
    if not verdict.is_event and verdict.confidence >= cfg.drop_confidence_threshold:
        return False
    return True


def decide_scan(gap: Interval, verdict: VisionVerdict | None, cfg: VisionConfig) -> GlobalPeak | None:
    """Whether a negative-space gap's scan verdict should synthesize a new
    peak, and where. Returns None if there's nothing to add."""
    if verdict is None or not verdict.is_event:
        return None
    if verdict.confidence < cfg.add_confidence_threshold:
        return None
    if verdict.frame_index is None:
        return None

    offsets = _evenly_spaced_offsets(gap.end_seconds - gap.start_seconds, cfg.frames_per_window)
    if not (0 <= verdict.frame_index < len(offsets)):
        return None
    return GlobalPeak(time_seconds=gap.start_seconds + offsets[verdict.frame_index], score=verdict.confidence)


def _peak_anchor_time(interval: Interval) -> float:
    """The moment within an interval most likely to show the actual event --
    its strongest detected peak, or the interval's midpoint if it somehow
    has none (shouldn't happen for a real audio-flagged interval)."""
    if not interval.peaks:
        return (interval.start_seconds + interval.end_seconds) / 2
    return max(interval.peaks, key=lambda p: p.score).time_seconds


class _HasPeakWindow(Protocol):
    peak_window_seconds: float


def _peak_sample_window(interval: Interval, cfg: _HasPeakWindow) -> Interval:
    """The tight sub-window (peak_window_seconds wide, centered on the
    interval's strongest peak) that both the Claude (stills) and Gemini
    (short video clip) implementations sample from -- shared so a
    provider comparison is looking at the identical time range, clamped
    to the original interval's bounds."""
    anchor = _peak_anchor_time(interval)
    half_window = cfg.peak_window_seconds / 2
    sample_start = max(interval.start_seconds, anchor - half_window)
    sample_end = min(interval.end_seconds, anchor + half_window)
    if sample_end <= sample_start:
        return Interval(start_seconds=interval.start_seconds, end_seconds=interval.end_seconds)
    return Interval(start_seconds=sample_start, end_seconds=sample_end)


_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def sanitize_caption_for_filename(caption: str, max_length: int = 80) -> str:
    """Make an LLM-generated caption safe to use as a Windows filename
    component: strip characters Windows forbids, collapse whitespace, and
    trim trailing dots/spaces (also disallowed). Falls back to "highlight"
    if nothing usable survives."""
    cleaned = _INVALID_FILENAME_CHARS.sub("", caption)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned[:max_length].rstrip(". ")
    return cleaned or "highlight"


ClassifyConfirmFn = Callable[[Interval, list[Chunk], VisionConfig], "VisionVerdict | None"]
ClassifyScanFn = Callable[[Interval, list[Chunk], VisionConfig], "VisionVerdict | None"]


def refine_with_vision(
    merged_intervals: list[Interval],
    chunks: list[Chunk],
    game_duration_seconds: float,
    timeline_cfg: TimelineConfig,
    vision_cfg: VisionConfig,
    classify_confirm_fn: ClassifyConfirmFn | None = None,
    classify_scan_fn: ClassifyScanFn | None = None,
) -> tuple[list[Interval], VisionRunLog]:
    """Confirm/reject every audio-flagged interval, then scan the gaps
    audio left uncovered for events it missed, and merge the result --
    the vision-refined counterpart to `tuning.detect_intervals`'s
    audio-only interval list. Pure w.r.t. its classify_*_fn arguments
    (default to the real ffmpeg+API-calling implementations below), so
    tests can inject fakes and never touch ffmpeg or the network."""
    classify_confirm_fn = classify_confirm_fn or classify_confirm
    classify_scan_fn = classify_scan_fn or classify_scan

    log = VisionRunLog()
    confirmed: list[Interval] = []
    for interval in merged_intervals:
        verdict = classify_confirm_fn(interval, chunks, vision_cfg)
        keep = decide_confirm(interval, verdict, vision_cfg)
        log.entries.append(VisionLogEntry("confirm", interval.start_seconds, interval.end_seconds, verdict, keep))
        if keep:
            confirmed.append(interval)

    covered = merge_intervals(merged_intervals, min_gap_seconds=0.0, min_interval_seconds=0.0)
    gaps = invert_intervals(
        covered, game_duration_seconds, vision_cfg.scan_chunk_min_seconds, vision_cfg.scan_chunk_max_seconds
    )

    new_peaks: list[GlobalPeak] = []
    for gap in gaps:
        verdict = classify_scan_fn(gap, chunks, vision_cfg)
        peak = decide_scan(gap, verdict, vision_cfg)
        log.entries.append(VisionLogEntry("scan", gap.start_seconds, gap.end_seconds, verdict, peak is not None))
        if peak is not None:
            new_peaks.append(peak)

    new_intervals = peaks_to_raw_intervals(new_peaks, timeline_cfg.lookback_seconds, timeline_cfg.post_peak_seconds)
    final = merge_intervals(confirmed + new_intervals, timeline_cfg.min_gap_seconds, timeline_cfg.min_interval_seconds)
    return final, log


# ---------------------------------------------------------------------------
# I/O -- ffmpeg frame extraction + Claude API calls. Not unit tested;
# validate against real footage once ANTHROPIC_API_KEY exists.
# ---------------------------------------------------------------------------


def _extract_frame_jpeg(source_path: Path, time_seconds: float, out_path: Path, max_width: int) -> None:
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-ss", f"{time_seconds:.3f}",
            "-i", str(source_path),
            "-frames:v", "1",
            "-vf", f"scale={max_width}:-2",
            "-q:v", "3",
            str(out_path),
        ],
        check=True,
    )


def extract_frame_samples(slices: list[ChunkSlice], count: int, tmp_dir: Path, max_width: int) -> list[Path]:
    """Grab `count` still frames evenly spaced across a (possibly
    chunk-boundary-spanning) window, allocated proportionally to each
    slice's share of the total duration."""
    total_duration = sum(cs.local_end_seconds - cs.local_start_seconds for cs in slices)
    if total_duration <= 0 or count <= 0:
        return []

    paths: list[Path] = []
    allocated = 0
    for i, cs in enumerate(slices):
        duration = cs.local_end_seconds - cs.local_start_seconds
        remaining = count - allocated
        alloc = remaining if i == len(slices) - 1 else min(remaining, max(1, round(count * duration / total_duration)))
        if alloc <= 0:
            continue
        source_path = cs.chunk.lrf_path or cs.chunk.mp4_path
        for offset in _evenly_spaced_offsets(duration, alloc):
            out_path = tmp_dir / f"frame_{len(paths):03d}.jpg"
            _extract_frame_jpeg(source_path, cs.local_start_seconds + offset, out_path, max_width)
            paths.append(out_path)
        allocated += alloc
    return paths


def _call_claude(image_paths: list[Path], prompt: str, cfg: VisionConfig) -> str:
    import anthropic

    api_key = os.environ.get(cfg.api_key_env)
    if not api_key:
        raise RuntimeError(f"{cfg.api_key_env} is not set -- see README's Vision AI section for setup")

    client = anthropic.Anthropic(api_key=api_key, timeout=cfg.request_timeout_seconds, max_retries=cfg.max_retries)
    content: list[dict] = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64.standard_b64encode(path.read_bytes()).decode("utf-8"),
            },
        }
        for path in image_paths
    ]
    content.append({"type": "text", "text": prompt})

    response = client.messages.create(
        model=cfg.model,
        max_tokens=300,
        messages=[{"role": "user", "content": content}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def classify_confirm(interval: Interval, chunks: list[Chunk], cfg: VisionConfig) -> VisionVerdict | None:
    """Classify one audio-flagged interval. Frames are sampled from a tight
    window centered on the interval's strongest peak (see
    `_peak_anchor_time`), not spread across the whole lookback/post_peak
    clip -- a real shot/strike is a sub-second transient that even spacing
    across a 12-20s clip usually misses entirely."""
    sample_window = _peak_sample_window(interval, cfg)

    try:
        with tempfile.TemporaryDirectory(prefix="soccer_hl_vision_") as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            slices = map_interval_to_chunks(sample_window, chunks)
            frame_paths = extract_frame_samples(slices, cfg.frames_per_window, tmp_dir, cfg.frame_max_width)
            if not frame_paths:
                return None
            prompt = _CONFIRM_PROMPT.format(duration=sample_window.end_seconds - sample_window.start_seconds)
            raw = _call_claude(frame_paths, prompt, cfg)
            return _parse_verdict_json(raw, mode="confirm")
    except Exception as exc:
        print(f"WARNING: vision confirm failed for [{interval.start_seconds:.1f}, {interval.end_seconds:.1f}]: {exc}")
        return None


def save_vision_log(log: VisionRunLog, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "kind": e.kind,
            "start_seconds": round(e.window_start_seconds, 2),
            "end_seconds": round(e.window_end_seconds, 2),
            "kept": e.kept,
            "verdict": None
            if e.verdict is None
            else {
                "is_event": e.verdict.is_event,
                "confidence": round(e.verdict.confidence, 3),
                "frame_index": e.verdict.frame_index,
                "rationale": e.verdict.rationale,
                "caption": e.verdict.caption,
            },
        }
        for e in log.entries
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def classify_scan(gap: Interval, chunks: list[Chunk], cfg: VisionConfig) -> VisionVerdict | None:
    try:
        with tempfile.TemporaryDirectory(prefix="soccer_hl_vision_") as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            slices = map_interval_to_chunks(gap, chunks)
            frame_paths = extract_frame_samples(slices, cfg.frames_per_window, tmp_dir, cfg.frame_max_width)
            if not frame_paths:
                return None
            prompt = _SCAN_PROMPT.format(
                duration=gap.end_seconds - gap.start_seconds, last_index=cfg.frames_per_window - 1
            )
            raw = _call_claude(frame_paths, prompt, cfg)
            return _parse_verdict_json(raw, mode="scan")
    except Exception as exc:
        print(f"WARNING: vision scan failed for [{gap.start_seconds:.1f}, {gap.end_seconds:.1f}]: {exc}")
        return None
