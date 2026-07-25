"""Fast, audio-only detection for sweeping many config variants against a
golden event set (see golden.py) without rendering clips or re-decoding
audio per candidate. Decode each chunk's audio once, then re-run detection
against the cached samples for as many DetectionConfig/TimelineConfig
combinations as needed."""

from __future__ import annotations

import numpy as np

from soccer_highlights.config import DetectionConfig, TimelineConfig
from soccer_highlights.detection import analyze
from soccer_highlights.discovery import Chunk
from soccer_highlights.timeline import GlobalPeak, Interval, merge_intervals, peaks_to_raw_intervals


def detect_intervals(
    samples_by_chunk: list[tuple[Chunk, np.ndarray]],
    sample_rate: int,
    strategy: str,
    detection_cfg: DetectionConfig,
    timeline_cfg: TimelineConfig,
) -> list[Interval]:
    """Run detection + timeline merge against already-decoded per-chunk
    samples. Pure function of its inputs -- no I/O, no printing -- so it's
    cheap to call in a tight sweep loop."""
    all_peaks: list[GlobalPeak] = []
    for chunk, samples in samples_by_chunk:
        trace = analyze(samples, sample_rate, strategy, detection_cfg)
        for event in trace.events:
            all_peaks.append(GlobalPeak(time_seconds=chunk.global_start_seconds + event.time_seconds, score=event.score))

    warmed_up_peaks = [p for p in all_peaks if p.time_seconds >= timeline_cfg.warmup_seconds]
    raw_intervals = peaks_to_raw_intervals(warmed_up_peaks, timeline_cfg.lookback_seconds, timeline_cfg.post_peak_seconds)
    return merge_intervals(raw_intervals, timeline_cfg.min_gap_seconds, timeline_cfg.min_interval_seconds)
