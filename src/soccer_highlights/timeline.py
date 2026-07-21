"""Turn peak events into merged highlight intervals on the global recording
timeline, then map those intervals back onto per-chunk source file ranges
for slicing."""

from __future__ import annotations

from dataclasses import dataclass, field

from soccer_highlights.discovery import Chunk


@dataclass
class GlobalPeak:
    time_seconds: float  # offset from the start of the whole recording session
    score: float


@dataclass
class Interval:
    start_seconds: float
    end_seconds: float
    peaks: list[GlobalPeak] = field(default_factory=list)


@dataclass
class ChunkSlice:
    chunk: Chunk
    local_start_seconds: float
    local_end_seconds: float


def peaks_to_raw_intervals(peaks: list[GlobalPeak], lookback_seconds: float, post_peak_seconds: float) -> list[Interval]:
    intervals = [
        Interval(
            start_seconds=max(0.0, peak.time_seconds - lookback_seconds),
            end_seconds=peak.time_seconds + post_peak_seconds,
            peaks=[peak],
        )
        for peak in sorted(peaks, key=lambda p: p.time_seconds)
    ]
    return intervals


def merge_intervals(intervals: list[Interval], min_gap_seconds: float, min_interval_seconds: float) -> list[Interval]:
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda iv: iv.start_seconds)
    merged = [ordered[0]]
    for current in ordered[1:]:
        last = merged[-1]
        if current.start_seconds - last.end_seconds <= min_gap_seconds:
            last.end_seconds = max(last.end_seconds, current.end_seconds)
            last.peaks.extend(current.peaks)
        else:
            merged.append(current)
    return [iv for iv in merged if iv.end_seconds - iv.start_seconds >= min_interval_seconds]


def map_interval_to_chunks(interval: Interval, chunks: list[Chunk]) -> list[ChunkSlice]:
    """Split a global-timeline interval into per-chunk local ranges, handling
    intervals that span a chunk boundary."""
    slices: list[ChunkSlice] = []
    for chunk in chunks:
        chunk_end = chunk.global_start_seconds + chunk.duration_seconds
        overlap_start = max(interval.start_seconds, chunk.global_start_seconds)
        overlap_end = min(interval.end_seconds, chunk_end)
        if overlap_end > overlap_start:
            slices.append(
                ChunkSlice(
                    chunk=chunk,
                    local_start_seconds=overlap_start - chunk.global_start_seconds,
                    local_end_seconds=overlap_end - chunk.global_start_seconds,
                )
            )
    return slices
