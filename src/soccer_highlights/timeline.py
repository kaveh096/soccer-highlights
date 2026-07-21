"""Turn peak events into merged highlight intervals on the global recording
timeline, then map those intervals back onto per-chunk source file ranges
for slicing."""

from __future__ import annotations

import math
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


def invert_intervals(
    covered: list[Interval], total_duration: float, min_length: float, max_length: float
) -> list[Interval]:
    """Return the gaps NOT covered by any interval in `covered` (already
    assumed merged/non-overlapping), chunked so no piece exceeds
    max_length and pieces shorter than min_length are dropped."""
    ordered = sorted(covered, key=lambda iv: iv.start_seconds)
    gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for iv in ordered:
        if iv.start_seconds > cursor:
            gaps.append((cursor, iv.start_seconds))
        cursor = max(cursor, iv.end_seconds)
    if cursor < total_duration:
        gaps.append((cursor, total_duration))

    negatives: list[Interval] = []
    for start, end in gaps:
        length = end - start
        if length < min_length:
            continue
        n_chunks = max(1, math.ceil(length / max_length))
        chunk_length = length / n_chunks
        for i in range(n_chunks):
            c_start = start + i * chunk_length
            c_end = end if i == n_chunks - 1 else start + (i + 1) * chunk_length
            negatives.append(Interval(start_seconds=c_start, end_seconds=c_end))
    return negatives


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
