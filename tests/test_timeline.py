from datetime import datetime
from pathlib import Path

from soccer_highlights.discovery import Chunk
from soccer_highlights.timeline import GlobalPeak, map_interval_to_chunks, merge_intervals, peaks_to_raw_intervals


def _chunk(sequence: int, global_start: float, duration: float) -> Chunk:
    return Chunk(
        sequence=sequence,
        start_time=datetime(2026, 7, 19, 8, 0, 0),
        mp4_path=Path(f"chunk_{sequence}.MP4"),
        lrf_path=None,
        duration_seconds=duration,
        global_start_seconds=global_start,
    )


def test_peaks_to_raw_intervals_clamps_lookback_at_zero():
    peaks = [GlobalPeak(time_seconds=5.0, score=1.0)]
    intervals = peaks_to_raw_intervals(peaks, lookback_seconds=45.0, post_peak_seconds=8.0)

    assert len(intervals) == 1
    assert intervals[0].start_seconds == 0.0
    assert intervals[0].end_seconds == 13.0


def test_merge_intervals_joins_close_intervals_and_drops_short_ones():
    peaks = [
        GlobalPeak(time_seconds=100.0, score=1.0),
        GlobalPeak(time_seconds=110.0, score=1.0),  # close enough to merge with the above
        GlobalPeak(time_seconds=500.0, score=1.0),  # far away, stays separate
    ]
    raw = peaks_to_raw_intervals(peaks, lookback_seconds=10.0, post_peak_seconds=5.0)
    merged = merge_intervals(raw, min_gap_seconds=5.0, min_interval_seconds=1.0)

    assert len(merged) == 2
    assert merged[0].start_seconds == 90.0
    assert merged[0].end_seconds == 115.0
    assert merged[1].start_seconds == 490.0


def test_merge_intervals_drops_intervals_below_min_length():
    peaks = [GlobalPeak(time_seconds=10.0, score=1.0)]
    raw = peaks_to_raw_intervals(peaks, lookback_seconds=1.0, post_peak_seconds=1.0)
    merged = merge_intervals(raw, min_gap_seconds=1.0, min_interval_seconds=10.0)

    assert merged == []


def test_map_interval_to_chunks_splits_across_boundary():
    chunks = [_chunk(1, global_start=0.0, duration=100.0), _chunk(2, global_start=100.0, duration=100.0)]
    peaks = [GlobalPeak(time_seconds=100.0, score=1.0)]
    raw = peaks_to_raw_intervals(peaks, lookback_seconds=10.0, post_peak_seconds=10.0)

    slices = map_interval_to_chunks(raw[0], chunks)

    assert len(slices) == 2
    assert slices[0].chunk.sequence == 1
    assert slices[0].local_start_seconds == 90.0
    assert slices[0].local_end_seconds == 100.0
    assert slices[1].chunk.sequence == 2
    assert slices[1].local_start_seconds == 0.0
    assert slices[1].local_end_seconds == 10.0


def test_map_interval_to_chunks_single_chunk_no_split():
    chunks = [_chunk(1, global_start=0.0, duration=100.0), _chunk(2, global_start=100.0, duration=100.0)]
    peaks = [GlobalPeak(time_seconds=50.0, score=1.0)]
    raw = peaks_to_raw_intervals(peaks, lookback_seconds=10.0, post_peak_seconds=10.0)

    slices = map_interval_to_chunks(raw[0], chunks)

    assert len(slices) == 1
    assert slices[0].chunk.sequence == 1
