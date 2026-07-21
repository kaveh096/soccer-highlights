from datetime import datetime
from pathlib import Path

from soccer_highlights.discovery import Chunk
from soccer_highlights.timeline import (
    GlobalPeak,
    Interval,
    invert_intervals,
    map_interval_to_chunks,
    merge_intervals,
    peaks_to_raw_intervals,
)


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


def test_invert_intervals_finds_gaps_before_between_and_after():
    covered = [Interval(start_seconds=50.0, end_seconds=100.0), Interval(start_seconds=150.0, end_seconds=180.0)]
    negatives = invert_intervals(covered, total_duration=200.0, min_length=1.0, max_length=1000.0)

    assert len(negatives) == 3
    assert (negatives[0].start_seconds, negatives[0].end_seconds) == (0.0, 50.0)
    assert (negatives[1].start_seconds, negatives[1].end_seconds) == (100.0, 150.0)
    assert (negatives[2].start_seconds, negatives[2].end_seconds) == (180.0, 200.0)


def test_invert_intervals_drops_short_gaps():
    covered = [Interval(start_seconds=0.0, end_seconds=100.0), Interval(start_seconds=103.0, end_seconds=200.0)]
    negatives = invert_intervals(covered, total_duration=200.0, min_length=5.0, max_length=1000.0)

    assert negatives == []


def test_invert_intervals_chunks_long_gaps():
    covered: list[Interval] = []
    negatives = invert_intervals(covered, total_duration=250.0, min_length=1.0, max_length=100.0)

    assert len(negatives) == 3
    total = sum(iv.end_seconds - iv.start_seconds for iv in negatives)
    assert total == 250.0
    for iv in negatives:
        assert iv.end_seconds - iv.start_seconds <= 100.0 + 1e-9
