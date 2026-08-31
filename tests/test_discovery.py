from datetime import datetime, timedelta
from pathlib import Path

from soccer_highlights.discovery import Chunk, wallclock_to_global


def _chunk(sequence: int, start_time: datetime, duration: float, global_start: float) -> Chunk:
    return Chunk(
        sequence=sequence,
        start_time=start_time,
        mp4_path=Path(f"chunk_{sequence}.MP4"),
        lrf_path=None,
        duration_seconds=duration,
        global_start_seconds=global_start,
    )


def test_wallclock_to_global_within_first_chunk():
    start = datetime(2026, 8, 23, 8, 0, 0)
    chunks = [_chunk(1, start, duration=100.0, global_start=0.0)]

    assert wallclock_to_global(start + timedelta(seconds=50), chunks) == 50.0


def test_wallclock_to_global_within_second_continuous_chunk():
    start1 = datetime(2026, 8, 23, 8, 0, 0)
    start2 = start1 + timedelta(seconds=100)  # back-to-back, no gap
    chunks = [
        _chunk(1, start1, duration=100.0, global_start=0.0),
        _chunk(2, start2, duration=100.0, global_start=100.0),
    ]

    assert wallclock_to_global(start2 + timedelta(seconds=10), chunks) == 110.0


def test_wallclock_to_global_returns_none_inside_unrecorded_gap():
    start1 = datetime(2026, 8, 23, 8, 0, 0)
    # chunk 1 ends at 8:01:40; chunk 2 doesn't start until 8:05:00 -- a real
    # stop/restart gap, the exact shape a camera swap or a pause leaves.
    start2 = datetime(2026, 8, 23, 8, 5, 0)
    chunks = [
        _chunk(1, start1, duration=100.0, global_start=0.0),
        _chunk(2, start2, duration=100.0, global_start=100.0),
    ]

    assert wallclock_to_global(datetime(2026, 8, 23, 8, 3, 0), chunks) is None


def test_wallclock_to_global_returns_none_before_first_and_after_last():
    start = datetime(2026, 8, 23, 8, 0, 0)
    chunks = [_chunk(1, start, duration=100.0, global_start=0.0)]

    assert wallclock_to_global(start - timedelta(seconds=1), chunks) is None
    assert wallclock_to_global(start + timedelta(seconds=100), chunks) is None
