from datetime import datetime, timedelta, timezone
from pathlib import Path

from soccer_highlights.discovery import Chunk, slice_start_epoch, wallclock_to_global


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


def _burned_in_wall_clock(epoch: float) -> datetime:
    """What ffmpeg's `%{pts:gmtime:<epoch>}` will actually render -- the epoch
    formatted as UTC, which is how the burned-in overlay is produced."""
    return datetime.fromtimestamp(epoch, timezone.utc).replace(tzinfo=None)


def test_slice_start_epoch_renders_the_chunks_own_local_wall_clock():
    # A Sunday-morning game starting 8:00 AM local. The overlay must read
    # 8:02:05 AM, NOT the UTC instant that corresponds to it -- burning in UTC
    # is the exact bug this asserts against (an 8:46 AM clip read "03.46 PM"
    # before this was fixed, 2026-08-31).
    chunk = _chunk(1, datetime(2026, 8, 23, 8, 0, 0), duration=600.0, global_start=0.0)

    assert _burned_in_wall_clock(slice_start_epoch(chunk, 125.5)) == datetime(2026, 8, 23, 8, 2, 5, 500000)


def test_slice_start_epoch_uses_the_owning_chunk_not_elapsed_footage_time():
    # The whole point of the feature: chunk 2 starts 41.5 min of wall-clock
    # after chunk 1's global offset would suggest (camera stopped between
    # them). An offset into chunk 2 must resolve through chunk 2's own
    # filename timestamp, ignoring global_start_seconds entirely.
    chunk2 = _chunk(2, datetime(2026, 8, 23, 8, 51, 30), duration=600.0, global_start=600.0)

    assert _burned_in_wall_clock(slice_start_epoch(chunk2, 30.0)) == datetime(2026, 8, 23, 8, 52, 0)


def test_slice_start_epoch_is_timezone_and_dst_independent():
    # Guard against someone "fixing" this to do a real local->UTC conversion:
    # the DJI timestamp already IS the local wall clock, so the same clock
    # reading must burn in identically on either side of a DST transition
    # (2026-11-01 is the US fall-back date) and regardless of the render
    # machine's OS timezone.
    summer = _chunk(1, datetime(2026, 8, 23, 9, 30, 0), duration=600.0, global_start=0.0)
    winter = _chunk(2, datetime(2026, 11, 1, 9, 30, 0), duration=600.0, global_start=600.0)

    assert _burned_in_wall_clock(slice_start_epoch(summer, 0.0)).time() == datetime(2026, 1, 1, 9, 30).time()
    assert _burned_in_wall_clock(slice_start_epoch(winter, 0.0)).time() == datetime(2026, 1, 1, 9, 30).time()
