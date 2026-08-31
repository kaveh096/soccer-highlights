from datetime import datetime
from pathlib import Path

import pytest

from soccer_highlights.config import MarksConfig, TimelineConfig
from soccer_highlights.discovery import Chunk
from soccer_highlights.marks import (
    RECORDING_TZ,
    Mark,
    load_marks_csv,
    resolve_marks,
    resolve_undos,
)
from soccer_highlights.timeline import GlobalPeak


def _chunk(sequence: int, start_time: datetime, duration: float, global_start: float) -> Chunk:
    return Chunk(
        sequence=sequence,
        start_time=start_time,
        mp4_path=Path(f"chunk_{sequence}.MP4"),
        lrf_path=None,
        duration_seconds=duration,
        global_start_seconds=global_start,
    )


def _local(y, mo, d, h, mi, s=0):
    return datetime(y, mo, d, h, mi, s, tzinfo=RECORDING_TZ)


def test_load_marks_csv_parses_rows_in_sequence_order(tmp_path):
    csv_path = tmp_path / "marks.csv"
    csv_path.write_text(
        "sequence,timestamp,category\n"
        "2,2026-08-23T08:10:05-07:00,black_goal\n"
        "1,2026-08-23T08:05:00-07:00,white_goal\n",
        encoding="utf-8",
    )

    marks = load_marks_csv(csv_path)

    assert [m.sequence for m in marks] == [1, 2]
    assert marks[0].category == "white_goal"


def test_load_marks_csv_rejects_naive_timestamp(tmp_path):
    csv_path = tmp_path / "marks.csv"
    csv_path.write_text("sequence,timestamp,category\n1,2026-08-23T08:05:00,white_goal\n", encoding="utf-8")

    with pytest.raises(ValueError, match="UTC offset"):
        load_marks_csv(csv_path)


def test_load_marks_csv_rejects_unknown_category(tmp_path):
    csv_path = tmp_path / "marks.csv"
    csv_path.write_text("sequence,timestamp,category\n1,2026-08-23T08:05:00-07:00,red_goal\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown category"):
        load_marks_csv(csv_path)


def test_resolve_undos_cancels_most_recent_active_mark_regardless_of_category():
    marks = [
        Mark(sequence=1, timestamp=_local(2026, 8, 23, 8, 0), category="white_goal"),
        Mark(sequence=2, timestamp=_local(2026, 8, 23, 8, 1), category="moment"),
        Mark(sequence=3, timestamp=_local(2026, 8, 23, 8, 2), category="undo"),  # cancels seq=2
    ]

    active = resolve_undos(marks)

    assert [m.sequence for m in active] == [1]


def test_resolve_undos_with_nothing_to_cancel_is_dropped_silently():
    marks = [Mark(sequence=1, timestamp=_local(2026, 8, 23, 8, 0), category="undo")]

    assert resolve_undos(marks) == []


def test_resolve_marks_snaps_to_nearby_audio_peak():
    start = datetime(2026, 8, 23, 8, 0, 0)
    chunks = [_chunk(1, start, duration=1000.0, global_start=0.0)]
    # Mark pressed 30s after the actual event (dead-ball lag); audio already
    # caught the transient at global t=100.
    mark = Mark(sequence=1, timestamp=_local(2026, 8, 23, 8, 2, 10), category="white_goal")
    audio_peaks = [GlobalPeak(time_seconds=100.0, score=1.0)]
    marks_cfg = MarksConfig(goal_lookback_seconds=60.0, goal_lookahead_seconds=5.0)
    timeline_cfg = TimelineConfig(lookback_seconds=6.0, post_peak_seconds=5.0)

    [resolved] = resolve_marks([mark], chunks, audio_peaks, marks_cfg, timeline_cfg)

    assert resolved.anchor == "audio_peak"
    assert resolved.interval.start_seconds == 94.0
    assert resolved.interval.end_seconds == 105.0


def test_resolve_marks_falls_back_to_fixed_window_when_no_peak_nearby():
    start = datetime(2026, 8, 23, 8, 0, 0)
    chunks = [_chunk(1, start, duration=1000.0, global_start=0.0)]
    mark = Mark(sequence=1, timestamp=_local(2026, 8, 23, 8, 2, 10), category="white_goal")  # global=130s
    marks_cfg = MarksConfig(goal_lookback_seconds=60.0, goal_lookahead_seconds=5.0)
    timeline_cfg = TimelineConfig(lookback_seconds=6.0, post_peak_seconds=5.0)

    [resolved] = resolve_marks([mark], chunks, [], marks_cfg, timeline_cfg)

    assert resolved.anchor == "fixed_window"
    assert resolved.interval.start_seconds == 70.0
    assert resolved.interval.end_seconds == 135.0


def test_resolve_marks_reports_unrecorded_gap():
    start1 = datetime(2026, 8, 23, 8, 0, 0)
    start2 = datetime(2026, 8, 23, 8, 5, 0)  # real gap after chunk 1 ends at 8:01:40
    chunks = [
        _chunk(1, start1, duration=100.0, global_start=0.0),
        _chunk(2, start2, duration=100.0, global_start=100.0),
    ]
    mark = Mark(sequence=1, timestamp=_local(2026, 8, 23, 8, 3, 0), category="moment")
    marks_cfg = MarksConfig()
    timeline_cfg = TimelineConfig()

    [resolved] = resolve_marks([mark], chunks, [], marks_cfg, timeline_cfg)

    assert resolved.anchor == "unrecorded_gap"
    assert resolved.global_seconds is None
    assert resolved.interval is None
