from datetime import datetime
from pathlib import Path

import pytest

from soccer_highlights.config import MarksConfig, TimelineConfig
from soccer_highlights.discovery import Chunk
from soccer_highlights.marks import (
    RECORDING_TZ,
    Mark,
    load_marks_csv,
    load_tallies_csv,
    measure_clock_offset,
    merge_tally_marks,
    resolve_marks,
    resolve_undos,
    union_with_audio,
)
from soccer_highlights.timeline import GlobalPeak, Interval


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


def test_load_tallies_csv_parses_the_real_bench_test_export(tmp_path):
    # Byte-for-byte the real 2026-09-02 Tallies export, so this test breaks
    # if the app's export format ever changes under us.
    csv_path = tmp_path / "white goal.csv"
    csv_path.write_text(
        "timestamp,count,count_change,action\n"
        "1788407270515,0,0,TALLY_CREATED\n"
        "1788407495354,1,1,CLICK\n"
        "1788407510932,-1,-1,CLICK\n"
        "1788407512546,-2,-1,CLICK\n"
        "1788407522303,-1,1,CLICK\n"
        "1788407522616,0,1,CLICK\n"
        "1788407550207,0,0,EDIT_TALLY\n"
        "1788407632369,1,1,CLICK\n"
        "1788407755313,2,1,CLICK\n"
        "1788407756222,1,-1,CLICK\n",
        encoding="utf-8",
    )

    marks = load_tallies_csv(csv_path, "white_goal")

    # +1 at ...495354, then two -1s (only one has anything left to cancel),
    # then +1,+1, then +1, then +1 -1. Net surviving: 3.
    assert [m.category for m in marks] == ["white_goal"] * 3
    assert [m.timestamp.timestamp() for m in marks] == [1788407522303 / 1000, 1788407522616 / 1000, 1788407632369 / 1000]


def test_load_tallies_csv_ignores_non_click_rows(tmp_path):
    csv_path = tmp_path / "t.csv"
    csv_path.write_text(
        "timestamp,count,count_change,action\n"
        "1788407270515,0,0,TALLY_CREATED\n"
        "1788407495354,1,1,CLICK\n"
        "1788407550207,0,0,EDIT_TALLY\n",
        encoding="utf-8",
    )

    assert len(load_tallies_csv(csv_path, "black_goal")) == 1


def test_load_tallies_csv_minus_is_scoped_to_its_own_counter_file(tmp_path):
    # The reason per-counter undo can't go through the global resolve_undos
    # stack: a minus on the white counter must cancel a WHITE goal, even
    # though a black goal was the most recent press overall.
    white = tmp_path / "white.csv"
    white.write_text(
        "timestamp,count,count_change,action\n1788407400000,1,1,CLICK\n1788407600000,0,-1,CLICK\n", encoding="utf-8"
    )
    black = tmp_path / "black.csv"
    black.write_text("timestamp,count,count_change,action\n1788407500000,1,1,CLICK\n", encoding="utf-8")

    merged = merge_tally_marks(load_tallies_csv(white, "white_goal") + load_tallies_csv(black, "black_goal"))

    assert [m.category for m in merged] == ["black_goal"]


def test_merge_tally_marks_orders_by_timestamp_across_files_and_numbers_sequences(tmp_path):
    white = tmp_path / "white.csv"
    white.write_text("timestamp,count,count_change,action\n1788407600000,1,1,CLICK\n", encoding="utf-8")
    black = tmp_path / "black.csv"
    black.write_text("timestamp,count,count_change,action\n1788407400000,1,1,CLICK\n", encoding="utf-8")

    merged = merge_tally_marks(load_tallies_csv(white, "white_goal") + load_tallies_csv(black, "black_goal"))

    assert [(m.sequence, m.category) for m in merged] == [(1, "black_goal"), (2, "white_goal")]


def test_load_tallies_csv_rejects_unknown_category(tmp_path):
    csv_path = tmp_path / "t.csv"
    csv_path.write_text("timestamp,count,count_change,action\n1788407400000,1,1,CLICK\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown category"):
        load_tallies_csv(csv_path, "red_goal")


def test_resolve_marks_snaps_to_nearby_audio_peak():
    start = datetime(2026, 8, 23, 8, 0, 0)
    chunks = [_chunk(1, start, duration=1000.0, global_start=0.0)]
    # Mark pressed at global t=130, 8s after the event audio caught at t=122
    # -- within the ~5-10s press delay Kaveh measured on the bench.
    mark = Mark(sequence=1, timestamp=_local(2026, 8, 23, 8, 2, 10), category="white_goal")
    audio_peaks = [GlobalPeak(time_seconds=122.0, score=1.0)]
    marks_cfg = MarksConfig(snap_lookback_seconds=15.0, snap_lookahead_seconds=2.0)
    timeline_cfg = TimelineConfig(lookback_seconds=6.0, post_peak_seconds=5.0)

    [resolved] = resolve_marks([mark], chunks, audio_peaks, marks_cfg, timeline_cfg)

    assert resolved.anchor == "audio_peak"
    assert resolved.interval.start_seconds == 116.0
    assert resolved.interval.end_seconds == 127.0


def test_resolve_marks_ignores_a_louder_peak_outside_the_tight_snap_window():
    """The reason the snap window is decoupled from the fallback window: a
    louder but far-older peak (an unrelated shout 40s earlier) must NOT win
    over a quieter peak inside the plausible press-delay window."""
    start = datetime(2026, 8, 23, 8, 0, 0)
    chunks = [_chunk(1, start, duration=1000.0, global_start=0.0)]
    mark = Mark(sequence=1, timestamp=_local(2026, 8, 23, 8, 2, 10), category="white_goal")  # global=130s
    audio_peaks = [
        GlobalPeak(time_seconds=90.0, score=9.9),  # much louder, but 40s back
        GlobalPeak(time_seconds=124.0, score=0.6),  # the real one
    ]
    marks_cfg = MarksConfig(snap_lookback_seconds=15.0, snap_lookahead_seconds=2.0)
    timeline_cfg = TimelineConfig(lookback_seconds=6.0, post_peak_seconds=5.0)

    [resolved] = resolve_marks([mark], chunks, audio_peaks, marks_cfg, timeline_cfg)

    assert resolved.anchor == "audio_peak"
    assert resolved.interval.peaks[0].time_seconds == 124.0


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


def test_union_with_audio_is_a_no_op_without_marks():
    """The graceful-degradation guarantee: forget the watch, get exactly
    today's audio-only behavior."""
    audio = [Interval(start_seconds=10.0, end_seconds=20.0), Interval(start_seconds=50.0, end_seconds=60.0)]

    intervals, sources = union_with_audio(audio, [])

    assert intervals == audio
    assert sources == ["audio", "audio"]


def test_union_with_audio_labels_a_corroborated_candidate_both():
    start = datetime(2026, 8, 23, 8, 0, 0)
    chunks = [_chunk(1, start, duration=1000.0, global_start=0.0)]
    peak = GlobalPeak(time_seconds=122.0, score=1.0)
    audio = [Interval(start_seconds=116.0, end_seconds=127.0, peaks=[peak])]
    mark = Mark(sequence=1, timestamp=_local(2026, 8, 23, 8, 2, 10), category="white_goal")
    resolved = resolve_marks([mark], chunks, [peak], MarksConfig(), TimelineConfig())

    intervals, sources = union_with_audio(audio, resolved)

    assert len(intervals) == 1
    assert sources == ["both"]


def test_union_with_audio_adds_a_new_candidate_audio_missed():
    start = datetime(2026, 8, 23, 8, 0, 0)
    chunks = [_chunk(1, start, duration=1000.0, global_start=0.0)]
    audio = [Interval(start_seconds=10.0, end_seconds=20.0)]
    mark = Mark(sequence=1, timestamp=_local(2026, 8, 23, 8, 2, 10), category="white_goal")
    resolved = resolve_marks([mark], chunks, [], MarksConfig(), TimelineConfig())

    intervals, sources = union_with_audio(audio, resolved)

    assert sources == ["audio", "mark"]
    assert intervals[1].start_seconds == 70.0  # the wide fallback window


def test_union_with_audio_does_not_credit_audio_for_a_merely_overlapping_candidate():
    """A fixed_window mark's 60s fallback often overlaps an unrelated older
    candidate. Provenance follows the anchor, not geometry -- otherwise this
    would be mislabeled `both` and silently erase a recall miss."""
    start = datetime(2026, 8, 23, 8, 0, 0)
    chunks = [_chunk(1, start, duration=1000.0, global_start=0.0)]
    # An unrelated candidate at 75-85s, well inside the mark's 70-135s
    # fallback window but outside its 115-132s snap window.
    stale_peak = GlobalPeak(time_seconds=80.0, score=5.0)
    audio = [Interval(start_seconds=75.0, end_seconds=85.0, peaks=[stale_peak])]
    mark = Mark(sequence=1, timestamp=_local(2026, 8, 23, 8, 2, 10), category="white_goal")
    resolved = resolve_marks([mark], chunks, [stale_peak], MarksConfig(), TimelineConfig())

    intervals, sources = union_with_audio(audio, resolved)

    # Sorted by start, so the mark's wide window (from 70s) sorts first.
    assert sources == ["mark", "audio"]
    assert intervals[0].start_seconds == 70.0


def test_union_with_audio_skips_marks_in_unrecorded_gaps():
    chunks = [
        _chunk(1, datetime(2026, 8, 23, 8, 0, 0), duration=100.0, global_start=0.0),
        _chunk(2, datetime(2026, 8, 23, 8, 5, 0), duration=100.0, global_start=100.0),
    ]
    mark = Mark(sequence=1, timestamp=_local(2026, 8, 23, 8, 3, 0), category="white_goal")
    resolved = resolve_marks([mark], chunks, [], MarksConfig(), TimelineConfig())

    intervals, sources = union_with_audio([], resolved)

    assert intervals == []
    assert sources == []


def test_measure_clock_offset_recovers_a_known_skew():
    start = datetime(2026, 8, 23, 8, 0, 0)
    chunks = [_chunk(1, start, duration=1000.0, global_start=0.0)]
    # Clap tapped at 8:00:50 (global 50s), but the camera's own timeline puts
    # the clap transient at 47s -- a 3s skew the ritual exists to expose.
    clap = Mark(sequence=1, timestamp=_local(2026, 8, 23, 8, 0, 50), category="moment")
    audio_peaks = [GlobalPeak(time_seconds=47.0, score=9.0)]
    resolved = resolve_marks([clap], chunks, audio_peaks, MarksConfig(), TimelineConfig())

    offset = measure_clock_offset(resolved, audio_peaks, window_seconds=30.0)

    assert offset is not None
    assert offset.offset_seconds == pytest.approx(-3.0)


def test_measure_clock_offset_returns_none_when_skew_exceeds_the_window():
    start = datetime(2026, 8, 23, 8, 0, 0)
    chunks = [_chunk(1, start, duration=1000.0, global_start=0.0)]
    clap = Mark(sequence=1, timestamp=_local(2026, 8, 23, 8, 0, 50), category="moment")
    audio_peaks = [GlobalPeak(time_seconds=500.0, score=9.0)]
    resolved = resolve_marks([clap], chunks, audio_peaks, MarksConfig(), TimelineConfig())

    assert measure_clock_offset(resolved, audio_peaks, window_seconds=30.0) is None


def test_clock_offset_shifts_marks_onto_the_camera_timeline():
    start = datetime(2026, 8, 23, 8, 0, 0)
    chunks = [_chunk(1, start, duration=1000.0, global_start=0.0)]
    mark = Mark(sequence=1, timestamp=_local(2026, 8, 23, 8, 0, 50), category="white_goal")

    [uncorrected] = resolve_marks([mark], chunks, [], MarksConfig(), TimelineConfig())
    [corrected] = resolve_marks([mark], chunks, [], MarksConfig(), TimelineConfig(), clock_offset_seconds=-3.0)

    assert uncorrected.global_seconds == 50.0
    assert corrected.global_seconds == 47.0


def test_clock_offset_can_move_a_mark_into_an_unrecorded_gap():
    # Applied to the wall clock BEFORE the gap check, not to the mapped
    # offset after it -- otherwise a corrected mark could be reported as
    # recorded when it actually falls in the gap.
    chunks = [
        _chunk(1, datetime(2026, 8, 23, 8, 0, 0), duration=100.0, global_start=0.0),
        _chunk(2, datetime(2026, 8, 23, 8, 5, 0), duration=100.0, global_start=100.0),
    ]
    mark = Mark(sequence=1, timestamp=_local(2026, 8, 23, 8, 1, 30), category="white_goal")  # inside chunk 1

    [corrected] = resolve_marks([mark], chunks, [], MarksConfig(), TimelineConfig(), clock_offset_seconds=60.0)

    assert corrected.anchor == "unrecorded_gap"


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
