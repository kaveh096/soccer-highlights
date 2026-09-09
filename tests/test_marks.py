from datetime import datetime
from pathlib import Path

import pytest

from soccer_highlights.config import MarksConfig, TimelineConfig
from soccer_highlights.discovery import Chunk
from soccer_highlights.marks import (
    RECORDING_TZ,
    Mark,
    extend_for_score_flip,
    load_marks_csv,
    load_tallies_csv,
    measure_clock_offset,
    merge_tally_marks,
    resolve_marks,
    resolve_undos,
    is_near_cam_goal,
    marks_outside_their_clip,
    review_sort_key,
    review_tier,
    tap_claim,
    score_events,
    score_for_interval,
    union_with_audio,
    union_with_audio_detailed,
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

    # Decrements are NOT auto-resolved (2026-09-07) -- every positive press
    # survives as a Mark regardless of any -1 rows around it.
    assert [m.category for m in marks] == ["white_goal"] * 5
    assert [m.timestamp.timestamp() for m in marks] == [
        1788407495354 / 1000,
        1788407522303 / 1000,
        1788407522616 / 1000,
        1788407632369 / 1000,
        1788407755313 / 1000,
    ]


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


def test_load_tallies_csv_decrements_are_not_auto_resolved(tmp_path, capsys):
    # 2026-09-07 (Kaveh, after Sep-06's real data): a decrement is no longer
    # popped against the most-recently-active press -- it's counted and
    # warned about, but every positive press survives regardless.
    white = tmp_path / "white.csv"
    white.write_text(
        "timestamp,count,count_change,action\n1788407400000,1,1,CLICK\n1788407600000,0,-1,CLICK\n", encoding="utf-8"
    )

    marks = load_tallies_csv(white, "white_goal")

    assert [m.category for m in marks] == ["white_goal"]  # the +1 survives despite the later -1
    assert "1 decrement(s)" in capsys.readouterr().out


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
    assert intervals[1].start_seconds == 115.0  # the wide fallback window (15s lookback)


def test_union_with_audio_does_not_credit_audio_for_a_merely_overlapping_candidate():
    """A fixed_window mark's fallback can still overlap an unrelated older
    candidate on a wide-enough config. Provenance follows the anchor, not
    geometry -- otherwise this would be mislabeled `both` and silently erase
    a recall miss. Uses an explicit wide lookback (the pre-2026-09-07
    default) so the scenario reproduces regardless of the production
    default's current value."""
    start = datetime(2026, 8, 23, 8, 0, 0)
    chunks = [_chunk(1, start, duration=1000.0, global_start=0.0)]
    # An unrelated candidate at 75-85s, well inside the mark's 70-135s
    # fallback window but outside its 115-132s snap window.
    stale_peak = GlobalPeak(time_seconds=80.0, score=5.0)
    audio = [Interval(start_seconds=75.0, end_seconds=85.0, peaks=[stale_peak])]
    mark = Mark(sequence=1, timestamp=_local(2026, 8, 23, 8, 2, 10), category="white_goal")
    wide_cfg = MarksConfig(goal_lookback_seconds=60.0)
    resolved = resolve_marks([mark], chunks, [stale_peak], wide_cfg, TimelineConfig())

    intervals, sources = union_with_audio(audio, resolved)

    # Sorted by start, so the mark's wide window (from 70s) sorts first.
    assert sources == ["mark", "audio"]
    assert intervals[0].start_seconds == 70.0


def test_union_with_audio_detailed_merges_close_same_category_fixed_window_marks():
    """2026-09-07: two black_goal taps 6.8s apart on a real game (neither
    found an audio peak) rendered as two 90%-overlapping clips. Same-category
    fixed-window fallbacks that overlap should collapse into one clip."""
    chunks = [_chunk(1, datetime(2026, 8, 23, 8, 0, 0), duration=1000.0, global_start=0.0)]
    marks = [
        _goal_mark(1, 8, 8, 15, "black_goal"),  # global 495.0
        _goal_mark(2, 8, 8, 22, "black_goal"),  # global 502.0, 7s later
    ]
    resolved = resolve_marks(marks, chunks, [], MarksConfig(), TimelineConfig())

    combined = union_with_audio_detailed([], resolved)

    assert len(combined) == 1
    interval, source, owner = combined[0]
    assert source == "mark"
    assert owner.mark.sequence == 1  # the earlier tap represents the merged group
    # Spans both taps' individual fallback windows (15s lookback/5s lookahead).
    assert interval.start_seconds == 480.0  # 495 - 15
    assert interval.end_seconds == 507.0  # 502 + 5


def test_union_with_audio_detailed_does_not_merge_across_categories():
    """A white_goal and a black_goal close together are still two different
    events, even if their fallback windows overlap."""
    chunks = [_chunk(1, datetime(2026, 8, 23, 8, 0, 0), duration=1000.0, global_start=0.0)]
    marks = [
        _goal_mark(1, 8, 8, 15, "white_goal"),
        _goal_mark(2, 8, 8, 22, "black_goal"),
    ]
    resolved = resolve_marks(marks, chunks, [], MarksConfig(), TimelineConfig())

    combined = union_with_audio_detailed([], resolved)

    assert len(combined) == 2
    assert {owner.mark.category for _, _, owner in combined} == {"white_goal", "black_goal"}


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


# --- Score counter -------------------------------------------------------
#
# The burned-in score flips at the TAP, not at the audio peak (Kaveh's call,
# 2026-09-05): a broadcast graphic also lags the goal, and tap-anchoring
# behaves the same whether or not audio found a peak.


def _goal_mark(seq: int, h: int, mi: int, s: int, category: str = "white_goal") -> Mark:
    return Mark(sequence=seq, timestamp=_local(2026, 8, 23, h, mi, s), category=category)


def test_score_events_counts_only_goals_in_time_order():
    chunks = [_chunk(1, datetime(2026, 8, 23, 8, 0, 0), duration=1000.0, global_start=0.0)]
    marks_cfg, timeline_cfg = MarksConfig(), TimelineConfig()
    marks = [
        _goal_mark(1, 8, 2, 0, "black_goal"),  # global 120
        _goal_mark(2, 8, 1, 0, "white_goal"),  # global 60, tapped later but earlier in time
        _goal_mark(3, 8, 3, 0, "moment"),  # never scores
    ]

    resolved = resolve_marks(marks, chunks, [], marks_cfg, timeline_cfg)

    assert score_events(resolved) == [(60.0, "white"), (120.0, "black")]


def test_score_events_skips_a_mark_in_an_unrecorded_gap():
    # No timeline position exists for it, so it cannot be placed on a clip.
    chunks = [
        _chunk(1, datetime(2026, 8, 23, 8, 0, 0), duration=100.0, global_start=0.0),
        _chunk(2, datetime(2026, 8, 23, 8, 5, 0), duration=100.0, global_start=100.0),
    ]
    marks_cfg, timeline_cfg = MarksConfig(), TimelineConfig()
    resolved = resolve_marks([_goal_mark(1, 8, 3, 0)], chunks, [], marks_cfg, timeline_cfg)

    assert resolved[0].anchor == "unrecorded_gap"
    assert score_events(resolved) == []


def test_score_for_interval_reports_the_score_before_the_clip_and_flips_at_the_tap():
    events = [(50.0, "white"), (80.0, "black"), (125.0, "white")]
    interval = Interval(start_seconds=120.0, end_seconds=134.0)

    cols = score_for_interval(interval, events)

    assert (cols.white, cols.black) == (1, 1)  # the 125s goal has NOT landed yet at clip start
    assert cols.flip_seconds == 5.0
    assert cols.flip_team == "white"


def test_score_for_interval_is_static_when_no_goal_falls_inside():
    events = [(50.0, "white")]

    cols = score_for_interval(Interval(start_seconds=200.0, end_seconds=214.0), events)

    assert (cols.white, cols.black) == (1, 0)
    assert cols.flip_seconds is None
    assert cols.flip_team == ""


def test_score_for_interval_falls_back_to_the_midpoint_when_the_tap_is_outside_its_clip():
    """A press too slow for extend_for_score_flip to cover without blowing
    its cap. The clip still shows a goal, so the score still has to move --
    but neither the tap nor audio gives a trustworthy instant, so it goes to
    a position that depends on neither."""
    chunks = [_chunk(1, datetime(2026, 8, 23, 8, 0, 0), duration=1000.0, global_start=0.0)]
    mark = _goal_mark(1, 8, 2, 30, "black_goal")  # global 150
    audio_peaks = [GlobalPeak(time_seconds=140.0, score=1.0)]
    resolved = resolve_marks([mark], chunks, audio_peaks, MarksConfig(), TimelineConfig())
    interval = Interval(start_seconds=131.0, end_seconds=145.0)  # tap at 150 is past the end

    cols = score_for_interval(interval, score_events(resolved), resolved[0])

    assert cols.flip_seconds == 7.0  # midpoint of a 14s clip
    assert cols.flip_team == "black"


def test_review_tier_untapped_clip_still_falls_back_to_goal_this_end():
    # No tap at all -- nothing to "value over Gemini", so goal_this_end alone
    # still decides tier 1, same as before 2026-09-07.
    assert review_tier(mark_category="", gemini_score=1, goal_this_end=True) == 1
    assert review_tier(mark_category="", gemini_score=4, goal_this_end=False) == 3
    assert review_tier(mark_category="", gemini_score=2, goal_this_end=False) == 5


def test_review_tier_a_moment_tap_never_reaches_tier_1_even_if_gemini_calls_it_a_goal():
    # 2026-09-07 (r03/r07 on Sep-06): an explicit non-goal tap overrides a
    # contradicting Gemini goal claim outright -- it doesn't just fail to
    # promote it, moment can never be tier 1.
    assert review_tier(mark_category="moment", gemini_score=4, goal_this_end=True) == 2
    assert review_tier(mark_category="moment", gemini_score=2, goal_this_end=True) == 4


def test_review_tier_moment_and_score_bands_below_tier_1():
    assert review_tier(mark_category="moment", gemini_score=3, goal_this_end=False) == 2
    assert review_tier(mark_category="", gemini_score=4, goal_this_end=False) == 3
    assert review_tier(mark_category="moment", gemini_score=1, goal_this_end=False) == 4
    assert review_tier(mark_category="", gemini_score=2, goal_this_end=False) == 5


def test_review_tier_goal_tap_ignores_gemini_when_near_cam_info_is_available():
    # 2026-09-07: once a real goal tap AND the per-game camera config are
    # both available, goal_this_end is not consulted at all -- taps win the
    # conflict in both directions.
    assert review_tier(mark_category="black_goal", gemini_score=1, goal_this_end=False, is_near_cam_goal=True) == 1
    assert review_tier(mark_category="black_goal", gemini_score=5, goal_this_end=True, is_near_cam_goal=False) == 3


def test_review_tier_goal_tap_falls_back_to_goal_this_end_without_camera_config():
    # is_near_cam_goal=None means the per-game camera config wasn't given --
    # degrades to the pre-2026-09-07 goal_this_end-driven behavior.
    assert review_tier(mark_category="white_goal", gemini_score=2, goal_this_end=True, is_near_cam_goal=None) == 1
    assert review_tier(mark_category="white_goal", gemini_score=2, goal_this_end=False, is_near_cam_goal=None) == 5


def test_review_tier_the_clap_sync_moment_skips_the_moment_boost():
    # Same inputs as a normal promoted moment, except is_sync_clap=True --
    # the kickoff clap-sync tap is a timing reference, not a real candidate,
    # so it must NOT land in tier 2 or 4 just for being mark_category=moment.
    assert review_tier(mark_category="moment", gemini_score=3, goal_this_end=False, is_sync_clap=True) == 5
    assert review_tier(mark_category="moment", gemini_score=1, goal_this_end=False, is_sync_clap=True) == 5
    # A high enough plain score still earns tier 3 on its own merits.
    assert review_tier(mark_category="moment", gemini_score=4, goal_this_end=False, is_sync_clap=True) == 3


def test_review_sort_key_orders_tiers_then_score_then_time():
    clips = [
        ("far_goal_score2", "black_goal", 2, False, 500.0, False),
        ("near_goal_score3", "black_goal", 3, False, 300.0, True),
        ("moment_score3", "moment", 3, False, 100.0, False),
        ("near_goal_score5", "white_goal", 5, False, 10.0, True),
        ("plain_score4", "", 4, False, 50.0, False),
        ("moment_score1", "moment", 1, False, 20.0, False),
    ]
    ranked = sorted(
        clips,
        key=lambda c: review_sort_key(
            mark_category=c[1], gemini_score=c[2], goal_this_end=c[3], start_seconds=c[4], is_near_cam_goal=c[5]
        ),
    )
    assert [name for name, *_ in ranked] == [
        "near_goal_score5",  # tier 1, higher score
        "near_goal_score3",  # tier 1, lower score
        "moment_score3",  # tier 2
        "plain_score4",  # tier 3
        "moment_score1",  # tier 4
        "far_goal_score2",  # tier 5
    ]


def test_is_near_cam_goal_by_half_and_team():
    # 2026-09-06 setup: cam behind black's goal in half 1 (white_goal near),
    # swapped in half 2 (black_goal near).
    assert is_near_cam_goal("white_goal", 100.0, halftime_seconds=1000.0, near_cam_team_first_half="white") is True
    assert is_near_cam_goal("black_goal", 100.0, halftime_seconds=1000.0, near_cam_team_first_half="white") is False
    assert is_near_cam_goal("black_goal", 1500.0, halftime_seconds=1000.0, near_cam_team_first_half="white") is True
    assert is_near_cam_goal("white_goal", 1500.0, halftime_seconds=1000.0, near_cam_team_first_half="white") is False


def test_tap_claim_by_category():
    assert tap_claim("white_goal") == "the white team scored a goal"
    assert tap_claim("black_goal") == "the dark/black team scored a goal"
    assert tap_claim("moment") == "something notable happened here, but it was NOT tapped as a goal"
    assert tap_claim("") is None


def test_is_near_cam_goal_false_when_per_game_info_is_missing():
    assert is_near_cam_goal("white_goal", 100.0, halftime_seconds=None, near_cam_team_first_half="white") is False
    assert is_near_cam_goal("white_goal", 100.0, halftime_seconds=1000.0, near_cam_team_first_half=None) is False
    assert is_near_cam_goal("moment", 100.0, halftime_seconds=1000.0, near_cam_team_first_half="white") is False


def test_extend_for_score_flip_stretches_a_clip_to_cover_its_own_tap():
    chunks = [_chunk(1, datetime(2026, 8, 23, 8, 0, 0), duration=1000.0, global_start=0.0)]
    # Tapped 8s after the peak audio caught -- inside the 10s cap.
    mark = _goal_mark(1, 8, 2, 10)  # global 130
    peak = GlobalPeak(time_seconds=122.0, score=1.0)
    marks_cfg = MarksConfig(score_flip_cap_seconds=10.0, score_flip_tail_seconds=2.0)
    resolved = resolve_marks([mark], chunks, [peak], marks_cfg, TimelineConfig(lookback_seconds=9.0, post_peak_seconds=5.0))
    audio = [Interval(start_seconds=113.0, end_seconds=127.0, peaks=[peak])]

    combined = union_with_audio_detailed(audio, resolved)
    n = extend_for_score_flip(combined, marks_cfg)

    assert n == 1
    assert combined[0][1] == "both"
    assert combined[0][0].end_seconds == 132.0  # tap 130 + 2s tail, so the flip is readable


def test_extend_for_score_flip_respects_the_cap_on_a_slow_press():
    """Without the cap, a press forgotten for minutes would stretch a 14s
    highlight into a several-minute clip."""
    chunks = [_chunk(1, datetime(2026, 8, 23, 8, 0, 0), duration=1000.0, global_start=0.0)]
    mark = _goal_mark(1, 8, 2, 25)  # global 145, i.e. 23s after the peak
    peak = GlobalPeak(time_seconds=122.0, score=1.0)
    marks_cfg = MarksConfig(score_flip_cap_seconds=10.0, snap_lookback_seconds=30.0)
    resolved = resolve_marks([mark], chunks, [peak], marks_cfg, TimelineConfig(lookback_seconds=9.0, post_peak_seconds=5.0))
    audio = [Interval(start_seconds=113.0, end_seconds=127.0, peaks=[peak])]

    combined = union_with_audio_detailed(audio, resolved)

    assert extend_for_score_flip(combined, marks_cfg) == 0
    assert combined[0][0].end_seconds == 127.0  # untouched


def test_marks_outside_their_clip_flags_a_capped_slow_press():
    # Same scenario as the cap test above -- the tap at global 145 never
    # gets pulled inside the clip (ends at 127), so it should be flagged.
    chunks = [_chunk(1, datetime(2026, 8, 23, 8, 0, 0), duration=1000.0, global_start=0.0)]
    mark = _goal_mark(1, 8, 2, 25)  # global 145
    peak = GlobalPeak(time_seconds=122.0, score=1.0)
    marks_cfg = MarksConfig(score_flip_cap_seconds=10.0, snap_lookback_seconds=30.0)
    resolved = resolve_marks([mark], chunks, [peak], marks_cfg, TimelineConfig(lookback_seconds=9.0, post_peak_seconds=5.0))
    audio = [Interval(start_seconds=113.0, end_seconds=127.0, peaks=[peak])]
    combined = union_with_audio_detailed(audio, resolved)
    extend_for_score_flip(combined, marks_cfg)

    outside = marks_outside_their_clip(combined)

    assert len(outside) == 1
    assert outside[0].mark.mark.sequence == 1
    assert outside[0].distance_seconds == 145.0 - 127.0  # positive: tap lands after the clip ends


def test_marks_outside_their_clip_is_empty_when_the_tap_is_covered():
    # The extend-for-score-flip success case from the earlier test: the tap
    # ends up inside the (now-extended) clip, so nothing should be flagged.
    chunks = [_chunk(1, datetime(2026, 8, 23, 8, 0, 0), duration=1000.0, global_start=0.0)]
    mark = _goal_mark(1, 8, 2, 10)  # global 130, 8s after the peak -- inside the cap
    peak = GlobalPeak(time_seconds=122.0, score=1.0)
    marks_cfg = MarksConfig(score_flip_cap_seconds=10.0, score_flip_tail_seconds=2.0)
    resolved = resolve_marks([mark], chunks, [peak], marks_cfg, TimelineConfig(lookback_seconds=9.0, post_peak_seconds=5.0))
    audio = [Interval(start_seconds=113.0, end_seconds=127.0, peaks=[peak])]
    combined = union_with_audio_detailed(audio, resolved)
    extend_for_score_flip(combined, marks_cfg)

    assert marks_outside_their_clip(combined) == []


def test_marks_outside_their_clip_ignores_fixed_window_marks():
    # A fixed_window mark's own tap is inside its window by construction --
    # never flagged, even with no audio at all.
    chunks = [_chunk(1, datetime(2026, 8, 23, 8, 0, 0), duration=1000.0, global_start=0.0)]
    resolved = resolve_marks([_goal_mark(1, 8, 2, 10)], chunks, [], MarksConfig(), TimelineConfig())
    combined = union_with_audio_detailed([], resolved)

    assert marks_outside_their_clip(combined) == []


def test_extend_for_score_flip_leaves_fixed_window_clips_alone():
    """A no-peak clip already ends goal_lookahead_seconds after its own tap,
    so the tap is inside it by construction."""
    chunks = [_chunk(1, datetime(2026, 8, 23, 8, 0, 0), duration=1000.0, global_start=0.0)]
    marks_cfg = MarksConfig()
    resolved = resolve_marks([_goal_mark(1, 8, 2, 10)], chunks, [], marks_cfg, TimelineConfig())

    combined = union_with_audio_detailed([], resolved)
    end_before = combined[0][0].end_seconds

    assert extend_for_score_flip(combined, marks_cfg) == 0
    assert combined[0][0].end_seconds == end_before
    assert combined[0][1] == "mark"


def test_undone_goal_never_reaches_the_score():
    marks = [_goal_mark(1, 8, 1, 0), Mark(sequence=2, timestamp=_local(2026, 8, 23, 8, 1, 5), category="undo")]
    chunks = [_chunk(1, datetime(2026, 8, 23, 8, 0, 0), duration=1000.0, global_start=0.0)]

    resolved = resolve_marks(resolve_undos(marks), chunks, [], MarksConfig(), TimelineConfig())

    assert score_events(resolved) == []
