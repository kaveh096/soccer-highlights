import csv
import json
from pathlib import Path

from soccer_highlights.golden import (
    GoldenEvent,
    build_golden_events,
    load_golden_events,
    save_golden_events,
    score_intervals_against_golden,
)
from soccer_highlights.timeline import Interval


def _write_strategy(review_root: Path, name: str, rows: list[tuple[float, float, str]]) -> None:
    """rows: list of (start, end, verdict). Peak time is placed mid-clip."""
    strategy_dir = review_root / name
    strategy_dir.mkdir(parents=True, exist_ok=True)
    events = [
        {
            "start_seconds": start,
            "end_seconds": end,
            "duration_seconds": end - start,
            "peaks": [{"time_seconds": (start + end) / 2, "score": 1.0}],
        }
        for start, end, _ in rows
    ]
    with open(strategy_dir / "events.json", "w", encoding="utf-8") as f:
        json.dump(events, f)

    with open(strategy_dir / "review_sheet.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["clip_file", "start_seconds", "end_seconds", "duration_seconds", "max_peak_score", "verdict", "notes"])
        for i, (start, end, verdict) in enumerate(rows, start=1):
            writer.writerow([f"clip_{i:03d}.mp4", start, end, end - start, 1.0, verdict, ""])


def _write_negatives(review_root: Path, rows: list[tuple[float, float, str, str]]) -> None:
    """rows: list of (start, end, verdict, notes)."""
    neg_dir = review_root / "negatives"
    neg_dir.mkdir(parents=True, exist_ok=True)
    events = [
        {"start_seconds": start, "end_seconds": end, "duration_seconds": end - start, "peaks": []}
        for start, end, _, _ in rows
    ]
    with open(neg_dir / "events.json", "w", encoding="utf-8") as f:
        json.dump(events, f)

    with open(neg_dir / "review_sheet.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["clip_file", "start_seconds", "end_seconds", "duration_seconds", "max_peak_score", "verdict", "notes"])
        for i, (start, end, verdict, notes) in enumerate(rows, start=1):
            writer.writerow([f"clip_{i:03d}.mp4", start, end, end - start, 0.0, verdict, notes])


def test_build_golden_events_merges_close_tp_anchors_and_includes_fn_offsets(tmp_path):
    _write_strategy(tmp_path, "strike_loose", [(100.0, 111.0, "TP"), (200.0, 211.0, "FP")])
    _write_strategy(tmp_path, "crowd_loose", [(101.0, 112.0, "TP")])  # same real event as strike_loose's first TP
    _write_negatives(tmp_path, [(300.0, 350.0, "FN", "20 seconds in there was a shot"), (400.0, 450.0, "TN", "")])

    events = build_golden_events(tmp_path, cluster_window_seconds=12.0)
    times = sorted(e.time_seconds for e in events)

    assert len(events) == 2
    assert abs(times[0] - 106.0) < 1.0  # mean of the two close TP peak times (~105.5, ~106.5)
    assert times[1] == 320.0  # 300 + 20


def test_build_golden_events_fn_without_offset_raises(tmp_path):
    _write_negatives(tmp_path, [(0.0, 50.0, "FN", "no offset mentioned here")])
    try:
        build_golden_events(tmp_path)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_save_and_load_golden_events_roundtrip(tmp_path):
    events = [GoldenEvent(time_seconds=12.5, sources=["a/clip_001.mp4"]), GoldenEvent(time_seconds=99.0, sources=["b/clip_002.mp4", "c/clip_003.mp4"])]
    path = tmp_path / "golden.json"
    save_golden_events(events, path)
    loaded = load_golden_events(path)
    assert loaded == events


def test_score_intervals_against_golden_counts_tp_fp_fn():
    golden = [GoldenEvent(time_seconds=10.0, sources=[]), GoldenEvent(time_seconds=50.0, sources=[]), GoldenEvent(time_seconds=90.0, sources=[])]
    intervals = [
        Interval(start_seconds=5.0, end_seconds=15.0),  # contains golden[0] -> TP
        Interval(start_seconds=20.0, end_seconds=30.0),  # contains nothing -> FP
        Interval(start_seconds=48.0, end_seconds=52.0),  # contains golden[1] -> TP
        # golden[2] at 90.0 is never covered -> FN
    ]

    score = score_intervals_against_golden(intervals, golden)

    assert score.total_clips == 3
    assert score.true_positives == 2
    assert score.false_positives == 1
    assert score.false_negatives == 1
    assert score.precision == 2 / 3
    assert score.recall == 2 / 3
    assert score.f1 is not None


def test_score_intervals_against_golden_empty_intervals():
    golden = [GoldenEvent(time_seconds=10.0, sources=[])]
    score = score_intervals_against_golden([], golden)
    assert score.precision is None
    assert score.recall == 0.0
    assert score.false_negatives == 1


def test_score_intervals_against_golden_reports_duration_for_giant_blob():
    # A single interval spanning almost the whole recording trivially
    # contains every golden event -- perfect P/R/F1 despite being useless.
    # Duration fields must expose that so it isn't mistaken for a good result.
    golden = [GoldenEvent(time_seconds=t, sources=[]) for t in [10.0, 500.0, 990.0]]
    blob = [Interval(start_seconds=0.0, end_seconds=1000.0)]

    score = score_intervals_against_golden(blob, golden)

    assert score.precision == 1.0
    assert score.recall == 1.0
    assert score.f1 == 1.0
    assert score.total_clips == 1
    assert score.max_clip_duration_seconds == 1000.0
    assert score.mean_clip_duration_seconds == 1000.0
