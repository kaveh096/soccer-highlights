import json

from soccer_highlights.config import TimelineConfig, VisionConfig
from soccer_highlights.timeline import GlobalPeak, Interval
from soccer_highlights.vision import (
    VisionVerdict,
    _evenly_spaced_offsets,
    _parse_verdict_json,
    decide_confirm,
    decide_scan,
    refine_with_vision,
)


def test_evenly_spaced_offsets_midpoints():
    assert _evenly_spaced_offsets(12.0, 3) == [2.0, 6.0, 10.0]


def test_evenly_spaced_offsets_single_frame_is_midpoint():
    assert _evenly_spaced_offsets(10.0, 1) == [5.0]


def test_evenly_spaced_offsets_five_frames():
    offsets = _evenly_spaced_offsets(10.0, 5)
    assert offsets == [1.0, 3.0, 5.0, 7.0, 9.0]


def test_evenly_spaced_offsets_zero_duration():
    assert _evenly_spaced_offsets(0.0, 3) == [0.0, 0.0, 0.0]


def test_evenly_spaced_offsets_zero_count():
    assert _evenly_spaced_offsets(10.0, 0) == []


def test_parse_verdict_json_confirm_well_formed():
    raw = json.dumps({"is_event": True, "confidence": 0.9, "rationale": "looks real"})
    verdict = _parse_verdict_json(raw, mode="confirm")
    assert verdict.is_event is True
    assert verdict.confidence == 0.9
    assert verdict.frame_index is None
    assert verdict.rationale == "looks real"


def test_parse_verdict_json_scan_well_formed():
    raw = json.dumps({"is_event": True, "frame_index": 2, "confidence": 0.8, "rationale": "goal"})
    verdict = _parse_verdict_json(raw, mode="scan")
    assert verdict.frame_index == 2


def test_parse_verdict_json_strips_code_fence():
    raw = "```json\n" + json.dumps({"is_event": False, "confidence": 0.6, "rationale": "practice shot"}) + "\n```"
    verdict = _parse_verdict_json(raw, mode="confirm")
    assert verdict.is_event is False
    assert verdict.confidence == 0.6


def test_parse_verdict_json_malformed_text_raises():
    try:
        _parse_verdict_json("not json at all", mode="confirm")
        assert False, "expected an exception"
    except (json.JSONDecodeError, ValueError):
        pass


def test_parse_verdict_json_missing_field_raises():
    try:
        _parse_verdict_json(json.dumps({"confidence": 0.5}), mode="confirm")
        assert False, "expected an exception"
    except KeyError:
        pass


def test_parse_verdict_json_confidence_out_of_range_raises():
    try:
        _parse_verdict_json(json.dumps({"is_event": True, "confidence": 1.5}), mode="confirm")
        assert False, "expected an exception"
    except ValueError:
        pass


def _cfg(**overrides) -> VisionConfig:
    cfg = VisionConfig()
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_decide_confirm_fails_open_on_none_verdict():
    interval = Interval(start_seconds=0.0, end_seconds=10.0)
    assert decide_confirm(interval, None, _cfg()) is True


def test_decide_confirm_keeps_low_confidence_false_positive():
    interval = Interval(start_seconds=0.0, end_seconds=10.0)
    verdict = VisionVerdict(is_event=False, confidence=0.5)  # below default 0.75 threshold
    assert decide_confirm(interval, verdict, _cfg(drop_confidence_threshold=0.75)) is True


def test_decide_confirm_drops_high_confidence_false_positive():
    interval = Interval(start_seconds=0.0, end_seconds=10.0)
    verdict = VisionVerdict(is_event=False, confidence=0.9)
    assert decide_confirm(interval, verdict, _cfg(drop_confidence_threshold=0.75)) is False


def test_decide_confirm_keeps_true_event_regardless_of_confidence():
    interval = Interval(start_seconds=0.0, end_seconds=10.0)
    verdict = VisionVerdict(is_event=True, confidence=0.99)
    assert decide_confirm(interval, verdict, _cfg()) is True


def test_decide_scan_returns_none_for_no_verdict_or_non_event():
    gap = Interval(start_seconds=100.0, end_seconds=112.0)
    assert decide_scan(gap, None, _cfg()) is None
    assert decide_scan(gap, VisionVerdict(is_event=False, confidence=0.9), _cfg()) is None


def test_decide_scan_returns_none_below_add_threshold():
    gap = Interval(start_seconds=100.0, end_seconds=112.0)
    verdict = VisionVerdict(is_event=True, confidence=0.5, frame_index=1)
    assert decide_scan(gap, verdict, _cfg(add_confidence_threshold=0.75)) is None


def test_decide_scan_returns_none_without_frame_index():
    gap = Interval(start_seconds=100.0, end_seconds=112.0)
    verdict = VisionVerdict(is_event=True, confidence=0.9, frame_index=None)
    assert decide_scan(gap, verdict, _cfg()) is None


def test_decide_scan_returns_none_for_out_of_range_frame_index():
    gap = Interval(start_seconds=100.0, end_seconds=112.0)
    verdict = VisionVerdict(is_event=True, confidence=0.9, frame_index=99)
    assert decide_scan(gap, verdict, _cfg(frames_per_window=5)) is None


def test_decide_scan_maps_frame_index_to_absolute_time():
    gap = Interval(start_seconds=100.0, end_seconds=112.0)  # 12s window
    verdict = VisionVerdict(is_event=True, confidence=0.9, frame_index=1)
    cfg = _cfg(frames_per_window=3, add_confidence_threshold=0.75)  # offsets: [2, 6, 10]

    peak = decide_scan(gap, verdict, cfg)

    assert peak == GlobalPeak(time_seconds=106.0, score=0.9)


def test_refine_with_vision_drops_confirmed_fp_and_adds_scan_hit():
    # Two audio-flagged intervals: one real (kept), one a confident FP (dropped).
    intervals = [
        Interval(start_seconds=10.0, end_seconds=20.0, peaks=[GlobalPeak(time_seconds=15.0, score=1.0)]),
        Interval(start_seconds=100.0, end_seconds=110.0, peaks=[GlobalPeak(time_seconds=105.0, score=1.0)]),
    ]
    timeline_cfg = TimelineConfig(
        lookback_seconds=6.0, post_peak_seconds=5.0, min_gap_seconds=5.0, min_interval_seconds=1.0, warmup_seconds=0.0
    )
    vision_cfg = _cfg(
        frames_per_window=3, drop_confidence_threshold=0.75, add_confidence_threshold=0.75,
        scan_chunk_min_seconds=5.0, scan_chunk_max_seconds=200.0,
    )
    game_duration = 400.0

    def fake_confirm(interval, chunks, cfg):
        if interval.start_seconds == 10.0:
            return VisionVerdict(is_event=True, confidence=0.95, rationale="real shot")
        return VisionVerdict(is_event=False, confidence=0.9, rationale="practice shot during break")

    # One gap will span [20.0, 100.0] (80s) and another [110.0, 400.0] (290s,
    # chunked by scan_chunk_max_seconds=200.0). Only flag an event in the
    # first gap, at its middle frame.
    def fake_scan(gap, chunks, cfg):
        if gap.start_seconds == 20.0:
            return VisionVerdict(is_event=True, confidence=0.9, frame_index=1, rationale="quiet far-side goal")
        return VisionVerdict(is_event=False, confidence=0.9, rationale="nothing here")

    refined, log = refine_with_vision(
        intervals, chunks=[], game_duration_seconds=game_duration, timeline_cfg=timeline_cfg, vision_cfg=vision_cfg,
        classify_confirm_fn=fake_confirm, classify_scan_fn=fake_scan,
    )

    starts = sorted(iv.start_seconds for iv in refined)
    # kept: the real audio interval [10, 20]; dropped: the FP at [100, 110];
    # added: a new windowed interval around the scan hit in gap [20, 100].
    assert len(refined) == 2
    assert starts[0] == 10.0
    assert not any(iv.start_seconds == 100.0 for iv in refined)

    confirm_entries = [e for e in log.entries if e.kind == "confirm"]
    scan_entries = [e for e in log.entries if e.kind == "scan"]
    assert len(confirm_entries) == 2
    assert any(e.kept for e in confirm_entries)
    assert any(not e.kept for e in confirm_entries)
    assert any(e.kept for e in scan_entries)  # the flagged gap
