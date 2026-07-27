import json

from soccer_highlights.golden import GoldenEvent
from soccer_highlights.timeline import Interval
from soccer_highlights.vision import VisionVerdict
from soccer_highlights.vision_eval import collect_verdicts, sweep_drop_threshold


def test_collect_verdicts_calls_classify_fn_and_caches(tmp_path):
    intervals = [Interval(start_seconds=0.0, end_seconds=10.0), Interval(start_seconds=20.0, end_seconds=30.0)]
    calls = []

    def fake_classify(interval, chunks):
        calls.append(interval.start_seconds)
        return VisionVerdict(is_event=True, confidence=0.9, rationale="real", caption="Shot on goal")

    cache_path = tmp_path / "verdicts.json"
    results = collect_verdicts(intervals, chunks=[], classify_fn=fake_classify, cache_path=cache_path)

    assert calls == [0.0, 20.0]
    assert len(results) == 2
    assert all(v.is_event for v in results)
    assert cache_path.exists()


def test_collect_verdicts_resumes_from_cache_without_recalling(tmp_path):
    intervals = [Interval(start_seconds=0.0, end_seconds=10.0), Interval(start_seconds=20.0, end_seconds=30.0)]
    cache_path = tmp_path / "verdicts.json"
    cache_path.write_text(
        json.dumps(
            [
                {
                    "start_seconds": 0.0,
                    "end_seconds": 10.0,
                    "verdict": {"is_event": False, "confidence": 0.8, "frame_index": None, "rationale": "fp", "caption": ""},
                }
            ]
        ),
        encoding="utf-8",
    )
    calls = []

    def fake_classify(interval, chunks):
        calls.append(interval.start_seconds)
        return VisionVerdict(is_event=True, confidence=0.9)

    results = collect_verdicts(intervals, chunks=[], classify_fn=fake_classify, cache_path=cache_path)

    assert calls == [20.0]  # only the uncached interval was classified
    assert results[0].is_event is False  # loaded from cache
    assert results[1].is_event is True  # freshly classified


def test_collect_verdicts_retries_cached_none_entries(tmp_path):
    intervals = [Interval(start_seconds=0.0, end_seconds=10.0), Interval(start_seconds=20.0, end_seconds=30.0)]
    cache_path = tmp_path / "verdicts.json"
    cache_path.write_text(
        json.dumps(
            [
                {"start_seconds": 0.0, "end_seconds": 10.0, "verdict": None},  # a failed/rate-limited attempt
                {
                    "start_seconds": 20.0,
                    "end_seconds": 30.0,
                    "verdict": {"is_event": True, "confidence": 0.9, "frame_index": None, "rationale": "", "caption": ""},
                },
            ]
        ),
        encoding="utf-8",
    )
    calls = []

    def fake_classify(interval, chunks):
        calls.append(interval.start_seconds)
        return VisionVerdict(is_event=False, confidence=0.95, rationale="now succeeded")

    results = collect_verdicts(intervals, chunks=[], classify_fn=fake_classify, cache_path=cache_path)

    assert calls == [0.0]  # only the previously-failed (None) entry was retried
    assert results[0].rationale == "now succeeded"
    assert results[1].is_event is True  # real cached verdict reused, not retried


def test_collect_verdicts_detects_cache_mismatch(tmp_path):
    intervals = [Interval(start_seconds=99.0, end_seconds=110.0)]
    cache_path = tmp_path / "verdicts.json"
    cache_path.write_text(
        json.dumps([{"start_seconds": 0.0, "end_seconds": 10.0, "verdict": None}]), encoding="utf-8"
    )

    try:
        collect_verdicts(intervals, chunks=[], classify_fn=lambda i, c: None, cache_path=cache_path)
        assert False, "expected a cache mismatch error"
    except ValueError:
        pass


def test_collect_verdicts_caches_none_verdicts_too(tmp_path):
    intervals = [Interval(start_seconds=0.0, end_seconds=10.0)]
    cache_path = tmp_path / "verdicts.json"

    results = collect_verdicts(intervals, chunks=[], classify_fn=lambda i, c: None, cache_path=cache_path)

    assert results == [None]
    cached = json.loads(cache_path.read_text())
    assert cached[0]["verdict"] is None


def test_sweep_drop_threshold_matches_decide_confirm_behavior():
    golden = [GoldenEvent(time_seconds=5.0, sources=[]), GoldenEvent(time_seconds=25.0, sources=[])]
    intervals = [
        Interval(start_seconds=0.0, end_seconds=10.0),  # contains golden[0], real event (always kept)
        Interval(start_seconds=20.0, end_seconds=30.0),  # contains golden[1], but a confident false-positive verdict
    ]
    verdicts = [
        VisionVerdict(is_event=True, confidence=0.4),
        VisionVerdict(is_event=False, confidence=0.8),
    ]

    points = sweep_drop_threshold(intervals, verdicts, golden, thresholds=[0.9, 0.5])

    # threshold=0.9: 0.8 confidence FP doesn't clear it -> both kept
    assert points[0].kept == 2
    assert points[0].score.false_negatives == 0
    # threshold=0.5: 0.8 confidence FP clears it -> dropped, golden[1] now missed
    assert points[1].kept == 1
    assert points[1].score.false_negatives == 1
