import json
from pathlib import Path

from soccer_highlights.label_audit import (
    AuditRow,
    DescribeResult,
    JudgeVerdict,
    LabeledRow,
    _parse_describe_json,
    _parse_judge_json,
    is_flagged,
    parse_review_sheet_rows,
    run_audit,
    run_describe_only,
    sort_by_disagreement,
    write_report_csv,
)
from soccer_highlights.timeline import Interval


def test_parse_review_sheet_rows_builds_labeled_rows():
    csv_rows = [
        {"clip_file": "clip_001.mp4", "start_seconds": "10.0", "end_seconds": "20.0", "verdict": "TP", "notes": "nice shot"},
        {"clip_file": "clip_002.mp4", "start_seconds": "30.0", "end_seconds": "40.0", "verdict": "fp", "notes": ""},
    ]
    strategy_dir = Path("output/review/strike_loose")
    rows = parse_review_sheet_rows("strike_loose", csv_rows, strategy_dir)

    assert len(rows) == 2
    assert rows[0] == LabeledRow(
        strategy="strike_loose", clip_file="clip_001.mp4", clip_path=strategy_dir / "clip_001.mp4",
        interval=Interval(start_seconds=10.0, end_seconds=20.0), verdict="TP", notes="nice shot",
    )
    assert rows[1].verdict == "FP"  # normalized to uppercase


def test_parse_review_sheet_rows_skips_unlabeled_rows():
    csv_rows = [{"clip_file": "clip_001.mp4", "start_seconds": "0", "end_seconds": "10", "verdict": "", "notes": ""}]
    assert parse_review_sheet_rows("negatives", csv_rows, Path("output/review/negatives")) == []


def test_parse_describe_json_well_formed():
    raw = json.dumps({"score": 4, "caption": "Goal from close range", "description": "A clear goal is scored.", "rationale": "clear goal"})
    result = _parse_describe_json(raw)
    assert result == DescribeResult(score=4, caption="Goal from close range", description="A clear goal is scored.", rationale="clear goal")


def test_parse_describe_json_strips_code_fence():
    raw = "```json\n" + json.dumps({"score": 1, "caption": "Break in play", "description": "Players stand around."}) + "\n```"
    result = _parse_describe_json(raw)
    assert result.score == 1
    assert result.caption == "Break in play"


def test_parse_describe_json_empty_description_raises():
    try:
        _parse_describe_json(json.dumps({"score": 2, "caption": "x", "description": "   "}))
        assert False, "expected an exception"
    except ValueError:
        pass


def test_parse_describe_json_empty_caption_raises():
    try:
        _parse_describe_json(json.dumps({"score": 2, "caption": "  ", "description": "something happens"}))
        assert False, "expected an exception"
    except ValueError:
        pass


def test_parse_describe_json_score_out_of_range_raises():
    try:
        _parse_describe_json(json.dumps({"score": 6, "caption": "x", "description": "y"}))
        assert False, "expected an exception"
    except ValueError:
        pass


def test_parse_judge_json_well_formed():
    raw = json.dumps({"agreement": "consistent", "distance_score": 0.1, "rationale": "both agree"})
    verdict = _parse_judge_json(raw)
    assert verdict.agreement == "consistent"
    assert verdict.distance_score == 0.1


def test_parse_judge_json_invalid_agreement_raises():
    try:
        _parse_judge_json(json.dumps({"agreement": "sort_of", "distance_score": 0.5}))
        assert False, "expected an exception"
    except ValueError:
        pass


def test_parse_judge_json_distance_out_of_range_raises():
    try:
        _parse_judge_json(json.dumps({"agreement": "consistent", "distance_score": 1.5}))
        assert False, "expected an exception"
    except ValueError:
        pass


def _row(strategy="strike_loose", clip="clip_001.mp4", verdict="TP") -> LabeledRow:
    return LabeledRow(
        strategy=strategy, clip_file=clip, clip_path=Path(f"output/review/{strategy}/{clip}"),
        interval=Interval(0.0, 10.0), verdict=verdict, notes="",
    )


def _describe(score=4, caption="a caption", description="a description") -> DescribeResult:
    return DescribeResult(score=score, caption=caption, description=description, rationale="")


def test_is_flagged_none_judge_is_flagged():
    assert is_flagged(None, threshold=0.5) is True


def test_is_flagged_consistent_low_distance_not_flagged():
    verdict = JudgeVerdict(agreement="consistent", distance_score=0.1, rationale="")
    assert is_flagged(verdict, threshold=0.5) is False


def test_is_flagged_consistent_but_high_distance_is_flagged():
    verdict = JudgeVerdict(agreement="consistent", distance_score=0.6, rationale="")
    assert is_flagged(verdict, threshold=0.5) is True


def test_is_flagged_inconsistent_agreement_is_flagged_regardless_of_distance():
    verdict = JudgeVerdict(agreement="human_likely_wrong", distance_score=0.2, rationale="")
    assert is_flagged(verdict, threshold=0.5) is True


def test_sort_by_disagreement_orders_descending_and_puts_failures_first():
    rows = [
        AuditRow(row=_row(clip="low.mp4"), description=_describe(), judge=JudgeVerdict("consistent", 0.1, "")),
        AuditRow(row=_row(clip="failed.mp4"), description=None, judge=None),
        AuditRow(row=_row(clip="high.mp4"), description=_describe(), judge=JudgeVerdict("ai_likely_wrong", 0.9, "")),
    ]
    ordered = sort_by_disagreement(rows)
    clip_order = [ar.row.clip_file for ar in ordered]
    assert clip_order[0] in ("failed.mp4", "high.mp4")  # both are distance 1.0 vs 0.9, failed treated as 1.0
    assert clip_order[-1] == "low.mp4"


def test_run_audit_calls_describe_then_judge_and_caches(tmp_path):
    rows = [_row(clip="clip_001.mp4", verdict="TP"), _row(clip="clip_002.mp4", verdict="FP")]
    describe_calls = []
    judge_calls = []

    def fake_describe(clip_path, duration, cfg):
        describe_calls.append(clip_path.name)
        return _describe(description=f"description for {clip_path.name}")

    def fake_judge(row, description, cfg):
        judge_calls.append((row.clip_file, description))
        return JudgeVerdict(agreement="consistent", distance_score=0.0, rationale="ok")

    cache_path = tmp_path / "audit.json"
    results = run_audit(rows, gemini_cfg=None, vision_cfg=None, cache_path=cache_path, describe_fn=fake_describe, judge_fn=fake_judge)

    assert describe_calls == ["clip_001.mp4", "clip_002.mp4"]
    assert len(judge_calls) == 2
    assert all(ar.judge.agreement == "consistent" for ar in results)
    assert cache_path.exists()


def test_run_audit_retries_only_missing_judge_not_description(tmp_path):
    rows = [_row(clip="clip_001.mp4")]
    cache_path = tmp_path / "audit.json"
    cache_path.write_text(
        json.dumps([{
            "strategy": "strike_loose", "clip_file": "clip_001.mp4", "start_seconds": 0.0, "end_seconds": 10.0,
            "verdict": "TP", "notes": "",
            "describe": {"score": 4, "caption": "cap", "description": "already described", "rationale": ""},
            "judge": None,
        }]),
        encoding="utf-8",
    )
    describe_calls = []
    judge_calls = []

    def fake_describe(clip_path, duration, cfg):
        describe_calls.append(clip_path.name)
        return _describe(description="should not be called")

    def fake_judge(row, description, cfg):
        judge_calls.append(description.description)
        return JudgeVerdict(agreement="ambiguous", distance_score=0.5, rationale="")

    results = run_audit(rows, gemini_cfg=None, vision_cfg=None, cache_path=cache_path, describe_fn=fake_describe, judge_fn=fake_judge)

    assert describe_calls == []  # description was cached, not re-fetched
    assert judge_calls == ["already described"]  # judge was retried with the cached description
    assert results[0].judge.agreement == "ambiguous"


def test_write_report_csv_includes_blank_new_label_columns(tmp_path):
    audit_rows = [
        AuditRow(row=_row(), description=_describe(score=5, caption="a caption", description="a description"), judge=JudgeVerdict("consistent", 0.1, "matches")),
    ]
    out_path = tmp_path / "report.csv"
    write_report_csv(audit_rows, out_path)

    content = out_path.read_text(encoding="utf-8")
    assert "new_label" in content
    assert "new_notes" in content
    assert "gemini_score" in content
    assert "gemini_caption" in content
    assert "a description" in content
    assert "a caption" in content
    assert content.strip().endswith(",")  # last two columns (new_label, new_notes) are blank


def test_run_describe_only_calls_describe_for_every_row_no_judge(tmp_path):
    rows = [_row(clip="clip_001.mp4"), _row(clip="clip_002.mp4")]
    describe_calls = []

    def fake_describe(clip_path, duration, cfg):
        describe_calls.append(clip_path.name)
        return _describe(description=f"description for {clip_path.name}")

    cache_path = tmp_path / "descriptions.json"
    results = run_describe_only(rows, gemini_cfg=None, cache_path=cache_path, describe_fn=fake_describe)

    assert describe_calls == ["clip_001.mp4", "clip_002.mp4"]
    assert [r.description for r in results] == ["description for clip_001.mp4", "description for clip_002.mp4"]
    assert cache_path.exists()


def test_run_describe_only_resumes_and_retries_only_failed_entries(tmp_path):
    rows = [_row(clip="clip_001.mp4"), _row(clip="clip_002.mp4")]
    cache_path = tmp_path / "descriptions.json"
    cache_path.write_text(
        json.dumps(
            [
                {"strategy": "strike_loose", "clip_file": "clip_001.mp4", "start_seconds": 0.0, "end_seconds": 10.0,
                 "describe": {"score": 3, "caption": "cap1", "description": "already done", "rationale": ""}},
                {"strategy": "strike_loose", "clip_file": "clip_002.mp4", "start_seconds": 0.0, "end_seconds": 10.0, "describe": None},
            ]
        ),
        encoding="utf-8",
    )
    describe_calls = []

    def fake_describe(clip_path, duration, cfg):
        describe_calls.append(clip_path.name)
        return _describe(description="freshly described")

    results = run_describe_only(rows, gemini_cfg=None, cache_path=cache_path, describe_fn=fake_describe)

    assert describe_calls == ["clip_002.mp4"]  # only the previously-failed one was retried
    assert [r.description for r in results] == ["already done", "freshly described"]


def test_run_describe_only_detects_cache_mismatch(tmp_path):
    rows = [_row(clip="clip_999.mp4")]
    cache_path = tmp_path / "descriptions.json"
    cache_path.write_text(
        json.dumps([{"strategy": "strike_loose", "clip_file": "clip_001.mp4", "start_seconds": 0.0, "end_seconds": 10.0, "describe": None}]),
        encoding="utf-8",
    )
    try:
        run_describe_only(rows, gemini_cfg=None, cache_path=cache_path, describe_fn=lambda p, d, g: _describe())
        assert False, "expected a cache mismatch error"
    except ValueError:
        pass
